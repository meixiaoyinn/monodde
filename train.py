# import time
#
# import mindspore as ms
# import mindspore.nn as nn
import mindspore.communication as comm
# from mindspore import amp
# from mindspore.amp import init_status, all_finite
from config import cfg
import datetime
from tqdm import tqdm
import logging
import shutil
import warnings
from model_utils.config import config,default_setup
# from utils.backup_files import sync_root
from src.kitti_dataset import create_kitti_dataset
from model_utils.utils import *
from src.monodde import *
from src.optimizer import *
from src.eval_net import EvalWrapper


# ms.set_seed(1)
# numpy.random.seed(1)
# random.seed(1)


def init_distribute():
    if cfg.is_distributed:
        comm.init()
        cfg.rank = comm.get_rank()   #获取当前进程的排名
        cfg.group_size = comm.get_group_size()  #获取当前通信组大小
        cfg.local_rank=comm.get_local_rank()
        ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                     device_num=cfg.group_size)   #配置自动并行计算  parameter_broadcast=True
        cfg.SOLVER.IMS_PER_BATCH=cfg.SOLVER.IMS_PER_BATCH // cfg.group_size
    else:
        cfg.MODEL.USE_SYNC_BN = False
    return cfg


def setup(args):
    '''load default config from config\defaults'''
    cfg.merge_from_file(args.config_path)
    # cfg.merge_from_list(args.opts)

    cfg.SEED = args.seed
    cfg.DATASETS.DATASET = args.dataset
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.DATALOADER.NUM_WORKERS = args.num_work
    cfg.TEST.EVAL_DIS_IOUS = args.eval_iou
    cfg.TEST.EVAL_DEPTH = args.eval_depth
    cfg.TEST.SURVEY_DEPTH = args.survey_depth

    cfg.MODEL.COOR_ATTRIBUTE = args.Coor_Attribute
    cfg.MODEL.COOR_UNCERN = args.Coor_Uncern
    cfg.MODEL.GRM_ATTRIBUTE = args.GRM_Attribute
    cfg.MODEL.GRM_UNCERN = args.GRM_Uncern
    cfg.MODEL.BACKBONE.CONV_BODY = args.backbone
    cfg.MODEL.DEVICE=args.device

    if args.vis_thre > 0:
        cfg.TEST.VISUALIZE_THRESHOLD = args.vis_thre

    if args.output is not None:
        cfg.OUTPUT_DIR = args.output

    if args.test:
        cfg.DATASETS.TEST_SPLIT = 'test'
        cfg.DATASETS.TEST = ("kitti_test",)
    cfg.is_training=args.is_training
    cfg.MODEL.PRETRAIN=args.pretrained
    cfg.is_distributed = args.distributed

    if args.demo:
        cfg.DATASETS.TRAIN = ("kitti_demo",)
        cfg.DATASETS.TEST = ("kitti_demo",)

    if args.data_root is not None:
        cfg.DATASETS.DATA_ROOT = args.data_root

    if args.debug:
        cfg.DATALOADER.NUM_WORKERS = 0
        cfg.TEST.DEBUG = args.debug

    cfg.START_TIME = datetime.datetime.strftime(datetime.datetime.now(), '%m-%d %H:%M:%S')
    default_setup(cfg, args)
    return cfg


_grad_scale = ops.MultitypeFuncGraph("grad_scale")


# class ClipGradients(nn.Cell):
#     """
#     Clip gradients.
#     Inputs:
#         grads (tuple[Tensor]): Gradients.
#         clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
#         clip_value (float): Specifies how much to clip.
#     Outputs:
#         tuple[Tensor], clipped gradients.
#     """
#     def __init__(self):
#         super(ClipGradients, self).__init__()
#         self.clip_by_norm = nn.ClipByNorm()
#         self.cast = ops.Cast()
#         self.dtype = ops.DType()
#
#     def construct(self,grads,clip_type,clip_value):
#         if clip_type != 0 and clip_type != 1:
#             return grads
#         new_grads = ()
#         for grad in grads:
#             dt = self.dtype(grad)
#             if clip_type == 0:
#                 t = ops.clip_by_value(grad, self.cast(ops.tuple_to_array((clip_value,)),dt))
#             else:
#                 t = self.clip_by_norm(grad, self.cast(ops.tuple_to_array((clip_value,)),dt))
#                 new_grads = new_grads + (t,)
#         return new_grads


class TrainOneStepCell(nn.TrainOneStepWithLossScaleCell):
    """
    Network training package class.

    Append an optimizer to the training network after that the construct function
    can be called to create the backward graph.

    Args:
        network (Cell): The training network.
        optimizer (Cell): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default value is 1.0.
        grad_clip (bool): Whether clip gradients. Default value is False.
    """
    def __init__(self, network, optimizer, scale_sense=1, grad_clip=False):
        if isinstance(scale_sense, (int, float)):
            scale_sense = ms.Tensor(scale_sense, ms.float32)
        super(TrainOneStepCell, self).__init__(network, optimizer, scale_sense)
        self.grad_clip = grad_clip
        # self.clip_gradients = ClipGradients()


    def construct(self, images, edge_infor, targets_heatmap, targets_variables,iteration):
        weights = self.weights
        # wmean=0
        # for i in weights:
        #     if ops.isnan(i.mean()):
        #         print(i)
        #     wmean+=i.mean()
        # print('weights',wmean/len(weights))

        loss = self.network(images, edge_infor, targets_heatmap, targets_variables,iteration)
        scaling_sens = self.scale_sense

        _, scaling_sens = self.start_overflow_check(loss, scaling_sens)

        scaling_sens_filled = ops.ones_like(loss) * ops.cast(scaling_sens, ops.dtype(loss))
        grads = self.grad(self.network, weights)(images, edge_infor, targets_heatmap, targets_variables,iteration, scaling_sens_filled)


        if loss < 10000:
            # for i in grads:
            #     print(i)
            #     if ops.isnan(i.mean()):
            #         print('have nan')
            # status = init_status()
            # is_finite = all_finite(grads, status)

            # if is_finite:
            # grads = self.hyper_map(ops.partial(_grad_scale, scaling_sens), grads)
            # apply grad reducer on grads
            # grads = self.grad_reducer(grads)
            # if self.grad_clip:
                # grads = self.clip_gradients(grads, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE)
                # grads = ops.clip_by_global_norm(grads)
            loss = ops.depend(loss, self.optimizer(grads))
        return loss


def train_preprocess():
    cfg=setup(config)
    if cfg.MODEL.DEVICE=='Ascend':
        device_id = get_device_id()
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target=cfg.MODEL.DEVICE, device_id=device_id,pynative_synchronize=True)  #pynative_synchronize=True,save_graphs=True,save_graphs_path='output/graph'
        ms.set_context(enable_compile_cache=True, compile_cache_path="output/my_compile_cache")
        # ms.set_context(mode=ms.GRAPH_MODE, device_target=cfg.MODEL.DEVICE, device_id=device_id,print_file_path='log.data')
        # ms.set_context(save_graphs=True, save_graphs_path="output/graph")
    else:
        ms.context.set_context(mode=ms.PYNATIVE_MODE, device_target=cfg.MODEL.DEVICE, device_id=0)
    device=ms.get_context("device_target")
    cfg=init_distribute()  # init distributed
    return cfg


def load_parameters(val_network, train_network):
    logging.info("Load parameters of train network")
    param_dict_new = {}
    for key, values in train_network.parameters_and_names():
        if key.startswith('moments.'):
            continue
        elif key.startswith('yolo_network.'):
            param_dict_new[key[13:]] = values
        else:
            param_dict_new[key] = values
    ms.load_param_into_net(val_network, param_dict_new)
    logging.info('Load train network success')


def get_val_dataset(cfg,is_train=False):
    datasets=create_kitti_dataset(cfg,is_train)
    return datasets


# @moxing_wrapper(pre_process=modelarts_pre_process, post_process=modelarts_post_process, pre_args=[config])
def train():
    cfg=train_preprocess()
    dataset=create_kitti_dataset(cfg,cfg.is_training)
    steps_per_epoch = dataset.get_dataset_size()
    data_loader = dataset.create_tuple_iterator(do_copy=False)

    total_iters_each_epoch = data_loader.dataset.dataset_size // cfg.SOLVER.IMS_PER_BATCH
    # use epoch rather than iterations for saving checkpoint and validation
    if cfg.SOLVER.EVAL_AND_SAVE_EPOCH:
        cfg.SOLVER.MAX_ITERATION = cfg.SOLVER.MAX_EPOCHS * total_iters_each_epoch
        cfg.SOLVER.SAVE_CHECKPOINT_INTERVAL = total_iters_each_epoch * cfg.SOLVER.SAVE_CHECKPOINT_EPOCH_INTERVAL
        cfg.SOLVER.EVAL_INTERVAL = total_iters_each_epoch * cfg.SOLVER.EVAL_EPOCH_INTERVAL
        cfg.SOLVER.STEPS = [total_iters_each_epoch * x for x in cfg.SOLVER.DECAY_EPOCH_STEPS]
        cfg.SOLVER.WARMUP_STEPS = cfg.SOLVER.WARMUP_EPOCH * total_iters_each_epoch
    # cfg.freeze()

    meters = MetricLogger(delimiter=" ",)

    network = Mono_net(cfg)
    # network=amp.auto_mixed_precision(network, "O2")
    #val_network = Mono_net(cfg)
    # network = MonoddeWithLossCell(network,cfg)
    loss_fn=LossNet()
    opt=get_optim(cfg,network,steps_per_epoch)
    # network = nn.TrainOneStepCell(network, opt,sens=2)
    # network=TrainOneStepCell(network, opt, scale_sense=cfg.SOLVER.loss_scale,grad_clip=True)

    network.set_train()
    logger = logging.getLogger("monoflex.trainer")
    logger.info("Start training")
    max_iter = cfg.SOLVER.MAX_ITERATION
    start_training_time = time.time()
    end = time.time()
    ckpt_queue = deque()

    # ds_val = get_val_dataset(cfg)
    # eval_wrapper = EvalWrapper(cfg, val_network,ds_val)

    default_depth_method = cfg.MODEL.HEAD.OUTPUT_DEPTH
    if cfg.local_rank == 0:
        best_mAP = 0
        best_result_str = None
        best_iteration = 0
        eval_iteration = 0
        record_metrics = ['Car_bev_', 'Car_3d_']

    iter_per_epoch=cfg.SOLVER.IMS_PER_BATCH

    # summary_collect_frequency = 1

    # Define forward function
    def forward_fn(images, edge_infor, targets_heatmap, targets_original,targets_select,calibs,iteration):
        # images = ops.transpose(images, (0, 3, 1, 2))
        output = network(images, edge_infor,targets_heatmap,targets_original,targets_select,calibs,iteration)
        # weights=network.weight
        loss = loss_fn(output)
        print("loss = {}".format(loss))
        return loss,output

    # Get gradient function
    grad_fn = ops.value_and_grad(forward_fn, None, opt.parameters, has_aux=True)

    # @ms.jit
    def train_step(images, edge_infor, targets_heatmap, targets_original,targets_select,calibs,iteration):
        (loss,_), grads = grad_fn(images, edge_infor, targets_heatmap, targets_original,targets_select,calibs,iteration)
        # if loss.asnumpy() != float("inf") and loss.asnumpy() <10000:
        # loss = ops.depend(loss, opt(grads))
        return loss

    # with ms.SummaryRecord('./summary_dir/summary_04', network=network) as summary_record:
    for data, iteration in tqdm(zip(data_loader,range(0, max_iter))):
        data_time = time.time() - end
        images, edge_infor, targets_heatmap, targets_original,targets_select,calibs = prepare_targets(data,cfg)
        if iteration>=2:
            print(iteration)
        loss = train_step(images, edge_infor, targets_heatmap, targets_original,targets_select,calibs,iteration)
        meters.update(loss=loss.asnumpy())
        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)
        # if iteration % summary_collect_frequency == 0:
        #     summary_record.add_value('scalar', 'loss', loss)
        #     summary_record.record(iteration)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        if iteration % 10 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.8f} \n",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=cfg.SOLVER.BASE_LR,
                )
            )

        if cfg.rank == 0 and (iteration % cfg.SOLVER.SAVE_CHECKPOINT_INTERVAL == 0):
            logger.info('iteration = {}, saving checkpoint ...'.format(iteration))
            ckpt_name = os.path.join(cfg.OUTPUT_DIR, "MonoDDE_{}_{}.ckpt".format(iteration, cfg.SOLVER.IMS_PER_BATCH))
            ms.save_checkpoint(network, ckpt_name)
            if len(ckpt_queue) == cfg.SOLVER.SAVE_CHECKPOINT_MAX_NUM:
                ckpt_to_remove = ckpt_queue.popleft()
                # shutil.rmtree(ckpt_to_remove)
            ckpt_queue.append(ckpt_name)
        if iteration == max_iter and cfg.rank == 0:
            ckpt_name = os.path.join(cfg.OUTPUT_DIR,"MonoDDE_{}_{}.ckpt".format(iteration + 1, iter_per_epoch))
            ms.save_checkpoint(network, ckpt_name)

        if iteration % cfg.SOLVER.EVAL_INTERVAL == 0 and iteration<0:
            if cfg.SOLVER.EVAL_AND_SAVE_EPOCH:
                cur_epoch = iteration // iter_per_epoch
                logger.info('epoch = {}, evaluate model on validation set with depth {}'.format(cur_epoch,
                                                                                                default_depth_method))
            else:
                logger.info('iteration = {}, evaluate model on validation set with depth {}'.format(iteration,
                                                                                                    default_depth_method))

            val_types = ("detection",)
            dataset_name = cfg.DATASETS.TEST[0]

            if cfg.OUTPUT_DIR:
                output_folder = os.path.join(cfg.OUTPUT_DIR, dataset_name, "inference_{}".format(iteration))
                os.makedirs(output_folder, exist_ok=True)
            #load_parameters(val_network, train_network=network)
            # result_dict, result_str, dis_ious  = eval_wrapper.inference(cur_epoch=iteration + 1, cur_step=1)
            #result_dict, dis_ious = eval_wrapper.inference(iteration)

            if comm.get_local_rank() == 0:
                # only record more accurate R40 results
                result_dict = result_dict[0]

                # record the best model according to the AP_3D, Car, Moderate, IoU=0.7
                important_key = '{}_3d_{:.2f}/moderate'.format('Car', 0.7)
                eval_mAP = float(result_dict[important_key])
                if eval_mAP >= best_mAP:
                    # save best mAP and corresponding iterations
                    best_mAP = eval_mAP
                    best_iteration = iteration
                    # best_result_str = result_str
                    ckpt_name = os.path.join(cfg.OUTPUT_DIR,"model_moderate_best_{}.ckpt".format(default_depth_method))
                    ms.save_checkpoint(network, ckpt_name)
            #
                    if cfg.SOLVER.EVAL_AND_SAVE_EPOCH:
                        logger.info(
                            'epoch = {}, best_mAP = {:.2f}, updating best checkpoint for depth {} \n'.format(
                                cur_epoch, eval_mAP, default_depth_method))
                    else:
                        logger.info(
                            'iteration = {}, best_mAP = {:.2f}, updating best checkpoint for depth {} \n'.format(
                                iteration, eval_mAP, default_depth_method))

                eval_iteration += 1
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    if cfg.rank == 0:
        logger.info(
            "Total training time: {} ({:.4f} s / it), best model is achieved at iteration = {}".format(
                total_time_str, total_training_time / (max_iter), best_iteration,
            )
        )

        logger.info('The best performance is as follows')
        logger.info('\n' + best_result_str)


if __name__ == '__main__':
    # ms.set_context(save_graphs=True, save_graphs_path="src")
    train()