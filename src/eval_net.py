from collections import defaultdict
import logging
import os
import shutil

from tqdm import tqdm

from .evaluation.kitti.kitti_eval import generate_kitti_3d_detection

# from .loss import Anno_Encoder
from .net_utils import Converter_key2channel
from model_utils.utils import *
import mindspore as ms
import mindspore.ops as ops
from model_utils.utils import prepare_targets

from model_utils.timer import Timer, get_time_str
from .net_utils import select_point_of_interest,box_iou
from .evaluation import evaluate_python
from .loss import *


class EvalWrapper:
    def __init__(self, cfg, network, dataset):
        super(EvalWrapper, self).__init__()
        self.ms_type=ms.float32
        self.cfg = cfg
        self.network = network
        self.dataset = dataset[0]
        self.device_num = cfg.group_size
        # self.anno_encoder = Anno_Encoder(cfg)
        # self.key2channel = Converter_key2channel(keys=cfg.MODEL.HEAD.REGRESSION_HEADS,
        #                                          channels=cfg.MODEL.HEAD.REGRESSION_CHANNELS)
        self.keys = [key for key_group in cfg.MODEL.HEAD.REGRESSION_HEADS for key in key_group]
        self.channels = [channel for channel_groups in cfg.MODEL.HEAD.REGRESSION_CHANNELS for channel in channel_groups]

        self.det_threshold = cfg.TEST.DETECTIONS_THRESHOLD
        self.max_detection = cfg.TEST.DETECTIONS_PER_IMG
        self.eval_dis_iou = cfg.TEST.EVAL_DIS_IOUS
        self.eval_depth = cfg.TEST.EVAL_DEPTH

        self.survey_depth = cfg.TEST.SURVEY_DEPTH
        self.depth_statistics_path = os.path.join(self.cfg.OUTPUT_DIR, 'depth_statistics')
        if self.survey_depth:
            if os.path.exists(self.depth_statistics_path):
                shutil.rmtree(self.depth_statistics_path)
            os.makedirs(self.depth_statistics_path)

        self.output_width = cfg.INPUT.WIDTH_TRAIN // cfg.MODEL.BACKBONE.DOWN_RATIO
        self.output_height = cfg.INPUT.HEIGHT_TRAIN // cfg.MODEL.BACKBONE.DOWN_RATIO
        self.output_depth = cfg.MODEL.HEAD.OUTPUT_DEPTH
        self.pred_2d = cfg.TEST.PRED_2D

        self.pred_direct_depth = 'depth' in self.keys
        self.depth_with_uncertainty = 'depth_uncertainty' in self.keys
        self.regress_keypoints = 'corner_offset' in self.keys
        self.keypoint_depth_with_uncertainty = 'corner_uncertainty' in self.keys

        self.GRM_with_uncertainty = 'GRM_uncern' in self.keys or 'GRM1_uncern' in self.keys
        self.predict_IOU3D_as_conf = 'IOU3D_predict' in self.keys

        # use uncertainty to guide the confidence
        self.uncertainty_as_conf = cfg.TEST.UNCERTAINTY_AS_CONFIDENCE
        self.uncertainty_range = cfg.MODEL.HEAD.UNCERTAINTY_RANGE

        self.variance_list = []
        # if cfg.eval_parallel:
        #     self.reduce = AllReduce()

        self.concat=ops.Concat(axis=1)
        self.tile=ops.Tile()
        self.ones=ops.Ones()
        self.cast=ops.Cast()
        self.nonzero=ops.NonZero()
        self.softmax=ops.Softmax()
        self.exp = ops.Exp()
        self.sigmoid = ops.Sigmoid()
        self.zeros = ops.Zeros()
        self.l2_norm = ops.L2Normalize()
        self.softmax_axis1 = nn.Softmax(axis=1)
        self.softmax_axis2 = nn.Softmax(axis=2)
        self.gather_nd = ops.GatherNd()
        self.atan2 = ops.Atan2()
        self.relu = ops.ReLU()
        self.expand_dims=ops.ExpandDims()
        self.reducesum = ops.ReduceSum()
        self.reducesum_t=ops.ReduceSum(True)
        self.reducemean = ops.ReduceMean()
        self.argminwithvalue=ops.ArgMinWithValue()

        self.EPS = 1e-9
        # center related
        self.num_cls = len(cfg.DATASETS.DETECT_CLASSES)
        self.min_radius = cfg.DATASETS.MIN_RADIUS
        self.max_radius = cfg.DATASETS.MAX_RADIUS
        self.center_ratio = cfg.DATASETS.CENTER_RADIUS_RATIO
        self.target_center_mode = cfg.INPUT.HEATMAP_CENTER
        # if mode == 'max', centerness is the larger value, if mode == 'area', assigned to the smaller bbox
        self.center_mode = cfg.MODEL.HEAD.CENTER_MODE

        # depth related
        self.depth_mode = cfg.MODEL.HEAD.DEPTH_MODE
        self.depth_range = ms.Tensor(cfg.MODEL.HEAD.DEPTH_RANGE, self.ms_type)
        self.depth_ref = ms.Tensor(cfg.MODEL.HEAD.DEPTH_REFERENCE, self.ms_type)

        # dimension related
        self.dim_mean = ms.Tensor(cfg.MODEL.HEAD.DIMENSION_MEAN, self.ms_type)
        self.dim_std = ms.Tensor(cfg.MODEL.HEAD.DIMENSION_STD, self.ms_type)
        self.dim_modes = cfg.MODEL.HEAD.DIMENSION_REG

        # orientation related
        self.alpha_centers = ms.Tensor(np.array([0, PI / 2, PI, - PI / 2]), self.ms_type)
        self.multibin = (cfg.INPUT.ORIENTATION == 'multi-bin')
        self.orien_bin_size = cfg.INPUT.ORIENTATION_BIN_SIZE

        # offset related
        self.offset_mean = cfg.MODEL.HEAD.REGRESSION_OFFSET_STAT[0]
        self.offset_std = cfg.MODEL.HEAD.REGRESSION_OFFSET_STAT[1]

        # output info
        self.down_ratio = cfg.MODEL.BACKBONE.DOWN_RATIO
        self.output_height = cfg.INPUT.HEIGHT_TRAIN // self.down_ratio
        self.output_width = cfg.INPUT.WIDTH_TRAIN // self.down_ratio
        self.K = self.output_width * self.output_height
        self.updates = ms.Tensor(np.array([PI]), self.ms_type)
        self.bbox_index = ms.Tensor([[4, 5, 0, 1, 6, 7, 2, 3],
                                     [0, 1, 2, 3, 4, 5, 6, 7],
                                     [4, 0, 1, 5, 6, 2, 3, 7]], ms.int32)

    def key2channel(self, key):
        # find the corresponding index
        index = self.keys.index(key)

        s = sum(self.channels[:index])
        e = s + self.channels[index]

        return slice(s, e, 1)

    def prepare_targets(self,data):
        # print('prepare_targets')
        per_batch = self.cfg.SOLVER.IMS_PER_BATCH
        down_ratio = self.cfg.MODEL.BACKBONE.DOWN_RATIO
        corner_loss_depth = self.cfg.MODEL.HEAD.CORNER_LOSS_DEPTH
        edge_infor = [data[-3], data[-2]]
        calibs = []
        for i in range(per_batch):
            calibs.append(
                dict(P=data[22][i, :, :], R0=data[23][i, :, :], C2V=data[24][i, :, :], c_u=data[25][i], c_v=data[26][i],
                     f_u=data[27][i],
                     f_v=data[28][i], b_x=data[29][i], b_y=data[30][i]))
        reg_mask = self.cast(data[9], ms.bool_)
        ori_imgs = self.cast(data[14], ms.int32)
        trunc_mask = self.cast(data[16], ms.int32)
        flatten_reg_mask_gt = reg_mask.view(-1).asnumpy().tolist()  # flatten_reg_mask_gt shape: (B * num_objs)

        # the corresponding image_index for each object, used for finding pad_size, calib and so on
        batch_idxs = ops.arange(per_batch, dtype=ms.int32).view(-1, 1).expand_as(reg_mask).reshape(-1)  # batch_idxs shape: (B * num_objs)
        batch_idxs = batch_idxs[flatten_reg_mask_gt]  # Only reserve the features of valid objects.
        valid_targets_bbox_points = data[4].view(-1, 2)[flatten_reg_mask_gt]  # valid_targets_bbox_points shape: (valid_objs, 2)

        # fcos-style targets for 2D
        target_bboxes_2D = data[12].view(-1, 4)[flatten_reg_mask_gt]  # target_bboxes_2D shape: (valid_objs, 4). 4 -> (x1, y1, x2, y2)
        target_bboxes_height = target_bboxes_2D[:, 3] - target_bboxes_2D[:, 1]  # target_bboxes_height shape: (valid_objs,)
        target_bboxes_width = target_bboxes_2D[:, 2] - target_bboxes_2D[:, 0]  # target_bboxes_width shape: (valid_objs,)

        target_regression_2D = self.concat((valid_targets_bbox_points - target_bboxes_2D[:, :2], target_bboxes_2D[:, 2:] - valid_targets_bbox_points),axis=1)  # offset to 2D bbox boundaries.
        mask_regression_2D = ops.logical_and(target_bboxes_height > 0, target_bboxes_width > 0)
        mask_regression_2D = mask_regression_2D.asnumpy().tolist()
        target_regression_2D = target_regression_2D[mask_regression_2D]  # target_regression_2D shape: (valid_objs, 4)

        # targets for 3D
        target_clses = data[3].view(-1)[flatten_reg_mask_gt]  # target_clses shape: (val_objs,)
        target_depths_3D = data[8][..., -1].view(-1)[flatten_reg_mask_gt]  # target_depths_3D shape: (val_objs,)
        target_rotys_3D = data[15].view(-1)[flatten_reg_mask_gt]  # target_rotys_3D shape: (val_objs,)
        target_alphas_3D = data[17].view(-1)[flatten_reg_mask_gt]  # target_alphas_3D shape: (val_objs,)
        target_offset_3D = data[11].view(-1, 2)[flatten_reg_mask_gt]  # The offset from target centers to projected 3D centers. target_offset_3D shape: (val_objs, 2)
        target_dimensions_3D = data[7].view(-1, 3)[flatten_reg_mask_gt]  # target_dimensions_3D shape: (val_objs, 3)

        target_orientation_3D = data[18].view(-1, data[18].shape[-1])[flatten_reg_mask_gt]  # target_orientation_3D shape: (num_objs, 8)
        target_locations_3D = decode_location_flatten(valid_targets_bbox_points, target_offset_3D,
                                                      target_depths_3D, calibs, data[13], batch_idxs,
                                                      down_ratio)  # target_locations_3D shape: (valid_objs, 3)
        target_corners_3D = encode_box3d(target_rotys_3D, target_dimensions_3D, target_locations_3D)  # target_corners_3D shape: (valid_objs, 8, 3)
        target_bboxes_3D = self.concat((target_locations_3D, target_dimensions_3D, target_rotys_3D[:, None]),axis=1)  # target_bboxes_3D shape: (valid_objs, 7)

        target_trunc_mask = trunc_mask.view(-1)[flatten_reg_mask_gt]  # target_trunc_mask shape(valid_objs,)
        obj_weights = data[10].view(-1)[flatten_reg_mask_gt]  # obj_weights shape: (valid_objs,)
        target_corner_keypoints = data[5].view(len(flatten_reg_mask_gt), -1, 3)[flatten_reg_mask_gt]  # target_corner_keypoints shape: (val_objs, 10, 3)
        target_corner_depth_mask = data[6].view(-1, 3)[flatten_reg_mask_gt]

        keypoints_visible = data[21].view(-1, data[21].shape[-1])[flatten_reg_mask_gt]  # keypoints_visible shape: (valid_objs, 11)
        if corner_loss_depth == 'GRM':
            keypoints_visible = self.tile(self.expand_dims(keypoints_visible, 2), (1, 1, 2)).reshape((keypoints_visible.shape[0], -1))  # The effectness of first 22 GRM equations.
            GRM_valid_items = self.concat((keypoints_visible, ops.ones((keypoints_visible.shape[0], 3), ms.bool_)))  # GRM_valid_items shape: (valid_objs, 25)
        elif corner_loss_depth == 'soft_GRM':
            keypoints_visible = self.tile(self.expand_dims(keypoints_visible[:, 0:8], 2), (1, 1, 2)).reshape((keypoints_visible.shape[0],-1))  # The effectiveness of the first 16 equations. shape: (valid_objs, 16)
            direct_depth_visible = self.ones((keypoints_visible.shape[0], 1), ms.bool_)
            veritical_group_visible = self.cast(data[6].view(-1, 3)[flatten_reg_mask_gt], ms.bool_)  # veritical_group_visible shape: (valid_objs, 3)
            GRM_valid_items = self.concat((self.cast(keypoints_visible, ms.bool_), direct_depth_visible, veritical_group_visible),axis=1)  # GRM_valid_items shape: (val_objs, 20)
        else:
            GRM_valid_items = None
            # preparing outputs
        return_dict = {'cls_ids': data[3], 'pad_size': data[13], 'target_centers': data[4], 'calib': calibs,
                       'reg_2D': target_regression_2D, 'offset_3D': target_offset_3D, 'depth_3D': target_depths_3D,
                       'orien_3D': target_orientation_3D, 'valid_targets_bbox_points': valid_targets_bbox_points,
                       'dims_3D': target_dimensions_3D, 'corners_3D': target_corners_3D,
                       'width_2D': target_bboxes_width,
                       'rotys_3D': target_rotys_3D, 'target_clses': target_clses,
                       'cat_3D': target_bboxes_3D, 'trunc_mask_3D': target_trunc_mask,
                       'height_2D': target_bboxes_height,
                       'GRM_valid_items': GRM_valid_items.asnumpy().tolist(),
                       'target_corner_depth_mask': target_corner_depth_mask,
                       'locations': target_locations_3D, 'obj_weights': obj_weights,
                       'target_corner_keypoints': target_corner_keypoints, 'mask_regression_2D': mask_regression_2D,
                       'flatten_reg_mask_gt': flatten_reg_mask_gt, 'batch_idxs': batch_idxs, 'keypoints': data[5],
                       'keypoints_depth_mask': data[6],
                       'ori_imgs': ori_imgs
                       }
        # return_dict = dict(cls_ids=data[3], target_centers=data[4], bboxes=data[12], keypoints=data[5],
        #                    dimensions=data[7],
        #                    locations=data[8], rotys=data[15], alphas=data[17], calib=calibs, pad_size=data[13],
        #                    reg_mask=reg_mask, reg_weight=data[10],
        #                    offset_3D=data[11], ori_imgs=ori_imgs, trunc_mask=trunc_mask, orientations=data[18],
        #                    keypoints_depth_mask=data[6],
        #                    GRM_keypoints_visible=data[21]
        #                    )
        flatten_mask_sum = Tensor(np.array(flatten_reg_mask_gt).sum(), ms.int32)
        GRM_valid_items_sum = Tensor(GRM_valid_items.asnumpy().sum(), ms.int32)
        GRM_valid_items_inverssum = Tensor((~GRM_valid_items).asnumpy().sum(), ms.int32)
        sums = ops.stack((flatten_mask_sum, GRM_valid_items_sum, GRM_valid_items_inverssum))
        return data[0], edge_infor, data[19], return_dict, sums

    def synchronize(self):
        sync = Tensor(np.array([1]).astype(np.int32))
        # sync = self.reduce(sync)    # For synchronization
        sync = sync.asnumpy()[0]
        if sync != self.device_num:
            raise ValueError(
                f"Sync value {sync} is not equal to number of device {self.device_num}. "
                f"There might be wrong with devices."
            )

    def inference(self, iteration, save_all_results=True, metrics=['R40'], dataset_name='kitti'):
        if self.cfg.OUTPUT_DIR:
            output_folder = os.path.join(self.cfg.OUTPUT_DIR, dataset_name, "inference_{}".format(iteration))
            os.makedirs(output_folder, exist_ok=True)
        num_devices = self.cfg.group_size
        logger = logging.getLogger("monoflex.inference")

        logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(self.dataset)))
        predict_folder = os.path.join(output_folder, 'data')

        if os.path.exists(predict_folder):
            shutil.rmtree(predict_folder)
        os.makedirs(predict_folder, exist_ok=True)

        total_timer = Timer()
        inference_timer = Timer()
        total_timer.tic()

        dis_ious = defaultdict(list)
        depth_errors = defaultdict(list)
        for index, data in enumerate(tqdm(self.dataset.create_dict_iterator(output_numpy=True, num_epochs=1))):
            output, eval_utils, visualize_preds, image_ids = self.inference_once(data, iteration)
            dis_iou = eval_utils['dis_ious']
            if dis_iou is not None:
                for key in dis_iou: dis_ious[key] += dis_iou[key].tolist()

            depth_error = eval_utils['depth_errors']
            if depth_error is not None:
                for key in depth_error: depth_errors[key] += depth_error[key].tolist()

            # if vis:
            #     show_image_with_boxes(vis_target.get_field('ori_img'), output, vis_target,
            #                           visualize_preds, vis_scores=eval_utils['vis_scores'],
            #                           vis_save_path=os.path.join(vis_folder, image_ids[0] + '.png'))

            # For the validation of the training phase, $save_all_results is True and all files should be saved.
            # During the validation of evaluation phase, $save_all_results is False and only the files containining detected tartgets are saved.
            if save_all_results or output.shape[0] != 0:
                predict_txt = image_ids.asnumpy()[0] + '.txt'
                predict_txt = os.path.join(predict_folder, predict_txt)
                generate_kitti_3d_detection(output, predict_txt)
            # disentangling IoU
        for key, value in dis_ious.items():
            mean_iou = sum(value) / len(value)
            dis_ious[key] = mean_iou

        for key, value in depth_errors.items():
            value = np.array(value)
            value[value > 1] = 1  # Limit the uncertainty below 1. Some estimated value could be really large.
            value = value.tolist()
            mean_error = sum(value) / len(value)
            depth_errors[key] = mean_error

        for key, value in dis_ious.items():
            logger.info("{}, MEAN IOU = {:.4f}".format(key, value))

        for key, value in depth_errors.items():
            logger.info("{}, MEAN ERROR/UNCERTAINTY = {:.4f}".format(key, value))

        total_time = total_timer.toc()
        total_time_str = get_time_str(total_time)
        logger.info(
            "Total run time: {} ({} s / img per device, on {} devices)".format(
                total_time_str, total_time * num_devices / len(self.dataset), num_devices
            )
        )

        if save_all_results is False:
            return None, None, None

        total_infer_time = get_time_str(inference_timer.total_time)
        logger.info(
            "Model inference time: {} ({} s / img per device, on {} devices)".format(
                total_infer_time,
                inference_timer.total_time * num_devices / len(self.dataset),
                num_devices,
            )
        )
        # if not comm.is_main_process():
        #     return None, None, None

        logger.info('Finishing generating predictions, start evaluating ...')
        ret_dicts = []

        # ! We observe bugs in the evaluation code of MonoFlex. Thus, the results of this inference process should only be for inference.
        for metric in metrics:
            root = './kitti\\training'
            result, ret_dict = evaluate_python(label_path=os.path.join(root, "label_2"),
                                               result_path=predict_folder,
                                               label_split_file=os.path.join(root, "ImageSets",
                                                                             "{}.txt".format(self.split)),
                                               current_class=self.cfg.DATASETS.DETECT_CLASSES,
                                               metric=metric)

            logger.info('metric = {}'.format(metric))
            logger.info('\n' + result)

            ret_dicts.append(ret_dict)
        return ret_dicts, dis_ious
        # return ret_dicts, result, dis_ious

    def inference_once(self, data, iteration):
        images, edge_infor, targets_heatmap, targets_variables, val_objs = self.prepare_targets(data)
        # images, edge_infor, targets_variables,image_ids = self.prepare_targets(data)
        image_ids=data[-1]
        predictions = self.network(images, targets_variables, edge_infor, iteration)
        pred_heatmap, pred_regression = predictions['cls'], predictions['reg']
        batch = pred_heatmap.shape[0]

        calib, pad_size = targets_variables['calib'], targets_variables['pad_size']
        img_size = targets_variables['size']

        # evaluate the disentangling IoU for each components in (3D center offset, depth, dimension, orientation)
        dis_ious = self.evaluate_3D_detection(targets_variables, pred_regression) if self.eval_dis_iou else None

        # evaluate the accuracy of predicted depths
        depth_errors = self.evaluate_3D_depths(targets_variables, pred_regression) if self.eval_depth else None

        if self.survey_depth: self.survey_depth_statistics(targets_variables, pred_regression, image_ids)

        # max-pooling as nms for heat-map
        heatmap = nms_hm(pred_heatmap)
        visualize_preds = {'heat_map': pred_heatmap}

        # select top-k of the predicted heatmap
        scores, indexs, clses, ys, xs = select_topk(heatmap, K=self.max_detection)

        pred_bbox_points = self.concat([xs.view(-1, 1), ys.view(-1, 1)])
        pred_regression_pois = select_point_of_interest(batch, indexs, pred_regression).view(-1, pred_regression.shape[1])

        # For debug
        self.det_threshold = 0

        # thresholding with score
        scores = scores.view(-1)
        if self.cfg.TEST.DEBUG:
            valid_mask = scores >= 0
        else:
            valid_mask = scores >= self.det_threshold

        # No valid predictions and not the debug mode.
        if valid_mask.sum() == 0:
            result = scores.new_zeros((1, 14))
            visualize_preds['keypoints'] = scores.new_zeros((1, 20))
            visualize_preds['proj_center'] = scores.new_zeros((1, 2))
            eval_utils = {'dis_ious': dis_ious, 'depth_errors': depth_errors, 'vis_scores': scores.new_zeros((1)),
                          'uncertainty_conf': scores.new_zeros((1)), 'estimated_depth_error': scores.new_zeros((1))}

            return result, eval_utils, visualize_preds, image_ids

        scores = scores[valid_mask]

        clses = clses.view(-1)[valid_mask]
        pred_bbox_points = pred_bbox_points[valid_mask]
        pred_regression_pois = pred_regression_pois[valid_mask]

        pred_2d_reg = ops.relu(pred_regression_pois[:, self.key2channel('2d_dim')])
        pred_offset_3D = pred_regression_pois[:, self.key2channel('3d_offset')]
        pred_dimensions_offsets = pred_regression_pois[:, self.key2channel('3d_dim')]
        pred_orientation = self.concat((pred_regression_pois[:, self.key2channel('ori_cls')],
                                    pred_regression_pois[:, self.key2channel('ori_offset')]))
        visualize_preds['proj_center'] = pred_bbox_points + pred_offset_3D

        pred_box2d = self.decode_box2d_fcos(pred_bbox_points, pred_2d_reg, pad_size, img_size)
        pred_dimensions = self.decode_dimension(ops.cast(clses, ms.int32), pred_dimensions_offsets)

        if self.pred_direct_depth:
            pred_depths_offset = pred_regression_pois[:, self.key2channel('depth')].squeeze(-1)
            pred_direct_depths = self.decode_depth(pred_depths_offset)

        if self.depth_with_uncertainty:
            pred_direct_uncertainty = pred_regression_pois[:, self.key2channel('depth_uncertainty')].exp()
            visualize_preds['depth_uncertainty'] = pred_regression[:, self.key2channel('depth_uncertainty'), ...].squeeze(1)

        if self.regress_keypoints:
            pred_keypoint_offset = pred_regression_pois[:, self.key2channel('corner_offset')]
            pred_keypoint_offset = pred_keypoint_offset.view(-1, 10, 2)
            # solve depth from estimated key-points
            pred_keypoints_depths = self.decode_depth_from_keypoints_batch(pred_keypoint_offset, pred_dimensions, calib)
            visualize_preds['keypoints'] = pred_keypoint_offset

        if self.keypoint_depth_with_uncertainty:
            pred_keypoint_uncertainty = self.exp(pred_regression_pois[:, self.key2channel('corner_uncertainty')])

        estimated_depth_error = None
        # For debug
        # reg_mask_gt = targets_variables["reg_mask"]	# reg_mask_gt shape: (B, num_objs)
        # flatten_reg_mask_gt = ops.cast(reg_mask_gt.view(-1),ms.bool_)	# flatten_reg_mask_gt shape: (B * num_objs)
        # pred_bbox_points = targets_variables['target_centers'].view(-1, 2)[flatten_reg_mask_gt]	# target centers
        # pred_offset_3D = targets_variables['offset_3D'].view(-1, 2)[flatten_reg_mask_gt]	# Offset from target centers to 3D ceneters
        # pred_dimensions = targets_variables['dimensions'].view(-1, 3)[flatten_reg_mask_gt]	# dimensions
        # target_orientation_3D = targets_variables['orientations'].view(-1, 8)[flatten_reg_mask_gt]	# The label of orientation
        # pred_orientation = ops.zeros((target_orientation_3D.shape[0], 16), dtype = ms.float32)	# gt_ori_code shape: (num_objs, 16)
        # pred_orientation[:, 0:8:2] = 0.1
        # pred_orientation[:, 1:8:2] = target_orientation_3D[:, 0:4]
        # pred_orientation[:, 8::2] = ops.sin(target_orientation_3D[:, 4:8])
        # pred_orientation[:, 9::2] = ops.cos(target_orientation_3D[:, 4:8])	# Orientation
        # pred_keypoint_offset = targets_variables['keypoints'][0, :, :, 0:2]
        # pred_keypoint_offset = pred_keypoint_offset[flatten_reg_mask_gt]	# Offset from target centers to keypoints.
        # pred_direct_depths = targets_variables['locations'][0, :, -1][flatten_reg_mask_gt]	# Direct depth estimation.
        # pred_keypoints_depths = ops.expand_dims(pred_direct_depths,1).tile((1, 3))	# pred_keypoints_depths shape: (num_objs, 3)
        # clses = targets_variables['cls_ids'][0][flatten_reg_mask_gt]	# Category information
        # pred_box2d = targets_variables['bboxes'].view(-1, 4)[flatten_reg_mask_gt] # 2D bboxes
        # pred_box2d = pred_box2d * self.cfg.MODEL.BACKBONE.DOWN_RATIO - pad_size.tile((1, 2))
        # scores = ops.ones((pred_bbox_points.shape[0],), dtype = ms.float32)	# 2D confidence

        '''reg_mask_gt = targets_variables["reg_mask"]	# reg_mask_gt shape: (B, num_objs)
        flatten_reg_mask_gt = ops.cast(reg_mask_gt.view(-1),ms.bool_)	# flatten_reg_mask_gt shape: (B * num_objs)
        target_depths = targets_variables['locations'][0, :, -1][flatten_reg_mask_gt]'''

        if self.output_depth == 'GRM':
            if not self.GRM_with_uncertainty:
                raise Exception("When compute depth with GRM, the GRM head of network must be loaded.")
            GRM_uncern = pred_regression_pois[:, self.key2channel('GRM_uncern')]  # (valid_objs, 25)
            GRM_uncern = self.exp(ops.clip_by_value(GRM_uncern, self.uncertainty_range[0], self.uncertainty_range[1]))

            # For debug !!!
            # GRM_uncern = 0.01 * self.ones((pred_bbox_points.shape[0], 25),  ms.float32)

            info_dict = {'target_centers': pred_bbox_points, 'offset_3D': pred_offset_3D, 'pad_size': pad_size,
                         'calib': calib, 'batch_idxs': None}
            GRM_rotys, GRM_alphas = self.decode_axes_orientation(pred_orientation,
                                                                              dict_for_3d_center=info_dict)

            GRM_locations, _, _ = self.decode_from_GRM(self.expand_dims(GRM_rotys, 1), pred_dimensions,
                                                                    pred_keypoint_offset.reshape(-1, 20),
                                                                    pred_direct_depths.unsqueeze(1),
                                                                    targets_dict=info_dict,
                                                                    GRM_uncern=GRM_uncern)  # pred_locations_3D shape: (valid_objs, 3)
            pred_depths = GRM_locations[:, 2]

            weights = 1 / GRM_uncern  # weights shape: (total_num_objs, 25)
            weights = weights / ops.ReduceSum(True)(weights, axis=1)
            estimated_depth_error = self.reducesum(weights * GRM_uncern, axis=1)  # estimated_depth_error shape: (valid_objs,)

        elif self.output_depth == 'soft_GRM':
            if not self.GRM_with_uncertainty:
                raise Exception("When compute depth with GRM, the GRM head of network must be loaded.")
            if 'GRM_uncern' in self.keys:
                GRM_uncern = pred_regression_pois[:, self.key2channel('GRM_uncern')]  # GRM_uncern shape: (valid_objs, 20)
            elif 'GRM1_uncern' in self.keys:
                uncern_GRM1 = pred_regression_pois[:, self.key2channel('GRM1_uncern')]  # uncern_GRM1 shape: (valid_objs, 8)
                uncern_GRM2 = pred_regression_pois[:, self.key2channel('GRM2_uncern')]  # uncern_GRM1 shape: (valid_objs, 8)
                uncern_Mono_Direct = pred_regression_pois[:, self.key2channel('Mono_Direct_uncern')]  # uncern_Mono_Direct shape: (valid_objs, 1)
                uncern_Mono_Keypoint = pred_regression_pois[:, self.key2channel('Mono_Keypoint_uncern')]  # uncern_Mono_Keypoint shape: (valid_objs, 3)
                GRM_uncern = ops.concat((self.expand_dims(uncern_GRM1, 2), self.expand_dims(uncern_GRM2, 2)), axis=2).view(-1, 16)
                GRM_uncern = self.concat((GRM_uncern, uncern_Mono_Direct, uncern_Mono_Keypoint))  # GRM_uncern shape: (valid_objs, 20)
            GRM_uncern = self.exp(ops.clip_by_value(GRM_uncern, self.uncertainty_range[0], self.uncertainty_range[1]))
            assert GRM_uncern.shape[1] == 20

            # For debug !!!
            # GRM_uncern = 0.01 * ops.ones((pred_bbox_points.shape[0], 20), dtype = ms.float32)

            pred_combined_depths = self.concat((self.expand_dims(pred_direct_depths, 1), pred_keypoints_depths))  # pred_combined_depths shape: (valid_objs, 4)

            info_dict = {'target_centers': pred_bbox_points, 'offset_3D': pred_offset_3D, 'pad_size': pad_size,
                         'calib': calib, 'batch_idxs': None}
            GRM_rotys, GRM_alphas = self.decode_axes_orientation(pred_orientation, dict_for_3d_center=info_dict)

            # For debug !!!
            # GRM_rotys = targets_variables['rotys'].view(-1)[flatten_reg_mask_gt]

            pred_vertex_offset = pred_keypoint_offset[:, 0:8, :]  # Do not use the top center and bottom center.
            pred_depths, separate_depths = self.decode_from_SoftGRM(self.expand_dims(GRM_rotys, 1),
                                                                                 pred_dimensions,
                                                                                 pred_vertex_offset.reshape(-1, 16),
                                                                                 pred_combined_depths,
                                                                                 targets_dict=info_dict,
                                                                                 GRM_uncern=GRM_uncern)  # pred_depths shape: (total_num_objs,)

            ### For the experiments of ablation study on depth estimation ###
            '''separate_depths = ops.cat((separate_depths[:, 0:16], separate_depths[:, 19:20]), axis = 1)
            GRM_uncern = ops.cat((GRM_uncern[:, 0:16], GRM_uncern[:, 19:20]), axis = 1)
            self.variance_list.append(ops.var(separate_depths).item())
            print('Mean variance:', sum(self.variance_list) / len(self.variance_list))'''

            if self.cfg.TEST.UNCERTAINTY_3D == "GRM_uncern":
                estimated_depth_error = error_from_uncertainty(GRM_uncern)  # estimated_depth_error shape: (valid_objs,)
            elif self.cfg.TEST.UNCERTAINTY_3D == "combined_depth_uncern":
                combined_depth_uncern = pred_regression_pois[:, self.key2channel('combined_depth_uncern')]
                combined_depth_uncern = self.exp(ops.clip_by_value(combined_depth_uncern, self.uncertainty_range[0], self.uncertainty_range[1]))
                estimated_depth_error = error_from_uncertainty(combined_depth_uncern)
            elif self.cfg.TEST.UNCERTAINTY_3D == 'corner_loss_uncern':
                corner_loss_uncern = pred_regression_pois[:, self.key2channel('corner_loss_uncern')]
                corner_loss_uncern = self.exp(ops.clip_by_value(corner_loss_uncern, self.uncertainty_range[0], self.uncertainty_range[1]))
                estimated_depth_error = error_from_uncertainty(corner_loss_uncern)
            elif self.cfg.TEST.UNCERTAINTY_3D == 'uncern_soft_avg':
                GRM_error = error_from_uncertainty(GRM_uncern)

                combined_depth_uncern = pred_regression_pois[:, self.key2channel('combined_depth_uncern')]
                combined_depth_uncern = self.exp(ops.clip_by_value(combined_depth_uncern, self.uncertainty_range[0], self.uncertainty_range[1]))
                combined_depth_error = error_from_uncertainty(combined_depth_uncern)

                corner_loss_uncern = pred_regression_pois[:, self.key2channel('corner_loss_uncern')]
                corner_loss_uncern = self.exp(ops.clip_by_value(corner_loss_uncern, self.uncertainty_range[0], self.uncertainty_range[1]))
                corner_loss_error = error_from_uncertainty(corner_loss_uncern)

                estimated_depth_error = self.concat(
                    (self.expand_dims(GRM_error,1), ops.expand_dims(combined_depth_error[:GRM_error.shape[0]], 1),
                     self.expand_dims(corner_loss_error[:GRM_error.shape[0]], 1)))
                estimated_depth_error = error_from_uncertainty(estimated_depth_error)

            # Uncertainty guided pruning to filter the unreasonable estimation results.
            if self.cfg.TEST.UNCERTAINTY_GUIDED_PRUNING:
                pred_depths, _ = uncertainty_guided_prune(separate_depths, GRM_uncern, cfg=self.cfg,
                                                          depth_range=self.depth_range,
                                                          initial_use_uncern=True)

        ### For obtain the Oracle results ###
        # pred_depths, estimated_depth_error = self.get_oracle_depths(pred_box2d, clses, separate_depths, GRM_uncern, targets[0])

        elif self.output_depth == 'direct':
            pred_depths = pred_direct_depths

            if self.depth_with_uncertainty: estimated_depth_error = pred_direct_uncertainty.squeeze(axis=1)

        elif self.output_depth.find('keypoints') >= 0:
            if self.output_depth == 'keypoints_avg':
                pred_depths = self.reducemean(pred_keypoints_depths,axis=1)
                if self.keypoint_depth_with_uncertainty: estimated_depth_error = self.reducemean(pred_keypoint_uncertainty,axis=1)

            elif self.output_depth == 'keypoints_center':
                pred_depths = pred_keypoints_depths[:, 0]
                if self.keypoint_depth_with_uncertainty: estimated_depth_error = pred_keypoint_uncertainty[:, 0]

            elif self.output_depth == 'keypoints_02':
                pred_depths = pred_keypoints_depths[:, 1]
                if self.keypoint_depth_with_uncertainty: estimated_depth_error = pred_keypoint_uncertainty[:, 1]

            elif self.output_depth == 'keypoints_13':
                pred_depths = pred_keypoints_depths[:, 2]
                if self.keypoint_depth_with_uncertainty: estimated_depth_error = pred_keypoint_uncertainty[:, 2]

            else:
                raise ValueError

        # hard ensemble, soft ensemble and simple average
        elif self.output_depth in ['hard', 'soft', 'mean', 'oracle']:
            if self.pred_direct_depth and self.depth_with_uncertainty:
                pred_combined_depths = self.concat((self.expand_dims(pred_direct_depths,1), pred_keypoints_depths))
                pred_combined_uncertainty = self.concat((pred_direct_uncertainty, pred_keypoint_uncertainty))
            else:
                pred_combined_depths = pred_keypoints_depths
                pred_combined_uncertainty = pred_keypoint_uncertainty

            depth_weights = 1 / pred_combined_uncertainty
            visualize_preds['min_uncertainty'] = ops.argmax(depth_weights,1)

            if self.output_depth == 'hard':
                pred_depths = pred_combined_depths[ops.arange(pred_combined_depths.shape[0]), ops.argmax(depth_weights,1)]

                # the uncertainty after combination
                estimated_depth_error = pred_combined_uncertainty.min(axis=1)

            elif self.output_depth == 'soft':
                depth_weights = depth_weights / self.reducesum_t(depth_weights,axis=1)
                pred_depths = self.reducesum(pred_combined_depths * depth_weights, axis=1)

                # the uncertainty after combination
                estimated_depth_error = self.reducesum(depth_weights * pred_combined_uncertainty, axis=1)

            elif self.output_depth == 'mean':
                pred_depths = self.reducemean(pred_combined_depths,1)

                # the uncertainty after combination
                estimated_depth_error = pred_combined_uncertainty.mean(axis=1)

            # the best estimator is always selected
            elif self.output_depth == 'oracle':
                pred_depths, estimated_depth_error = self.get_oracle_depths(pred_box2d, clses, pred_combined_depths,
                                                                            pred_combined_uncertainty, data[0])

        batch_idxs = self.zeros(pred_depths.shape[0], dtype=ms.int32)
        pred_locations = self.decode_location_flatten(pred_bbox_points, pred_offset_3D, pred_depths, calib, pad_size, batch_idxs)
        pred_center_locations = pred_locations
        pred_rotys, pred_alphas = self.decode_axes_orientation(pred_orientation, locations=pred_locations)

        pred_locations[:, 1] += pred_dimensions[:, 1] / 2
        clses = clses.view(-1, 1)
        pred_alphas = pred_alphas.view(-1, 1)
        pred_rotys = pred_rotys.view(-1, 1)
        scores = scores.view(-1, 1)
        # change dimension back to h,w,l
        pred_dimensions = pred_dimensions.roll(shifts=-1, dims=1)

        # the uncertainty of depth estimation can reflect the confidence for 3D object detection
        vis_scores = scores

        if self.predict_IOU3D_as_conf:
            IOU3D_predict = pred_regression_pois[:, self.key2channel('IOU3D_predict')]
            scores = scores * self.sigmoid(IOU3D_predict)
            uncertainty_conf, estimated_depth_error = None, None

        elif self.uncertainty_as_conf and estimated_depth_error is not None:
            '''[bias_thre = (pred_dimensions[:, 0] + pred_dimensions[:, 2]) / 2 * 0.3
            conf_list = []
            for i in range(bias_thre.shape[0]):
                conf = 2 * stats.norm.cdf(bias_thre[i].item(), 0, estimated_depth_error[i].item()) - 1
                conf_list.append(conf)
            uncertainty_conf = ms.Tensor(conf_list)
            uncertainty_conf = ops.clamp(uncertainty_conf, min=0.01, max=1)'''

            uncertainty_conf = 1 - ops.clip_by_value(estimated_depth_error, ms.Tensor(0.01,ms.float32), ms.Tensor(1,ms.float32))
            scores = scores * uncertainty_conf.view(-1, 1)
        else:
            uncertainty_conf, estimated_depth_error = None, None

        # kitti output format
        result = self.concat([clses, pred_alphas, pred_box2d, pred_dimensions, pred_locations, pred_rotys, scores])

        if self.cfg.TEST.USE_NMS:
            bboxes = self.encode_box3d(pred_rotys, pred_dimensions, pred_center_locations)
            result = nms_3d(result, bboxes, scores.squeeze(1), iou_threshold=self.cfg.TEST.NMS_THRESHOLD)

        eval_utils = {'dis_ious': dis_ious, 'depth_errors': depth_errors, 'uncertainty_conf': uncertainty_conf,
                      'estimated_depth_error': estimated_depth_error, 'vis_scores': vis_scores}

        # Filter 2D confidence * 3D confidence
        result_mask = (result[:, -1] > self.cfg.TEST.DETECTIONS_3D_THRESHOLD).asnumpy().tolist()
        result = result[result_mask]

        return result, eval_utils, visualize_preds, image_ids


    def encode_box3d(self, rotys, dims, locs):
        '''
        construct 3d bounding box for each object.
        Args:
                rotys: rotation in shape N
                dims: dimensions of objects
                locs: locations of objects

        Returns:

        '''
        if len(rotys.shape) == 2:
            rotys = rotys.flatten()
        if len(dims.shape) == 3:
            dims = dims.view(-1, 3)
        if len(locs.shape) == 3:
            locs = locs.view(-1, 3)

        # device = rotys.device
        N = rotys.shape[0]
        ry = self.rad_to_matrix(rotys, N)

        # l, h, w
        dims_corners = ops.tile(dims.view((-1, 1)),(1, 8))
        dims_corners = dims_corners * 0.5
        dims_corners[:, 4:] = -dims_corners[:, 4:]

        # dims_corners[:, 4:] = -dims_corners[:, 4:]
        index = ops.tile(self.bbox_index,(N, 1))

        box_3d_object = ops.gather_elements(dims_corners, 1, index)
        b = box_3d_object.view((N, 3, -1))
        box_3d = ops.matmul(ry, b)  # ry:[11,3,3]   box_3d_object:[11,3,8]
        box_3d =box_3d + ops.tile(ops.expand_dims(locs, -1),(1, 1, 8))

        return ops.transpose(box_3d, (0, 2, 1))

    @staticmethod
    def rad_to_matrix(rotys, N):
        # device = rotys.device

        cos, sin = ops.cos(rotys), ops.sin(rotys)

        i_temp = ops.Tensor([[1, 0, 1],
                             [0, 1, 0],
                             [-1, 0, 1]], dtype=ms.float32)

        ry = ops.tile(i_temp, (N, 1)).view(N, -1, 3)

        ry[:, 0, 0] *= cos
        ry[:, 0, 2] *= sin
        ry[:, 2, 0] *= sin
        ry[:, 2, 2] *= cos

        return ry

    def decode_location_flatten(self, points, offsets, depths, calibs, pad_size, batch_idxs):
        batch_size = len(calibs)
        gts = ops.unique(batch_idxs)[0]
        locations = self.zeros((points.shape[0], 3), self.ms_type)
        points = (points + offsets) * self.down_ratio - pad_size[batch_idxs]  # Left points: The 3D centers in original images.

        for idx, gt in enumerate(gts):
            corr_pts_idx = self.nonzero(batch_idxs == gt).squeeze(-1)
            calib = calibs[gt]
            # concatenate uv with depth
            corr_pts_depth = self.concat((points[corr_pts_idx], depths[corr_pts_idx, None]))
            # locations = ops.tensor_scatter_add(locations, corr_pts_idx, corr_pts_depth)
            locations[corr_pts_idx] = project_image_to_rect(corr_pts_depth, calib)
        return locations

    def get_oracle_depths(self, pred_bboxes, pred_clses, pred_combined_depths, pred_combined_uncertainty, target):
        calib = target['calib']
        pad_size = target['pad_size']
        pad_w, pad_h = pad_size

        valid_mask = self.cast(target['reg_mask'],ms.bool_).asnumpy()
        num_gt = valid_mask.sum()
        valid_mask=valid_mask.tolist()
        gt_clses = target['cls_ids'][valid_mask]
        gt_boxes = target['gt_bboxes'][valid_mask]
        gt_locs = target['locations'][valid_mask]

        gt_depths = gt_locs[:, -1]
        gt_boxes_center = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2

        iou_thresh = 0.5

        # initialize with the average values
        oracle_depth = self.reducemean(pred_combined_depths,1)
        estimated_depth_error = self.reducemean(pred_combined_uncertainty,1)

        for i in range(pred_bboxes.shape[0]):
            # find the corresponding object bounding boxes
            box2d = pred_bboxes[i]
            box2d_center = (box2d[:2] + box2d[2:]) / 2
            img_dis = self.reducesum((box2d_center.reshape(1, 2) - gt_boxes_center) ** 2, 1)
            same_cls_mask = gt_clses == pred_clses[i]
            img_dis[~same_cls_mask] = 9999
            near_idx = ops.argmin(img_dis)  # Find the gt_boxes_center most matched with the pred bbox.
            # iou 2d
            iou_2d = box_iou(box2d.asnumpy(), gt_boxes[near_idx].asnumpy())

            if iou_2d < iou_thresh:
                # match failed, simply choose the default average
                continue
            else:
                estimator_index = ops.argmin(ops.abs(pred_combined_depths[i] - gt_depths[near_idx]))  # Use the estimated depth most match with the ground truth.
                oracle_depth[i] = pred_combined_depths[i, estimator_index]
                estimated_depth_error[i] = pred_combined_uncertainty[i, estimator_index]

        return oracle_depth, estimated_depth_error

    def decode_3D_centers(self, target_centers, offset_3D, pad_size, batch_idxs):
        '''
        Description:
            Decode the 2D points that 3D centers projected on the original image rather than the heatmap.
        Input:
            target_centers: The points that represent targets. shape: (total_num_objs, 2)
            offset_3D: The offset from target_centers to 3D centers. shape: (total_num_objs, 2)
            pad_size: The size of padding. shape: (B, 2)
            batch_idxs: The batch index of various objects. shape: (total_num_objs,)
        Output:
            target_centers: The 2D points that 3D centers projected on the 2D plane. shape: (total_num_objs, 2)
        '''
        centers_3D = self.zeros(target_centers.shape,self.ms_type)
        offset_3D=self.cast(offset_3D,self.ms_type)
        target_centers=self.cast(target_centers,self.ms_type)

        for idx, gt_idx in enumerate(ops.unique(batch_idxs)[0]):
            corr_idx = self.nonzero(batch_idxs == gt_idx)
            centers=self.gather_nd(target_centers,corr_idx)
            offset_3D_select = self.gather_nd(offset_3D, corr_idx)
            corr_idx=self.cast(corr_idx,ms.int32)
            centers_3D_i=(centers + offset_3D_select) * self.down_ratio - pad_size[idx]
            centers_3D = ops.tensor_scatter_add(centers_3D, corr_idx, centers_3D_i)
        return centers_3D


    def decode_2D_keypoints(self, target_centers, pred_keypoint_offset, pad_size, batch_idxs):
        '''
        Description:
            Calculate the positions of keypoints on original image.
        Args:
            target_centers: The position of target centers on heatmap. shape: (total_num_objs, 2)
            pred_keypoint_offset: The offset of keypoints related to target centers on the feature maps. shape: (total_num_objs, 2 * keypoint_num)
            pad_size: The padding size of the original image. shape: (B, 2)
            batch_idxs: The batch index of various objects. shape: (total_num_objs,)
        Returns:
            pred_keypoints: The decoded 2D keypoints. shape: (total_num_objs, 10, 2)
        '''
        total_num_objs, _ = pred_keypoint_offset.shape

        pred_keypoint_offset = pred_keypoint_offset.reshape((total_num_objs, -1, 2))  # It could be 8 or 10 keypoints.
        pred_keypoints = ops.expand_dims(target_centers,1) + pred_keypoint_offset  # pred_keypoints shape: (total_num_objs, keypoint_num, 2)

        for idx, gt_idx in enumerate(ops.unique(ops.cast(batch_idxs, ms.int32))[0]):
            corr_idx = self.cast(self.nonzero(batch_idxs == gt_idx).squeeze(-1), ms.int32)
            pred_keypoints[corr_idx][ :] = pred_keypoints[corr_idx][ :] * self.down_ratio - pad_size[idx]

        return pred_keypoints

    def decode_from_GRM(self, pred_rotys, pred_dimensions, pred_keypoint_offset, pred_direct_depths, targets_dict,
                        GRM_uncern=None, GRM_valid_items=None,
                        batch_idxs=None, cfg=None):
        '''
        Description:
            Compute the 3D locations based on geometric constraints.
        Input:
            pred_rotys: The predicted global orientation. shape: (total_num_objs, 1)
            pred_dimensions: The predicted dimensions. shape: (num_objs, 3)
            pred_keypoint_offset: The offset of keypoints related to target centers on the feature maps. shape: (total_num_obs, 20)
            pred_direct_depths: The directly estimated depth of targets. shape: (total_num_objs, 1)
            targets_dict: The dictionary that contains somre required information. It must contains the following 4 items:
                targets_dict['target_centers']: Target centers. shape: (valid_objs, 2)
                targets_dict['offset_3D']: The offset from target centers to 3D centers. shape: (valid_objs, 2)
                targets_dict['pad_size']: The pad size for the original image. shape: (B, 2)
                targets_dict['calib']: A list contains calibration objects. Its length is B.
            GRM_uncern: The estimated uncertainty of 25 equations in GRM. shape: None or (total_num_objs, 25).
            GRM_valid_items: The effectiveness of 25 equations. shape: None or (total_num_objs, 25)
            batch_idxs: The batch index of various objects. shape: None or (total_num_objs,)
            cfg: The config object. It could be None.
        Output:
            pinv: The decoded positions of targets. shape: (total_num_objs, 3)
            A: Matrix A of geometric constraints. shape: (total_num_objs, 25, 3)
            B: Matrix B of geometric constraints. shape: (total_num_objs, 25, 1)
        '''
        target_centers = targets_dict['target_centers']  # The position of target centers on heatmap. shape: (total_num_objs, 2)
        offset_3D = targets_dict['offset_3D']  # shape: (total_num_objs, 2)
        calibs = [targets_dict['calib']]  # The list contains calibration objects. Its length is B.
        pad_size = targets_dict['pad_size']  # shape: (B, 2)

        if GRM_uncern is None:
            GRM_uncern = self.ones((pred_rotys.shape[0], 25), self.ms_type)

        if len(calibs) == 1:  # Batch size is 1.
            batch_idxs = self.ones((pred_rotys.shape[0],), ms.uint8)

        if GRM_valid_items is None:  # All equations of GRM is valid.
            GRM_valid_items = self.ones((pred_rotys.shape[0], 25), ms.bool_)

        # For debug
        '''pred_keypoint_offset = targets_dict['keypoint_offset'][:, :, 0:2].contiguous().view(-1, 20)
        pred_rotys = targets_dict['rotys'].view(-1, 1)
        pred_dimensions = targets_dict['dimensions']
        locations = targets_dict['locations']
        pred_direct_depths = locations[:, 2].view(-1, 1)'''

        pred_keypoints = self.decode_2D_keypoints(target_centers, pred_keypoint_offset,
                                                  targets_dict['pad_size'],
                                                  batch_idxs=batch_idxs)  # pred_keypoints: The 10 keypoint position original image. shape: (total_num_objs, 10, 2)

        c_u = ops.stack([calib['c_u'] for calib in calibs],axis=0)  # c_u shape: (B,)
        c_v = ops.stack([calib['c_v'] for calib in calibs],axis=0)
        f_u = ops.stack([calib['f_u'] for calib in calibs],axis=0)
        f_v = ops.stack([calib['f_v'] for calib in calibs],axis=0)
        b_x = ops.stack([calib['b_x'] for calib in calibs],axis=0)
        b_y = ops.stack([calib['b_y'] for calib in calibs],axis=0)

        n_pred_keypoints = pred_keypoints  # n_pred_keypoints shape: (total_num_objs, 10, 2)

        for idx, gt_idx in enumerate(ops.unique(ops.cast(batch_idxs, ms.int32))[0]):
            # corr_idx = ops.nonzero(batch_idxs == gt_idx).squeeze(-1).astype(ms.float64)
            corr_idx = self.cast(self.nonzero(batch_idxs == gt_idx).squeeze(1),ms.int32)
            n_pred_keypoints[corr_idx][:, :, 0] = (n_pred_keypoints[corr_idx][:, :, 0] - c_u[idx]) / f_u[idx]
            n_pred_keypoints[corr_idx][:, :, 1] = (n_pred_keypoints[corr_idx][:, :, 1] - c_v[idx]) / f_v[idx]

        total_num_objs = n_pred_keypoints.shape[0]

        centers_3D = self.decode_3D_centers(target_centers, offset_3D, pad_size, batch_idxs)  # centers_3D: The positions of 3D centers of objects projected on original image.
        n_centers_3D = centers_3D
        for idx, gt_idx in enumerate(ops.unique(ops.cast(batch_idxs, ms.int32))[0]):
            corr_idx = self.cast(self.nonzero(batch_idxs == gt_idx).squeeze(1),ms.int32)
            n_centers_3D[corr_idx][ :,0] = (n_centers_3D[corr_idx][ :,0] - c_u[idx]) / f_u[idx]  # n_centers_3D shape: (total_num_objs, 2)
            n_centers_3D[corr_idx][ :,1] = (n_centers_3D[corr_idx][ :,1] - c_v[idx]) / f_v[idx]

        kp_group = self.concat([(n_pred_keypoints.reshape((total_num_objs, 20)), n_centers_3D,
                             self.zeros((total_num_objs, 2), self.ms_type))])  # kp_group shape: (total_num_objs, 24)
        coe = self.zeros((total_num_objs, 24, 2), self.ms_type)
        coe[:, 0:: 2, 0] = -1
        coe[:, 1:: 2, 1] = -1
        A = ops.concat((coe, self.expand_dims(kp_group, 2)), axis=2)
        coz = self.zeros((total_num_objs, 1, 3), self.ms_type)
        coz[:, :, 2] = 1
        A = self.concat((A, coz))  # A shape: (total_num_objs, 25, 3)

        pred_rotys = pred_rotys.reshape(total_num_objs, 1)
        cos = ops.cos(pred_rotys)  # cos shape: (total_num_objs, 1)
        sin = ops.sin(pred_rotys)  # sin shape: (total_num_objs, 1)

        pred_dimensions = pred_dimensions.reshape(total_num_objs, 3)
        l = pred_dimensions[:, 0: 1]  # h shape: (total_num_objs, 1). The same as w and l.
        h = pred_dimensions[:, 1: 2]
        w = pred_dimensions[:, 2: 3]

        B = self.zeros((total_num_objs, 25, 1), self.ms_type)
        B[:, 0, :] = l / 2 * cos + w / 2 * sin
        B[:, 2, :] = l / 2 * cos - w / 2 * sin
        B[:, 4, :] = -l / 2 * cos - w / 2 * sin
        B[:, 6, :] = -l / 2 * cos + w / 2 * sin
        B[:, 8, :] = l / 2 * cos + w / 2 * sin
        B[:, 10, :] = l / 2 * cos - w / 2 * sin
        B[:, 12, :] = -l / 2 * cos - w / 2 * sin
        B[:, 14, :] = -l / 2 * cos + w / 2 * sin
        B[:, 1: 8: 2, :] = self.expand_dims((h / 2), 1)
        B[:, 9: 16: 2, :] = -self.expand_dims((h / 2), 1)
        B[:, 17, :] = h / 2
        B[:, 19, :] = -h / 2

        total_num_objs = n_pred_keypoints.shape[0]
        pred_direct_depths = pred_direct_depths.reshape(total_num_objs, )
        for idx, gt_idx in enumerate(ops.unique(batch_idxs.astype(ms.int32))[0]):
            corr_idx = self.cast(self.nonzero(batch_idxs == gt_idx).squeeze(-1),ms.int32)
            B[corr_idx][:, 22, 0] = -(centers_3D[corr_idx][:, 0] - c_u[idx]) * pred_direct_depths[corr_idx] / f_u[idx] - b_x[idx]
            B[corr_idx][:, 23, 0] = -(centers_3D[corr_idx][:, 1] - c_v[idx]) * pred_direct_depths[corr_idx] / f_v[idx] - b_y[idx]
            B[corr_idx][:, 24, 0] = pred_direct_depths[corr_idx]

        C = self.zeros((total_num_objs, 25, 1), self.ms_type)
        kps = n_pred_keypoints.view(total_num_objs, 20)  # kps_x shape: (total_num_objs, 20)
        C[:, 0, :] = kps[:, 0: 1] * (-l / 2 * sin + w / 2 * cos)
        C[:, 1, :] = kps[:, 1: 2] * (-l / 2 * sin + w / 2 * cos)
        C[:, 2, :] = kps[:, 2: 3] * (-l / 2 * sin - w / 2 * cos)
        C[:, 3, :] = kps[:, 3: 4] * (-l / 2 * sin - w / 2 * cos)
        C[:, 4, :] = kps[:, 4: 5] * (l / 2 * sin - w / 2 * cos)
        C[:, 5, :] = kps[:, 5: 6] * (l / 2 * sin - w / 2 * cos)
        C[:, 6, :] = kps[:, 6: 7] * (l / 2 * sin + w / 2 * cos)
        C[:, 7, :] = kps[:, 7: 8] * (l / 2 * sin + w / 2 * cos)
        C[:, 8, :] = kps[:, 8: 9] * (-l / 2 * sin + w / 2 * cos)
        C[:, 9, :] = kps[:, 9: 10] * (-l / 2 * sin + w / 2 * cos)
        C[:, 10, :] = kps[:, 10: 11] * (-l / 2 * sin - w / 2 * cos)
        C[:, 11, :] = kps[:, 11: 12] * (-l / 2 * sin - w / 2 * cos)
        C[:, 12, :] = kps[:, 12: 13] * (l / 2 * sin - w / 2 * cos)
        C[:, 13, :] = kps[:, 13: 14] * (l / 2 * sin - w / 2 * cos)
        C[:, 14, :] = kps[:, 14: 15] * (l / 2 * sin + w / 2 * cos)
        C[:, 15, :] = kps[:, 15: 16] * (l / 2 * sin + w / 2 * cos)

        B = B - C  # B shape: (total_num_objs, 25, 1)

        # A = A[:, 22:25, :]
        # B = B[:, 22:25, :]

        weights = 1 / GRM_uncern  # weights shape: (total_num_objs, 25)

        ##############  Block the invalid equations ##############
        # A = A * GRM_valid_items.unsqueeze(2)
        # B = B * GRM_valid_items.unsqueeze(2)

        ##############  Solve pinv for Coordinate loss ##############
        A_coor = A
        B_coor = B
        if cfg is not None and not cfg.MODEL.COOR_ATTRIBUTE:  # Do not use Coordinate loss to train attributes.
            A_coor = A_coor.asnumpy()
            B_coor = B_coor.asnumpy()

        weights_coor = weights
        if cfg is not None and not cfg.MODEL.COOR_UNCERN:  # Do not use Coordinate loss to train uncertainty.
            weights_coor = weights_coor.asnumpy()

        A_coor = A_coor * self.expand_dims(weights_coor, 2)
        B_coor = B_coor * self.expand_dims(weights_coor, 2)

        AT_coor = ops.transpose(A_coor, (0, 2, 1))  # A shape: (total_num_objs, 25, 3)
        pinv = ops.bmm(AT_coor, A_coor)
        pinv = ops.inverse(pinv)
        pinv = ops.bmm(pinv, AT_coor)
        pinv = ops.bmm(pinv, B_coor)

        ##############  Solve A_uncern and B_uncern for GRM loss ##############
        A_uncern = A
        B_uncern = B
        if cfg is not None and not cfg.MODEL.GRM_ATTRIBUTE:  # Do not use GRM loss to train attributes.
            A_uncern = A_uncern.asnumpy()  # .detach()
            B_uncern = B_uncern.asnumpy()  # .detach()

        weights_uncern = weights
        if cfg is not None and not cfg.MODEL.GRM_UNCERN:  # Do not use GRM loss to train uncertainty.
            weights_uncern = weights_uncern.asnumpy()  # .detach()

        A_uncern = A_uncern * self.expand_dims(weights_uncern, 2)
        B_uncern = B_uncern * self.expand_dims(weights_uncern, 2)

        return pinv.view(-1, 3), A_uncern, B_uncern

    def decode_dimension(self, cls_id, dims_offset):
        '''
        retrieve object dimensions
        Args:
            cls_id: each object id
            dims_offset: dimension offsets, shape = (N, 3)
        Returns:
        '''
        cls_dimension_mean = self.dim_mean[cls_id]
        if self.dim_modes[0] == 'exp':
            dims_offset = self.exp(dims_offset)
        if self.dim_modes[2]:
            cls_dimension_std = self.dim_std[cls_id, :]
            dimensions = dims_offset * cls_dimension_std + cls_dimension_mean
        else:
            dimensions = dims_offset * cls_dimension_mean
        return dimensions


    def decode_depth(self, depths_offset):
        if self.depth_mode == 'exp':
            depth = self.exp(depths_offset)
        elif self.depth_mode == 'linear':
            depth = depths_offset * self.depth_ref[1] + self.depth_ref[0]
        elif self.depth_mode == 'inv_sigmoid':
            depth = 1 / self.sigmoid(depths_offset) - 1
        else:
            raise ValueError
        if self.depth_range is not None:
            depth = ops.clip_by_value(depth, self.depth_range[0], self.depth_range[1])

        return depth

    def decode_axes_orientation(self, vector_ori, locations=None, dict_for_3d_center=None):
        '''
        Description:
            Compute global orientation (rotys) and relative angle (alphas). Relative angle is calculated based on $vector_ori.
            When $locations is provided, we use locations_x and locations_z to compute $rays ($rotys-$alphas). If $dict_for_3d_center
            is provided, rays is derived from $center_3D_x, $f_x and $c_u.
        Args:
            vector_ori: local orientation in [axis_cls, head_cls, sin, cos] format. shape: (valid_objs, 16)
            locations: object location. shape: None or (valid_objs, 3)
            dict_for_3d_center: A dictionary that contains information relative to $rays. If not None, its components are as follows:
                dict_for_3d_center['target_centers']: Target centers. shape: (valid_objs, 2)
                dict_for_3d_center['offset_3D']: The offset from target centers to 3D centers. shape: (valid_objs, 2)
                dict_for_3d_center['pad_size']: The pad size for the original image. shape: (B, 2)
                dict_for_3d_center['calib']: A list contains calibration objects. Its length is B.
                dict_for_3d_center['batch_idxs']: The bacth index of input batch. shape: None or (valid_objs,)

        Returns: for training we only need roty
                         for testing we need both alpha and roty

        '''
        if self.multibin:
            pred_bin_cls = vector_ori[:, : self.orien_bin_size * 2].reshape(-1, self.orien_bin_size, 2)
            pred_bin_cls = self.softmax_axis2(pred_bin_cls)[..., 1]
            pred_bin_cls_arg = ops.argmax(pred_bin_cls,1)
            orientations = self.zeros((vector_ori.shape[0],), vector_ori.dtype)
            for i in range(self.orien_bin_size):
                indx=ops.cast((pred_bin_cls_arg == i),ms.float32)
                if indx.sum() == 0: continue
                mask_i = self.nonzero(pred_bin_cls_arg == i)
                s = self.orien_bin_size * 2 + i * 2
                e = s + 2
                pred_bin_offset=self.gather_nd(vector_ori,mask_i)
                pred_bin_offset=pred_bin_offset[:,s: e]
                bin_offset=self.atan2(pred_bin_offset[:, 0], pred_bin_offset[:, 1]) + self.alpha_centers[i]
                mask_i=self.cast(mask_i, ms.int32)
                orientations=ops.tensor_scatter_add(orientations, mask_i, bin_offset)
        else:
            axis_cls = self.softmax(vector_ori[:, :2])
            axis_cls = axis_cls[:, 0] < axis_cls[:, 1]
            head_cls = self.softmax(vector_ori[:, 2:4])
            head_cls = head_cls[:, 0] < head_cls[:, 1]
            # cls axis
            orientations = self.alpha_centers[axis_cls + head_cls * 2]
            sin_cos_offset = self.l2_norm(vector_ori[:, 4:])
            orientations += ops.atan(sin_cos_offset[:, 0] / sin_cos_offset[:, 1])
        if locations is not None:  # Compute rays based on 3D locations.
            locations = locations.view(-1, 3)
            rays = self.atan2(locations[:, 0], locations[:, 2])
        elif dict_for_3d_center is not None:  # Compute rays based on 3D centers projected on 2D plane.
            if len(dict_for_3d_center['calib']) == 1:  # Batch size is 1.
                batch_idxs = self.zeros((vector_ori.shape[0],), ms.uint8)
            else:
                batch_idxs = dict_for_3d_center['batch_idxs']
            centers_3D = self.decode_3D_centers(dict_for_3d_center['target_centers'], dict_for_3d_center['offset_3D'],
                                                dict_for_3d_center['pad_size'], batch_idxs)
            centers_3D_x = centers_3D[:, 0]  # centers_3D_x shape: (total_num_objs,)

            c_u = ops.stack([calib['c_u'] for calib in dict_for_3d_center['calib']],axis=0)
            f_u = ops.stack([calib['f_u'] for calib in dict_for_3d_center['calib']],axis=0)
            # b_x = ops.stack([calib['b_x'] for calib in dict_for_3d_center['calib']],axis=0)

            rays = ops.zeros_like(orientations)
            for idx, gt_idx in enumerate(ops.unique(batch_idxs)[0]):
                corr_idx = self.nonzero(batch_idxs == gt_idx)
                corr_idx=self.cast(corr_idx,ms.int32)
                centers_3D_x=self.gather_nd(centers_3D_x,corr_idx)
                centers_3D_x_atan=self.atan2(centers_3D_x - c_u[idx], f_u[idx])
                rays=ops.tensor_scatter_add(rays, corr_idx, centers_3D_x_atan)  # This is exactly an approximation.
        else:
            raise Exception("locations and dict_for_3d_center should not be None simultaneously.")
        alphas = orientations
        rotys = alphas + rays

        larger_idx = self.cast(self.nonzero(rotys > PI),ms.int32)
        small_idx = self.cast(self.nonzero(rotys < -PI),ms.int32)
        if larger_idx.shape[0]>0:
            larger_idx=larger_idx.squeeze(1)
            rotys[larger_idx]-=2 * self.updates
        if small_idx.shape[0] > 0:
            small_idx=small_idx.squeeze(1)
            rotys[small_idx]+=2 * self.updates

        larger_idx = self.cast(self.nonzero(alphas > PI),ms.int32)
        small_idx = self.cast(self.nonzero(alphas < -PI),ms.int32)
        if larger_idx.shape[0]>0:
            larger_idx = larger_idx.squeeze(1)
            alphas[larger_idx] -= 2 * PI
        if small_idx.shape[0] > 0:
            small_idx = small_idx.squeeze(1)
            alphas[small_idx] += 2 * PI
        return rotys, alphas


    def decode_depth_from_keypoints_batch(self, pred_keypoints, pred_dimensions, calibs, batch_idxs=None):
        # pred_keypoints: K x 10 x 2, 8 vertices, bottom center and top center (bottom first)
        # pred_keypoints[k,10,2]
        # pred_dimensions[k,3]
        pred_height_3D = pred_dimensions[:, 1]  # [k,]
        batch_size = len(calibs)
        if batch_size == 1:
            batch_idxs = self.zeros(pred_dimensions.shape[0], self.ms_type)

        center_height = pred_keypoints[:, -2, 1] - pred_keypoints[:, -1, 1]  # [2]

        corner_02_height = pred_keypoints[:, [0, 2], 1] - pred_keypoints[:, [4, 6], 1]  # [2,2]
        corner_13_height = pred_keypoints[:, [1, 3], 1] - pred_keypoints[:, [5, 7], 1]  # [2,2]

        pred_keypoint_depths = {'center': [], 'corner_02': [], 'corner_13': []}

        for idx, gt_idx in enumerate(ops.unique(ops.cast(batch_idxs, ms.int32))[0]):
            calib = calibs[idx]
            corr_pts_idx = self.cast(self.nonzero(batch_idxs == gt_idx).squeeze(-1),ms.int32)
            center_depth = calib['f_v'] * pred_height_3D[corr_pts_idx] / (
                    self.relu(center_height[corr_pts_idx]) * self.down_ratio + self.EPS)
            corner_02_depth = calib['f_v'] * ops.expand_dims(pred_height_3D[corr_pts_idx], -1) / (
                    self.relu(corner_02_height[corr_pts_idx]) * self.down_ratio + self.EPS)
            corner_13_depth = calib['f_v'] * ops.expand_dims(pred_height_3D[corr_pts_idx], -1) / (
                    self.relu(corner_13_height[corr_pts_idx]) * self.down_ratio + self.EPS)

            corner_02_depth = self.reducemean(corner_02_depth,1)
            corner_13_depth = self.reducemean(corner_13_depth,1)

            pred_keypoint_depths['center'].append(center_depth)
            pred_keypoint_depths['corner_02'].append(corner_02_depth)
            pred_keypoint_depths['corner_13'].append(corner_13_depth)

        for key, depths in pred_keypoint_depths.items():
            pred_keypoint_depths[key] = ops.clip_by_value(ops.concat(depths), self.depth_range[0], self.depth_range[1])
        pred_depths = ops.stack(([depth for depth in pred_keypoint_depths.values()]), axis=1)

        return pred_depths

    def evaluate_3D_detection(self, targets, pred_regression):
        # computing disentangling 3D IoU for offset, depth, dimension, orientation
        batch, channel = pred_regression.shape[:2]

        # 1. extract prediction in points of interest
        target_points = self.cast(targets['target_centers'],ms.float32)
        pred_regression_pois = select_point_of_interest(  # pred_regression_pois shape: (B, num_objs, C)
            batch, target_points, pred_regression)

        # 2. get needed predictions
        pred_regression_pois = pred_regression_pois.view(-1, channel)
        reg_mask = targets['reg_mask'].view(-1)
        reg_mask=self.cast(reg_mask,ms.bool_).asnumpy().tolist()
        pred_regression_pois = pred_regression_pois[reg_mask]
        target_points = target_points[0][reg_mask]

        pred_offset_3D = pred_regression_pois[:, self.key2channel('3d_offset')]
        pred_dimensions_offsets = pred_regression_pois[:, self.key2channel('3d_dim')]
        pred_orientation = self.concat((pred_regression_pois[:, self.key2channel('ori_cls')], pred_regression_pois[:, self.key2channel('ori_offset')]))
        pred_keypoint_offset = pred_regression_pois[:, self.key2channel('corner_offset')].view(-1, 10, 2)

        # 3. get ground-truth
        target_clses = ops.masked_select(targets['cls_ids'].view(-1),reg_mask)
        target_offset_3D = targets['offset_3D'].view(-1, 2)[reg_mask]
        target_locations = targets['locations'].view(-1, 3)[reg_mask]
        target_dimensions = targets['dimensions'].view(-1, 3)[reg_mask]
        target_rotys = ops.masked_select(targets['rotys'].view(-1),reg_mask)

        target_depths = target_locations[:, -1]

        # 4. decode prediction
        pred_dimensions = self.decode_dimension(target_clses, pred_dimensions_offsets)

        pred_depths_offset = pred_regression_pois[:, self.key2channel('depth')].squeeze(-1)

        if self.output_depth == 'GRM':
            if not self.GRM_with_uncertainty:
                raise Exception("When compute depth with GRM, the GRM head of network must be loaded.")
            GRM_uncern = pred_regression_pois[:, self.key2channel('GRM_uncern')].exp()  # (valid_objs, 25)
            info_dict = {'target_centers': target_points, 'offset_3D': pred_offset_3D, 'pad_size': targets["pad_size"],
                         'calib': targets['calib'], 'batch_idxs': None}
            pred_rotys, _ = self.decode_axes_orientation(pred_orientation, dict_for_3d_center=info_dict)
            pred_direct_depths = self.decode_depth(pred_depths_offset)
            pred_locations, _, _ = self.decode_from_GRM(pred_rotys.unsqueeze(1), pred_dimensions,
                                                                     pred_keypoint_offset.reshape(-1, 20),
                                                                     pred_direct_depths.unsqueeze(1),
                                                                     targets_dict=info_dict,
                                                                     GRM_uncern=GRM_uncern)  # pred_locations_3D shape: (valid_objs, 3)
            pred_depths = pred_locations[:, 2]

        elif self.output_depth == 'direct':
            pred_depths = self.decode_depth(pred_depths_offset)

        elif self.output_depth == 'keypoints':
            pred_depths = self.decode_depth_from_keypoints(pred_offset_3D, pred_keypoint_offset,
                                                                        pred_dimensions, targets['calib'])
            pred_uncertainty = self.exp(pred_regression_pois[:, self.key2channel('corner_uncertainty')])
            pred_depths = pred_depths[ops.arange(pred_depths.shape[0]), ops.argmin(pred_uncertainty,1)]  # Use the depth estimation with the smallest uncertainty.

        elif self.output_depth == 'combine':
            pred_direct_depths = self.decode_depth(pred_depths_offset)
            pred_keypoints_depths = self.decode_depth_from_keypoints(pred_offset_3D, pred_keypoint_offset,
                                                                                  pred_dimensions, targets['calib'])
            pred_combined_depths = self.concat((pred_direct_depths.unsqueeze(1), pred_keypoints_depths))

            pred_direct_uncertainty = self.exp(pred_regression_pois[:, self.key2channel('depth_uncertainty')],)
            pred_keypoint_uncertainty = self.exp(pred_regression_pois[:, self.key2channel('corner_uncertainty')],)
            pred_combined_uncertainty = self.concat((pred_direct_uncertainty, pred_keypoint_uncertainty))
            pred_depths = pred_combined_depths[ops.arange(pred_combined_depths.shape[0]), ops.argmin(pred_combined_uncertainty,1)]  # # Use the depth estimation with the smallest uncertainty.

        elif self.output_depth == 'soft':
            pred_direct_depths = self.decode_depth(pred_depths_offset)
            pred_keypoints_depths = self.decode_depth_from_keypoints(pred_offset_3D, pred_keypoint_offset, pred_dimensions, targets['calib'])
            pred_combined_depths = self.concat((pred_direct_depths.unsqueeze(1), pred_keypoints_depths))

            pred_direct_uncertainty = self.exp(pred_regression_pois[:, self.key2channel('depth_uncertainty')])
            pred_keypoint_uncertainty = self.exp(pred_regression_pois[:, self.key2channel('corner_uncertainty')])
            pred_combined_uncertainty = self.concat((pred_direct_uncertainty, pred_keypoint_uncertainty))

            depth_weights = 1 / pred_combined_uncertainty
            depth_weights = depth_weights / self.reducesum_t(depth_weights,1)
            pred_depths = self.reducesum(pred_combined_depths * depth_weights, 1)

        batch_idxs = self.zeros(pred_depths.shape[0],ms.float32)
        # Decode 3D location from using ground truth target points, target center offset (estimation or label) and depth (estimation or label).
        pred_locations_offset = self.decode_location_flatten(target_points, pred_offset_3D, target_depths,
                                                                          # pred_offset_3D is the target center offset.
                                                                          targets['calib'], targets["pad_size"],
                                                                          batch_idxs)

        pred_locations_depth = self.decode_location_flatten(target_points, target_offset_3D, pred_depths,
                                                                         targets['calib'], targets["pad_size"],
                                                                         batch_idxs)

        pred_locations = self.decode_location_flatten(target_points, pred_offset_3D, pred_depths,
                                                                   targets['calib'], targets["pad_size"], batch_idxs)

        pred_rotys, _ = self.decode_axes_orientation(pred_orientation, target_locations)

        fully_pred_rotys, _ = self.decode_axes_orientation(pred_orientation, pred_locations)

        # fully predicted
        pred_bboxes_3d = self.concat((pred_locations, pred_dimensions, fully_pred_rotys[:, None]))
        # ground-truth
        target_bboxes_3d = self.concat((target_locations, target_dimensions, target_rotys[:, None]))
        # disentangling
        offset_bboxes_3d = self.concat((pred_locations_offset, target_dimensions, target_rotys[:, None]))  # The offset is target center offset.
        depth_bboxes_3d = self.concat((pred_locations_depth, target_dimensions, target_rotys[:, None]))
        dims_bboxes_3d = self.concat((target_locations, pred_dimensions, target_rotys[:, None]))
        orien_bboxes_3d = self.concat((target_locations, target_dimensions, pred_rotys[:, None]))

        # 6. compute 3D IoU
        pred_IoU = get_iou3d(pred_bboxes_3d, target_bboxes_3d)
        offset_IoU = get_iou3d(offset_bboxes_3d, target_bboxes_3d)
        depth_IoU = get_iou3d(depth_bboxes_3d, target_bboxes_3d)
        dims_IoU = get_iou3d(dims_bboxes_3d, target_bboxes_3d)
        orien_IoU = get_iou3d(orien_bboxes_3d, target_bboxes_3d)
        output = dict(pred_IoU=pred_IoU, offset_IoU=offset_IoU, depth_IoU=depth_IoU, dims_IoU=dims_IoU,
                      orien_IoU=orien_IoU)

        return output


    def decode_depth_from_keypoints(self, pred_offsets, pred_keypoints, pred_dimensions, calibs, avg_center=False):
        # pred_keypoints: K x 10 x 2, 8 vertices, bottom center and top center
        assert len(calibs) == 1  # for inference, batch size is always 1

        calib = calibs[0]
        # we only need the values of y
        pred_height_3D = pred_dimensions[:, 1]
        pred_keypoints = pred_keypoints.view(-1, 10, 2)
        # center height -> depth
        if avg_center:
            updated_pred_keypoints = pred_keypoints - pred_offsets.view(-1, 1, 2)
            center_height = updated_pred_keypoints[:, -2:, 1]
            center_depth = calib['f_v'] * ops.expand_dims(pred_height_3D, -1) / (center_height.abs() * self.down_ratio * 2)
            center_depth = self.reducesum(center_depth, 1)
        else:
            center_height = pred_keypoints[:, -2, 1] - pred_keypoints[:, -1, 1]
            center_depth = calib['f_v'] * pred_height_3D / (center_height.abs() * self.down_ratio)

        # corner height -> depth
        corner_02_height = pred_keypoints[:, [0, 2], 1] - pred_keypoints[:, [4, 6], 1]
        corner_13_height = pred_keypoints[:, [1, 3], 1] - pred_keypoints[:, [5, 7], 1]
        corner_02_depth = calib['f_v'] * ops.expand_dims(pred_height_3D,-1) / (corner_02_height * self.down_ratio)
        corner_13_depth = calib['f_v'] * ops.expand_dims(pred_height_3D,-1) / (corner_13_height * self.down_ratio)
        corner_02_depth = self.reducesum(corner_02_depth, 1)
        corner_13_depth = self.reducesum(corner_13_depth, 1)
        # K x 3
        pred_depths = ops.stack((center_depth, corner_02_depth, corner_13_depth),axis=1)

        return pred_depths

    def evaluate_3D_depths(self, targets, pred_regression):
        # computing disentangling 3D IoU for offset, depth, dimension, orientation
        batch, channel = pred_regression.shape[:2]

        # 1. extract prediction in points of interest
        target_points = targets['target_centers']  # target_points shape: (num_objs, 2). (x, y)
        pred_regression_pois = select_point_of_interest(batch, target_points, pred_regression)

        pred_regression_pois = pred_regression_pois.view(-1, channel)
        reg_mask = self.cast(targets['reg_mask'].view(-1), ms.bool_)
        reg_mask=reg_mask.asnumpy().tolist()
        pred_regression_pois = pred_regression_pois[reg_mask]
        target_points = ops.masked_select(target_points[0],reg_mask)
        target_offset_3D = targets['offset_3D'].view(-1, 2)[reg_mask]

        # depth predictions
        pred_depths_offset = pred_regression_pois[:, self.key2channel('depth')]  # pred_direct_depths shape: (num_objs,)
        pred_keypoint_offset = pred_regression_pois[:, self.key2channel('corner_offset')]  # pred_keypoint_offset: (num_objs, 20)

        # Orientatiion predictions
        pred_orientation = self.concat((pred_regression_pois[:, self.key2channel('ori_cls')],
                                    pred_regression_pois[:, self.key2channel('ori_offset')]),)  # pred_orientation shape: (num_objs, 16)
        info_dict = {'target_centers': target_points, 'offset_3D': target_offset_3D, 'pad_size': targets['pad_size'],
                     'calib': targets['calib'], 'batch_idxs': None}
        pred_rotys, _ = self.decode_axes_orientation(pred_orientation, dict_for_3d_center=info_dict)

        if self.depth_with_uncertainty:
            pred_direct_uncertainty = self.exp(pred_regression_pois[:, self.key2channel('depth_uncertainty')],)
        if self.keypoint_depth_with_uncertainty:
            pred_keypoint_uncertainty = self.exp(pred_regression_pois[:, self.key2channel('corner_uncertainty')],)
        if self.GRM_with_uncertainty:
            if 'GRM_uncern' in self.keys:
                GRM_uncern = pred_regression_pois[:, self.key2channel('GRM_uncern')]  # GRM_uncern shape: (valid_objs, 20)
            elif 'GRM1_uncern' in self.keys:
                uncern_GRM1 = pred_regression_pois[:, self.key2channel('GRM1_uncern')]  # uncern_GRM1 shape: (valid_objs, 8)
                uncern_GRM2 = pred_regression_pois[:, self.key2channel('GRM2_uncern')]  # uncern_GRM1 shape: (valid_objs, 8)
                uncern_Mono_Direct = pred_regression_pois[:, self.key2channel('Mono_Direct_uncern')]  # uncern_Mono_Direct shape: (valid_objs, 1)
                uncern_Mono_Keypoint = pred_regression_pois[:, self.key2channel('Mono_Keypoint_uncern')]  # uncern_Mono_Keypoint shape: (valid_objs, 3)
                GRM_uncern = ops.concat((self.expand_dims(uncern_GRM1,2), self.expand_dims(uncern_GRM2,2)), axis=2).view(-1, 16)
                GRM_uncern = self.concat((GRM_uncern, uncern_Mono_Direct, uncern_Mono_Keypoint))  # GRM_uncern shape: (valid_objs, 20)
            GRM_uncern = self.exp(GRM_uncern,)

        # dimension predictions
        target_clses = ops.masked_select(targets['cls_ids'].view(-1),reg_mask)
        pred_dimensions_offsets = pred_regression_pois[:, self.key2channel('3d_dim')]
        pred_dimensions = self.decode_dimension(target_clses, pred_dimensions_offsets, )  # pred_dimensions shape: (num_objs, 3)
        # direct
        pred_direct_depths = self.decode_depth(pred_depths_offset.squeeze(-1))  # pred_direct_depths shape: (num_objs,)
        # three depths from keypoints
        pred_keypoints_depths = self.decode_depth_from_keypoints_batch(pred_keypoint_offset.view(-1, 10, 2), pred_dimensions, targets['calib'])  # pred_keypoints_depths shape: (num_objs, 3)
        # The depth solved by original MonoFlex.
        pred_combined_depths = self.concat((pred_direct_depths.unsqueeze(1), pred_keypoints_depths))  # pred_combined_depths shape: (num_objs, 4)

        MonoFlex_depth_flag = self.depth_with_uncertainty and self.keypoint_depth_with_uncertainty
        if MonoFlex_depth_flag:
            # combined uncertainty
            pred_combined_uncertainty = self.concat((pred_direct_uncertainty, pred_keypoint_uncertainty))
            # min-uncertainty
            pred_uncertainty_min_depth = pred_combined_depths[
                ops.arange(pred_combined_depths.shape[0]), ops.argmin(pred_combined_uncertainty,1)]  # Select the depth with the smallest uncertainty.
            # inv-uncertainty weighted
            pred_uncertainty_weights = 1 / pred_combined_uncertainty
            pred_uncertainty_weights = pred_uncertainty_weights / self.reducesum_t(pred_uncertainty_weights,1)

            pred_uncertainty_softmax_depth = self.reducesum(pred_combined_depths * pred_uncertainty_weights, axis=1)  # Depth produced by soft weighting.

        # Decode depth based on geometric constraints.
        if self.output_depth == 'GRM':
            pred_location_GRM, _, _ = self.decode_from_GRM(self.expand_dims(pred_rotys,1),
                                                                        pred_dimensions,
                                                                        pred_keypoint_offset.reshape(-1, 20),
                                                                        ops.expand_dims(pred_direct_depths, 1),
                                                                        targets_dict=info_dict,
                                                                        GRM_uncern=GRM_uncern)
            pred_depth_GRM = pred_location_GRM[:, 2]
        elif self.output_depth == 'soft_GRM':
            pred_keypoint_offset = pred_keypoint_offset[:, 0:16]
            SoftGRM_depth, separate_depths = self.decode_from_SoftGRM(pred_rotys.unsqueeze(1),
                                                                                   # separate_depths shape: (val_objs, 20)
                                                                                   pred_dimensions,
                                                                                   pred_keypoint_offset.reshape(-1, 16),
                                                                                   pred_combined_depths,
                                                                                   targets_dict=info_dict,
                                                                                   GRM_uncern=GRM_uncern)

            # Uncertainty guided pruning to filter the unreasonable estimation results.
            if self.cfg.TEST.UNCERTAINTY_GUIDED_PRUNING:
                SoftGRM_depth, _ = uncertainty_guided_prune(separate_depths, GRM_uncern, self.cfg, depth_range=self.depth_range)

        # 3. get ground-truth
        target_locations = targets['locations'].view(-1, 3)[reg_mask]
        target_depths = target_locations[:, -1]

        Mono_pred_combined_error = (pred_combined_depths - target_depths[:, None]).abs()
        Mono_pred_direct_error = Mono_pred_combined_error[:, 0]
        Mono_pred_keypoints_error = Mono_pred_combined_error[:, 1:]
        pred_mean_depth = self.reducemean(pred_combined_depths,1)
        pred_mean_error = (pred_mean_depth - target_depths).abs()
        # upper-bound
        pred_min_error = self.argminwithvalue(Mono_pred_combined_error,1)[0]

        pred_errors = {
            'Mono direct error': Mono_pred_direct_error,
            'Mono keypoint_center error': Mono_pred_keypoints_error[:, 0],
            'Mono keypoint_02 error': Mono_pred_keypoints_error[:, 1],
            'Mono keypoint_13 error': Mono_pred_keypoints_error[:, 2],
            'Mono mean error': pred_mean_error,
            'Mono min error': pred_min_error,
        }

        # abs error
        if MonoFlex_depth_flag:
            pred_uncertainty_min_error = (pred_uncertainty_min_depth - target_depths).abs()
            pred_uncertainty_softmax_error = (pred_uncertainty_softmax_depth - target_depths).abs()

            # ops.clamp(estimated_depth_error, min=0.01, max=1)
            pred_errors.update({
                'Mono sigma_min error': pred_uncertainty_min_error,
                'Mono sigma_weighted error': pred_uncertainty_softmax_error,
                'target depth': target_depths,

                'Mono direct_sigma': pred_direct_uncertainty[:, 0],

                'Mono keypoint_center_sigma': pred_keypoint_uncertainty[:, 0],
                'Mono keypoint_02_sigma': pred_keypoint_uncertainty[:, 1],
                'Mono keypoint_13_sigma': pred_keypoint_uncertainty[:, 2]
            })

        if self.output_depth == 'GRM':
            pred_GRM_error = (pred_depth_GRM - target_depths).abs()
            pred_errors.update({
                'pred_GRM_error': pred_GRM_error
            })

        if self.output_depth == 'soft_GRM':
            separate_error = (
                    separate_depths - target_depths.unsqueeze(1)).abs()  # separate_error shape: (val_objs, 20)

            height_depths = self.concat((separate_depths[:, 1:16:2], separate_depths[:, 16:20]))
            height_uncern = self.concat((GRM_uncern[:, 1:16:2], GRM_uncern[:, 16:20]))
            height_w = 1 / height_uncern
            height_w = height_w / self.reducesum(height_w,1)
            height_depth = self.reducesum((height_depths * height_w),1)

            SoftGRM_uncertainty_min_error = separate_error[ops.arange(separate_error.shape[0]), ops.argmin(GRM_uncern,1)]
            for i in range(separate_depths.shape[1]):
                error_key = 'SoftGRM error' + str(i)
                uncern_key = 'SoftGRM uncern' + str(i)
                pred_errors.update({error_key: separate_error[:, i].abs(),
                                    uncern_key: GRM_uncern[:, i]})
            pred_errors.update({'SoftGRM weighted error': (SoftGRM_depth - target_depths).abs()})
            pred_errors.update({'SoftGRM sigma_min error': SoftGRM_uncertainty_min_error})
            pred_errors.update({'SoftGRM height solved error': (height_depth - target_depths).abs()})
            pred_errors.update({'SoftGRM min error': separate_error.min(axis=1)[0]})

        return pred_errors


    def decode_depth_from_keypoints_batch(self, pred_keypoints, pred_dimensions, calibs, batch_idxs=None):
        # pred_keypoints: K x 10 x 2, 8 vertices, bottom center and top center (bottom first)
        # pred_keypoints[k,10,2]
        # pred_dimensions[k,3]
        pred_height_3D = pred_dimensions[:, 1]  # [k,]
        batch_size = len(calibs)
        if batch_size == 1:
            batch_idxs = self.zeros(pred_dimensions.shape[0], self.ms_type)

        center_height = pred_keypoints[:, -2, 1] - pred_keypoints[:, -1, 1]  # [2]

        corner_02_height = pred_keypoints[:, [0, 2], 1] - pred_keypoints[:, [4, 6], 1]  # [2,2]
        corner_13_height = pred_keypoints[:, [1, 3], 1] - pred_keypoints[:, [5, 7], 1]  # [2,2]

        pred_keypoint_depths = {'center': [], 'corner_02': [], 'corner_13': []}

        for idx, gt_idx in enumerate(ops.unique(ops.cast(batch_idxs, ms.int32))[0]):
            calib = calibs[idx]
            corr_pts_idx = self.cast(self.nonzero(batch_idxs == gt_idx).squeeze(-1),ms.int32)
            center_depth = calib['f_v'] * pred_height_3D[corr_pts_idx] / (
                    self.relu(center_height[corr_pts_idx]) * self.down_ratio + self.EPS)
            corner_02_depth = calib['f_v'] * ops.expand_dims(pred_height_3D[corr_pts_idx], -1) / (
                    self.relu(corner_02_height[corr_pts_idx]) * self.down_ratio + self.EPS)
            corner_13_depth = calib['f_v'] * ops.expand_dims(pred_height_3D[corr_pts_idx], -1) / (
                    self.relu(corner_13_height[corr_pts_idx]) * self.down_ratio + self.EPS)

            corner_02_depth = self.reducemean(corner_02_depth,1)
            corner_13_depth = self.reducemean(corner_13_depth,1)

            pred_keypoint_depths['center'].append(center_depth)
            pred_keypoint_depths['corner_02'].append(corner_02_depth)
            pred_keypoint_depths['corner_13'].append(corner_13_depth)

        for key, depths in pred_keypoint_depths.items():
            pred_keypoint_depths[key] = ops.clip_by_value(ops.concat(depths), self.depth_range[0], self.depth_range[1])
        pred_depths = ops.stack(([depth for depth in pred_keypoint_depths.values()]), axis=1)

        return pred_depths


    def decode_from_SoftGRM(self, pred_rotys, pred_dimensions, pred_keypoint_offset, pred_combined_depth,
                            targets_dict, GRM_uncern=None, GRM_valid_items=None,
                            batch_idxs=None):
        '''
        Description:
            Decode depth from geometric constraints directly.
        Input:
            pred_rotys: The predicted global orientation. shape: (val_objs, 1)
            pred_dimensions: The predicted dimensions. shape: (val_objs, 3)
            pred_keypoint_offset: The offset of keypoints related to target centers on the feature maps. shape: (val_objs, 16)
            pred_combined_depth: The depth decoded from direct regression and keypoints. (val_objs, 4)
            targets_dict: The dictionary that contains somre required information. It must contains the following 4 items:
                targets_dict['target_centers']: Target centers. shape: (valid_objs, 2)
                targets_dict['offset_3D']: The offset from target centers to 3D centers. shape: (valid_objs, 2)
                targets_dict['pad_size']: The pad size for the original image. shape: (B, 2)
                targets_dict['calib']: A list contains calibration objects. Its length is B.
            GRM_uncern: The estimated uncertainty of 20 equations in GRM. shape: None or (val_objs, 20).
            GRM_valid_items: The effectiveness of 20 equations. shape: None or (val_objs, 20)
            batch_idxs: The batch index of various objects. shape: None or (val_objs,)
            weighted_sum: Whether to directly weighted add the depths decoded by 20 equations separately.
        Output:
            depth: The depth solved considering all geometric constraints. shape: (val_objs)
            separate_depths: The depths produced by 24 equations, respectively. shape: (val_objs, 24).
        '''
        val_objs_num, _ = pred_combined_depth.shape

        target_centers = targets_dict['target_centers']  # The position of target centers on heatmap. shape: (val_objs, 2)
        offset_3D = targets_dict['offset_3D']  # shape: (val_objs, 2)
        calibs = targets_dict['calib']  # The list contains calibration objects. Its length is B.
        pad_size = targets_dict['pad_size']  # shape: (B, 2)

        if len(calibs) == 1:  # Batch size is 1.
            batch_idxs = self.zeros((val_objs_num,), ms.uint8)

        if GRM_uncern is not None:
            assert GRM_uncern.shape[1] == 20
        else:
            GRM_uncern = self.ones((val_objs_num, 20), self.ms_type)

        if GRM_valid_items is not None:
            assert GRM_valid_items.shape[1] == 20
        else:
            GRM_valid_items = self.ones((val_objs_num, 20), self.ms_type)

        assert pred_keypoint_offset.shape[1] == 16  # Do not use the bottom center and top center. Only 8 vertexes.

        pred_keypoints = self.decode_2D_keypoints(target_centers, pred_keypoint_offset, targets_dict['pad_size'],
                                                  batch_idxs=batch_idxs)  # pred_keypoints: The 10 keypoint position original image. shape: (total_num_objs, 8, 2)

        c_u = ops.stack([calib['c_u'] for calib in calibs],axis=0)  # c_u shape: (B,)
        c_v = ops.stack([calib['c_v'] for calib in calibs],axis=0)
        f_u = ops.stack([calib['f_u'] for calib in calibs],axis=0)
        f_v = ops.stack([calib['f_v'] for calib in calibs],axis=0)
        # b_x = ops.stack([calib['b_x'] for calib in calibs],axis=0)
        # b_y = ops.stack([calib['b_y'] for calib in calibs],axis=0)

        n_pred_keypoints = pred_keypoints  # n_pred_keypoints shape: (total_num_objs, 8, 2)
        # n_pred_keypoints=self.zeros(pred_keypoints.shape,self.ms_type)
        for idx, gt_idx in enumerate(ops.unique(batch_idxs)[0]):
            corr_idx = self.cast(self.nonzero(batch_idxs == gt_idx).squeeze(1),ms.int32)
            n_pred_keypoints[corr_idx][ :, 0] = (n_pred_keypoints[corr_idx][ :, 0] - c_u[idx]) / f_u[idx]  # The normalized keypoint coordinates on 2D plane. shape: (total_num_objs, 8, 2)
            n_pred_keypoints[corr_idx][ :, 1] = (n_pred_keypoints[corr_idx][ :, 1] - c_v[idx]) / f_v[idx]

        centers_3D = self.decode_3D_centers(target_centers, offset_3D, pad_size, batch_idxs)  # centers_3D: The positions of 3D centers of objects projected on original image.
        n_centers_3D = centers_3D
        for idx, gt_idx in enumerate(ops.unique(batch_idxs)[0]):
            corr_idx = self.cast(ops.nonzero(batch_idxs == gt_idx).squeeze(-1),ms.int32)
            n_centers_3D[corr_idx][ :,0] = (n_centers_3D[corr_idx][:, 0] - c_u[idx]) / f_u[idx]  # n_centers_3D: The normalized 3D centers on 2D plane. shape: (total_num_objs, 2)
            n_centers_3D[corr_idx][ :,1] = (n_centers_3D[corr_idx][:,1] - c_v[idx]) / f_v[idx]

        A = self.zeros((val_objs_num, 20, 1), self.ms_type)
        B = self.zeros((val_objs_num, 20, 1), self.ms_type)
        C = self.zeros((val_objs_num, 20, 1), self.ms_type)

        A[:, 0:16, 0] = (n_pred_keypoints - ops.expand_dims(n_centers_3D, 1)).view(val_objs_num, 16)
        A[:, 16:20, 0] = 1

        cos = ops.cos(pred_rotys)  # cos shape: (val_objs, 1)
        sin = ops.sin(pred_rotys)  # sin shape: (val_objs, 1)

        l = pred_dimensions[:, 0: 1]  # h shape: (total_num_objs, 1). The same as w and l.
        h = pred_dimensions[:, 1: 2]
        w = pred_dimensions[:, 2: 3]

        B[:, 0, :] = l / 2 * cos + w / 2 * sin
        B[:, 2, :] = l / 2 * cos - w / 2 * sin
        B[:, 4, :] = -l / 2 * cos - w / 2 * sin
        B[:, 6, :] = -l / 2 * cos + w / 2 * sin
        B[:, 8, :] = l / 2 * cos + w / 2 * sin
        B[:, 10, :] = l / 2 * cos - w / 2 * sin
        B[:, 12, :] = -l / 2 * cos - w / 2 * sin
        B[:, 14, :] = -l / 2 * cos + w / 2 * sin
        B[:, 1: 8: 2, :] = ops.expand_dims((h / 2), 1)
        B[:, 9: 16: 2, :] = -ops.expand_dims((h / 2), 1)
        B[:, 16:20, 0] = pred_combined_depth  # Direct first keypoint next

        kps = n_pred_keypoints.reshape(val_objs_num, 16)  # kps_x shape: (total_num_objs, 16)
        C[:, 0, :] = kps[:, 0: 1] * (-l / 2 * sin + w / 2 * cos)
        C[:, 1, :] = kps[:, 1: 2] * (-l / 2 * sin + w / 2 * cos)
        C[:, 2, :] = kps[:, 2: 3] * (-l / 2 * sin - w / 2 * cos)
        C[:, 3, :] = kps[:, 3: 4] * (-l / 2 * sin - w / 2 * cos)
        C[:, 4, :] = kps[:, 4: 5] * (l / 2 * sin - w / 2 * cos)
        C[:, 5, :] = kps[:, 5: 6] * (l / 2 * sin - w / 2 * cos)
        C[:, 6, :] = kps[:, 6: 7] * (l / 2 * sin + w / 2 * cos)
        C[:, 7, :] = kps[:, 7: 8] * (l / 2 * sin + w / 2 * cos)
        C[:, 8, :] = kps[:, 8: 9] * (-l / 2 * sin + w / 2 * cos)
        C[:, 9, :] = kps[:, 9: 10] * (-l / 2 * sin + w / 2 * cos)
        C[:, 10, :] = kps[:, 10: 11] * (-l / 2 * sin - w / 2 * cos)
        C[:, 11, :] = kps[:, 11: 12] * (-l / 2 * sin - w / 2 * cos)
        C[:, 12, :] = kps[:, 12: 13] * (l / 2 * sin - w / 2 * cos)
        C[:, 13, :] = kps[:, 13: 14] * (l / 2 * sin - w / 2 * cos)
        C[:, 14, :] = kps[:, 14: 15] * (l / 2 * sin + w / 2 * cos)
        C[:, 15, :] = kps[:, 15: 16] * (l / 2 * sin + w / 2 * cos)

        B = B - C  # B shape: (total_num_objs, 24, 1)

        weights = 1 / (GRM_uncern)  # weights shape: (val_objs, 20)

        separate_depths = (B / (A + self.EPS)).squeeze(2)  # separate_depths: The depths produced by 24 equations, respectively. shape: (val_objs, 20).
        separate_depths = ops.clip_by_value(separate_depths, self.depth_range[0], self.depth_range[1])

        weights = (weights / ops.ReduceSum(keep_dims=True)(weights, 1))
        depth = self.reducesum(weights * separate_depths, 1)

        return depth, separate_depths

    def survey_depth_statistics(self, targets, pred_regression, image_ids):
        ID_TYPE_CONVERSION = {
            0: 'Car',
            1: 'Pedestrian',
            2: 'Cyclist',
        }
        batch, channel = pred_regression.shape[:2]
        output_path = self.cfg.OUTPUT_DIR

        # 1. extract prediction in points of interest
        target_points = targets['target_centers']  # target_points shape: (B, objs_per_batch, 2). (x, y)
        pred_regression_pois = select_point_of_interest(batch, target_points, pred_regression)

        pred_regression_pois = pred_regression_pois.view(-1, channel)
        reg_mask = ops.cast(targets['reg_mask'].view(-1), ms.bool_)  # reg_mask_gt shape: (B * num_objs)
        pred_regression_pois = pred_regression_pois[reg_mask]
        target_points = target_points.view(-1, 2)[reg_mask]  # Left target_points shape: (num_objs, 2)

        pred_offset_3D = pred_regression_pois[:, self.key2channel('3d_offset')]  # pred_offset_3D shape: (num_objs, 2)
        pred_orientation = self.concat((pred_regression_pois[:, self.key2channel('ori_cls')],
                                    pred_regression_pois[:, self.key2channel('ori_offset')]))  # pred_orientation shape: (num_objs, 16)

        # depth predictions
        pred_depths_offset = pred_regression_pois[:, self.key2channel('depth')]  # pred_direct_depths shape: (num_objs,)
        pred_keypoint_offset = pred_regression_pois[:, self.key2channel('corner_offset')]  # pred_keypoint_offset: (num_objs, 20)

        if self.depth_with_uncertainty:
            pred_direct_uncertainty = self.exp(pred_regression_pois[:, self.key2channel('depth_uncertainty')])  # pred_direct_uncertainty shape: (num_objs, 1)
        else:
            pred_direct_uncertainty = None
        if self.keypoint_depth_with_uncertainty:
            pred_keypoint_uncertainty = self.exp(pred_regression_pois[:, self.key2channel('corner_uncertainty')])  # pred_keypoint_uncertainty shape: (num_objs, 3)
        else:
            pred_keypoint_uncertainty = None

        # dimension predictions
        target_clses = targets['cls_ids'].view(-1)[reg_mask]  # target_clses shape: (num_objs,)
        pred_dimensions_offsets = pred_regression_pois[:, self.key2channel('3d_dim')]
        pred_dimensions = self.decode_dimension(target_clses, pred_dimensions_offsets, )  # pred_dimensions shape: (num_objs, 3)

        dimension_residual = (pred_dimensions - targets['dimensions'][0, reg_mask, :]) / targets['dimensions'][0, reg_mask, :]  # dimension_residual shape: (num_objs, 3). (l, h, w)
        target_keypoint_offset = targets['keypoints'][0, :, :, 0:2]
        target_keypoint_offset = target_keypoint_offset[reg_mask].view(-1, 20)  # Offset from target centers to keypoints.
        target_vertex_offset = target_keypoint_offset[:, 0:16]

        # direct
        pred_direct_depths = self.expand_dims(self.decode_depth(pred_depths_offset.squeeze(-1)), 1)  # pred_direct_depths shape: (num_objs, 1)
        # three depths from keypoints
        pred_keypoints_depths = self.decode_depth_from_keypoints_batch(pred_keypoint_offset.view(-1, 10, 2), pred_dimensions, targets['calib'])  # pred_keypoints_depths shape: (num_objs, 3)

        target_locations = targets['locations'].view(-1, 3)[reg_mask]
        target_depth = target_locations[:, 2]  # target_depth shape: (num_objs)

        # Decode depth with GRM.
        pred_combined_depths = self.concat((pred_direct_depths, pred_keypoints_depths))  # pred_combined_depths shape: (valid_objs, 4)
        info_dict = {'target_centers': target_points, 'offset_3D': pred_offset_3D, 'pad_size': targets['pad_size'],
                     'calib': targets['calib'], 'batch_idxs': None}
        GRM_rotys, GRM_alphas = self.decode_axes_orientation(pred_orientation, dict_for_3d_center=info_dict)

        # For debug
        '''info_dict['target_centers'] = targets['target_centers'][0, reg_mask, :].float()
        info_dict['offset_3D'] = targets['offset_3D'][0, reg_mask, :]
        GRM_rotys = targets['rotys'][0, reg_mask]
        pred_dimensions = targets['dimensions'][0, reg_mask, :]
        pred_keypoint_offset = targets['keypoints'][0, reg_mask, :, 0:2].view(-1, 20)'''

        pred_vertex_offset = pred_keypoint_offset[:, 0:16]  # Do not use the top center and bottom center.
        pred_depths, separate_depths = self.decode_from_SoftGRM(self.expand_dims(GRM_rotys,1),
                                                                             pred_dimensions, pred_vertex_offset,
                                                                             pred_combined_depths,
                                                                             targets_dict=info_dict)  # pred_depths shape: (total_num_objs,). separate_depths shape: (num_objs, 20)

        if 'GRM_uncern' in self.keys:
            GRM_uncern = pred_regression_pois[:, self.key2channel('GRM_uncern')]  # GRM_uncern shape: (valid_objs, 20)
        elif 'GRM1_uncern' in self.keys:
            uncern_GRM1 = pred_regression_pois[:, self.key2channel('GRM1_uncern')]  # uncern_GRM1 shape: (valid_objs, 8)
            uncern_GRM2 = pred_regression_pois[:, self.key2channel('GRM2_uncern')]  # uncern_GRM1 shape: (valid_objs, 8)
            uncern_Mono_Direct = pred_regression_pois[:, self.key2channel('Mono_Direct_uncern')]  # uncern_Mono_Direct shape: (valid_objs, 1)
            uncern_Mono_Keypoint = pred_regression_pois[:, self.key2channel('Mono_Keypoint_uncern')]  # uncern_Mono_Keypoint shape: (valid_objs, 3)
            GRM_uncern = self.concat((self.expand_dims(uncern_GRM1,2), self.expand_dims(uncern_GRM2,2))).view(-1, 16)
            GRM_uncern = self.concat((GRM_uncern, uncern_Mono_Direct, uncern_Mono_Keypoint))  # GRM_uncern shape: (valid_objs, 20)

        GRM_uncern = self.exp(ops.clip_by_value(GRM_uncern, self.uncertainty_range[0], self.uncertainty_range[1]))

        f = open(os.path.join(self.depth_statistics_path, image_ids[0] + '.txt'), 'w')
        for i in range(target_points.shape[0]):
            Mono_depth = ops.concat((pred_direct_depths[i], pred_keypoints_depths[i]), axis=0)
            category = target_clses[i]
            category = ID_TYPE_CONVERSION[category]
            output_str = "Object: {}, Category: {}, Depth label: {}\nDirect depth: {}, Keypoint depth 1: {}, " \
                         "keypoint depth 2: {} keypoint depth 3: {}, Mono Depth mean: {}, Mono Depth std: {}\n".format(
                i, category, target_depth[i], pred_direct_depths[i, 0], pred_keypoints_depths[i, 0],
                pred_keypoints_depths[i, 1], pred_keypoints_depths[i, 2], Mono_depth.mean(), Mono_depth.std())
            for j in range(separate_depths.shape[1]):
                if j < target_vertex_offset.shape[1]:
                    output_str += 'Keypoint prediction bias: {} '.format(target_vertex_offset[i][j].item() - pred_vertex_offset[i][j])
                output_str += 'GRM_depth {}: {}, GRM_uncern {}: {}\n'.format(j, separate_depths[i][j], j, GRM_uncern[i][j])
            output_str += 'GRM final depth: {}'.format(pred_depths[i])
            f.write(output_str)
            f.write('\n\n')
        f.close()
