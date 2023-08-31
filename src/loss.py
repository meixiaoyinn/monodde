import mindspore.ops as ops
import mindspore.nn as nn
import mindspore as ms
# import mindspore.numpy as mnp
import numpy as np
# import pdb
from .net_utils import select_point_of_interest,project_image_to_rect,construct_zeros,construct_tensor
from .loss_patial import *
# from shapely.geometry import Polygon
# from .net_utils import Converter_key2channel, project_image_to_rect

# PI = np.pi


def exp_rampup(rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""

    def warpper(iteration):
        if iteration < rampup_length:
            iteration = np.clip(iteration, 0.0, rampup_length)
            phase = 1.0 - iteration / rampup_length
            # weight increase from 0.007~1
            return float(np.exp(-5.0 * phase * phase))
        else:
            return 1.0

    return warpper
'''depth loss'''


class Inverse_Sigmoid_Loss(nn.Cell):
    def __init__(self):
        super(Inverse_Sigmoid_Loss, self).__init__()
        self.l1loss = nn.L1Loss(reduction='none')
        self.sigmoid = ops.Sigmoid()

    def construct(self, prediction, target, weight=None):
        trans_prediction = 1 / self.sigmoid(target) - 1
        loss = self.l1loss(trans_prediction, target)
        if weight is not None:
            loss = loss * weight

        return loss


class Log_L1_Loss(nn.Cell):
    def __init__(self):
        super(Log_L1_Loss, self).__init__()
        self.log = ops.Log()
        self.l1loss = nn.L1Loss(reduction='none')

    def construct(self, prediction, target, weight=None):
        loss = self.l1loss(self.log(prediction), self.log(target))

        if weight is not None:
            loss = loss * weight
        return loss


''''''


class FocalLoss(nn.Cell):
    def __init__(self, alpha=2, beta=4,loss_weight=1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.pow = ops.Pow()
        self.log=ops.Log()
        self.weight=loss_weight
        # self.num_hm_pos=ms.Parameter(ms.Tensor(0, ms.float32), requires_grad=False)
        self.clip_min = ms.Tensor(1, ms.float32)

    def construct(self, prediction, target):
        positive_index = ops.cast(ops.equal(target,1),ms.float32)
        negative_index = ops.cast(ops.logical_and(ops.less(target,1) , ops.ge(target,0)),ms.float32)
        # ignore_index = ops.equal(target, -1)  # ignored pixels

        negative_weights = self.pow(1 - target, self.beta)
        # loss = 0.

        positive_loss = self.log(prediction) * self.pow(1 - prediction, self.alpha) * positive_index
        # ops.print_('positive_loss:',positive_loss)
        negative_loss = self.log(1 - prediction) * self.pow(prediction, self.alpha) * negative_weights * negative_index
        # ops.print_('negative_loss:', negative_loss)
        num_hm_pos = positive_index.sum()
        positive_loss = positive_loss.sum()
        negative_loss = negative_loss.sum()

        loss = - negative_loss - positive_loss
        # ops.print_('loss:', loss)
        # num_hm_pos=ms.Tensor(num_positive, ms.float32)

        # hm_loss = self.weight * loss  # Heatmap loss.
        hm_loss = self.weight * loss / ops.clip_by_value(num_hm_pos, self.clip_min)

        return hm_loss


class Mono_loss(nn.Cell):
    def __init__(self, cfg):
        super(Mono_loss, self).__init__()
        self.cfg = cfg
        # self.key2channel = Converter_key2channel(keys=cfg.MODEL.HEAD.REGRESSION_HEADS,
        #                                          channels=cfg.MODEL.HEAD.REGRESSION_CHANNELS)
        self.ct_keys = [key for key_group in cfg.MODEL.HEAD.REGRESSION_HEADS for key in key_group]
        self.ct_channels = [channel for channel_groups in cfg.MODEL.HEAD.REGRESSION_CHANNELS for channel in channel_groups]
        self.loss_weight_ramper = exp_rampup(cfg.SOLVER.RAMPUP_ITERATIONS)
        self.ms_type = ms.float32

        self.max_objs = cfg.DATASETS.MAX_OBJECTS
        self.center_sample = cfg.MODEL.HEAD.CENTER_SAMPLE
        self.regress_area = cfg.MODEL.HEAD.REGRESSION_AREA
        self.heatmap_type = cfg.MODEL.HEAD.HEATMAP_TYPE
        self.corner_depth_sp = cfg.MODEL.HEAD.SUPERVISE_CORNER_DEPTH
        self.loss_keys = cfg.MODEL.HEAD.LOSS_NAMES

        self.dim_weight = ms.Tensor(cfg.MODEL.HEAD.DIMENSION_WEIGHT,self.ms_type).view(1, 3)
        self.uncertainty_range = cfg.MODEL.HEAD.UNCERTAINTY_RANGE

        self.loss_weights = {}
        for key, weight in zip(cfg.MODEL.HEAD.LOSS_NAMES, cfg.MODEL.HEAD.INIT_LOSS_WEIGHT): self.loss_weights[key] = weight

        # loss functions
        loss_types = cfg.MODEL.HEAD.LOSS_TYPE
        # self.cls_loss_fnc = FocalLoss(cfg.MODEL.HEAD.LOSS_PENALTY_ALPHA,
        #                               cfg.MODEL.HEAD.LOSS_BETA,self.loss_weights['hm_loss'])  # penalty-reduced focal loss
        # self.iou_loss = IOULoss(loss_type=loss_types[2])  # iou loss for 2D detection

        # depth loss
        # if loss_types[3] == 'berhu':
        #     self.depth_loss = Berhu_Loss()
        # if loss_types[3] == 'inv_sig':
        #     self.depth_loss = Inverse_Sigmoid_Loss()
        # elif loss_types[3] == 'log':
        #     self.depth_loss = Log_L1_Loss()
        # elif loss_types[3] == 'L1':
        # self.depth_loss = nn.L1Loss(reduction='none')
        # else:
        #     raise ValueError

        # regular regression loss
        self.reg_loss = loss_types[1]
        self.reg_loss_fnc = nn.L1Loss(reduction='none') if loss_types[1] == 'L1' else nn.SmoothL1Loss
        self.keypoint_loss_fnc = nn.L1Loss(reduction='none')

        # multi-bin loss setting for orientation estimation
        self.multibin = (cfg.INPUT.ORIENTATION == 'multi-bin')
        self.orien_bin_size = cfg.INPUT.ORIENTATION_BIN_SIZE
        self.trunc_offset_loss_type = cfg.MODEL.HEAD.TRUNCATION_OFFSET_LOSS
        self.td_dim_index = self.key2channel('2d_dim')  #self.key2channel('2d_dim')
        self.tid_dim_index = self.key2channel('3d_dim')  #(49,52,1)
        self.td_offset_index = self.key2channel('3d_offset') #(4,6,1)
        self.ori_cls_index = self.key2channel('ori_cls')  #(52,60,1)
        self.ori_offset_index = self.key2channel('ori_offset') #(60,68,1)
        self.corner_offset_index = self.key2channel('corner_offset')  #(6,26,1)
        self.combined_depth_uncern_index = self.key2channel('combined_depth_uncern')  #(70,71,1)
        self.corner_loss_uncern_index = self.key2channel('corner_loss_uncern') #(70,72,1)
        self.corner_uncertainty_index = self.key2channel('corner_uncertainty')  #(26,29,1)
        # if 'GRM_uncern' in self.ct_keys:
        #     self.GRM_uncern_index = self.key2channel('GRM_uncern')
        self.GRM1_uncern_index = self.key2channel('GRM1_uncern')  #(29,37,1)
        self.GRM2_uncern_index = self.key2channel('GRM2_uncern')  #(37,45,1)
        self.Mono_Direct_uncern_index = self.key2channel('Mono_Direct_uncern')  #(45,46,1)
        self.Mono_Keypoint_uncern_index = self.key2channel('Mono_Keypoint_uncern')  #(46,49,1)
        self.depth_index = self.key2channel('depth')  #(68,69,1)
        self.depth_uncertainty_index = self.key2channel('depth_uncertainty')  #(69,70,1)

        # whether to compute corner loss
        self.compute_direct_depth_loss = 'depth_loss' in self.loss_keys
        self.compute_keypoint_depth_loss = 'keypoint_depth_loss' in self.loss_keys
        self.compute_weighted_depth_loss = 'weighted_avg_depth_loss' in self.loss_keys
        self.compute_corner_loss = 'corner_loss' in self.loss_keys
        self.separate_trunc_offset = 'trunc_offset_loss' in self.loss_keys
        self.compute_combined_depth_loss = 'combined_depth_loss' in self.loss_keys
        self.compute_GRM_loss = 'GRM_loss' in self.loss_keys
        self.compute_SoftGRM_loss = 'SoftGRM_loss' in self.loss_keys
        self.compute_IOU3D_predict_loss = 'IOU3D_predict_loss' in self.loss_keys

        # corner_with_uncertainty is whether to use corners with uncertainty to solve the depth, rather than directly applying uncertainty to corner estimation itself.
        self.pred_direct_depth = 'depth' in self.ct_keys
        self.depth_with_uncertainty = 'depth_uncertainty' in self.ct_keys
        self.compute_keypoint_corner = 'corner_offset' in self.ct_keys
        self.corner_with_uncertainty = 'corner_uncertainty' in self.ct_keys

        self.corner_offset_uncern = 'corner_offset_uncern' in self.ct_keys
        self.dim_uncern = '3d_dim_uncern' in self.ct_keys
        self.combined_depth_uncern = 'combined_depth_uncern' in self.ct_keys
        self.corner_loss_uncern = 'corner_loss_uncern' in self.ct_keys

        self.perdict_IOU3D = 'IOU3D_predict' in self.ct_keys

        self.uncertainty_weight = cfg.MODEL.HEAD.UNCERTAINTY_WEIGHT  # 1.0
        self.keypoint_xy_weights = cfg.MODEL.HEAD.KEYPOINT_XY_WEIGHT  # [1, 1]
        self.keypoint_norm_factor = cfg.MODEL.HEAD.KEYPOINT_NORM_FACTOR  # 1.0
        self.modify_invalid_keypoint_depths = cfg.MODEL.HEAD.MODIFY_INVALID_KEYPOINT_DEPTH

        # depth used to compute 8 corners
        self.corner_loss_depth = cfg.MODEL.HEAD.CORNER_LOSS_DEPTH
        self.SoftGRM_loss_weight = ms.Tensor(self.cfg.MODEL.HEAD.SOFTGRM_LOSS_WEIGHT,self.ms_type)
        self.dynamic_thre = cfg.SOLVER.DYNAMIC_THRESHOLD
        self.clip_min=ms.Tensor(1,self.ms_type)

        self.exp = ops.Exp()
        self.expand_dims=ops.ExpandDims()
        self.log = ops.Log()
        self.div = ops.Div()
        self.concat = ops.Concat(axis=1)
        self.sin = ops.Sin()
        self.cos = ops.Cos()
        self.cast=ops.Cast()
        self.minimum=ops.Minimum()
        self.maximum=ops.Maximum()
        self.zeros=ops.Zeros()
        self.relu=ops.ReLU()

        self.reducesum = ops.ReduceSum()
        self.reducesum_t=ops.ReduceSum(True)
        self.reducemean = ops.ReduceMean()
        self.ones = ops.Ones()
        self.concat_0=ops.Concat(0)

        self.sigmoid = ops.Sigmoid()
        self.zeros = ops.Zeros()
        self.l2_norm = ops.L2Normalize()
        self.softmax_axis1 = nn.Softmax(axis=1)
        self.softmax_axis2 = nn.Softmax(axis=2)
        self.gather_nd = ops.GatherNd()
        self.atan2 = ops.Atan2()
        self.nonzero = ops.NonZero()
        self.cast = ops.Cast()
        self.stack_1=ops.Stack(1)
        self.stack_0 = ops.Stack(0)

        self.EPS = 1
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
        self.PI=np.pi

        # orientation related
        self.alpha_centers = ms.Tensor(np.array([0, self.PI / 2, self.PI, - self.PI / 2]), self.ms_type)
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
        self.print=ops.Print()
        self.tile=ops.Tile()
        self.reshape=ops.Reshape()
        # self.gather_nd=ops.GatherNd()
        self.muti_offset_loss=Muti_offset_loss(self.reg_loss_fnc,self.loss_weights['offset_loss'], self.loss_weights['trunc_offset_loss'])
        self.reg2D_loss=Reg2D_loss(loss_types[2],self.loss_weights['bbox_loss'])
        self.depth3D_loss=Depth3D_loss(self.key2channel('depth'),self.key2channel('depth_uncertainty'),
                                       cfg.MODEL.HEAD.UNCERTAINTY_RANGE,self.loss_weights['depth_loss'],cfg.MODEL.HEAD.DEPTH_MODE,
                                       self.depth_ref,self.depth_range,self.depth_with_uncertainty)
        self.orien3D_loss=Orien3D_loss(self.loss_weights['orien_loss'],self.orien_bin_size)
        self.dim_mean = ops.Tensor(cfg.MODEL.HEAD.DIMENSION_MEAN)
        self.dim3D_loss=Dim3D_loss(self.reg_loss_fnc,self.key2channel('3d_dim'),self.loss_weights['dims_loss'],self.dim_weight,self.dim_modes,self.dim_mean,self.dim_std)
        self.corner3D_loss=Corner3D_loss(self.reg_loss_fnc,self.loss_weights['corner_loss'])
        self.keypoint_loss=Keypoint_loss(self.keypoint_loss_fnc,self.loss_weights['keypoint_loss'])
        self.keypoint_depth_loss=Keypoint_depth_loss(self.reg_loss_fnc,self.loss_weights['keypoint_depth_loss'])


    def key2channel(self, key):
        # find the corresponding index
        index = self.ct_keys.index(key)

        s = sum(self.ct_channels[:index])
        e = s + self.ct_channels[index]

        return slice(s, e, 1)


    def construct(self, targets_original,targets_select,calibs, pred_regression, iteration):
        self.print('compute loss')
        batch, channel, feat_h, feat_w = pred_regression.shape

        # 2. extract corresponding predictions
        flatten_reg_mask_gt=targets_select[19]
        mask_regression_2D=targets_select[18]
        ops.print_('pred_regression max:', pred_regression.max())
        pred_regression_pois_3D=self.reshape(select_point_of_interest(batch, targets_original[1], pred_regression),(-1, channel))
        ops.print_('pred_regression_pois_3D max:', pred_regression_pois_3D.max())
        pred_regression_pois_3D=self.gather_nd(pred_regression_pois_3D,flatten_reg_mask_gt)   # pred_regression_pois_3D shape: (valid_objs, C)
        pred_regression_2D = self.gather_nd(pred_regression_pois_3D, mask_regression_2D)[::, self.td_dim_index]
        pred_regression_2D = self.relu(pred_regression_2D)  # pred_regression_2D shape: (valid_objs, 4)

        pred_offset_3D = pred_regression_pois_3D[::, self.td_offset_index]  # pred_offset_3D shape: (valid_objs, 2)
        pred_dimensions_offsets_3D = pred_regression_pois_3D[::, self.tid_dim_index]  # pred_dimensions_offsets_3D shape: (valid_objs, 3)
        # decode the pred residual dimensions to real dimensions
        pred_dimensions_3D = self.decode_dimension(targets_select[9], pred_dimensions_offsets_3D)

        pred_orientation_3D = self.concat((pred_regression_pois_3D[::, self.ori_cls_index], pred_regression_pois_3D[::,
                                                                                            self.ori_offset_index]))  # pred_orientation_3D shape: (valid_objs, 16)
        pred_depths_offset_3D = pred_regression_pois_3D[:, self.depth_index].squeeze(-1)
        pred_direct_depths_3D = self.decode_depth(pred_depths_offset_3D)
        preds_depth_uncertainty = pred_regression_pois_3D[:, self.depth_uncertainty_index].squeeze(1)  # preds['depth_uncertainty'] shape: (val_objs,)

        pred_keypoints_3D = pred_regression_pois_3D[::, self.corner_offset_index]
        pred_keypoints_3D = self.reshape(pred_keypoints_3D,(flatten_reg_mask_gt.shape[0], -1, 2))
        preds_keypoints = pred_keypoints_3D  # pred_keypoints_3D shape: (val_objs, 10, 2)
        pred_keypoints_depths_3D = self.decode_depth_from_keypoints_batch(pred_keypoints_3D,
                                                                          pred_dimensions_3D,
                                                                          calibs,
                                                                          targets_select[20])
        preds_keypoints_depths = pred_keypoints_depths_3D  # pred_keypoints_depths_3D shape: (val_objs, 3)

        # Optimize combined_depth with uncertainty
        # if self.combined_depth_uncern:
        combined_depth_uncern = pred_regression_pois_3D[::, self.combined_depth_uncern_index]
        preds_combined_depth_uncern = self.exp(ops.clip_by_value(combined_depth_uncern, self.uncertainty_range[0], self.uncertainty_range[1]))

        # Optimize corner coordinate loss with uncertainty
        # if self.corner_loss_uncern:
        corner_loss_uncern = pred_regression_pois_3D[::, self.corner_loss_uncern_index]
        preds_corner_loss_uncern = self.exp(ops.clip_by_value(corner_loss_uncern, self.uncertainty_range[0], self.uncertainty_range[1]))

        # predict the uncertainties of the solved depths from groups of keypoints
        # if self.corner_with_uncertainty:
        preds_corner_offset_uncertainty = pred_regression_pois_3D[::, self.corner_uncertainty_index]  # preds['corner_offset_uncertainty'] shape: (val_objs, 3)

        # if self.uncertainty_range is not None:
        preds_corner_offset_uncertainty = ops.clip_by_value(preds_corner_offset_uncertainty, self.uncertainty_range[0],
                                                            self.uncertainty_range[1])

        uncern_GRM1 = pred_regression_pois_3D[::, self.GRM1_uncern_index]  # uncern_GRM1 shape: (num_objs, 8)
        uncern_GRM2 = pred_regression_pois_3D[::, self.GRM2_uncern_index]  # uncern_GRM1 shape: (num_objs, 8)
        uncern_Mono_Direct = pred_regression_pois_3D[::,
                             self.Mono_Direct_uncern_index]  # uncern_Mono_Direct shape: (num_objs, 1)
        uncern_Mono_Keypoint = pred_regression_pois_3D[::,
                               self.Mono_Keypoint_uncern_index]  # uncern_Mono_Keypoint shape: (num_objs, 3)
        GRM_uncern = self.concat((uncern_GRM1, uncern_GRM2))
        GRM_uncern = self.concat((GRM_uncern, uncern_Mono_Direct, uncern_Mono_Keypoint))  # GRM_uncern shape: (num_objs, 20)
        GRM_uncern = self.exp(ops.clip_by_value(GRM_uncern, self.uncertainty_range[0], self.uncertainty_range[1]))
        assert GRM_uncern.shape[1] == 20

        pred_combined_depths = self.concat((ops.expand_dims(pred_direct_depths_3D, 1), pred_keypoints_depths_3D))  # pred_combined_depths shape: (valid_objs, 4)

        info_dict = (targets_select[4], targets_select[1],targets_original[4],calibs, targets_select[20])
        GRM_rotys, _ = self.decode_axes_orientation(pred_orientation_3D, dict_for_3d_center=info_dict)

        pred_vertex_offset = pred_keypoints_3D[:, 0:8, :]  # Do not use the top center and bottom center.
        pred_corner_depth_3D, separate_depths = self.decode_from_SoftGRM(ops.expand_dims(GRM_rotys, 1),
                                                                         pred_dimensions_3D,
                                                                         pred_vertex_offset.reshape(-1, 16),
                                                                         pred_combined_depths,
                                                                         targets_dict=info_dict,
                                                                         GRM_uncern=GRM_uncern,
                                                                         batch_idxs=targets_select[20])  # pred_corner_depth_3D shape: (val_objs,), separate_depths shape: (val_objs, 20)

        pred_locations_3D = self.decode_location_flatten(targets_select[4], pred_offset_3D,
                                                         pred_corner_depth_3D,
                                                         calibs,
                                                         targets_original[4],
                                                         targets_select[20])  # pred_locations_3D shape: (val_objs, 3)
        # decode rotys and alphas
        pred_rotys_3D, _ = self.decode_axes_orientation(pred_orientation_3D,
                                                        locations=pred_locations_3D)  # pred_rotys_3D shape: (val_objs,)
        # encode corners
        pred_corners_3D = self.encode_box3d(pred_rotys_3D, pred_dimensions_3D,
                                            pred_locations_3D)  # pred_corners_3D shape: (val_objs, 8, 3)
        # concatenate all predictions
        # pred_bboxes_3D = self.concat((pred_locations_3D, pred_dimensions_3D, pred_rotys_3D[::, None]))  # pred_bboxes_3D shape: (val_objs, 7)

        target_corner_keypoints = self.gather_nd(targets_original[2].view((self.max_objs * batch, -1, 3)), flatten_reg_mask_gt)  # target_corner_keypoints shape: (val_objs, 10, 3)
        targets_keypoints = target_corner_keypoints[..., :2]  # targets['keypoints'] shape: (val_objs, 10, 2)
        targets_keypoints_mask = target_corner_keypoints[..., -1]  # targets['keypoints_mask'] shape: (val_objs, 10)
        # mask for whether depth should be computed from certain group of keypoints
        target_corner_depth_mask = self.gather_nd(targets_original[3].view(-1, 3), flatten_reg_mask_gt)
        # targets_keypoints_depth_mask = target_corner_depth_mask  # target_corner_depth_mask shape: (val_objs, 3)
        #
        num_reg_2D = targets_select[18].shape[0]
        num_reg_3D = targets_select[19].shape[0]

        trunc_mask = targets_select[11]
        output=[]
        # IoU loss for 2D detection
        if num_reg_3D > 0:
            self.print('pred_direct_depths_3D',pred_direct_depths_3D)
            depth_3D_loss=self.depth3D_loss(pred_direct_depths_3D,preds_depth_uncertainty,targets_select[2])
            self.print('depth_3D_loss:', depth_3D_loss)
            offset_3D_trunc_loss=self.muti_offset_loss(targets_select[1],pred_offset_3D,trunc_mask)
            self.print('offset_3D_trunc_loss:', offset_3D_trunc_loss)
            # orientation loss
            orien_3D_loss=self.orien3D_loss(pred_orientation_3D,targets_select[3])
            self.print('orien_3D_loss:', orien_3D_loss)
            # dimension loss
            dims_3D_loss=self.dim3D_loss(pred_dimensions_3D,targets_select[5])
            self.print('dims_3D_loss:', dims_3D_loss)
            # corner loss
            # if self.compute_corner_loss:
                # N x 8 x 3
            weight_ramper=self.loss_weight_ramper(iteration)
            corner_3D_loss=self.corner3D_loss(pred_corners_3D,preds_corner_loss_uncern,targets_select[6],weight_ramper)
            self.print('corner_3D_loss:',corner_3D_loss)
            # output = [depth_3D_loss, offset_3D_trunc_loss, dims_3D_loss, corner_3D_loss]
            output=[depth_3D_loss,offset_3D_trunc_loss,orien_3D_loss,dims_3D_loss,corner_3D_loss]

            # else:
                # N x K x 3
            self.print('preds_keypoints',preds_keypoints)
            preds_keypoints=ops.clip_by_value(preds_keypoints,clip_value_max=ms.Tensor(10000,ms.float32))
            keypoint_loss=self.keypoint_loss(preds_keypoints, targets_keypoints,targets_keypoints_mask)
            output.append(keypoint_loss)
            self.print('keypoint_loss:',keypoint_loss)
            # if self.compute_keypoint_depth_loss:
            self.print('preds_keypoints_depths', preds_keypoints_depths)
            keypoint_depth_loss=self.keypoint_depth_loss(preds_keypoints_depths,target_corner_depth_mask,preds_corner_offset_uncertainty,targets_select[2])
            keypoint_depth_loss=ops.clip_by_value(keypoint_depth_loss,clip_value_max=ms.Tensor(1000,self.ms_type))
            output.append(keypoint_depth_loss)
            self.print('keypoint_depth_loss:', keypoint_depth_loss)

            # if self.corner_with_uncertainty:
            if self.pred_direct_depth and self.depth_with_uncertainty:
                combined_depth = self.concat((ops.expand_dims(pred_direct_depths_3D, 1), (preds_keypoints_depths)))
                depth_uncertainty=ops.expand_dims(preds_depth_uncertainty, 1)
                uncertainty=self.concat((depth_uncertainty,preds_corner_offset_uncertainty))
                combined_uncertainty = self.exp(uncertainty)
            else:
                combined_depth = preds_keypoints_depths
                combined_uncertainty = self.exp(preds_corner_offset_uncertainty)

            combined_weights = 1 / combined_uncertainty
            combined_weights = combined_weights / combined_weights.sum(axis=1, keepdims=True)
            soft_depths = self.reducesum(combined_depth * combined_weights, 1)

            if self.compute_weighted_depth_loss:
                soft_depth_loss = self.loss_weights['weighted_avg_depth_loss'] * \
                                  self.reg_loss_fnc(soft_depths, targets_select[2])
                output.append(soft_depth_loss)
                self.print('soft_depth_loss:',soft_depth_loss)

            # if self.compute_combined_depth_loss:  # The loss for final estimated depth.
            combined_depth_loss = self.reg_loss_fnc(pred_corner_depth_3D, targets_select[2])
            # if self.combined_depth_uncern:
            combined_depth_uncern = preds_combined_depth_uncern.squeeze(1)
            combined_depth_loss = combined_depth_loss / combined_depth_uncern + self.log(combined_depth_uncern)
            # if self.cfg.SOLVER.DYNAMIC_WEIGHT:
            #     combined_depth_loss = self.reweight_loss(combined_depth_loss, objs_weight)
            combined_depth_loss = self.loss_weight_ramper(iteration) * self.loss_weights['combined_depth_loss'] * self.reducemean(combined_depth_loss)
            output.append(combined_depth_loss)
            self.print('combined_depth_loss:',combined_depth_loss)

            # if self.compute_SoftGRM_loss:
            GRM_valid_items = targets_select[13]  # GRM_valid_items shape: (val_objs, 20)
            # GRM_valid_item_sum = ms.Tensor(np.array([GRM_valid_items]).sum(), self.ms_type)
            GRM_valid_item_sum=self.cast(GRM_valid_items,self.ms_type).sum()
            GRM_valid_items_inverse_sum=self.cast(~GRM_valid_items,self.ms_type).sum()
            GRM_valid_items_inverse = self.cast(self.nonzero((~GRM_valid_items)),ms.int32)
            # GRM_valid_items_inverse_sum=np.array(GRM_valid_items_inverse).sum()
            GRM_valid_items=self.cast(self.nonzero(GRM_valid_items),ms.int32)
            # GRM_valid_items_inverse=GRM_valid_items_inverse.tolist()
            separate_depths = separate_depths  # separate_depths shape: (val_objs, 20)
            depth_3D=ops.expand_dims(targets_select[2], 1).expand_as(separate_depths)
            valid_target_depth = self.gather_nd(depth_3D,GRM_valid_items)  # shape: (valid_equas,)
            # sd_shape=separate_depths.shape
            valid_separate_depths = self.gather_nd(separate_depths,GRM_valid_items)  # shape: (valid_equas,)
            # GRM_uncern = GRM_uncern
            SoftGRM_weight = ops.function.broadcast_to(ops.expand_dims(self.SoftGRM_loss_weight, 0), separate_depths.shape)
            valid_SoftGRM_weight = self.gather_nd(SoftGRM_weight,GRM_valid_items)

            valid_uncern = self.gather_nd(GRM_uncern, GRM_valid_items)  # shape: (valid_equas,)
            valid_SoftGRM_loss = self.reg_loss_fnc(valid_separate_depths, valid_target_depth) / valid_uncern + self.log(
                valid_uncern)
            valid_SoftGRM_loss = (valid_SoftGRM_loss * valid_SoftGRM_weight).sum() / ops.clip_by_value(
                GRM_valid_item_sum, self.clip_min)

            if GRM_valid_items_inverse_sum>0:
                invalid_SoftGRM_weight = self.gather_nd(SoftGRM_weight,GRM_valid_items_inverse)
                invalid_target_depth = self.gather_nd(depth_3D,GRM_valid_items_inverse)  # shape: (invalid_equas,) problem~
                invalid_separate_depths = self.gather_nd(separate_depths,GRM_valid_items_inverse)  # shape: (invalid_equas,) ~
                invalid_separate_depths=ops.stop_gradient(invalid_separate_depths)
                invalid_uncern = self.gather_nd(GRM_uncern,GRM_valid_items_inverse)  # shape: (invalid_equas,)
                invalid_SoftGRM_loss = self.reg_loss_fnc(invalid_separate_depths, invalid_target_depth), invalid_uncern
                invalid_SoftGRM_loss = (invalid_SoftGRM_loss * invalid_SoftGRM_weight).sum()/ ops.clip_by_value(GRM_valid_items_inverse_sum, self.clip_min)  # Avoid the occasion that no invalid equations and the returned value is NaN.
                SoftGRM_loss = (self.loss_weight_ramper(iteration) * self.loss_weights['SoftGRM_loss'] * (
                            valid_SoftGRM_loss + invalid_SoftGRM_loss))
            else:
                SoftGRM_loss = self.loss_weight_ramper(iteration) * self.loss_weights['SoftGRM_loss'] * valid_SoftGRM_loss
            output.append(SoftGRM_loss)
            self.print('SoftGRM_loss:', SoftGRM_loss)
        if num_reg_2D > 0:
            reg_2D_loss = self.reg2D_loss(pred_regression_2D, targets_select[0], mask_regression_2D)
            self.print('reg_2D_loss:', reg_2D_loss)
            output.append(reg_2D_loss)

        return output

    def reweight_loss(self,loss, weight):
        '''
        Description:
            Reweight loss by weight.
        Input:
            loss: Loss vector. shape: (val_objs,).
            weight: Weight vector. shape: (val_objs,).
        Output:
            w_loss: Float.
        '''
        w_loss = loss * weight
        return w_loss

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
        index = ops.tile(ms.Tensor([[4, 5, 0, 1, 6, 7, 2, 3],
                                     [0, 1, 2, 3, 4, 5, 6, 7],
                                     [4, 0, 1, 5, 6, 2, 3, 7]], ms.int32),(N, 1))

        box_3d_object = ops.gather_elements(dims_corners, 1, index)
        b = box_3d_object.view((N, 3, -1))
        box_3d = ops.matmul(ry, b)  # ry:[11,3,3]   box_3d_object:[11,3,8]
        box_3d =box_3d + ops.tile(ops.expand_dims(locs, -1),(1, 1, 8))

        return ops.transpose(box_3d, (0, 2, 1))

    # def decode_box2d_fcos(self, centers, pred_offset, pad_size=None, out_size=None):
    #     box2d_center = centers.view(-1, 2)
    #     box2d = self.zeros((box2d_center.shape[0], 4),self.ms_type)
    #     # left, top, right, bottom
    #     box2d[:, :2] = box2d_center - pred_offset[:, :2]
    #     box2d[:, 2:] = box2d_center + pred_offset[:, 2:]
    #
    #     # for inference
    #     if pad_size is not None:
    #         N = box2d.shape[0]
    #         # upscale and subtract the padding
    #         box2d = box2d * self.down_ratio - ops.tile(pad_size,(1, 2))
    #         # clamp to the image bound
    #         box2d[:, 0::2]=ops.clip_by_value(box2d[:, 0::2],ms.Tensor(0,self.ms_type), ms.Tensor(out_size[0] - 1,self.ms_type))
    #         box2d[:, 1::2]=ops.clip_by_value(box2d[:, 1::2],ms.Tensor(0,self.ms_type), ms.Tensor(out_size[1] - 1,self.ms_type))
    #
    #     return box2d

    @staticmethod
    def rad_to_matrix(rotys, N):
        # device = rotys.device

        cos, sin = ops.cos(rotys), ops.sin(rotys)

        i_temp = ops.Tensor([[1, 0, 1],
                             [0, 1, 0],
                             [-1, 0, 1]], dtype=ms.float32)

        ry = ops.tile(i_temp,(N, 1)).view(N, -1, 3)

        ry[:, 0, 0] *= cos
        ry[:, 0, 2] *= sin
        ry[:, 2, 0] *= sin
        ry[:, 2, 2] *= cos

        return ry

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

    def decode_location_flatten(self, points, offsets, depths, calibs, pad_size, batch_idxs):
        # batch_size = len(calibs)
        gts = ops.unique(batch_idxs)[0].asnumpy().tolist()
        locations = self.zeros((points.shape[0], 3), ms.float32)
        points = (points + offsets) * self.down_ratio - pad_size[batch_idxs]  # Left points: The 3D centers in original images.

        for idx, gt in enumerate(gts):
            corr_pts_idx = ops.cast(ops.nonzero((batch_idxs == gt)),ms.int32).squeeze(-1)
            calib = calibs[gt]
            # concatenate uv with depth
            corr_pts_depth = self.concat((points[corr_pts_idx], depths[corr_pts_idx, None]))
            locations[corr_pts_idx] = project_image_to_rect(corr_pts_depth, calib)
        return locations

    def decode_depth_from_keypoints(self, pred_offsets, pred_keypoints, pred_dimensions, calibs, avg_center=False):
        # pred_keypoints: K x 10 x 2, 8 vertices, bottom center and top center
        assert len(calibs) == 1  # for inference, batch size is always 1

        calib = calibs[0]
        # we only need the values of y
        pred_height_3D = pred_dimensions[:, 1]
        pred_keypoints = pred_keypoints.view((-1, 10, 2))
        # center height -> depth
        if avg_center:
            updated_pred_keypoints = pred_keypoints - pred_offsets.view(-1, 1, 2)
            center_height = updated_pred_keypoints[:, -2:, 1]
            center_depth = calib['f_v'] * ops.expand_dims(pred_height_3D, -1) / (center_height.abs() * self.down_ratio * 2)
            center_depth = center_depth.mean(1)
        else:
            center_height = pred_keypoints[:, -2, 1] - pred_keypoints[:, -1, 1]
            center_depth = calib['f_v'] * pred_height_3D / (center_height.abs() * self.down_ratio)

        # corner height -> depth
        corner_02_height = pred_keypoints[:, [0, 2], 1] - pred_keypoints[:, [4, 6], 1]
        corner_13_height = pred_keypoints[:, [1, 3], 1] - pred_keypoints[:, [5, 7], 1]
        corner_02_depth = calib['f_v'] * ops.expand_dims(pred_height_3D,-1) / (corner_02_height * self.down_ratio)
        corner_13_depth = calib['f_v'] * ops.expand_dims(pred_height_3D,-1) / (corner_13_height * self.down_ratio)
        corner_02_depth = corner_02_depth.mean(1)
        corner_13_depth = corner_13_depth.mean(1)
        # K x 3
        pred_depths = self.stack_1((center_depth, corner_02_depth, corner_13_depth))

        return pred_depths


    def decode_depth_from_keypoints_batch(self, pred_keypoints, pred_dimensions, calibs, batch_idxs=None):
        # pred_keypoints: K x 10 x 2, 8 vertices, bottom center and top center (bottom first)
        # pred_keypoints[k,10,2]
        # pred_dimensions[k,3]
        pred_height_3D = pred_dimensions[:, 1]  # [k,]
        batch_size = len(calibs)
        if batch_size == 1:
            batch_idxs = self.zeros(pred_dimensions.shape[0], self.ms_type)

        center_height = pred_keypoints[::, -2, 1] - pred_keypoints[::, -1, 1]  # [2]

        corner_02_height = pred_keypoints[::, [0, 2], 1] - pred_keypoints[::, [4, 6], 1]  # [2,2]
        corner_13_height = pred_keypoints[::, [1, 3], 1] - pred_keypoints[::, [5, 7], 1]  # [2,2]

        center= []
        corner_02= []
        corner_13= []
        emu_item=ops.unique(ops.cast(batch_idxs, ms.int32))[0].asnumpy().tolist()
        idx=0
        for gt_idx in emu_item:
            calib = calibs[idx]
            corr_pts_idx = self.cast(self.nonzero(batch_idxs == gt_idx),ms.int32)
            center_depth = calib['f_v'] * self.gather_nd(pred_height_3D,corr_pts_idx) / (
                    self.relu(self.gather_nd(center_height,corr_pts_idx)) * self.down_ratio + self.EPS)
            corner_02_depth = calib['f_v'] * ops.expand_dims(self.gather_nd(pred_height_3D,corr_pts_idx), -1) / (
                    self.relu(self.gather_nd(corner_02_height,corr_pts_idx)) * self.down_ratio + self.EPS)
            corner_13_depth = calib['f_v'] * ops.expand_dims(self.gather_nd(pred_height_3D,corr_pts_idx), -1) / (
                    self.relu(self.gather_nd(corner_13_height,corr_pts_idx)) * self.down_ratio + self.EPS)

            corner_02_depth = corner_02_depth.mean(1)
            corner_13_depth = corner_13_depth.mean(1)

            center.append(center_depth)
            corner_02.append(corner_02_depth)
            corner_13.append(corner_13_depth)
            idx+=1
        # for items in pred_keypoint_depths:
        pred_keypoint_depths_center = ops.clip_by_value(self.concat_0(center), self.depth_range[0], self.depth_range[1])
        pred_keypoint_depths_corner_02 = ops.clip_by_value(self.concat_0(corner_02), self.depth_range[0], self.depth_range[1])
        pred_keypoint_depths_corner_13 = ops.clip_by_value(self.concat_0(corner_13), self.depth_range[0], self.depth_range[1])
        # for key, depths in pred_keypoint_depths.items():
        #     pred_keypoint_depths[key] = ops.clip_by_value(ops.concat(depths), self.depth_range[0], self.depth_range[1])
        # pred_depths = ops.stack(([depth for depth in pred_keypoint_depths.values()]), axis=1)
        pred_depths=self.stack_1((pred_keypoint_depths_center,pred_keypoint_depths_corner_02,pred_keypoint_depths_corner_13))

        return pred_depths


    def decode_dimension(self, cls_id, dims_offset):
        '''
        retrieve object dimensions
        Args:
            cls_id: each object id
            dims_offset: dimension offsets, shape = (N, 3)

        Returns:

        '''
        cls_dimension_mean = self.dim_mean[cls_id]

        # if self.dim_modes[0] == 'exp':
        dims_offset = self.exp(dims_offset)

        # if self.dim_modes[2]:
        #     cls_dimension_std = self.dim_std[cls_id, :]
        #     dimensions = dims_offset * cls_dimension_std + cls_dimension_mean
        # else:
        dimensions = dims_offset * cls_dimension_mean

        return dimensions

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
        # if self.multibin:
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

        if locations is not None:  # Compute rays based on 3D locations.
            locations = locations.view(-1, 3)
            rays = self.atan2(locations[:, 0], locations[:, 2])
        elif dict_for_3d_center is not None:  # Compute rays based on 3D centers projected on 2D plane.
            if len(dict_for_3d_center[3]) == 1:  # Batch size is 1.
                batch_idxs = self.zeros((vector_ori.shape[0],), ms.uint8)
            else:
                batch_idxs = dict_for_3d_center[4]
            ops.print_('offset_3D:', dict_for_3d_center[1])
            centers_3D = self.decode_3D_centers(dict_for_3d_center[0], dict_for_3d_center[1],
                                                dict_for_3d_center[2], batch_idxs)
            ops.print_('centers_3D:',centers_3D)
            centers_3D_x = centers_3D[:, 0]  # centers_3D_x shape: (total_num_objs,)

            c_u = self.stack_0([calib['c_u'] for calib in dict_for_3d_center[3]])
            f_u = self.stack_0([calib['f_u'] for calib in dict_for_3d_center[3]])
            # b_x = ops.stack([calib['b_x'] for calib in dict_for_3d_center['calib']],axis=0)

            rays = self.zeros(orientations.shape,self.ms_type)
            for idx, gt_idx in enumerate(ops.unique(batch_idxs)[0]):
                corr_idx = self.nonzero(batch_idxs == gt_idx).squeeze(1)
                corr_idx=self.cast(corr_idx,ms.int32)
                centers_3D_x=centers_3D_x[corr_idx]
                centers_3D_x_atan=self.atan2(centers_3D_x - c_u[idx], f_u[idx])
                # print()
                corr_idx=self.expand_dims(corr_idx,1)
                rays=ops.tensor_scatter_add(rays, corr_idx, centers_3D_x_atan)  # This is exactly an approximation.
        else:
            locations = locations.view(-1, 3)
            rays = self.zeros(orientations.shape, self.ms_type)
        #     raise Exception("locations and dict_for_3d_center should not be None simultaneously.")
        alphas = orientations
        rotys = alphas + rays

        larger_idx = self.cast(self.nonzero(rotys > self.PI),ms.int32)
        small_idx = self.cast(self.nonzero(rotys < -self.PI),ms.int32)
        if larger_idx.shape[0]>0:
            larger_idx=larger_idx.squeeze(1)
            rotys[larger_idx]-=2 * self.PI
        if small_idx.shape[0] > 0:
            small_idx=small_idx.squeeze(1)
            rotys[small_idx]+=2 * self.PI

        larger_idx = self.cast(self.nonzero(alphas > self.PI),ms.int32)
        small_idx = self.cast(self.nonzero(alphas < -self.PI),ms.int32)
        if larger_idx.shape[0]>0:
            larger_idx = larger_idx.squeeze(1)
            alphas[larger_idx] -= 2 * self.PI
        if small_idx.shape[0] > 0:
            small_idx = small_idx.squeeze(1)
            alphas[small_idx] += 2 * self.PI
        return rotys, alphas

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

        for idx, gt_idx in enumerate(ops.unique(batch_idxs)[0].asnumpy().tolist()):
            corr_idx = self.nonzero((batch_idxs == gt_idx))
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

        for idx, gt_idx in enumerate(ops.unique(ops.cast(batch_idxs, ms.int32))[0].asnumpy().tolist()):
            corr_idx = self.cast(self.nonzero(batch_idxs == gt_idx), ms.int32)
            pred_keypoints_i=self.gather_nd(pred_keypoints,corr_idx)
            corr_idx=corr_idx.squeeze(1)
            pred_keypoints[corr_idx] = pred_keypoints_i * self.down_ratio - pad_size[idx]

        return pred_keypoints

    # def decode_from_GRM(self, pred_rotys, pred_dimensions, pred_keypoint_offset, pred_direct_depths, targets_dict,
    #                     GRM_uncern=None, GRM_valid_items=None,
    #                     batch_idxs=None, cfg=None):
    #     '''
    #     Description:
    #         Compute the 3D locations based on geometric constraints.
    #     Input:
    #         pred_rotys: The predicted global orientation. shape: (total_num_objs, 1)
    #         pred_dimensions: The predicted dimensions. shape: (num_objs, 3)
    #         pred_keypoint_offset: The offset of keypoints related to target centers on the feature maps. shape: (total_num_obs, 20)
    #         pred_direct_depths: The directly estimated depth of targets. shape: (total_num_objs, 1)
    #         targets_dict: The dictionary that contains somre required information. It must contains the following 4 items:
    #             targets_dict['target_centers']: Target centers. shape: (valid_objs, 2)
    #             targets_dict['offset_3D']: The offset from target centers to 3D centers. shape: (valid_objs, 2)
    #             targets_dict['pad_size']: The pad size for the original image. shape: (B, 2)
    #             targets_dict['calib']: A list contains calibration objects. Its length is B.
    #         GRM_uncern: The estimated uncertainty of 25 equations in GRM. shape: None or (total_num_objs, 25).
    #         GRM_valid_items: The effectiveness of 25 equations. shape: None or (total_num_objs, 25)
    #         batch_idxs: The batch index of various objects. shape: None or (total_num_objs,)
    #         cfg: The config object. It could be None.
    #     Output:
    #         pinv: The decoded positions of targets. shape: (total_num_objs, 3)
    #         A: Matrix A of geometric constraints. shape: (total_num_objs, 25, 3)
    #         B: Matrix B of geometric constraints. shape: (total_num_objs, 25, 1)
    #     '''
    #     target_centers = targets_dict['target_centers']  # The position of target centers on heatmap. shape: (total_num_objs, 2)
    #     offset_3D = targets_dict['offset_3D']  # shape: (total_num_objs, 2)
    #     calibs = [targets_dict['calib']]  # The list contains calibration objects. Its length is B.
    #     pad_size = targets_dict['pad_size']  # shape: (B, 2)
    #
    #     if GRM_uncern is None:
    #         GRM_uncern = self.ones((pred_rotys.shape[0], 25), self.ms_type)
    #
    #     if len(calibs) == 1:  # Batch size is 1.
    #         batch_idxs = self.ones((pred_rotys.shape[0],), ms.uint8)
    #
    #     if GRM_valid_items is None:  # All equations of GRM is valid.
    #         GRM_valid_items = self.ones((pred_rotys.shape[0], 25), ms.bool_)
    #
    #     # For debug
    #     '''pred_keypoint_offset = targets_dict['keypoint_offset'][:, :, 0:2].contiguous().view(-1, 20)
    #     pred_rotys = targets_dict['rotys'].view(-1, 1)
    #     pred_dimensions = targets_dict['dimensions']
    #     locations = targets_dict['locations']
    #     pred_direct_depths = locations[:, 2].view(-1, 1)'''
    #
    #     pred_keypoints = self.decode_2D_keypoints(target_centers, pred_keypoint_offset,
    #                                               targets_dict['pad_size'],
    #                                               batch_idxs=batch_idxs)  # pred_keypoints: The 10 keypoint position original image. shape: (total_num_objs, 10, 2)
    #
    #     c_u = self.stack_0([calib['c_u'] for calib in calibs])  # c_u shape: (B,)
    #     c_v = self.stack_0([calib['c_v'] for calib in calibs])
    #     f_u = self.stack_0([calib['f_u'] for calib in calibs])
    #     f_v = self.stack_0([calib['f_v'] for calib in calibs])
    #     b_x = self.stack_0([calib['b_x'] for calib in calibs])
    #     b_y = self.stack_0([calib['b_y'] for calib in calibs])
    #
    #     n_pred_keypoints = pred_keypoints  # n_pred_keypoints shape: (total_num_objs, 10, 2)
    #
    #     for idx, gt_idx in enumerate(ops.unique(ops.cast(batch_idxs, ms.int32))[0].asnumpy().tolist()):
    #         # corr_idx = ops.nonzero(batch_idxs == gt_idx).squeeze(-1).astype(ms.float64)
    #         corr_idx = self.cast(self.nonzero(batch_idxs == gt_idx).squeeze(1),ms.int32)
    #         n_pred_keypoints[corr_idx][ :, 0] = (n_pred_keypoints[corr_idx][ :, 0] - c_u[idx]) / f_u[idx]
    #         n_pred_keypoints[corr_idx][ :, 1] = (n_pred_keypoints[corr_idx][ :, 1] - c_v[idx]) / f_v[idx]
    #
    #     total_num_objs = n_pred_keypoints.shape[0]
    #
    #     centers_3D = self.decode_3D_centers(target_centers, offset_3D, pad_size, batch_idxs)  # centers_3D: The positions of 3D centers of objects projected on original image.
    #     n_centers_3D = centers_3D
    #     for idx, gt_idx in enumerate(ops.Unique()(ops.cast(batch_idxs, ms.int32))[0].asnumpy().tolist()):
    #         corr_idx = self.cast(self.nonzero(batch_idxs == gt_idx).squeeze(1),ms.int32)
    #         n_centers_3D[corr_idx][ :,0] = (n_centers_3D[corr_idx][ :,0] - c_u[idx]) / f_u[idx]  # n_centers_3D shape: (total_num_objs, 2)
    #         n_centers_3D[corr_idx][ :,1] = (n_centers_3D[corr_idx][ :,1] - c_v[idx]) / f_v[idx]
    #
    #     kp_group = ops.concat([(ops.Reshape(n_pred_keypoints, (total_num_objs, 20)), n_centers_3D,
    #                          self.zeros((total_num_objs, 2), self.ms_type))],axis=1)  # kp_group shape: (total_num_objs, 24)
    #     coe = self.zeros((total_num_objs, 24, 2), self.ms_type)
    #     coe[:, 0:: 2, 0] = -1
    #     coe[:, 1:: 2, 1] = -1
    #     A = ops.concat((coe, ops.expand_dims(kp_group, 2)), axis=2)
    #     coz = self.zeros((total_num_objs, 1, 3), self.ms_type)
    #     coz[:, :, 2] = 1
    #     A = ops.concat((A, coz), axis=1)  # A shape: (total_num_objs, 25, 3)
    #
    #     pred_rotys = pred_rotys.reshape(total_num_objs, 1)
    #     cos = ops.cos(pred_rotys)  # cos shape: (total_num_objs, 1)
    #     sin = ops.sin(pred_rotys)  # sin shape: (total_num_objs, 1)
    #
    #     pred_dimensions = pred_dimensions.reshape(total_num_objs, 3)
    #     l = pred_dimensions[:, 0: 1]  # h shape: (total_num_objs, 1). The same as w and l.
    #     h = pred_dimensions[:, 1: 2]
    #     w = pred_dimensions[:, 2: 3]
    #
    #     B = self.zeros((total_num_objs, 25, 1), self.ms_type)
    #     B[:, 0, :] = l / 2 * cos + w / 2 * sin
    #     B[:, 2, :] = l / 2 * cos - w / 2 * sin
    #     B[:, 4, :] = -l / 2 * cos - w / 2 * sin
    #     B[:, 6, :] = -l / 2 * cos + w / 2 * sin
    #     B[:, 8, :] = l / 2 * cos + w / 2 * sin
    #     B[:, 10, :] = l / 2 * cos - w / 2 * sin
    #     B[:, 12, :] = -l / 2 * cos - w / 2 * sin
    #     B[:, 14, :] = -l / 2 * cos + w / 2 * sin
    #     B[:, 1: 8: 2, :] = ops.expand_dims((h / 2), 1)
    #     B[:, 9: 16: 2, :] = -ops.expand_dims((h / 2), 1)
    #     B[:, 17, :] = h / 2
    #     B[:, 19, :] = -h / 2
    #
    #     total_num_objs = n_pred_keypoints.shape[0]
    #     pred_direct_depths = pred_direct_depths.reshape(total_num_objs, )
    #     for idx, gt_idx in enumerate(ops.unique(self.cast(batch_idxs,ms.int32))[0]):
    #         corr_idx = self.cast(ops.nonzero(batch_idxs == gt_idx).squeeze(-1),ms.int32)
    #         B[corr_idx][ 22, 0] = -(centers_3D[corr_idx][ 0] - c_u[idx]) * pred_direct_depths[corr_idx] / f_u[idx] - b_x[idx]
    #         B[corr_idx][ 23, 0] = -(centers_3D[corr_idx][ 1] - c_v[idx]) * pred_direct_depths[corr_idx] / f_v[idx] - b_y[idx]
    #         B[corr_idx][ 24, 0] = pred_direct_depths[corr_idx]
    #
    #     C = self.zeros((total_num_objs, 25, 1), self.ms_type)
    #     kps = n_pred_keypoints.view(total_num_objs, 20)  # kps_x shape: (total_num_objs, 20)
    #     C[:, 0, :] = kps[:, 0: 1] * (-l / 2 * sin + w / 2 * cos)
    #     C[:, 1, :] = kps[:, 1: 2] * (-l / 2 * sin + w / 2 * cos)
    #     C[:, 2, :] = kps[:, 2: 3] * (-l / 2 * sin - w / 2 * cos)
    #     C[:, 3, :] = kps[:, 3: 4] * (-l / 2 * sin - w / 2 * cos)
    #     C[:, 4, :] = kps[:, 4: 5] * (l / 2 * sin - w / 2 * cos)
    #     C[:, 5, :] = kps[:, 5: 6] * (l / 2 * sin - w / 2 * cos)
    #     C[:, 6, :] = kps[:, 6: 7] * (l / 2 * sin + w / 2 * cos)
    #     C[:, 7, :] = kps[:, 7: 8] * (l / 2 * sin + w / 2 * cos)
    #     C[:, 8, :] = kps[:, 8: 9] * (-l / 2 * sin + w / 2 * cos)
    #     C[:, 9, :] = kps[:, 9: 10] * (-l / 2 * sin + w / 2 * cos)
    #     C[:, 10, :] = kps[:, 10: 11] * (-l / 2 * sin - w / 2 * cos)
    #     C[:, 11, :] = kps[:, 11: 12] * (-l / 2 * sin - w / 2 * cos)
    #     C[:, 12, :] = kps[:, 12: 13] * (l / 2 * sin - w / 2 * cos)
    #     C[:, 13, :] = kps[:, 13: 14] * (l / 2 * sin - w / 2 * cos)
    #     C[:, 14, :] = kps[:, 14: 15] * (l / 2 * sin + w / 2 * cos)
    #     C[:, 15, :] = kps[:, 15: 16] * (l / 2 * sin + w / 2 * cos)
    #
    #     B = B - C  # B shape: (total_num_objs, 25, 1)
    #
    #     # A = A[:, 22:25, :]
    #     # B = B[:, 22:25, :]
    #
    #     weights = 1 / GRM_uncern  # weights shape: (total_num_objs, 25)
    #
    #     ##############  Block the invalid equations ##############
    #     # A = A * GRM_valid_items.unsqueeze(2)
    #     # B = B * GRM_valid_items.unsqueeze(2)
    #
    #     ##############  Solve pinv for Coordinate loss ##############
    #     A_coor = A
    #     B_coor = B
    #     if cfg is not None and not cfg.MODEL.COOR_ATTRIBUTE:  # Do not use Coordinate loss to train attributes.
    #         A_coor = A_coor.asnumpy()
    #         B_coor = B_coor.asnumpy()
    #
    #     weights_coor = weights
    #     if cfg is not None and not cfg.MODEL.COOR_UNCERN:  # Do not use Coordinate loss to train uncertainty.
    #         weights_coor = weights_coor.asnumpy()
    #
    #     A_coor = A_coor * ops.expand_dims(weights_coor, 2)
    #     B_coor = B_coor * ops.expand_dims(weights_coor, 2)
    #
    #     AT_coor = ops.transpose(A_coor, (0, 2, 1))  # A shape: (total_num_objs, 25, 3)
    #     pinv = ops.bmm(AT_coor, A_coor)
    #     pinv = ops.inverse(pinv)
    #     pinv = ops.bmm(pinv, AT_coor)
    #     pinv = ops.bmm(pinv, B_coor)
    #
    #     ##############  Solve A_uncern and B_uncern for GRM loss ##############
    #     A_uncern = A
    #     B_uncern = B
    #     if cfg is not None and not cfg.MODEL.GRM_ATTRIBUTE:  # Do not use GRM loss to train attributes.
    #         A_uncern = A_uncern.asnumpy()  # .detach()
    #         B_uncern = B_uncern.asnumpy()  # .detach()
    #
    #     weights_uncern = weights
    #     if cfg is not None and not cfg.MODEL.GRM_UNCERN:  # Do not use GRM loss to train uncertainty.
    #         weights_uncern = weights_uncern.asnumpy()  # .detach()
    #
    #     A_uncern = A_uncern * ops.expand_dims(weights_uncern, 2)
    #     B_uncern = B_uncern * ops.expand_dims(weights_uncern, 2)
    #
    #     return pinv.view(-1, 3), A_uncern, B_uncern

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

        target_centers = targets_dict[0]  # The position of target centers on heatmap. shape: (val_objs, 2)
        offset_3D = targets_dict[1]  # shape: (val_objs, 2)
        calibs = targets_dict[3]  # The list contains calibration objects. Its length is B.
        pad_size = targets_dict[2]  # shape: (B, 2)

        if len(calibs) == 1:  # Batch size is 1.
            batch_idxs = self.zeros((val_objs_num,), ms.int32)

        if GRM_uncern is not None:
            assert GRM_uncern.shape[1] == 20
        else:
            GRM_uncern = self.ones((val_objs_num, 20), self.ms_type)

        if GRM_valid_items is not None:
            assert GRM_valid_items.shape[1] == 20
        else:
            GRM_valid_items = self.ones((val_objs_num, 20), self.ms_type)

        assert pred_keypoint_offset.shape[1] == 16  # Do not use the bottom center and top center. Only 8 vertexes.

        pred_keypoints = self.decode_2D_keypoints(target_centers, pred_keypoint_offset, targets_dict[2],
                                                  batch_idxs=batch_idxs)  # pred_keypoints: The 10 keypoint position original image. shape: (total_num_objs, 8, 2)

        c_u = self.stack_0([calib['c_u'] for calib in calibs])  # c_u shape: (B,)
        c_v = self.stack_0([calib['c_v'] for calib in calibs])
        f_u = self.stack_0([calib['f_u'] for calib in calibs])
        f_v = self.stack_0([calib['f_v'] for calib in calibs])
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

        l = pred_dimensions[::, 0: 1]  # h shape: (total_num_objs, 1). The same as w and l.
        h = pred_dimensions[::, 1: 2]
        w = pred_dimensions[::, 2: 3]

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

        weights = (weights / self.reducesum_t(weights, 1))
        depth = self.reducesum(weights * separate_depths, 1)

        return depth, separate_depths


class LossNet(nn.Cell):
    """Monodde loss method"""

    def construct(self, loss):
        # loss_sum=0
        # for l in loss:
        #     loss_sum+=l
        return sum(loss)


