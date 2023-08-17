import mindspore.ops as ops
import mindspore.nn as nn
import mindspore as ms
import mindspore.numpy as mnp
import numpy as np
import pdb
from shapely.geometry import Polygon
# from .net_utils import Converter_key2channel, project_image_to_rect

PI = np.pi


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
class Berhu_Loss(nn.Cell):
    def __init__(self):
        super(Berhu_Loss, self).__init__()
        # according to ECCV18 Joint taskrecursive learning for semantic segmentation and depth estimation
        self.c = 0.2

    def construct(self, prediction, target):
        pdb.set_trace()
        differ = (prediction - target).abs()
        c = ops.clip_by_value(self.c, differ.max() * self.c, 1e-4, )  # 有疑问
        # larger than c: l2 loss
        # smaller than c: l1 loss
        large_idx = ops.nonzero(differ > c)
        small_idx = ops.nonzero(differ <= c)

        loss = differ[small_idx].sum() + ((differ[large_idx] ** 2) / c + c).sum() / 2

        return loss


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

        negative_loss = self.log(1 - prediction) * self.pow(prediction, self.alpha) * negative_weights * negative_index

        num_hm_pos = ops.reduce_sum(positive_index)
        positive_loss = ops.reduce_sum(positive_loss)
        negative_loss = ops.reduce_sum(negative_loss)

        loss = - negative_loss - positive_loss
        # num_hm_pos=ms.Tensor(num_positive, ms.float32)

        hm_loss = self.weight * loss  # Heatmap loss.
        hm_loss = hm_loss / ops.clip_by_value(num_hm_pos, self.clip_min)

        return hm_loss


'''iou loss'''


class IOULoss(nn.Cell):
    def __init__(self, loss_type="iou"):
        super(IOULoss, self).__init__()
        self.loss_type = loss_type
        self.min = ops.Minimum()
        self.max = ops.Maximum()
        self.log = ops.Log()

    def construct(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_area = (target_left + target_right) * (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

        w_intersect = self.min(pred_left, target_left) + self.min(pred_right, target_right)
        h_intersect = self.min(pred_bottom, target_bottom) + self.min(pred_top, target_top)
        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        # ious = (area_intersect) / (area_union)
        ious = (area_intersect + 1.0) / (area_union + 1.0)
        if self.loss_type == 'iou':
            losses = -self.log(ious)
        elif self.loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loss_type == 'giou':
            g_w_intersect = self.max(pred_left, target_left) + self.max(pred_right, target_right)
            g_h_intersect = self.max(pred_bottom, target_bottom) + self.max(pred_top, target_top)
            ac_uion = g_w_intersect * g_h_intersect + 1e-5
            gious = ious - (ac_uion - area_union) / ac_uion
            losses = 1 - gious
            # losses = gious
        else:
            raise NotImplementedError
        return losses


class Mono_loss(nn.Cell):
    def __init__(self, cfg):
        super(Mono_loss, self).__init__()
        self.cfg = cfg
        # self.key2channel = Converter_key2channel(keys=cfg.MODEL.HEAD.REGRESSION_HEADS,
        #                                          channels=cfg.MODEL.HEAD.REGRESSION_CHANNELS)
        self.ct_keys = [key for key_group in cfg.MODEL.HEAD.REGRESSION_HEADS for key in key_group]
        # self.ct_channels = [channel for channel_groups in cfg.MODEL.HEAD.REGRESSION_CHANNELS for channel in channel_groups]
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
        self.cls_loss_fnc = FocalLoss(cfg.MODEL.HEAD.LOSS_PENALTY_ALPHA,
                                      cfg.MODEL.HEAD.LOSS_BETA,self.loss_weights['hm_loss'])  # penalty-reduced focal loss
        self.iou_loss = IOULoss(loss_type=loss_types[2])  # iou loss for 2D detection

        # depth loss
        if loss_types[3] == 'berhu':
            self.depth_loss = Berhu_Loss()
        elif loss_types[3] == 'inv_sig':
            self.depth_loss = Inverse_Sigmoid_Loss()
        elif loss_types[3] == 'log':
            self.depth_loss = Log_L1_Loss()
        elif loss_types[3] == 'L1':
            self.depth_loss = nn.L1Loss(reduction='none')
        else:
            raise ValueError

        # regular regression loss
        self.reg_loss = loss_types[1]
        self.reg_loss_fnc = nn.L1Loss(reduction='none') if loss_types[1] == 'L1' else nn.SmoothL1Loss
        self.keypoint_loss_fnc = nn.L1Loss(reduction='none')

        # multi-bin loss setting for orientation estimation
        self.multibin = (cfg.INPUT.ORIENTATION == 'multi-bin')
        self.orien_bin_size = cfg.INPUT.ORIENTATION_BIN_SIZE
        self.trunc_offset_loss_type = cfg.MODEL.HEAD.TRUNCATION_OFFSET_LOSS

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
        self.eps = 1e-1
        self.SoftGRM_loss_weight = ms.Tensor(self.cfg.MODEL.HEAD.SOFTGRM_LOSS_WEIGHT,self.ms_type)
        self.dynamic_thre = cfg.SOLVER.DYNAMIC_THRESHOLD
        self.clip_min=ms.Tensor(1,self.ms_type)

        self.reducesum = ops.ReduceSum(keep_dims=False)
        self.nonzero=ops.NonZero()
        self.exp = ops.Exp()
        self.expand_dims=ops.ExpandDims()
        self.log = ops.Log()
        self.div = ops.Div()
        self.reducmean = ops.ReduceMean()
        self.concat = ops.Concat(axis=1)
        # self.argminwithvalue=ops.ArgMinWithValue(axis=1)

        # self.reg_cnt = ms.Parameter(ms.Tensor(0, self.ms_type), requires_grad=False)
        self.l1loss = nn.L1Loss(reduction='none')
        self.sin = ops.Sin()
        self.cos = ops.Cos()
        self.l2normalize=ops.L2Normalize()
        self.cast=ops.Cast()
        self.minimum=ops.Minimum()
        self.maximum=ops.Maximum()
        self.zeros=ops.Zeros()


    def construct(self, targets_heatmap, predictions, pred_targets, preds, reg_nums, weights, iteration):
        ops.print_('compute loss')
        pred_heatmap = predictions['cls']
        # heatmap loss
        hm_loss= self.cls_loss_fnc(pred_heatmap, targets_heatmap)   # Heatmap loss.
        # hm_loss=ops.stop_gradient(hm_loss)
        loss_list=hm_loss
        ops.print_('heatmap loss:',hm_loss)
        num_reg_2D = reg_nums['reg_2D']
        num_reg_3D = reg_nums['reg_3D']

        trunc_mask = pred_targets['trunc_mask_3D']
        num_trunc = self.reducesum(self.cast(trunc_mask,self.ms_type))
        trunc_mask_inverse=self.cast(self.nonzero(trunc_mask==0).squeeze(1),ms.int32)

        # IoU loss for 2D detection
        if num_reg_2D > 0:
            reg_2D_loss = self.iou_loss(preds['reg_2D'], pred_targets['reg_2D'])
            reg_2D_loss = self.loss_weights['bbox_loss'] * self.reducmean(reg_2D_loss)
            loss_list+=reg_2D_loss
            ops.print_('reg_2D_loss:',reg_2D_loss)

        if num_reg_3D > 0:
            # direct depth loss
            if self.compute_direct_depth_loss:
                depth_3D_loss = self.loss_weights['depth_loss'] * self.depth_loss(preds['depth_3D'], pred_targets['depth_3D'])

                if self.depth_with_uncertainty:
                    depth_3D_loss = depth_3D_loss * self.exp(- preds['depth_uncertainty']) + \
                                    preds['depth_uncertainty'] * self.loss_weights['depth_loss']

                depth_3D_loss = self.reducmean(depth_3D_loss)
                # depth_3D_loss=ops.stop_gradient(depth_3D_loss)
                loss_list+=depth_3D_loss
                ops.print_('depth_3D_loss:',depth_3D_loss)

            # offset_3D loss
            offset_3D_loss = self.reducesum(self.reg_loss_fnc(preds['offset_3D'], pred_targets['offset_3D']),1)  # offset_3D_loss shape: (val_objs,)

            # use different loss functions for inside and outside objects
            if self.separate_trunc_offset:
                if self.trunc_offset_loss_type == 'L1':
                    trunc_offset_loss = offset_3D_loss[trunc_mask]
                    loss_list = loss_list + trunc_offset_loss

                elif self.trunc_offset_loss_type == 'log':
                    if num_trunc > 0:
                        trunc_mask = self.cast(self.nonzero(trunc_mask == 1).squeeze(1), ms.int32)
                        trunc_offset_loss = self.log((1 + offset_3D_loss[trunc_mask]))

                        trunc_offset_loss = self.loss_weights['trunc_offset_loss'] * self.reducesum(trunc_offset_loss) / ops.clip_by_value(num_trunc, self.clip_min)
                        if num_trunc!=1:
                            loss_list += self.reducesum(trunc_offset_loss)
                        else:
                            loss_list+=trunc_offset_loss
                offset_3D_loss = self.loss_weights['offset_loss'] * self.reducmean(offset_3D_loss[trunc_mask_inverse])
                loss_list += offset_3D_loss
            else:
                loss_list = loss_list+self.loss_weights['offset_loss'] * self.reducmean(offset_3D_loss)
            ops.print_('offset_3D_loss:',offset_3D_loss)
            # orientation loss
            if self.multibin:
                orien_3D_loss = self.loss_weights['orien_loss'] * \
                                self.Real_MultiBin_loss(preds['orien_3D'], pred_targets['orien_3D'],num_bin=self.orien_bin_size)
                loss_list+=orien_3D_loss
                # orien_3D_loss=ops.stop_gradient(orien_3D_loss)
                ops.print_('orien_3D_loss:',orien_3D_loss)
            # dimension loss
            dims_3D_loss = self.reg_loss_fnc(preds['dims_3D'], pred_targets['dims_3D']) * self.dim_weight
            if self.dim_uncern:
                dims_3D_loss = dims_3D_loss / preds['dim_uncern'] + self.log(preds['dim_uncern'])
            dims_3D_loss = self.reducesum(dims_3D_loss,1)
            dims_3D_loss = self.loss_weights['dims_loss'] * self.reducmean(dims_3D_loss)
            loss_list+=dims_3D_loss
            ops.print_('dims_3D_loss:',dims_3D_loss)

            # corner loss
            if self.compute_corner_loss:
                # N x 8 x 3
                corner_3D_loss = self.reducesum(self.reg_loss_fnc(preds['corners_3D'], pred_targets['corners_3D']),2)
                corner_3D_loss=self.reducmean(corner_3D_loss,1)
                if self.corner_loss_uncern:
                    corner_loss_uncern = preds['corner_loss_uncern'].squeeze(1)
                    corner_3D_loss = corner_3D_loss / corner_loss_uncern + self.log(corner_loss_uncern)
                corner_3D_loss = self.loss_weight_ramper(iteration) * self.loss_weights['corner_loss'] * self.reducmean(corner_3D_loss)
                loss_list+=corner_3D_loss
                ops.print_('corner_3D_loss:',corner_3D_loss)

            if self.compute_keypoint_corner:
                if self.corner_offset_uncern:
                    keypoint_loss = self.keypoint_loss_fnc(preds['keypoints'], pred_targets['keypoints'])  # keypoint_loss shape: (val_objs, 10, 2)
                    keypoint_loss=keypoint_loss.view(-1)
                    keypoint_loss_mask = self.cast(self.expand_dims(pred_targets['keypoints_mask'], 2).expand_as(keypoint_loss), ms.bool_).view(-1)  # keypoint_loss_mask shape: (val_objs, 10, 2)
                    keypoint_loss_uncern = preds['corner_offset_uncern'].view(-1, 10,2)  # keypoint_loss_uncern shape: (val_objs, 10, 2)

                    valid_keypoint_loss = keypoint_loss[keypoint_loss_mask]  # valid_keypoint_loss shape: (valid_equas,)
                    invalid_keypoint_loss = keypoint_loss[(~keypoint_loss_mask)]  # invalid_keypoint_loss shape: (invalid_equas,)
                    invalid_keypoint_loss=ops.stop_gradient(invalid_keypoint_loss)
                    valid_keypoint_uncern = keypoint_loss_uncern[keypoint_loss_mask]  # valid_keypoint_uncern shape: (valid_equas,)
                    invalid_keypoint_uncern = keypoint_loss_uncern[(~keypoint_loss_mask)]  # invalid_keypoint_uncern: (invalid_equas,)
                    invalid_keypoint_uncern = ops.stop_gradient(invalid_keypoint_uncern)

                    valid_keypoint_loss = valid_keypoint_loss / valid_keypoint_uncern + self.log(valid_keypoint_uncern)
                    valid_keypoint_loss = self.reducesum(valid_keypoint_loss) / ops.clip_by_value(self.reducesum(keypoint_loss_mask), self.clip_min)
                    invalid_keypoint_loss = invalid_keypoint_loss / invalid_keypoint_uncern
                    invalid_keypoint_loss = self.reducesum(invalid_keypoint_loss) / ops.clip_by_value(self.reducesum(invalid_keypoint_loss), self.clip_min)
                    if self.modify_invalid_keypoint_depths:
                        keypoint_loss = self.loss_weights['keypoint_loss'] * (valid_keypoint_loss + invalid_keypoint_loss)
                    else:
                        keypoint_loss = self.loss_weights['keypoint_loss'] * valid_keypoint_loss
                else:
                    # N x K x 3
                    keypoint_loss = self.reducesum(self.keypoint_loss_fnc(preds['keypoints'], pred_targets['keypoints']),2) * \
                                    pred_targets['keypoints_mask']
                    keypoint_loss = self.reducesum(keypoint_loss,1)  # Left keypoints_loss shape: (val_objs,)
                    keypoint_loss = self.loss_weights['keypoint_loss'] * self.reducesum(keypoint_loss) / ops.clip_by_value(self.reducesum(pred_targets['keypoints_mask']), self.clip_min)
                # keypoint_loss=ops.stop_gradient(keypoint_loss)
                loss_list+=keypoint_loss
                ops.print_('keypoint_loss:',keypoint_loss)
                if self.compute_keypoint_depth_loss:
                    pred_keypoints_depth, keypoints_depth_mask = preds['keypoints_depths'], pred_targets['keypoints_depth_mask'].view(-1)

                    keypoints_depth_mask_inverse=self.nonzero((~keypoints_depth_mask)).shape[0]
                    target_keypoints_depth = ops.tile(ops.expand_dims(pred_targets['depth_3D'], -1),(1, 3)).view(-1)

                    pred_keypoints_depth=pred_keypoints_depth.view(-1)
                    valid_pred_keypoints_depth =ops.masked_select(pred_keypoints_depth,keypoints_depth_mask)


                    # valid and non-valid
                    target_keypoints_depth_mask=ops.masked_select(target_keypoints_depth,keypoints_depth_mask)
                    valid_keypoint_depth_loss = self.loss_weights['keypoint_depth_loss'] * self.reg_loss_fnc(valid_pred_keypoints_depth, target_keypoints_depth_mask)

                    if keypoints_depth_mask_inverse> 0:
                        target_keypoints_depth_inversmask = ops.masked_select(target_keypoints_depth, (~keypoints_depth_mask))
                        invalid_pred_keypoints_depth = ops.masked_select(pred_keypoints_depth,(~keypoints_depth_mask))  # The depths decoded from invalid keypoints are not used for updating networks.
                        invalid_pred_keypoints_depth=ops.stop_gradient(invalid_pred_keypoints_depth)
                        invalid_keypoint_depth_loss = self.loss_weights['keypoint_depth_loss'] * self.reg_loss_fnc(invalid_pred_keypoints_depth, target_keypoints_depth_inversmask)
                    else:
                        invalid_keypoint_depth_loss = ms.Tensor(0, self.ms_type)

                    if self.corner_with_uncertainty:
                        # center depth, corner 0246 depth, corner 1357 depth
                        pred_keypoint_depth_uncertainty = preds['corner_offset_uncertainty'].view(-1)

                        valid_uncertainty = ops.masked_select(pred_keypoint_depth_uncertainty,keypoints_depth_mask)
                        if keypoints_depth_mask_inverse > 0:
                            invalid_uncertainty = ops.masked_select(pred_keypoint_depth_uncertainty,(~keypoints_depth_mask))
                            invalid_keypoint_depth_loss = invalid_keypoint_depth_loss * self.exp(- invalid_uncertainty)  # Lead to infinite uncertainty for invisible keypoints.
                            invalid_keypoint_depth_loss = self.reducesum(invalid_keypoint_depth_loss) / ops.clip_by_value(ms.Tensor(np.array(keypoints_depth_mask_inverse).sum(), self.ms_type), self.clip_min)
                        else:
                            invalid_keypoint_depth_loss = ms.Tensor(0, self.ms_type)

                        valid_keypoint_depth_loss = valid_keypoint_depth_loss * self.exp(- valid_uncertainty) + \
                                                    self.loss_weights['keypoint_depth_loss'] * valid_uncertainty

                    # average
                    valid_keypoint_depth_loss = self.reducesum(valid_keypoint_depth_loss) / ops.clip_by_value(self.reducesum(self.cast(keypoints_depth_mask,self.ms_type)),self.clip_min)

                    # the gradients of invalid depths are not back-propagated
                    if self.modify_invalid_keypoint_depths:
                        keypoint_depth_loss = (valid_keypoint_depth_loss + invalid_keypoint_depth_loss)
                    else:
                        keypoint_depth_loss = valid_keypoint_depth_loss
                    # keypoint_depth_loss = ops.stop_gradient(keypoint_depth_loss)
                    loss_list+=keypoint_depth_loss
                    ops.print_('keypoint_depth_loss:',keypoint_depth_loss)

                if self.corner_with_uncertainty:
                    if self.pred_direct_depth and self.depth_with_uncertainty:
                        combined_depth = self.concat((ops.expand_dims(preds['depth_3D'], 1), (preds['keypoints_depths'])))
                        depth_uncertainty=ops.expand_dims(preds['depth_uncertainty'], 1)
                        uncertainty=self.concat((depth_uncertainty,preds['corner_offset_uncertainty']))
                        combined_uncertainty = self.exp(uncertainty)
                    else:
                        combined_depth = preds['keypoints_depths']
                        combined_uncertainty = self.exp(preds['corner_offset_uncertainty'])

                    combined_weights = 1 / combined_uncertainty
                    combined_weights = combined_weights / combined_weights.sum(axis=1, keepdims=True)
                    soft_depths = self.reducesum(combined_depth * combined_weights, 1)

                    if self.compute_weighted_depth_loss:
                        soft_depth_loss = self.loss_weights['weighted_avg_depth_loss'] * \
                                          self.reg_loss_fnc(soft_depths, pred_targets['depth_3D'])
                        loss_list+=soft_depth_loss
                        ops.print_('soft_depth_loss:',soft_depth_loss)

            if self.compute_combined_depth_loss:  # The loss for final estimated depth.
                combined_depth_loss = self.reg_loss_fnc(preds['combined_depth'], pred_targets['depth_3D'])
                if self.combined_depth_uncern:
                    combined_depth_uncern = preds['combined_depth_uncern'].squeeze(1)
                    combined_depth_loss = combined_depth_loss / combined_depth_uncern + self.log(combined_depth_uncern)
                # if self.cfg.SOLVER.DYNAMIC_WEIGHT:
                #     combined_depth_loss = self.reweight_loss(combined_depth_loss, objs_weight)
                combined_depth_loss = self.loss_weight_ramper(iteration) * self.loss_weights['combined_depth_loss'] * self.reducmean(combined_depth_loss)
                loss_list+=combined_depth_loss
                ops.print_('combined_depth_loss:',combined_depth_loss)

            if self.compute_GRM_loss:
                GRM_valid_items = pred_targets['GRM_valid_items']  # GRM_valid_items shape: (val_objs, 25)
                GRM_valid_item_sum=ms.Tensor(np.array([GRM_valid_items]).sum(),self.ms_type)
                GRM_valid_items_inverse=(np.array(GRM_valid_items)==False).tolist()
                GRM_valid_item_inversum=ms.Tensor(np.array(GRM_valid_items_inverse).sum(),self.ms_type)
                valid_GRM_A = preds['GRM_A'][GRM_valid_items]  # valid_GRM_A shape: (valid_equas, 3)
                valid_GRM_B = preds['GRM_B'][GRM_valid_items]  # valid_GRM_B shape: (valid_equas, 1)
                invalid_GRM_A = preds['GRM_A'][GRM_valid_items_inverse]  # invalid_GRM_A shape: (invalid_equas, 3)
                invalid_GRM_A=ops.stop_gradient(invalid_GRM_A)
                invalid_GRM_B = preds['GRM_B'][GRM_valid_items_inverse]  # invalid_GRM_B shape: (invalid_equas, 1)
                invalid_GRM_B = ops.stop_gradient(invalid_GRM_B)
                valid_target_location = ops.expand_dims(pred_targets['locations'], 1).expand_as(preds['GRM_A'])[GRM_valid_items] # valid_target_location shape: (valid_equas, 3)
                invalid_target_location = ops.expand_dims(pred_targets['locations'], 1).expand_as(preds['GRM_A'])[GRM_valid_items_inverse]  # # valid_target_location shape: (invalid_equas, 3)
                valid_uncern = preds['GRM_uncern'][GRM_valid_items]  # shape: (valid_equas,)
                invalid_uncern = preds['GRM_uncern'][GRM_valid_items_inverse]  # shape: (invalid_equas,)~

                valid_GRM_loss = self.reg_loss_fnc(self.reducesum((valid_GRM_A * valid_target_location), 1), valid_GRM_B.squeeze(1))  # valid_GRM_loss shape: (valid_equas, 1)
                valid_GRM_loss = valid_GRM_loss / (valid_uncern+1) + self.log(valid_uncern)
                valid_GRM_loss = self.reducesum(valid_GRM_loss) / ops.clip_by_value(GRM_valid_item_sum, self.clip_min)

                invalid_GRM_loss = self.reg_loss_fnc(self.reducesum((invalid_GRM_A * invalid_target_location), 1),invalid_GRM_B.squeeze(1))  # invalid_GRM_loss shape: (invalid_equas, 1)
                invalid_GRM_loss = invalid_GRM_loss / (invalid_uncern+1)
                invalid_GRM_loss = self.reducesum(invalid_GRM_loss) / ops.clip_by_value(GRM_valid_item_inversum, self.clip_min)

                if self.modify_invalid_keypoint_depths:
                    GRM_loss = self.loss_weights['GRM_loss'] * (valid_GRM_loss + invalid_GRM_loss)
                else:
                    GRM_loss = self.loss_weights['GRM_loss'] * valid_GRM_loss
                loss_list+=GRM_loss

            if self.compute_SoftGRM_loss:
                GRM_valid_items = np.array(pred_targets['GRM_valid_items']).reshape(-1)  # GRM_valid_items shape: (val_objs, 20)
                GRM_valid_item_sum = ms.Tensor(np.array([GRM_valid_items]).sum(), self.ms_type)
                GRM_valid_items_inverse = (GRM_valid_items == False)
                GRM_valid_items_inverse_sum=np.array(GRM_valid_items_inverse).sum()
                GRM_valid_items=GRM_valid_items.tolist()
                GRM_valid_items_inverse=GRM_valid_items_inverse.tolist()
                separate_depths = preds['separate_depths']  # separate_depths shape: (val_objs, 20)
                depth_3D=ops.expand_dims(pred_targets['depth_3D'], 1).expand_as(separate_depths).view(-1)
                valid_target_depth = depth_3D[GRM_valid_items]  # shape: (valid_equas,)
                sd_shape=separate_depths.shape
                separate_depths=separate_depths.view(-1)
                valid_separate_depths = separate_depths[GRM_valid_items]  # shape: (valid_equas,)
                GRM_uncern = preds['GRM_uncern'].view(-1)
                SoftGRM_weight = ops.function.broadcast_to(ops.expand_dims(self.SoftGRM_loss_weight, 0), sd_shape).view(-1)
                valid_SoftGRM_weight = SoftGRM_weight[GRM_valid_items]
                if GRM_valid_items_inverse_sum>0:
                    invalid_SoftGRM_weight = SoftGRM_weight[GRM_valid_items_inverse]
                    invalid_target_depth = depth_3D[GRM_valid_items_inverse]  # shape: (invalid_equas,) problem~
                    invalid_separate_depths = separate_depths[GRM_valid_items_inverse]  # shape: (invalid_equas,) ~
                    invalid_separate_depths=ops.stop_gradient(invalid_separate_depths)
                    invalid_uncern = GRM_uncern[GRM_valid_items_inverse]  # shape: (invalid_equas,)
                    invalid_SoftGRM_loss = self.reg_loss_fnc(invalid_separate_depths, invalid_target_depth), invalid_uncern
                    invalid_SoftGRM_loss = self.reducesum(invalid_SoftGRM_loss * invalid_SoftGRM_weight)/ ops.clip_by_value(ms.Tensor(GRM_valid_items_inverse_sum,self.ms_type), self.clip_min)  # Avoid the occasion that no invalid equations and the returned value is NaN.

                else:
                    invalid_SoftGRM_loss = ms.Tensor(0, self.ms_type)
                valid_uncern = GRM_uncern[GRM_valid_items]  # shape: (valid_equas,)

                valid_SoftGRM_loss = self.reg_loss_fnc(valid_separate_depths, valid_target_depth) / valid_uncern + self.log(valid_uncern)

                valid_SoftGRM_loss = self.reducesum(valid_SoftGRM_loss * valid_SoftGRM_weight)/ops.clip_by_value(GRM_valid_item_sum, self.clip_min)
                if self.modify_invalid_keypoint_depths:
                    SoftGRM_loss = (self.loss_weight_ramper(iteration) * self.loss_weights['SoftGRM_loss'] * (valid_SoftGRM_loss + invalid_SoftGRM_loss))
                else:
                    SoftGRM_loss = self.loss_weight_ramper(iteration) * self.loss_weights['SoftGRM_loss'] * valid_SoftGRM_loss
                loss_list+=SoftGRM_loss
                ops.print_('SoftGRM_loss:',SoftGRM_loss)

        return loss_list

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


    '''multibin loss'''
    def Real_MultiBin_loss(self,vector_ori, gt_ori, num_bin=4):
        gt_ori = gt_ori.view((-1, gt_ori.shape[-1]))  # bin1 cls, bin1 offset, bin2 cls, bin2 offst
        cls_losses = ms.Tensor(0, self.ms_type)
        reg_losses = ms.Tensor(0, self.ms_type)
        reg_cnt=ms.Tensor(0, self.ms_type)

        for i in range(num_bin):
            # bin cls loss
            cls_ce_loss = ops.cross_entropy(vector_ori[:, (i * 2): (i * 2 + 2)], ops.cast(gt_ori[:, i],ms.int32), reduction='none')  # gt_ori  p
            # regression loss
            valid_mask_i = (gt_ori[:, i] == 1).asnumpy()
            # valid_mask_i=self.cast(valid_mask_i,ms.int32).squeeze(1)
            cls_losses += self.reducmean(cls_ce_loss)
            # valid_mask_i_sum=valid_mask_i.shape[0]
            valid_mask_i_sum = valid_mask_i.sum()
            valid_mask_i = valid_mask_i.tolist()
            # valid_mask_i_2d=ops.expand_dims(valid_mask_i, 1)
            if valid_mask_i_sum > 0:
                s = num_bin * 2 + i * 2
                e = s + 2
                pred_offset = self.l2normalize(vector_ori[valid_mask_i][:, s: e])
                reg_loss = self.l1loss(pred_offset[:, 0], self.sin(gt_ori[valid_mask_i][:, num_bin + i])) + \
                           self.l1loss(pred_offset[:, 1], self.cos(gt_ori[valid_mask_i][:, num_bin + i]))

                reg_losses += self.reducesum(reg_loss)
                reg_cnt += valid_mask_i_sum

        return cls_losses / num_bin + reg_losses / reg_cnt
