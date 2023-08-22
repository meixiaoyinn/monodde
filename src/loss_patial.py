import mindspore.ops as ops
import mindspore.nn as nn
import mindspore as ms
# import mindspore.numpy as mnp
import numpy as np


class Muti_offset_loss(nn.Cell):
    def __init__(self,loss_fnc,weights_offset_loss,weights_trunc_offset):
        super(Muti_offset_loss,self).__init__()
        self.ms_type=ms.float32
        self.reg_loss_fnc=loss_fnc
        self.weights_offset_loss=weights_offset_loss
        self.weights_trunc_offset=weights_trunc_offset
        self.clip_min=ms.Tensor(1,ms.float32)
        self.int_type=ms.int32

        self.nonzero=ops.NonZero()
        self.log=ops.Log()
        self.cast=ops.Cast()
        self.gather_nd=ops.GatherNd()
        self.reducemean=ops.ReduceMean()


    def construct(self, target_offset_3D,pred_offset_3D,trunc_mask):
        num_trunc = self.cast(trunc_mask, self.ms_type).sum()
        trunc_mask_inverse = self.cast(self.nonzero(trunc_mask == 0), self.int_type)
        # pred_offset_3D = pred_regression_pois_3D[::, self.td_offset_index]  # pred_offset_3D shape: (valid_objs, 2)
        offset_3D_loss = self.reg_loss_fnc(pred_offset_3D, target_offset_3D).sum(1)  # offset_3D_loss shape: (val_objs,)

        # use different loss functions for inside and outside objects
        if num_trunc > 0:
            trunc_mask = self.cast(self.nonzero(trunc_mask == 1), self.int_type)
            trunc_offset_loss = self.log(1 + self.gather_nd(offset_3D_loss, trunc_mask))

            trunc_offset_loss = self.weights_trunc_offset * trunc_offset_loss.sum() / ops.clip_by_value(num_trunc, self.clip_min)
            # self.print('trunc_offset_loss:', trunc_offset_loss)
        else:trunc_offset_loss=(self.cast(trunc_mask,self.ms_type)*offset_3D_loss).sum()
        offset_3D_loss = self.weights_offset_loss * self.reducemean(self.gather_nd(offset_3D_loss, trunc_mask_inverse))
        return offset_3D_loss+trunc_offset_loss


class Reg2D_loss(nn.Cell):
    def __init__(self,loss_type,weight):
        super(Reg2D_loss,self).__init__()
        self.iou_loss = IOULoss(loss_type=loss_type)
        self.weight_bbox=weight
        self.gather_nd=ops.GatherNd()
        self.relu=nn.ReLU()

    def construct(self, pred_regression_2D, target_reg2D,mask_regression_2D):
        reg_2D_loss = self.iou_loss(pred_regression_2D, target_reg2D)
        reg_2D_loss = self.weight_bbox* reg_2D_loss.mean()
        return reg_2D_loss


class Depth3D_loss(nn.Cell):
    def __init__(self,depth_index,depth_uncertainty_index,uncertainty_range,loss_weights,depth_mode,depth_ref,depth_range,depth_with_uncertainty):
        super(Depth3D_loss,self).__init__()
        self.depth_index=depth_index
        self.depth_uncertainty_index=depth_uncertainty_index
        self.depth_loss=nn.L1Loss(reduction='none')
        self.uncertainty_range=uncertainty_range
        self.weight=loss_weights
        self.depth_mode=depth_mode
        self.depth_ref=depth_ref
        self.depth_range=depth_range
        self.depth_with_uncertainty=depth_with_uncertainty

        self.exp=ops.Exp()
        self.reducemean=ops.ReduceMean()
        self.sigmoid = ops.Sigmoid()


    def construct(self, pred_direct_depths_3D,preds_depth_uncertainty,targets_depth):
        # pred_depths_offset_3D = pred_regression_pois_3D[:, self.depth_index].squeeze(-1)
        # pred_direct_depths_3D = self.decode_depth(pred_depths_offset_3D)
        depth_3D_loss = self.weight * self.depth_loss(pred_direct_depths_3D, targets_depth)
        # preds_depth_uncertainty = pred_regression_pois_3D[:, self.depth_uncertainty_index].squeeze(1)  # preds['depth_uncertainty'] shape: (val_objs,)

        # if self.uncertainty_range is not None:
        preds_depth_uncertainty = ops.clip_by_value(preds_depth_uncertainty, self.uncertainty_range[0],self.uncertainty_range[1])
        if self.depth_with_uncertainty:
            depth_3D_loss = depth_3D_loss * self.exp(- preds_depth_uncertainty) + \
                            preds_depth_uncertainty * self.weight

        depth_3D_loss = self.reducemean(depth_3D_loss)
        return depth_3D_loss


class Orien3D_loss(nn.Cell):
    def __init__(self,weight,orien_bin_size):
        super(Orien3D_loss,self).__init__()
        self.weight=weight
        self.orien_bin_size=orien_bin_size
        self.ms_type=ms.float32
        self.nonzero=ops.NonZero()
        self.cast=ops.Cast()
        self.l2normalize=ops.L2Normalize()
        self.l1loss = nn.L1Loss(reduction='none')
        self.sin=ops.Sin()
        self.cos=ops.Cos()
        self.concat=ops.Concat(1)

    def construct(self,pred_orientation_3D,target_orien):
        # pred_orientation_3D = self.concat((pred_regression_pois_3D[::, self.ori_cls_index], pred_regression_pois_3D[::, self.ori_offset_index]))  # pred_orientation_3D shape: (valid_objs, 16)
        gt_ori = target_orien.view((-1, target_orien.shape[-1]))  # bin1 cls, bin1 offset, bin2 cls, bin2 offst
        cls_losses = ms.Tensor(0, self.ms_type)
        reg_losses = ms.Tensor(0, self.ms_type)
        reg_cnt = ms.Tensor(0, self.ms_type)

        for i in range(self.orien_bin_size):
            # bin cls loss
            cls_ce_loss = ops.cross_entropy(pred_orientation_3D[:, (i * 2): (i * 2 + 2)], ops.cast(gt_ori[:, i], ms.int32),
                                            reduction='none')  # gt_ori  p
            # regression loss
            valid_mask = (gt_ori[:, i] == 1)
            valid_mask_sum = self.cast(valid_mask, ms.float32).sum()
            if valid_mask_sum > 0:
                valid_mask_i = self.nonzero(valid_mask).squeeze(1)
                cls_losses += cls_ce_loss.mean()
                valid_mask_i = self.cast(valid_mask_i, ms.int32)
                s = self.orien_bin_size * 2 + i * 2
                e = s + 2
                pred_offset = self.l2normalize(pred_orientation_3D[valid_mask_i][:, s: e])
                reg_loss = self.l1loss(pred_offset[:, 0], self.sin(gt_ori[valid_mask_i][:, self.orien_bin_size + i])) + \
                           self.l1loss(pred_offset[:, 1], self.cos(gt_ori[valid_mask_i][:, self.orien_bin_size + i]))

                reg_losses += reg_loss.sum()
                reg_cnt += valid_mask_sum

        loss= cls_losses / self.orien_bin_size + reg_losses / reg_cnt
        orien_3D_loss = self.weight * loss
        return orien_3D_loss


class Dim3D_loss(nn.Cell):
    def __init__(self,reg_loss_fnc,index,weight_loss,dim_weight,dim_modes,dim_mean,dim_std):
        super(Dim3D_loss,self).__init__()
        self.weight_loss=weight_loss
        self.dim_weight=dim_weight
        self.reg_loss_fnc=reg_loss_fnc

    def construct(self,pred_dimensions_3D, target_dims_3D):
        dims_3D_loss = self.reg_loss_fnc(pred_dimensions_3D, target_dims_3D) * self.dim_weight
        dims_3D_loss = dims_3D_loss.sum(1)
        dims_3D_loss = self.weight_loss * dims_3D_loss.mean()
        # dims_3D_loss = ops.clip_by_value(dims_3D_loss, clip_value_max=ms.Tensor(1000, self.ms_type))
        return dims_3D_loss


class Corner3D_loss(nn.Cell):
    def __init__(self,reg_loss_fnc,weight):
        super(Corner3D_loss,self).__init__()
        self.loss_weights=weight
        self.log=ops.Log()
        self.reg_loss_fnc=reg_loss_fnc

    def construct(self, pred_corners_3D,preds_corner_loss_uncern,targets_corners_3D,weight_ramper):
        corner_3D_loss = self.reg_loss_fnc(pred_corners_3D, targets_corners_3D).sum(2)
        corner_3D_loss = corner_3D_loss.mean(1)
        # if self.corner_loss_uncern:
        corner_loss_uncern = preds_corner_loss_uncern.squeeze(1)
        corner_3D_loss = corner_3D_loss / corner_loss_uncern + self.log(corner_loss_uncern)
        corner_3D_loss = weight_ramper * self.loss_weights * corner_3D_loss.mean()
        return corner_3D_loss


class Keypoint_loss(nn.Cell):
    def __init__(self,keypoint_loss_fnc,weight):
        super(Keypoint_loss,self).__init__()
        self.keypoint_loss_fnc=keypoint_loss_fnc
        self.loss_weights=weight
        self.clip_min=ms.Tensor(1,ms.float32)

    def construct(self, preds_keypoints, targets_keypoints,targets_keypoints_mask):
        keypoint_loss = self.keypoint_loss_fnc(preds_keypoints, targets_keypoints).sum(2) * targets_keypoints_mask
        keypoint_loss = keypoint_loss.sum(1)  # Left keypoints_loss shape: (val_objs,)
        keypoint_loss = self.loss_weights * keypoint_loss.sum() / ops.clip_by_value(targets_keypoints_mask.sum(), self.clip_min)
        return keypoint_loss


class Keypoint_depth_loss(nn.Cell):
    def __init__(self,reg_loss_fnc,weight):
        super(Keypoint_depth_loss,self).__init__()
        self.reg_loss_fnc=reg_loss_fnc
        self.loss_weights=weight
        self.gather_nd=ops.GatherNd()
        self.cast=ops.Cast()
        self.ms_type=ms.float32
        self.nonzero=ops.NonZero()
        self.tile=ops.Tile()
        self.exp=ops.Exp()
        self.print=ops.Print()
        self.log=ops.Log()
        self.clip_min=ms.Tensor(1,self.ms_type)

    def construct(self, pred_keypoints_depth,keypoints_depth_mask,pred_keypoint_depth_uncertainty,target_depth3D):
        keypoints_depth_mask_sum = keypoints_depth_mask.sum()
        keypoints_depth_mask_inverse_sum = (1-keypoints_depth_mask).sum()
        keypoints_depth_mask_inverse = self.cast(self.nonzero((1-keypoints_depth_mask)), ms.int32)
        keypoints_depth_mask = self.cast(self.nonzero(keypoints_depth_mask), ms.int32)
        target_keypoints_depth = self.tile(ops.expand_dims(target_depth3D, -1), (1, 3))
        # pred_keypoints_depth=pred_keypoints_depth.view(-1)
        valid_pred_keypoints_depth = self.gather_nd(pred_keypoints_depth, keypoints_depth_mask)

        # valid and non-valid
        target_keypoints_depth_mask = self.gather_nd(target_keypoints_depth, keypoints_depth_mask)
        # self.print('target_keypoints_depth_mask:', target_keypoints_depth_mask)
        # self.print('valid_pred_keypoints_depth:', valid_pred_keypoints_depth)
        valid_keypoint_depth_loss = self.loss_weights * self.reg_loss_fnc(valid_pred_keypoints_depth, target_keypoints_depth_mask)

        valid_uncertainty = self.gather_nd(pred_keypoint_depth_uncertainty, keypoints_depth_mask)
        # self.print('valid_uncertainty:', valid_uncertainty)
        valid_keypoint_depth_loss = valid_keypoint_depth_loss * self.exp(- valid_uncertainty) + self.loss_weights * self.log(valid_uncertainty)
        # average
        valid_keypoint_depth_loss = valid_keypoint_depth_loss.sum() / ops.clip_by_value(self.cast(keypoints_depth_mask_sum, self.ms_type), self.clip_min)
        if keypoints_depth_mask_inverse_sum > 0:
            target_keypoints_depth_inversmask = self.gather_nd(target_keypoints_depth, keypoints_depth_mask_inverse)
            invalid_pred_keypoints_depth = self.gather_nd(pred_keypoints_depth, keypoints_depth_mask_inverse)  # The depths decoded from invalid keypoints are not used for updating networks.
            invalid_pred_keypoints_depth = ops.stop_gradient(invalid_pred_keypoints_depth)
            invalid_keypoint_depth_loss = self.loss_weights * self.reg_loss_fnc(invalid_pred_keypoints_depth, target_keypoints_depth_inversmask)

            invalid_uncertainty = self.gather_nd(pred_keypoint_depth_uncertainty, keypoints_depth_mask_inverse)
            invalid_keypoint_depth_loss = invalid_keypoint_depth_loss * self.exp(- invalid_uncertainty)
            invalid_keypoint_depth_loss = invalid_keypoint_depth_loss.sum() / self.cast(keypoints_depth_mask_inverse_sum, self.ms_type)
            keypoint_depth_loss = (valid_keypoint_depth_loss + invalid_keypoint_depth_loss)
        else:
            keypoint_depth_loss = valid_keypoint_depth_loss
        return keypoint_depth_loss

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

