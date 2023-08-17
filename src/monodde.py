import logging
import time

import numpy as np
import os
import shutil

import mindspore
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp

from .backbone import *
from .predictor import *
from .loss import *
from model_utils.utils import *


class Mono_net(nn.Cell):
    '''
    Generalized structure for keypoint based object detector.
    main parts:
    - backbone:include dla network and dcn-v2network
        -dla:deep layer aggregation module
        -dcn-v2:deformable convolutional networks
    - heads:predictor module
    '''

    def __init__(self, cfg):
        super(Mono_net, self).__init__()

        # dla_dcn.NORM_TYPE = cfg.MODEL.BACKBONE.NORM_TYPE

        if cfg.MODEL.BACKBONE.CONV_BODY == 'dla34':
            self.backbone = build_backbone(cfg)
        elif cfg.MODEL.BACKBONE.CONV_BODY == 'dla34_noDCN':
            # self.backbone = dla_noDCN.DLA(cfg)
            self.backbone.out_channels = 64

        self.heads = bulid_head(cfg, self.backbone.out_channels)

        self.test = cfg.DATASETS.TEST_SPLIT == 'test'
        self.training=cfg.is_training

    def construct(self, images,edge_infor):
        ops.print_('backbone')
        images=ops.transpose(images,(0,3,1,2))
        features = self.backbone(images)
        edge_count=edge_infor[0]
        edge_indices=edge_infor[-1]
        output = self.heads(features, edge_count,edge_indices)
        return output



class MonoddeWithLossCell(nn.Cell):
    '''MonoDDE loss'''
    def __init__(self,network,cfg):
        super(MonoddeWithLossCell, self).__init__()
        self.cfg=cfg
        self.ms_type=ms.float32
        # self.per_batch=cfg.SOLVER.IMS_PER_BATCH
        self.mono_network=network
        self.loss_block=Mono_loss(cfg)
        # self.anno_encoder = Anno_Encoder(cfg)
        self.corner_loss_depth=cfg.MODEL.HEAD.CORNER_LOSS_DEPTH

        # flatten keys and channels
        self.keys = [key for key_group in cfg.MODEL.HEAD.REGRESSION_HEADS for key in key_group]
        self.channels = [channel for channel_groups in cfg.MODEL.HEAD.REGRESSION_CHANNELS for channel in channel_groups]
        self.td_dim_index=self.key2channel('2d_dim')
        self.td_offset_index=self.key2channel('3d_offset')
        self.tid_dim_index=self.key2channel('3d_dim')
        self.ori_cls_index=self.key2channel('ori_cls')
        self.ori_offset_index=self.key2channel('ori_offset')
        self.depth_index=self.key2channel('depth')
        self.depth_uncertainty_index=self.key2channel('depth_uncertainty')
        self.corner_offset_index=self.key2channel('corner_offset')
        self.combined_depth_uncern_index=self.key2channel('combined_depth_uncern')
        self.corner_loss_uncern_index=self.key2channel('corner_loss_uncern')
        self.corner_uncertainty_index=self.key2channel('corner_uncertainty')
        if 'GRM_uncern' in self.keys:
            self.GRM_uncern_index=self.key2channel('GRM_uncern')
        self.GRM1_uncern_index=self.key2channel('GRM1_uncern')
        self.GRM2_uncern_index = self.key2channel('GRM2_uncern')
        self.Mono_Direct_uncern_index=self.key2channel('Mono_Direct_uncern')
        self.Mono_Keypoint_uncern_index=self.key2channel('Mono_Keypoint_uncern')

        # self.key2channel = Converter_key2channel(keys=cfg.MODEL.HEAD.REGRESSION_HEADS,
        #                                          channels=cfg.MODEL.HEAD.REGRESSION_CHANNELS)
        self.pred_direct_depth = 'depth' in self.keys
        self.depth_with_uncertainty = 'depth_uncertainty' in self.keys
        self.compute_keypoint_corner = 'corner_offset' in self.keys
        self.corner_with_uncertainty = 'corner_uncertainty' in self.keys

        self.corner_offset_uncern = 'corner_offset_uncern' in self.keys
        if self.corner_offset_uncern:
            self.corner_offset_uncern_index=self.key2channel('corner_offset_uncern')
        self.dim_uncern = '3d_dim_uncern' in self.keys
        if self.dim_uncern:
            self.td_dim_uncern_index=self.key2channel('3d_dim_uncern')
        self.combined_depth_uncern = 'combined_depth_uncern' in self.keys
        self.corner_loss_uncern = 'corner_loss_uncern' in self.keys
        self.uncertainty_range = cfg.MODEL.HEAD.UNCERTAINTY_RANGE
        self.perdict_IOU3D = 'IOU3D_predict' in self.keys
        if self.perdict_IOU3D:
            self.IOU3D_predict_index = self.key2channel('IOU3D_predict')

        self.concat=ops.Concat(axis=1)
        self.reducesum=ops.ReduceSum()
        self.reducemean=ops.ReduceMean()
        self.ones=ops.Ones()
        self.relu=ops.ReLU()

        self.exp = ops.Exp()
        self.sigmoid = ops.Sigmoid()
        self.zeros = ops.Zeros()
        self.l2_norm = ops.L2Normalize()
        self.softmax_axis1 = nn.Softmax(axis=1)
        self.softmax_axis2 = nn.Softmax(axis=2)
        self.gather_nd = ops.GatherNd()
        self.atan2 = ops.Atan2()
        self.nonzero = ops.NonZero()
        self.cast = ops.Cast()

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
        # self.updates = ms.Tensor(np.array([PI]), self.ms_type)
        # self.bbox_index = ms.Tensor([[4, 5, 0, 1, 6, 7, 2, 3],
        #                              [0, 1, 2, 3, 4, 5, 6, 7],
        #                              [4, 0, 1, 5, 6, 2, 3, 7]], ms.int32)

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

    def decode_box2d_fcos(self, centers, pred_offset, pad_size=None, out_size=None):
        box2d_center = centers.view(-1, 2)
        box2d = self.zeros((box2d_center.shape[0], 4),self.ms_type)
        # left, top, right, bottom
        box2d[:, :2] = box2d_center - pred_offset[:, :2]
        box2d[:, 2:] = box2d_center + pred_offset[:, 2:]

        # for inference
        if pad_size is not None:
            N = box2d.shape[0]
            # upscale and subtract the padding
            box2d = box2d * self.down_ratio - ops.tile(pad_size,(1, 2))
            # clamp to the image bound
            box2d[:, 0::2]=ops.clip_by_value(box2d[:, 0::2],ms.Tensor(0,self.ms_type), ms.Tensor(out_size[0] - 1,self.ms_type))
            box2d[:, 1::2]=ops.clip_by_value(box2d[:, 1::2],ms.Tensor(0,self.ms_type), ms.Tensor(out_size[1] - 1,self.ms_type))

        return box2d

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
            center_depth = self.reducemean(center_depth, 1)
        else:
            center_height = pred_keypoints[:, -2, 1] - pred_keypoints[:, -1, 1]
            center_depth = calib['f_v'] * pred_height_3D / (center_height.abs() * self.down_ratio)

        # corner height -> depth
        corner_02_height = pred_keypoints[:, [0, 2], 1] - pred_keypoints[:, [4, 6], 1]
        corner_13_height = pred_keypoints[:, [1, 3], 1] - pred_keypoints[:, [5, 7], 1]
        corner_02_depth = calib['f_v'] * ops.expand_dims(pred_height_3D,-1) / (corner_02_height * self.down_ratio)
        corner_13_depth = calib['f_v'] * ops.expand_dims(pred_height_3D,-1) / (corner_13_height * self.down_ratio)
        corner_02_depth = self.reducemean(corner_02_depth, 1)
        corner_13_depth = self.reducemean(corner_13_depth, 1)
        # K x 3
        pred_depths = ops.stack((center_depth, corner_02_depth, corner_13_depth),axis=1)

        return pred_depths


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

        center= []
        corner_02= []
        corner_13= []
        emu_item=ops.unique(ops.cast(batch_idxs, ms.int32))[0]
        idx=0
        for gt_idx in emu_item:
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

            center.append(center_depth)
            corner_02.append(corner_02_depth)
            corner_13.append(corner_13_depth)
            idx+=1
        # for items in pred_keypoint_depths:
        pred_keypoint_depths_center = ops.clip_by_value(ops.concat(center), self.depth_range[0], self.depth_range[1])
        pred_keypoint_depths_corner_02 = ops.clip_by_value(ops.concat(corner_02), self.depth_range[0], self.depth_range[1])
        pred_keypoint_depths_corner_13 = ops.clip_by_value(ops.concat(corner_13), self.depth_range[0], self.depth_range[1])
        # for key, depths in pred_keypoint_depths.items():
        #     pred_keypoint_depths[key] = ops.clip_by_value(ops.concat(depths), self.depth_range[0], self.depth_range[1])
        # pred_depths = ops.stack(([depth for depth in pred_keypoint_depths.values()]), axis=1)
        pred_depths=ops.stack((pred_keypoint_depths_center,pred_keypoint_depths_corner_02,pred_keypoint_depths_corner_13),1)

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

        if self.dim_modes[0] == 'exp':
            dims_offset = self.exp(dims_offset)

        if self.dim_modes[2]:
            cls_dimension_std = self.dim_std[cls_id, :]
            dimensions = dims_offset * cls_dimension_std + cls_dimension_mean
        else:
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
            rotys[larger_idx]-=2 * PI
        if small_idx.shape[0] > 0:
            small_idx=small_idx.squeeze(1)
            rotys[small_idx]+=2 * PI

        larger_idx = self.cast(self.nonzero(alphas > PI),ms.int32)
        small_idx = self.cast(self.nonzero(alphas < -PI),ms.int32)
        if larger_idx.shape[0]>0:
            larger_idx = larger_idx.squeeze(1)
            alphas[larger_idx] -= 2 * PI
        if small_idx.shape[0] > 0:
            small_idx = small_idx.squeeze(1)
            alphas[small_idx] += 2 * PI
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
            pred_keypoints[corr_idx] = pred_keypoints[corr_idx] * self.down_ratio - pad_size[idx]

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

        for idx, gt_idx in enumerate(ops.unique(ops.cast(batch_idxs, ms.int32))[0].asnumpy().tolist()):
            # corr_idx = ops.nonzero(batch_idxs == gt_idx).squeeze(-1).astype(ms.float64)
            corr_idx = self.cast(self.nonzero(batch_idxs == gt_idx).squeeze(1),ms.int32)
            n_pred_keypoints[corr_idx][ :, 0] = (n_pred_keypoints[corr_idx][ :, 0] - c_u[idx]) / f_u[idx]
            n_pred_keypoints[corr_idx][ :, 1] = (n_pred_keypoints[corr_idx][ :, 1] - c_v[idx]) / f_v[idx]

        total_num_objs = n_pred_keypoints.shape[0]

        centers_3D = self.decode_3D_centers(target_centers, offset_3D, pad_size, batch_idxs)  # centers_3D: The positions of 3D centers of objects projected on original image.
        n_centers_3D = centers_3D
        for idx, gt_idx in enumerate(ops.Unique()(ops.cast(batch_idxs, ms.int32))[0].asnumpy().tolist()):
            corr_idx = self.cast(self.nonzero(batch_idxs == gt_idx).squeeze(1),ms.int32)
            n_centers_3D[corr_idx][ :,0] = (n_centers_3D[corr_idx][ :,0] - c_u[idx]) / f_u[idx]  # n_centers_3D shape: (total_num_objs, 2)
            n_centers_3D[corr_idx][ :,1] = (n_centers_3D[corr_idx][ :,1] - c_v[idx]) / f_v[idx]

        kp_group = ops.concat([(ops.Reshape(n_pred_keypoints, (total_num_objs, 20)), n_centers_3D,
                             self.zeros((total_num_objs, 2), self.ms_type))],axis=1)  # kp_group shape: (total_num_objs, 24)
        coe = self.zeros((total_num_objs, 24, 2), self.ms_type)
        coe[:, 0:: 2, 0] = -1
        coe[:, 1:: 2, 1] = -1
        A = ops.concat((coe, ops.expand_dims(kp_group, 2)), axis=2)
        coz = self.zeros((total_num_objs, 1, 3), self.ms_type)
        coz[:, :, 2] = 1
        A = ops.concat((A, coz), axis=1)  # A shape: (total_num_objs, 25, 3)

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
        B[:, 1: 8: 2, :] = ops.expand_dims((h / 2), 1)
        B[:, 9: 16: 2, :] = -ops.expand_dims((h / 2), 1)
        B[:, 17, :] = h / 2
        B[:, 19, :] = -h / 2

        total_num_objs = n_pred_keypoints.shape[0]
        pred_direct_depths = pred_direct_depths.reshape(total_num_objs, )
        for idx, gt_idx in enumerate(ops.unique(self.cast(batch_idxs,ms.int32))[0]):
            corr_idx = self.cast(ops.nonzero(batch_idxs == gt_idx).squeeze(-1),ms.int32)
            B[corr_idx][ 22, 0] = -(centers_3D[corr_idx][ 0] - c_u[idx]) * pred_direct_depths[corr_idx] / f_u[idx] - b_x[idx]
            B[corr_idx][ 23, 0] = -(centers_3D[corr_idx][ 1] - c_v[idx]) * pred_direct_depths[corr_idx] / f_v[idx] - b_y[idx]
            B[corr_idx][ 24, 0] = pred_direct_depths[corr_idx]

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

        A_coor = A_coor * ops.expand_dims(weights_coor, 2)
        B_coor = B_coor * ops.expand_dims(weights_coor, 2)

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

        A_uncern = A_uncern * ops.expand_dims(weights_uncern, 2)
        B_uncern = B_uncern * ops.expand_dims(weights_uncern, 2)

        return pinv.view(-1, 3), A_uncern, B_uncern

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

    def key2channel(self, key):
        # find the corresponding index
        index = self.keys.index(key)

        s = sum(self.channels[:index])
        e = s + self.channels[index]

        return slice(s, e, 1)


    def prepare_predictions(self, targets_variables, predictions):
        target_corner_keypoints=ms.Tensor(0)
        pred_keypoints_3D=ms.Tensor(0)
        pred_direct_depths_3D=ms.Tensor(0)
        GRM_uncern=ms.Tensor(0)
        pred_keypoints_depths_3D=ms.Tensor(0)
        pred_corner_depth_3D=ms.Tensor(0)
        pred_regression = predictions['reg']
        batch, channel, feat_h, feat_w = pred_regression.shape

        # 2. extract corresponding predictions
        flatten_reg_mask_gt=targets_variables['flatten_reg_mask_gt']
        mask_regression_2D=targets_variables['mask_regression_2D']
        pred_regression_pois_3D = select_point_of_interest(batch, targets_variables['target_centers'], pred_regression).view(-1, channel)[flatten_reg_mask_gt] # pred_regression_pois_3D shape: (valid_objs, C)
        # pred_regression_pois_3D=pred_regression_pois_3D[mask_regression_2D]
        pred_regression_2D = self.relu(pred_regression_pois_3D[mask_regression_2D][:,self.td_dim_index])  # pred_regression_2D shape: (valid_objs, 4)
        pred_offset_3D = pred_regression_pois_3D[:,self.td_offset_index]  # pred_offset_3D shape: (valid_objs, 2)
        pred_dimensions_offsets_3D = pred_regression_pois_3D[:, self.tid_dim_index]  # pred_dimensions_offsets_3D shape: (valid_objs, 3)
        pred_orientation_3D = self.concat((pred_regression_pois_3D[:, self.ori_cls_index],pred_regression_pois_3D[:, self.ori_offset_index]))  # pred_orientation_3D shape: (valid_objs, 16)

        # decode the pred residual dimensions to real dimensions
        pred_dimensions_3D = self.decode_dimension(targets_variables['target_clses'], pred_dimensions_offsets_3D)

        # preparing outputs
        targets = {'reg_2D': targets_variables['reg_2D'], 'offset_3D': targets_variables['offset_3D'], 'depth_3D': targets_variables['depth_3D'],
                   'orien_3D': targets_variables['orien_3D'],
                   'dims_3D': targets_variables['dims_3D'], 'corners_3D': targets_variables['corners_3D'], 'width_2D': targets_variables['width_2D'],
                   'rotys_3D': targets_variables['rotys_3D'],
                   'cat_3D': targets_variables['cat_3D'], 'trunc_mask_3D': targets_variables['trunc_mask_3D'], 'height_2D': targets_variables['height_2D'],
                   'GRM_valid_items': targets_variables['GRM_valid_items'],
                   'locations': targets_variables['locations'],'flatten_reg_mask_gt':flatten_reg_mask_gt
                   }

        preds = {'reg_2D': pred_regression_2D, 'offset_3D': pred_offset_3D, 'orien_3D': pred_orientation_3D,
                 'dims_3D': pred_dimensions_3D, 'mask_regression_2D':mask_regression_2D}

        reg_nums = {'reg_2D': Tensor(np.array(mask_regression_2D).sum(),ms.int32), 'reg_3D': Tensor(np.array(flatten_reg_mask_gt).sum(),ms.int32)}
                    # 'reg_obj': val_objs[0]}
        weights = {'object_weights': targets_variables['obj_weights']}

        # predict the depth with direct regression
        if self.pred_direct_depth:
            pred_depths_offset_3D = pred_regression_pois_3D[:, self.depth_index].squeeze(-1)
            pred_direct_depths_3D = self.decode_depth(pred_depths_offset_3D)
            preds['depth_3D'] = pred_direct_depths_3D  # pred_direct_depths_3D shape: (valid_objs,)

        # predict the uncertainty of depth regression
        if self.depth_with_uncertainty:
            preds['depth_uncertainty'] = pred_regression_pois_3D[:, self.depth_uncertainty_index].squeeze(1)  # preds['depth_uncertainty'] shape: (val_objs,)

            if self.uncertainty_range is not None:
                preds['depth_uncertainty'] = ops.clip_by_value(preds['depth_uncertainty'],self.uncertainty_range[0],
                                                         self.uncertainty_range[1])

        # else:
        # 	print('depth_uncertainty: {:.2f} +/- {:.2f}'.format(
        # 		preds['depth_uncertainty'].mean().item(), preds['depth_uncertainty'].std().item()))

        # predict the keypoints
        if self.compute_keypoint_corner:
            # targets for keypoints
            target_corner_keypoints = targets_variables["keypoints"].view(len(flatten_reg_mask_gt), -1, 3)[flatten_reg_mask_gt]  # target_corner_keypoints shape: (val_objs, 10, 3)
            targets['keypoints'] = target_corner_keypoints[..., :2]  # targets['keypoints'] shape: (val_objs, 10, 2)
            targets['keypoints_mask'] = target_corner_keypoints[..., -1]  # targets['keypoints_mask'] shape: (val_objs, 10)
            reg_nums['keypoints'] = self.reducesum(targets['keypoints_mask'])
            # mask for whether depth should be computed from certain group of keypoints
            target_corner_depth_mask = targets_variables["keypoints_depth_mask"].view(-1, 3)[flatten_reg_mask_gt]
            targets['keypoints_depth_mask'] = self.cast(target_corner_depth_mask,ms.bool_)  # target_corner_depth_mask shape: (val_objs, 3)

            # predictions for keypoints
            pred_keypoints_3D = pred_regression_pois_3D[:, self.corner_offset_index]
            pred_keypoints_3D = pred_keypoints_3D.view((np.array(flatten_reg_mask_gt).sum(), -1, 2))
            preds['keypoints'] = pred_keypoints_3D  # pred_keypoints_3D shape: (val_objs, 10, 2)

            pred_keypoints_depths_3D = self.decode_depth_from_keypoints_batch(pred_keypoints_3D,
                                                                                           pred_dimensions_3D,
                                                                                           targets_variables['calib'],
                                                                                           targets_variables['batch_idxs'])
            preds['keypoints_depths'] = pred_keypoints_depths_3D  # pred_keypoints_depths_3D shape: (val_objs, 3)

        # Optimize keypoint offset with uncertainty.
        if self.corner_offset_uncern:
            corner_offset_uncern = pred_regression_pois_3D[:, self.corner_offset_uncern_index]
            preds['corner_offset_uncern'] = self.exp(ops.clip_by_value(corner_offset_uncern, self.uncertainty_range[0], self.uncertainty_range[1]))

        # Optimize dimension with uncertainty.
        if self.dim_uncern:
            dim_uncern = pred_regression_pois_3D[:, self.td_dim_uncern_index]
            preds['dim_uncern'] = self.exp(ops.clip_by_value(dim_uncern, self.uncertainty_range[0],self.uncertainty_range[1]))

        # Optimize combined_depth with uncertainty
        if self.combined_depth_uncern:
            combined_depth_uncern = pred_regression_pois_3D[:, self.combined_depth_uncern_index]
            preds['combined_depth_uncern'] = self.exp(ops.clip_by_value(combined_depth_uncern, self.uncertainty_range[0],self.uncertainty_range[1]))

        # Optimize corner coordinate loss with uncertainty
        if self.corner_loss_uncern:
            corner_loss_uncern = pred_regression_pois_3D[:, self.corner_loss_uncern_index]
            preds['corner_loss_uncern'] = self.exp(ops.clip_by_value(corner_loss_uncern, self.uncertainty_range[0],self.uncertainty_range[1]))

        # predict the uncertainties of the solved depths from groups of keypoints
        if self.corner_with_uncertainty:
            preds['corner_offset_uncertainty'] = pred_regression_pois_3D[:, self.corner_uncertainty_index]  # preds['corner_offset_uncertainty'] shape: (val_objs, 3)

            if self.uncertainty_range is not None:
                preds['corner_offset_uncertainty'] = ops.clip_by_value(preds['corner_offset_uncertainty'],self.uncertainty_range[0],self.uncertainty_range[1])

        if self.corner_loss_depth == 'GRM':
            GRM_uncern = pred_regression_pois_3D[:, self.GRM_uncern_index]  # GRM_uncern shape: (num_objs, 25)
            GRM_uncern = self.exp(ops.clip_by_value(GRM_uncern, self.uncertainty_range[0], self.uncertainty_range[1]))
            # Decode rot_y
            # Verify the correctness of orientation decoding.
            '''gt_ori_code = ops.zeros_like(pred_orientation_3D).to(pred_orientation_3D.device)	# gt_ori_code shape: (num_objs, 16)
            gt_ori_code[:, 0:8:2] = 0.1
            gt_ori_code[:, 1:8:2] = target_orientation_3D[:, 0:4]
            gt_ori_code[:, 8::2] = ops.sin(target_orientation_3D[:, 4:8])
            gt_ori_code[:, 9::2] = ops.cos(target_orientation_3D[:, 4:8])
            pred_orientation_3D = gt_ori_code'''
            info_dict = {'target_centers': targets_variables['valid_targets_bbox_points'], 'offset_3D': targets_variables['offset_3D'],
                         'pad_size': targets_variables['pad_size'],
                         'calib': targets_variables['calib'], 'batch_idxs': targets_variables['batch_idxs']}
            GRM_rotys, _ = self.decode_axes_orientation(pred_orientation_3D, dict_for_3d_center=info_dict)

            info_dict.update({'ori_imgs': targets_variables['ori_imgs'], 'keypoint_offset': target_corner_keypoints,
                              'locations': targets_variables['locations'],
                              'dimensions': targets_variables['dims_3D'], 'rotys': targets_variables['rotys_3D']})
            GRM_locations, GRM_A, GRM_B = self.decode_from_GRM(ops.expand_dims(GRM_rotys,1), pred_dimensions_3D,
                                                                            pred_keypoints_3D.view(-1, 20),
                                                                            pred_direct_depths_3D.view(-1, 1),
                                                                            GRM_uncern=GRM_uncern,
                                                                            GRM_valid_items=targets_variables['GRM_valid_items'],
                                                                            batch_idxs=targets_variables['batch_idxs'], cfg=self.cfg,
                                                                            targets_dict=info_dict)
            pred_corner_depth_3D = GRM_locations[:, 2]

            preds.update({'combined_depth': pred_corner_depth_3D, 'GRM_A': GRM_A, 'GRM_B': GRM_B, 'GRM_uncern': GRM_uncern})

        elif self.corner_loss_depth == 'soft_GRM':
            if 'GRM_uncern' in self.keys:
                GRM_uncern = pred_regression_pois_3D[:,self.GRM_uncern_index]  # GRM_uncern shape: (num_objs, 20)
            elif 'GRM1_uncern' in self.keys:
                uncern_GRM1 = pred_regression_pois_3D[:,self.GRM1_uncern_index]  # uncern_GRM1 shape: (num_objs, 8)
                uncern_GRM2 = pred_regression_pois_3D[:,self.GRM2_uncern_index]  # uncern_GRM1 shape: (num_objs, 8)
                uncern_Mono_Direct = pred_regression_pois_3D[:,self.Mono_Direct_uncern_index]  # uncern_Mono_Direct shape: (num_objs, 1)
                uncern_Mono_Keypoint = pred_regression_pois_3D[:, self.Mono_Keypoint_uncern_index]  # uncern_Mono_Keypoint shape: (num_objs, 3)
                GRM_uncern = self.concat((uncern_GRM1, uncern_GRM2))
                GRM_uncern = self.concat((GRM_uncern, uncern_Mono_Direct, uncern_Mono_Keypoint))  # GRM_uncern shape: (num_objs, 20)
            GRM_uncern = self.exp(ops.clip_by_value(GRM_uncern, self.uncertainty_range[0], self.uncertainty_range[1]))
            assert GRM_uncern.shape[1] == 20

            pred_combined_depths = self.concat((ops.expand_dims(pred_direct_depths_3D,1), pred_keypoints_depths_3D))  # pred_combined_depths shape: (valid_objs, 4)
            info_dict = {'target_centers': targets_variables['valid_targets_bbox_points'], 'offset_3D': targets_variables['offset_3D'],
                         'pad_size': targets_variables['pad_size'],
                         'calib': targets_variables['calib'], 'batch_idxs': targets_variables['batch_idxs']}
            GRM_rotys, _ = self.decode_axes_orientation(pred_orientation_3D, dict_for_3d_center=info_dict)

            pred_vertex_offset = pred_keypoints_3D[:, 0:8, :]  # Do not use the top center and bottom center.
            pred_corner_depth_3D, separate_depths = self.decode_from_SoftGRM(ops.expand_dims(GRM_rotys,1),
                                                                                          pred_dimensions_3D,
                                                                                          pred_vertex_offset.reshape(-1,16),
                                                                                          pred_combined_depths,
                                                                                          targets_dict=info_dict,
                                                                                          GRM_uncern=GRM_uncern,
                                                                                          batch_idxs=targets_variables['batch_idxs'])  # pred_corner_depth_3D shape: (val_objs,), separate_depths shape: (val_objs, 20)
            preds.update({'combined_depth': pred_corner_depth_3D, 'separate_depths': separate_depths, 'GRM_uncern': GRM_uncern})

        elif self.corner_loss_depth == 'direct':
            pred_corner_depth_3D = pred_direct_depths_3D  # Only use estimated depth.

        elif self.corner_loss_depth == 'keypoint_mean':
            pred_corner_depth_3D = self.reducemean(preds['keypoints_depths'],1)  # Only use depth solved by keypoints.

        else:
            assert self.corner_loss_depth in ['soft_combine', 'hard_combine']
            # make sure all depths and their uncertainties are predicted
            pred_combined_uncertainty = self.exp(self.concat((ops.expand_dims(preds['depth_uncertainty'],-1), preds['corner_offset_uncertainty'])))  # pred_combined_uncertainty shape: (val_objs, 4)
            pred_combined_depths = self.concat((ops.expand_dims(pred_direct_depths_3D,-1), preds['keypoints_depths']))  # pred_combined_depths shape: (val_objs, 4)

            if self.corner_loss_depth == 'soft_combine':  # Weighted sum.
                pred_uncertainty_weights = 1 / pred_combined_uncertainty
                pred_uncertainty_weights = pred_uncertainty_weights / ops.reduce_sum(pred_uncertainty_weights,keepdim=True,dim=1)
                pred_corner_depth_3D = self.reducesum(pred_combined_depths * pred_uncertainty_weights, 1)
                preds['weighted_depths'] = pred_corner_depth_3D  # pred_corner_depth_3D shape: (val_objs,)

            elif self.corner_loss_depth == 'hard_combine':  # Directly use the depth with the smallest uncertainty.
                pred_corner_depth_3D = pred_combined_depths[ops.arange(pred_combined_depths.shape[0]), pred_combined_uncertainty.argmin(axis=1)]

        if self.perdict_IOU3D:
            preds['IOU3D_predict'] = pred_regression_pois_3D[:, self.IOU3D_predict_index]

        pred_locations_3D = self.decode_location_flatten(targets_variables['valid_targets_bbox_points'], pred_offset_3D,
                                                                      pred_corner_depth_3D,
                                                                      targets_variables['calib'],
                                                                      targets_variables['pad_size'],
                                                                      targets_variables['batch_idxs'])  # pred_locations_3D shape: (val_objs, 3)
        # decode rotys and alphas
        pred_rotys_3D, _ = self.decode_axes_orientation(pred_orientation_3D,locations=pred_locations_3D)  # pred_rotys_3D shape: (val_objs,)
        # encode corners
        pred_corners_3D = self.encode_box3d(pred_rotys_3D, pred_dimensions_3D, pred_locations_3D)  # pred_corners_3D shape: (val_objs, 8, 3)
        # concatenate all predictions
        pred_bboxes_3D = self.concat((pred_locations_3D, pred_dimensions_3D, pred_rotys_3D[:, None]))  # pred_bboxes_3D shape: (val_objs, 7)

        preds.update({'corners_3D': pred_corners_3D, 'rotys_3D': pred_rotys_3D, 'cat_3D': pred_bboxes_3D})

        return targets, preds, reg_nums, weights


    def construct(self,images, edge_infor, targets_heatmap, targets_variables,iteration):
        predictions=self.mono_network(images, edge_infor)
        pred_targets, preds, reg_nums, weights = self.prepare_predictions(targets_variables, predictions)
        loss=self.loss_block(targets_heatmap, predictions, pred_targets, preds, reg_nums, weights, iteration)
        return loss
