import pdb
import time

import mindspore as ms
import numpy as np
from mindspore import nn
from mindspore import ops
from mindspore import Tensor
from .net_utils import *
from mindspore.common.initializer import initializer,XavierUniform
import math



# def bulid_head(cfg, in_channels):
#     return Detect_Head(cfg, in_channels)


# class Detect_Head(nn.Cell):
#     def __init__(self, cfg, in_channels):
#         super(Detect_Head, self).__init__()
#         self.predictor = Predictor(cfg, in_channels)
#
#
#     def construct(self, features, edge_count,edge_indices, test=False):
#         x = self.predictor(features, edge_count, edge_indices)
#
#         return x

class Class_head(nn.Cell):
    '''
    Cls Heads
    '''
    def __init__(self,cfg,in_channels):
        super(Class_head, self).__init__()
        self.classes = len(cfg.DATASETS.DETECT_CLASSES)
        self.ms_type = ms.float32
        self.head_conv = cfg.MODEL.HEAD.NUM_CHANNEL
        self.bn_momentum = cfg.MODEL.HEAD.BN_MOMENTUM
        norm_func = nn.BatchNorm2d

        self.head1_conv=nn.Conv2d(in_channels, self.head_conv, kernel_size=3, pad_mode='pad', padding=1, has_bias=False)
        self.head1_norm=norm_func(self.head_conv, momentum=self.bn_momentum)
        self.head1_act=nn.ReLU()

        self.head2_conv=nn.Conv2d(self.head_conv, self.classes, kernel_size=1, pad_mode='pad', padding=1 // 2, has_bias=True)
        s = ops.repeat_elements(ops.expand_dims(Tensor(- np.log(1 / cfg.MODEL.HEAD.INIT_P - 1), self.ms_type), 0), rep=3, axis=0)
        self.head2_conv.bias.set_data(s)

        # self.class_head1 = nn.SequentialCell(
        #     [nn.Conv2d(in_channels, self.head_conv, kernel_size=3, pad_mode='pad', padding=1, has_bias=False),
        #      norm_func(self.head_conv, momentum=self.bn_momentum), nn.ReLU()])
        # self.class_head2 = nn.Conv2d(self.head_conv, self.classes, kernel_size=1, pad_mode='pad', padding=1 // 2, has_bias=True)
        #
        # s = ops.repeat_elements(ops.expand_dims(Tensor(- np.log(1 / cfg.MODEL.HEAD.INIT_P - 1), self.ms_type), 0), rep=3, axis=0)
        # self.class_head2.bias.set_data(s)

    def construct(self, features):
        # feature_cls = self.head1_act(self.head1_norm(self.head1_conv(features)))
        output=self.head1_conv(features)
        feature_cls=self.head1_norm(output)
        feature_cls=self.head1_act(feature_cls)
        output_cls = self.head2_conv(feature_cls)

        return feature_cls, output_cls


class feat_layer(nn.Cell):
    def __init__(self,in_channels,head_conv,bn_momentum):
        super().__init__()
        self.conv=nn.Conv2d(in_channels, head_conv, kernel_size=3, pad_mode='pad', padding=1)
        self.norm=nn.BatchNorm2d(head_conv, momentum=bn_momentum)
        self.relu=nn.ReLU()
    def construct(self, inputs):
        output_c=self.conv(inputs)
        output=self.norm(output_c)
        return self.relu(output)

class Edge_Feature(nn.Cell):
    def __init__(self, cfg):
        super(Edge_Feature, self).__init__()
        self.classes = len(cfg.DATASETS.DETECT_CLASSES)
        self.enable_edge_fusion = cfg.MODEL.HEAD.ENABLE_EDGE_FUSION
        self.edge_fusion_kernel_size = cfg.MODEL.HEAD.EDGE_FUSION_KERNEL_SIZE
        self.edge_fusion_relu = cfg.MODEL.HEAD.EDGE_FUSION_RELU
        self.head_conv = cfg.MODEL.HEAD.NUM_CHANNEL
        self.bn_momentum = cfg.MODEL.HEAD.BN_MOMENTUM

        self.output_width = cfg.INPUT.WIDTH_TRAIN // cfg.MODEL.BACKBONE.DOWN_RATIO
        self.output_height = cfg.INPUT.HEIGHT_TRAIN // cfg.MODEL.BACKBONE.DOWN_RATIO

        self.concat = ops.Concat(axis=1)

        trunc_activision_func = nn.ReLU() if self.edge_fusion_relu else nn.Identity()

        self.trunc_heatmap_conv = nn.Conv1d(self.head_conv, self.head_conv, kernel_size=self.edge_fusion_kernel_size,
                                            padding=self.edge_fusion_kernel_size // 2, pad_mode='pad')
        self.trunc_heatmap_norm = nn.BatchNorm1d(832 * self.head_conv, momentum=self.bn_momentum)  # (cfg.SOLVER.IMS_PER_BATCH*self.head_conv, momentum=self.bn_momentum)
        self.trunc_heatmap_act_conv = nn.SequentialCell([trunc_activision_func, nn.Conv1d(self.head_conv, self.classes, kernel_size=1, pad_mode='pad')])

        self.trunc_offset_conv = nn.Conv1d(self.head_conv, self.head_conv, kernel_size=self.edge_fusion_kernel_size,
                                           padding=self.edge_fusion_kernel_size // 2, pad_mode='pad')
        self.trunc_offset_norm = nn.BatchNorm1d(832 * self.head_conv,momentum=self.bn_momentum)  # (cfg.SOLVER.IMS_PER_BATCH*self.head_conv, momentum=self.bn_momentum)
        self.trunc_offset_act_conv = nn.SequentialCell([trunc_activision_func, nn.Conv1d(self.head_conv, 2, kernel_size=1)])

        self.cast = ops.Cast()

    def construct(self, b,edge_count,edge_indices,feature_cls,reg_feature,output_cls,output_reg):
        # edge_indices_stack = self.stack([edge_indices]).squeeze(0)# B x K x 2
        edge_lens = edge_count.squeeze(1)  # B

        # normalize
        grid_edge_indices = self.cast(edge_indices.view(b, -1, 1, 2),ms.float32)  # grid_edge_indices shape: (B, K, 1, 2)
        grid_edge_indices[..., 0] = grid_edge_indices[..., 0] / (self.output_width - 1) * 2 - 1  # Normalized to [-1, 1]
        grid_edge_indices[..., 1] = grid_edge_indices[..., 1] / (self.output_height - 1) * 2 - 1  # Normalized to [-1, 1]

        # apply edge fusion for both offset and heatmap
        feature_for_fusion = self.concat((feature_cls, reg_feature))  # feature_for_fusion shape: (B, C (C=512), H, W)
        edge_features = ops.grid_sample(feature_for_fusion, grid_edge_indices, align_corners=True).squeeze(-1)  # edge_features shape: (B, C, L)

        edge_cls_feature = edge_features[:, :self.head_conv, ...]  # edge_cls_feature: feature_cls on edges.
        edge_offset_feature = edge_features[:, self.head_conv:, ...]  # edge_offset_feature: reg_feature on edges.

        (cls_c, cls_h, cls_w) = edge_cls_feature.shape
        edge_cls_output = self.trunc_heatmap_conv(edge_cls_feature).reshape(cls_c, cls_h * cls_w)
        edge_cls_output = self.trunc_heatmap_norm(edge_cls_output).reshape((cls_c, cls_h, cls_w))
        edge_cls_output = self.trunc_heatmap_act_conv(edge_cls_output)

        (offset_c, offset_h, offset_w) = edge_offset_feature.shape
        edge_offset_output = self.trunc_offset_conv(edge_offset_feature).reshape((offset_c, offset_h * offset_w))
        edge_offset_output = self.trunc_offset_norm(edge_offset_output).reshape((offset_c, offset_h, offset_w))
        edge_offset_output = self.trunc_offset_act_conv(edge_offset_output)

        # edge_indices:[1,832,2],   edge_lens:[1]
        edge_lens = self.cast(edge_lens, ms.int64)
        for k in range(b):
            output_cls_inter = []
            edge_indice_k = edge_indices[k][:edge_lens[k]]  # [n,2]
            edge_indice_k = ops.stack((edge_indice_k[:, 1], edge_indice_k[:, 0]), 1)

            output_cls_k = output_cls[k]
            output_cls_k_updates = edge_cls_output[k][:, :edge_lens[k]]
            for m in range(self.classes):
                output_cls_inter.append(ops.tensor_scatter_add(output_cls_k[m], edge_indice_k, output_cls_k_updates[m]))
            output_cls_inter_s = ops.stack(output_cls_inter, 0)
            output_cls[k] = output_cls_inter_s

            output_reg_inter = []
            output_reg_k = output_reg[k]
            output_reg_k_updates = edge_offset_output[k][:, :edge_lens[k]]
            for m in range(2):
                output_reg_inter.append(ops.tensor_scatter_add(output_reg_k[m], edge_indice_k, output_reg_k_updates[m]))
            output_reg_inter_s = ops.stack(output_reg_inter, 0)

            output_reg[k] = output_reg_inter_s
            # output_reg = ops.clip_by_value(output_reg, clip_value_max=ms.Tensor(100000, ms.float32))
        return output_cls,output_reg



class Predictor(nn.Cell):
    def __init__(self, cfg, in_channels):
        super(Predictor, self).__init__()
        # ("Car", "Cyclist", "Pedestrian")
        self.classes = len(cfg.DATASETS.DETECT_CLASSES)
        self.ms_type = ms.float32
        self.head_conv = cfg.MODEL.HEAD.NUM_CHANNEL
        self.bn_momentum = cfg.MODEL.HEAD.BN_MOMENTUM
        self.regression_channel_cfg = cfg.MODEL.HEAD.REGRESSION_CHANNELS

        self.dim2d_reg=feat_layer(in_channels, self.head_conv,self.bn_momentum)
        self.offset3d_reg = feat_layer(in_channels, self.head_conv, self.bn_momentum)
        self.corrner_offset_reg = feat_layer(in_channels, self.head_conv, self.bn_momentum)
        self.corner_uncertainty_reg = feat_layer(in_channels, self.head_conv, self.bn_momentum)
        self.uncern_reg = feat_layer(in_channels, self.head_conv, self.bn_momentum)
        self.dim3d_reg = feat_layer(in_channels, self.head_conv, self.bn_momentum)
        self.ori_reg = feat_layer(in_channels, self.head_conv, self.bn_momentum)
        self.depth_reg = feat_layer(in_channels, self.head_conv, self.bn_momentum)
        self.depth_uncertainty_reg = feat_layer(in_channels, self.head_conv, self.bn_momentum)
        self.combined_depth_uncern_reg = feat_layer(in_channels, self.head_conv, self.bn_momentum)
        self.corner_loss_uncern_reg = feat_layer(in_channels, self.head_conv, self.bn_momentum)

        self.dim2d_head = nn.Conv2d(self.head_conv, self.regression_channel_cfg[0][0], kernel_size=1, pad_mode='pad',padding=1 // 2, has_bias=True)
        _fill_fc_weights(self.dim2d_head)
        self.offset3d_head = nn.Conv2d(self.head_conv, self.regression_channel_cfg[1][0], kernel_size=1, pad_mode='pad',padding=1 // 2, has_bias=True)
        _fill_fc_weights(self.offset3d_head)
        self.corner_offset_head = nn.Conv2d(self.head_conv, self.regression_channel_cfg[2][0], kernel_size=1, pad_mode='pad',padding=1 // 2, has_bias=True)
        _fill_fc_weights(self.corner_offset_head)
        self.corner_uncertainty_head = nn.Conv2d(self.head_conv, self.regression_channel_cfg[3][0], kernel_size=1, pad_mode='pad',padding=1 // 2, has_bias=True)
        self.corner_uncertainty_head.weight.set_data(initializer('xavier_normal', self.corner_uncertainty_head.weight.shape, self.corner_uncertainty_head.weight.dtype))
        self.GRM1uncern_head = nn.Conv2d(self.head_conv, self.regression_channel_cfg[4][0], kernel_size=1, pad_mode='pad',padding=1 // 2, has_bias=True)
        _fill_fc_weights(self.GRM1uncern_head)
        self.GRM2uncern_head = nn.Conv2d(self.head_conv, self.regression_channel_cfg[4][1], kernel_size=1, pad_mode='pad',padding=1 // 2, has_bias=True)
        _fill_fc_weights(self.GRM2uncern_head)
        self.Mono_Direct_uncern_head = nn.Conv2d(self.head_conv, self.regression_channel_cfg[4][2], kernel_size=1, pad_mode='pad',padding=1 // 2, has_bias=True)
        _fill_fc_weights(self.Mono_Direct_uncern_head)
        self.Mono_Keypoint_uncern_head = nn.Conv2d(self.head_conv, self.regression_channel_cfg[4][3], kernel_size=1, pad_mode='pad',padding=1 // 2, has_bias=True)
        _fill_fc_weights(self.Mono_Keypoint_uncern_head)
        self.dim3d_head = nn.Conv2d(self.head_conv, self.regression_channel_cfg[5][0], kernel_size=1, pad_mode='pad',padding=1 // 2, has_bias=True)
        _fill_fc_weights(self.dim3d_head)
        self.oricls_head = nn.Conv2d(self.head_conv, self.regression_channel_cfg[6][0], kernel_size=1, pad_mode='pad',padding=1 // 2, has_bias=True)
        _fill_fc_weights(self.oricls_head)
        self.orioffeset_head = nn.Conv2d(self.head_conv, self.regression_channel_cfg[6][1], kernel_size=1, pad_mode='pad', padding=1 // 2, has_bias=True)
        _fill_fc_weights(self.orioffeset_head)
        self.depth_head = nn.Conv2d(self.head_conv, self.regression_channel_cfg[7][0], kernel_size=1, pad_mode='pad',padding=1 // 2, has_bias=True)
        _fill_fc_weights(self.depth_head)
        self.depth_uncertainty_head = nn.Conv2d(self.head_conv, self.regression_channel_cfg[8][0], kernel_size=1, pad_mode='pad',padding=1 // 2, has_bias=True)
        self.depth_uncertainty_head.weight.set_data(initializer('xavier_normal', self.depth_uncertainty_head.weight.shape,self.depth_uncertainty_head.weight.dtype))
        self.combined_depth_uncern_head = nn.Conv2d(self.head_conv, self.regression_channel_cfg[9][0], kernel_size=1, pad_mode='pad',padding=1 // 2, has_bias=True)
        _fill_fc_weights(self.combined_depth_uncern_head)
        self.corner_loss_uncern_head = nn.Conv2d(self.head_conv, self.regression_channel_cfg[10][0], kernel_size=1, pad_mode='pad',padding=1 // 2, has_bias=True)
        _fill_fc_weights(self.corner_loss_uncern_head)

        self.edge_feature=Edge_Feature(cfg)
        self.concat=ops.Concat(axis=1)
        self.print = ops.Print()

    def construct(self, features,feature_cls, output_cls, edge_count,edge_indices):
        self.print('predict')
        b, c, h, w = features.shape
        output_regs=[]
        dim2d_reg_feature=self.dim2d_reg(features)
        out_dim2d_reg=self.dim2d_head(dim2d_reg_feature)
        output_regs.append(out_dim2d_reg)
        # ops.print_('features:',features)
        offset3d_reg_feature = self.offset3d_reg(features)
        out_offset3d_reg = self.offset3d_head(offset3d_reg_feature)
        output_cls,out_edge_reg=self.edge_feature(b,edge_count,edge_indices,feature_cls,offset3d_reg_feature,output_cls,out_offset3d_reg)
        output_regs.append(out_edge_reg)
        corrner_offset_reg_feature = self.corrner_offset_reg(features)
        out_corrner_offset_reg = self.corner_offset_head(corrner_offset_reg_feature)
        output_regs.append(out_corrner_offset_reg)
        corner_uncertainty_reg_feature = self.corner_uncertainty_reg(features)
        out_corner_uncertainty_reg = self.corner_uncertainty_head(corner_uncertainty_reg_feature)
        output_regs.append(out_corner_uncertainty_reg)
        reg_feature=self.uncern_reg(features)
        out_reg = self.GRM1uncern_head(reg_feature)
        output_regs.append(out_reg)
        out_reg = self.GRM2uncern_head(reg_feature)
        output_regs.append(out_reg)
        out_reg = self.Mono_Direct_uncern_head(reg_feature)
        output_regs.append(out_reg)
        out_reg = self.Mono_Keypoint_uncern_head(reg_feature)
        output_regs.append(out_reg)
        reg_feature = self.dim3d_reg(features)
        out_reg = self.dim3d_head(reg_feature)
        output_regs.append(out_reg)
        reg_feature = self.ori_reg(features)
        out_reg = self.oricls_head(reg_feature)
        output_regs.append(out_reg)
        out_reg = self.orioffeset_head(reg_feature)
        output_regs.append(out_reg)
        reg_feature = self.depth_reg(features)
        out_reg = self.depth_head(reg_feature)
        output_regs.append(out_reg)
        reg_feature = self.depth_uncertainty_reg(features)
        out_reg = self.depth_uncertainty_head(reg_feature)
        output_regs.append(out_reg)
        reg_feature = self.combined_depth_uncern_reg(features)
        out_reg = self.combined_depth_uncern_head(reg_feature)
        output_regs.append(out_reg)
        reg_feature = self.corner_loss_uncern_reg(features)
        out_reg = self.corner_loss_uncern_head(reg_feature)
        output_regs.append(out_reg)

        output_cls = sigmoid_hm(output_cls)
        output_regs = self.concat(output_regs)
        # output_final=ops.Concat(axis=1)((output_cls,output_regs))

        return  (output_cls, output_regs)


# class Predictor1(nn.Cell):
#     def __init__(self, cfg, in_channels):
#         super(Predictor1, self).__init__()
#         # ("Car", "Cyclist", "Pedestrian")
#         self.classes = len(cfg.DATASETS.DETECT_CLASSES)
#         self.ms_type = ms.float32
#
#         self.regression_head_cfg = cfg.MODEL.HEAD.REGRESSION_HEADS
#         self.regression_channel_cfg = cfg.MODEL.HEAD.REGRESSION_CHANNELS
#         self.output_width = Tensor(cfg.INPUT.WIDTH_TRAIN // cfg.MODEL.BACKBONE.DOWN_RATIO,ms.int32)
#         self.output_height = Tensor(cfg.INPUT.HEIGHT_TRAIN // cfg.MODEL.BACKBONE.DOWN_RATIO,ms.int32)
#
#         self.head_conv = cfg.MODEL.HEAD.NUM_CHANNEL
#
#         # use_norm = cfg.MODEL.HEAD.USE_NORMALIZATION
#         # if use_norm == 'BN':
#         norm_func = nn.BatchNorm2d
#         # elif use_norm == 'GN':
#         #     norm_func = group_norm
#         # else:
#         #     norm_func = nn.Identity
#
#         # the inplace-abn is applied to reduce GPU memory and slightly increase the batch-size
#         self.bn_momentum = cfg.MODEL.HEAD.BN_MOMENTUM
#
#         ###########################################
#         ###############  Cls Heads ################
#         ###########################################
#
#         # self.class_head1 = nn.SequentialCell([nn.Conv2d(in_channels, self.head_conv, kernel_size=3, pad_mode='pad', padding=1, has_bias=False),
#         #                norm_func(self.head_conv, momentum=self.bn_momentum),nn.ReLU()])
#         # self.class_head2 = nn.Conv2d(self.head_conv, self.classes, kernel_size=1, pad_mode='pad', padding=1 // 2, has_bias=True)
#         #
#         # s=ops.repeat_elements(ops.expand_dims(Tensor(- np.log(1 / cfg.MODEL.HEAD.INIT_P - 1),self.ms_type),0),rep=3,axis=0)
#         # self.class_head2.bias.set_data(s)
#
#         ###########################################
#         ############  Regression Heads ############
#         ###########################################
#
#         # init regression heads
#         self.reg_features = nn.CellList()
#         self.reg_heads = nn.CellList()  # 多个nn.ModuleList，每个list都包含一个3*3卷积
#
#         # init regression heads
#         for idx, regress_head_key in enumerate(self.regression_head_cfg):
#             feat_layer = nn.SequentialCell([nn.Conv2d(in_channels, self.head_conv, kernel_size=3, pad_mode='pad',padding=1),
#                                            norm_func(self.head_conv, momentum=self.bn_momentum), nn.ReLU()])
#
#             self.reg_features.append(feat_layer)
#             # init output head
#             head_channels = self.regression_channel_cfg[idx]
#             head_list = nn.CellList()
#
#             for key_index, key in enumerate(regress_head_key):
#                 key_channel = head_channels[key_index]
#
#                 output_head = nn.Conv2d(self.head_conv, key_channel, kernel_size=1, pad_mode='pad',padding=1 // 2, has_bias=True)
#
#                 if key.find('uncertainty') >= 0 and cfg.MODEL.HEAD.UNCERTAINTY_INIT:
#                     if isinstance(output_head, nn.SequentialCell):
#                         output_head[-1].weight.set_data(initializer('xavier_normal',output_head[-1].weight.shape, output_head[-1].weight.dtype))
#                     else:
#                         output_head.weight.set_data(initializer('xavier_normal',output_head.weight.shape, output_head.weight.dtype))
#
#                 # since the edge fusion is applied to the offset branch, we should save the index of this branch
#                 if key == '3d_offset': self.offset_index = [idx, key_index]
#
#                 _fill_fc_weights(output_head)
#                 head_list.append(output_head)
#
#             self.reg_heads.append(head_list)
#
#         ###########################################
#         ##############  Edge Feature ##############
#         ###########################################
#
#         # edge feature fusion
#         self.enable_edge_fusion = cfg.MODEL.HEAD.ENABLE_EDGE_FUSION
#         self.edge_fusion_kernel_size = cfg.MODEL.HEAD.EDGE_FUSION_KERNEL_SIZE
#         self.edge_fusion_relu = cfg.MODEL.HEAD.EDGE_FUSION_RELU
#
#         self.concat=ops.Concat(axis=1)
#         # self.stack=ops.Stack(axis=0)
#
#         # if self.enable_edge_fusion:
#         trunc_norm_func = nn.BatchNorm1d if cfg.MODEL.HEAD.EDGE_FUSION_NORM == 'BN' else nn.Identity
#         trunc_activision_func = nn.ReLU() if self.edge_fusion_relu else nn.Identity()
#
#         self.trunc_heatmap_conv=nn.Conv1d(self.head_conv, self.head_conv, kernel_size=self.edge_fusion_kernel_size,padding=self.edge_fusion_kernel_size // 2,pad_mode='pad')
#         self.trunc_heatmap_norm=trunc_norm_func(832*self.head_conv, momentum=self.bn_momentum)#(cfg.SOLVER.IMS_PER_BATCH*self.head_conv, momentum=self.bn_momentum)
#         self.trunc_heatmap_act_conv=nn.SequentialCell([trunc_activision_func, nn.Conv1d(self.head_conv, self.classes, kernel_size=1, pad_mode='pad')])
#
#         self.trunc_offset_conv=nn.Conv1d(self.head_conv, self.head_conv, kernel_size=self.edge_fusion_kernel_size,padding=self.edge_fusion_kernel_size // 2,pad_mode='pad')
#         self.trunc_offset_norm=trunc_norm_func(832*self.head_conv, momentum=self.bn_momentum)#(cfg.SOLVER.IMS_PER_BATCH*self.head_conv, momentum=self.bn_momentum)
#         self.trunc_offset_act_conv=nn.SequentialCell([trunc_activision_func,nn.Conv1d(self.head_conv, 2, kernel_size=1)])
#
#         self.cast=ops.Cast()
#
#     def construct(self, features,edge_count,edge_indices):
#         ops.print_('predict')
#         b, c, h, w = features.shape
#         # targets=(targets)f
#
#         # output classification
#         # feature_cls = features
#         feature_cls = self.class_head1(features)
#         output_cls = self.class_head2(feature_cls)
#
#         output_regs = []
#         # output regression
#         i=0
#         for reg_feature_head in self.reg_features:
#             reg_feature = reg_feature_head(features)
#
#             j=0
#             for reg_output_head in self.reg_heads[i]:
#                 output_reg = reg_output_head(reg_feature)
#
#                 # apply edge feature enhancement
#                 # if self.enable_edge_fusion and i == self.offset_index[0] and j == self.offset_index[1]:
#                 if i == self.offset_index[0] and j == self.offset_index[1]:
#                     # edge_indices_stack = self.stack([edge_indices]).squeeze(0)# B x K x 2
#                     edge_lens=edge_count.squeeze(1)    # B
#
#                     # normalize
#                     grid_edge_indices = self.cast(edge_indices.view(b, -1, 1, 2),self.ms_type) # grid_edge_indices shape: (B, K, 1, 2)
#                     grid_edge_indices[..., 0] = grid_edge_indices[..., 0] / (self.output_width - 1) * 2 - 1  # Normalized to [-1, 1]
#                     grid_edge_indices[..., 1] = grid_edge_indices[..., 1] / (self.output_height - 1) * 2 - 1  # Normalized to [-1, 1]
#
#                     # apply edge fusion for both offset and heatmap
#                     feature_for_fusion = self.concat((feature_cls, reg_feature))  # feature_for_fusion shape: (B, C (C=512), H, W)
#                     edge_features = ops.grid_sample(feature_for_fusion, grid_edge_indices, align_corners=True).squeeze(-1)  # edge_features shape: (B, C, L)
#
#                     edge_cls_feature = edge_features[:, :self.head_conv, ...]  # edge_cls_feature: feature_cls on edges.
#                     edge_offset_feature = edge_features[:, self.head_conv:,...]  # edge_offset_feature: reg_feature on edges.
#
#                     (cls_c,cls_h,cls_w)=edge_cls_feature.shape
#                     edge_cls_output = self.trunc_heatmap_conv(edge_cls_feature).view(cls_c,cls_h*cls_w)
#                     edge_cls_output = self.trunc_heatmap_norm(edge_cls_output).reshape((cls_c, cls_h, cls_w))
#                     # edge_cls_output=self.trunc_heatmap_norm(cls_h*cls_w, momentum=self.bn_momentum)(edge_cls_output).reshape((cls_c,cls_h,cls_w))
#                     # edge_cls_output = self.trunc_heatmap_norm(edge_cls_output).view((cls_c, cls_h, cls_w))
#                     edge_cls_output=self.trunc_heatmap_act_conv(edge_cls_output)
#
#                     (offset_c, offset_h, offset_w) = edge_offset_feature.shape
#                     edge_offset_output = self.trunc_offset_conv(edge_offset_feature).view((offset_c , offset_h* offset_w))
#                     edge_offset_output = self.trunc_offset_norm(edge_offset_output).reshape((offset_c, offset_h, offset_w))
#                     # edge_offset_output=self.trunc_offset_norm(offset_h*offset_w, momentum=self.bn_momentum)(edge_offset_output).reshape((offset_c, offset_h, offset_w))
#                     # edge_offset_output = self.trunc_offset_norm(edge_offset_output).view((offset_c, offset_h, offset_w))
#                     edge_offset_output=self.trunc_offset_act_conv(edge_offset_output)
#
#                     #edge_indices:[1,832,2],   edge_lens:[1]
#                     edge_lens=self.cast(edge_lens,ms.int64)
#                     for k in range(b):
#                         output_cls_inter=[]
#                         edge_indice_k = edge_indices[k][ :edge_lens[k]]  # [n,2]
#                         edge_indice_k=ops.stack((edge_indice_k[:,1],edge_indice_k[:,0]),1)
#
#                         output_cls_k = output_cls[k]
#                         output_cls_k_updates=edge_cls_output[k][:, :edge_lens[k]]
#                         for m in range(self.classes):
#                             output_cls_inter.append(ops.tensor_scatter_add(output_cls_k[m], edge_indice_k, output_cls_k_updates[m]))
#                         output_cls_inter=ops.stack(output_cls_inter,0)
#                         output_cls[k]=output_cls_inter
#
#                         output_reg_inter = []
#                         output_reg_k=output_reg[k]
#                         output_reg_k_updates=edge_offset_output[k][ :, :edge_lens[k]]
#                         for m in range(2):
#                             output_reg_inter.append(ops.tensor_scatter_add(output_reg_k[m], edge_indice_k, output_reg_k_updates[m]))
#                         output_reg_inter=ops.stack(output_reg_inter,0)
#
#                         output_reg[k]=output_reg_inter
#                         output_reg = ops.clip_by_value(output_reg, clip_value_max=ms.Tensor(100000, ms.float32))
#
#                 output_regs.append(output_reg)
#                 j+=1
#             i += 1
#         output_cls =sigmoid_hm(output_cls)
#         output_regs = self.concat(output_regs)
#         # output_final=ops.Concat(axis=1)((output_cls,output_regs))
#
#         return {'cls': output_cls, 'reg': output_regs}

def sigmoid_hm(hm_features):
    x = ops.sigmoid(hm_features)
    x = ops.clip_by_value(x, ms.Tensor(1e-4, ms.float32), ms.Tensor(1 - (1e-4), ms.float32))

    return x


def _fill_fc_weights(layers):
    for name, param in layers.parameters_and_names():
        if isinstance(param, nn.Conv2d):
            param.weight.set_data(initializer('xavier_normal', param.weight.shape))
            if 'bias' in name:
                param.bias.set_data(initializer('zeros', param.shape, param.dtype))