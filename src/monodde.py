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
from .predictor import Predictor,Class_head
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
        self.backbone = build_backbone(cfg)
        self.heads=Class_head(cfg,self.backbone.out_channels)
        self.predictor = Predictor(cfg, self.backbone.out_channels)

        self.test = cfg.DATASETS.TEST_SPLIT == 'test'
        self.training=cfg.is_training
        self.print=ops.Print()
        self.cls_loss_fnc = FocalLoss(cfg.MODEL.HEAD.LOSS_PENALTY_ALPHA,
                                      cfg.MODEL.HEAD.LOSS_BETA,
                                      0.2)  # penalty-reduced focal loss
        self.mono_loss=Mono_loss(cfg)
    def construct(self,images, edge_infor,targets_heatmap,targets_original,targets_select,calibs,iteration):
        self.print('backbone')
        features = self.backbone(images)
        edge_count=edge_infor[0]
        edge_indices=edge_infor[1]
        feature_cls, output_cls=self.heads(features)
        feature_cls = ops.clip_by_value(feature_cls, ms.Tensor(1e-4, ms.float32), ms.Tensor(1 - 1e-4, ms.float32))
        output_cls = ops.clip_by_value(output_cls, ms.Tensor(1e-4, ms.float32), ms.Tensor(1 - 1e-4, ms.float32))
        prediction = self.predictor(features,feature_cls, output_cls, edge_count,edge_indices)
        # if self.training:
        ops.print_('hm max:',prediction[0].max())
        hm_loss = self.cls_loss_fnc(prediction[0], targets_heatmap)
        # if hm_loss>200:
        #     hm_loss=hm_loss*0.0001
        hm_loss = ops.clip_by_value(hm_loss,ms.Tensor(0, ms.float32), ms.Tensor(100, ms.float32))
        ops.print_('hm_loss:',hm_loss)
        reg_loss_out=self.mono_loss(targets_original,targets_select,calibs, prediction[1], iteration)
        reg_loss_out.append(hm_loss)
        return reg_loss_out



