import math

import mindspore as ms
import numpy as np
from mindspore import nn
from mindspore import ops
from mindspore.common.initializer import initializer,XavierUniform
# from net_utils import *

from .dcnv2 import DeformConv2d
import logging

# BN_MOMENTUM = 0.9
# NORM_TYPE='BIN'
# BATCH_SIZE=1


def build_backbone(cfg):
    model = DLASeg(BN_MOMENTUM=cfg.MODEL.HEAD.BN_MOMENTUM,
        # base_name=cfg.MODEL.BACKBONE.CONV_BODY,
                   batchsize=cfg.SOLVER.IMS_PER_BATCH,
                   pretrained=cfg.MODEL.PRETRAIN,
                   down_ratio=cfg.MODEL.BACKBONE.DOWN_RATIO,
                   last_level=5
                   )
    # BATCH_SIZE=cfg.SOLVER.IMS_PER_BATCH
    return model


class DLASeg(nn.Cell):
    def __init__(self,BN_MOMENTUM,batchsize,pretrained,down_ratio,last_level):
        super(DLASeg, self).__init__()
        assert down_ratio in [2, 4, 8, 16]

        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.BN_MOMENTUM=BN_MOMENTUM
        # self.base = globals()[base_name](pretrained=pretrained)
        self.block=BasicBlock
        self.base=DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],batchsize=batchsize,
                block=BasicBlock)
        if pretrained:
            dla34_ckp = 'src/dla34' + '.ckpt'
            model_weights = ms.load_checkpoint(dla34_ckp)
            num_classes = len(model_weights[list(model_weights.keys())[-1]])
            # self.base.fc = nn.Conv2d(
            #     self.base.channels[-1], num_classes,
            #     kernel_size=1, stride=1, pad_mode='pad', padding=0, has_bias=True)
            ms.load_param_into_net(self.base, model_weights, strict_load=False)  # If we use BIN, the pre-trained weights cannot be fully loaded.
            logging.info("Load train network success")

        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)

        self.out_channels = channels[self.first_level]

        self.ida_up = IDAUp(self.out_channels, channels[self.first_level:self.last_level], [2 ** i for i in range(self.last_level - self.first_level)])

    def construct(self, x):
        # x: list of features with stride = 1, 2, 4, 8, 16, 32
        x = self.base(x)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i])
        y=self.ida_up(y, 0, len(y))

        ops.print_('features max:',y[-1].max())
        ops.print_('features nan:',ops.nonzero(ops.isnan(y[-1])).shape[0])
        y[-1]=ops.clip_by_value(y[-1],ms.Tensor(1e-4, ms.float32), ms.Tensor(1 - 1e-4, ms.float32))
        return y[-1]


class BasicBlock(nn.Cell):
    def __init__(self, inplanes, planes,batchsize, stride=1, dilation=1, norm_type = 'BIN',BN_MOMENTUM=0.9):
        super(BasicBlock, self).__init__()
        self.conv1= nn.Conv2d(inplanes,planes, kernel_size=3,
                               stride=stride, pad_mode='pad',padding=dilation, dilation=dilation)
        self.bn1=get_norm(planes,batchsize,norm_type=norm_type,momentum=BN_MOMENTUM)
        self.relu=nn.ReLU()
        self.conv2=nn.Conv2d(planes,planes, kernel_size=3,
                               stride=1, pad_mode='pad',padding=dilation, dilation=dilation)
        self.bn2=get_norm(planes,batchsize,norm_type=norm_type,momentum=BN_MOMENTUM)

    def construct(self, x, residual=None):
        if residual is None:
            residual = x
        out = self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out = self.conv2(out)
        out=self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


# def dla34(pretrained=True, **kwargs):  # DLA-34
#     model = DLA([1, 1, 1, 2, 2, 1],
#                 [16, 32, 64, 128, 256, 512],
#                 block=BasicBlock, **kwargs)
#     if pretrained:
#         dla34_ckp = 'src/dla34' +'.ckpt'
#         model_weights = ms.load_checkpoint(dla34_ckp)
#         num_classes = len(model_weights[list(model_weights.keys())[-1]])
#         model.fc = nn.Conv2d(
#             model.channels[-1], num_classes,
#             kernel_size=1, stride=1, pad_mode='pad', padding=0, has_bias=True)
#         ms.load_param_into_net(model,model_weights, strict_load=False)  # If we use BIN, the pre-trained weights cannot be fully loaded.
#         logging.info("Load train network success")

    # return model


class DLA(nn.Cell):
    def __init__(self, levels, channels,batchsize, num_classes=1000,
                 block=BasicBlock, residual_root=False, linear_root=False,BN_MOMENTUM=0.9):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.SequentialCell(nn.Conv2d(3, channels[0], kernel_size=7,
                              stride=1, pad_mode='pad', padding=3),nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),nn.ReLU())
        # self.base_layer=Con_norm_act(in_channel=3,out_channel=channels[0],kernel_size=7,stride=1,pad_mode='pad',padding=3,has_bias=False
        #                              ,BN_MOMENTUM=BN_MOMENTUM)
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], batchsize, 2,
                           level_root=False,
                           root_residual=residual_root,BN_MOMENTUM=BN_MOMENTUM)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], batchsize, 2,
                           level_root=True, root_residual=residual_root,BN_MOMENTUM=BN_MOMENTUM)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], batchsize, 2,
                           level_root=True, root_residual=residual_root,BN_MOMENTUM=BN_MOMENTUM)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], batchsize, 2,
                           level_root=True, root_residual=residual_root,BN_MOMENTUM=BN_MOMENTUM)
        self.dla_fn = [self.level0, self.level1, self.level2, self.level3, self.level4, self.level5]

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1,BN_MOMENTUM=0.9):
        modules = []
        for i in range(convs):
            modules.append([nn.Conv2d(inplanes, planes, kernel_size=3,
                              stride=stride if i == 0 else 1, pad_mode='pad', padding=dilation,
                            dilation=dilation),nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                            nn.ReLU()])
            inplanes = planes
        return nn.SequentialCell(*modules)


    def construct(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(len(self.channels)):
            x=self.dla_fn[i](x)
            x = ops.clip_by_value(x, ms.Tensor(1e-4, ms.float32), ms.Tensor(1 - 1e-4, ms.float32))
            y.append(x)
        return y


class Tree(nn.Cell):
    def __init__(self, levels, block, in_channels, out_channels, batchsize=1, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False,NORM_TYPE='BIN',BN_MOMENTUM=0.9):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels,batchsize, stride,
                               dilation=dilation, norm_type = NORM_TYPE)
            self.tree2 = block(out_channels, out_channels,batchsize, 1,
                               dilation=dilation, norm_type = NORM_TYPE)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,batchsize,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,batchsize,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual,BN_MOMENTUM)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.SequentialCell(
                [nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, has_bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)]
            )

    def construct(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class batch_instance_norm(nn.Cell):
    def __init__(self, planes,batchsize, momentum = 0.9):
        super(batch_instance_norm, self).__init__()
        self.BN = nn.BatchNorm2d(planes, momentum = momentum)
        self.BN2 = nn.BatchNorm2d(batchsize*planes, momentum=momentum)
        # self.BN_in = nn.BatchNorm2d(1, momentum=momentum)
        # self.IN = nn.InstanceNorm2d(planes, momentum = momentum)

    def construct(self, x):
        # return (self.BN(x) + self.IN(x)) / 2
        #in只能在gpu环境下运行
        n,c,h,w=x.shape
        x_in=x.view(-1,n*c,h,w)
        return (self.BN(x)+self.BN2(x_in).view(n,c,h,w)) / 2
        # n=x.shape[0]
        # IN_x_final=[]
        # for i in range(n):
        #     IN_x=ops.expand_dims(x[i],1)
        #     IN_x=self.BN_in(IN_x).transpose((1,0,2,3))
        #     IN_x_final.append(IN_x)
        # IN_x_final=ops.concat(IN_x_final,axis=0)
        # return (self.BN(x) + x) / 2


def get_norm(planes,batchsize, norm_type = 'BN', momentum=0.9):
    assert norm_type in ['BN', 'BIN']
    if norm_type == 'BN':
        return nn.BatchNorm2d(planes, momentum=momentum)
    elif norm_type == 'BIN':
        return batch_instance_norm(planes,batchsize,  momentum=momentum)



# class Con_norm_act(nn.Cell):
#     def __init__(self,in_channel,out_channel,kernel_size,stride=1,pad_mode='pad',padding=0,has_bias=False,BN_MOMENTUM=0.9):
#         super(Con_norm_act, self).__init__()
#         self.outc=out_channel
#         self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
#                                stride=stride, pad_mode=pad_mode, padding=padding,
#                                has_bias=has_bias)
#         self.bn = nn.BatchNorm2d(out_channel, momentum=BN_MOMENTUM)
#         self.act = nn.ReLU()
#     def construct(self,x):
#         out=self.conv(x)
#         out1=self.bn(out)
#         out1=self.act(out1)
#         return out1


class Root(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, residual,BN_MOMENTUM=0.9):
        super(Root, self).__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,has_bias=False
                               ,pad_mode='pad',padding=(kernel_size - 1) // 2)
        self.bn=nn.BatchNorm2d(out_channels,momentum=BN_MOMENTUM)
        self.relu = nn.ReLU()
        self.residual = residual
        self.concat = ops.Concat(axis=1)

    def construct(self, *x):
        children = x
        x = self.conv(self.concat(x))
        x=self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class DLAUp(nn.Cell):
    def __init__(self, startp, channels, scales, in_channels=None):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        self.ida = []
        for i in range(len(channels) - 1):
            j = -i - 2
            self.ida.append(IDAUp(channels[j], in_channels[j:], scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]
        self.ida_nfs = nn.CellList(self.ida)


    def construct(self, layers):
        out = [layers[-1]] # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = self.ida_nfs[i]
            layers = ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class IDAUp(nn.Cell):

    def __init__(self, o, channels, up_f,BN_MOMENTUM=0.9):
        super(IDAUp, self).__init__()
        proj_list = []
        up_list = []
        node_list = []
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = DeformConv(c, o,BN_MOMENTUM)
            node = DeformConv(o, o,BN_MOMENTUM)

            up = nn.Conv2dTranspose(o, o, f * 2, stride=f,pad_mode='pad',
                                    padding=f // 2,
                                    group=o, has_bias=False)
            fill_up_weights(up)
            proj_list.append(proj)
            up_list.append(up)
            node_list.append(node)
        self.projs = nn.CellList(proj_list)
        self.ups = nn.CellList(up_list)
        self.nodes = nn.CellList(node_list)
    def construct(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = self.ups[i - startp-1]
            project = self.projs[i - startp-1]
            layers[i] = upsample(project(layers[i]))
            layers[i] = ops.clip_by_value(layers[i], ms.Tensor(-30, ms.float32), ms.Tensor(30, ms.float32))
            node = self.nodes[i - startp-1]
            layers[i] = node(layers[i] + layers[i-1])
        return layers


def fill_up_weights(up):
    # w = up.weight.data
    w=np.zeros(up.weight.data.shape,np.float32)
    f = math.ceil(w.shape[2] / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.shape[2]):
        for j in range(w.shape[3]):
            w[:, :, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.shape[0]):
        w[c, 0, :, :] = w[0, 0, :, :]
    up.weight.set_data(ms.Tensor(w,ms.float32))


# class DeformConv2d(nn.Cell):
#     """
#     Deformable convolution opertor
#
#     Args:
#         inc(int): Input channel.
#         outc(int): Output channel.
#         kernel_size (int): Convolution window. Default: 3.
#         stride (int): The distance of kernel moving. Default: 1.
#         padding (int): Implicit paddings size on both sides of the input. Default: 1.
#         has_bias (bool): Specifies whether the layer uses a bias vector. Default: False.
#         modulation (bool): If True, modulated defomable convolution (Deformable ConvNets v2). Default: True.
#     Returns:
#         Tensor, detection of images(bboxes, score, keypoints and category id of each objects)
#     """
#     def __init__(self, inc, outc, kernel_size=3, stride=1, pad_mode='pad', padding=0, has_bias=False, modulation=True):
#         super().__init__()
#         self.stride = stride
#         self.modulation = modulation
#         self.zero_padding = nn.Pad(((0, 0), (0, 0), (padding, padding), (padding, padding)))
#         self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, pad_mode='valid', padding=0,
#                               stride=kernel_size, has_bias=has_bias)
#         self.conv.weight.set_data(initializer('xavier_normal', self.conv.weight.shape))
#         self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=kernel_size,
#                                 pad_mode=pad_mode, padding=padding, stride=stride, has_bias=False)
#         self.p_conv.weight.set_data(initializer('xavier_normal', self.p_conv.weight.shape))
#         if modulation:
#             self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=kernel_size,
#                                     pad_mode=pad_mode, padding=padding, stride=stride, has_bias=False)
#         if kernel_size % 2 == 0:
#             raise ValueError("Only odd number is supported, but current kernel sizeis {}".format(kernel_size))
#
#         self.floor=ops.Floor()
#         self.cast=ops.Cast()
#         self.expand_dims=ops.ExpandDims()
#         self.sigmoid=ops.Sigmoid()
#
#
#     def construct(self, x):
#         """deformed sampling locations with augmented offsets"""
#         # 0 ── h ──x
#         # |
#         # w
#         # |
#         # y
#
#         # (n, c, h_in, w_in)
#         x_shape = x.shape
#         # get learned shift for each pixels(the shift is relative to current pixel)
#         # (n, c, h_in, w_in) -> (n, 2*k*k, h, w)
#         offset = self.p_conv(x)
#
#         # get absolute position of each pixel w.r.s to input feature map without offset
#         # -> (1, 2*k*k, h, w)
#         p_base = _get_offset_base(offset.shape, self.stride)
#         # (1, 2*k*k, h, w) + (n, 2*k*k, h, w) -> (n, 2*k*k, h, w)
#         p = p_base + offset
#
#         # (n, 2*k*k, h, w) -> (n, h, w, 2*k*k)
#         p = p.transpose(0, 2, 3, 1)
#         p_lt = self.cast(self.floor(p),ms.int32)
#         p_rb = p_lt + 1
#
#         # (n, h, w, 2*k*k) -> (n, h, w, k*k), (n, h, w, k*k)
#         k2 = p.shape[-1] // 2
#         p_h = p[:, :, :, :k2].clip(0, x_shape[2] - 1)
#         p_w = p[:, :, :, k2:].clip(0, x_shape[3] - 1)
#
#         # (n, h, w, 2*k*k) -> (n, h, w, k*k), (n, h, w, k*k)
#         p_lt_h = p_lt[:, :, :, :k2].clip(0, x_shape[2] - 1)
#         p_lt_w = p_lt[:, :, :, k2:].clip(0, x_shape[3] - 1)
#
#         # (n, h, w, 2*k*k) -> (n, h, w, k*k), (n, h, w, k*k)
#         p_rb_h = p_rb[:, :, :, :k2].clip(0, x_shape[2] - 1)
#         p_rb_w = p_rb[:, :, :, k2:].clip(0, x_shape[3] - 1)
#
#         # perform bilinear interpolation
#         # (n, h, w, k*k) -> (n, h, w, k*k)
#         weight_lt = (1 - (p_h - p_lt_h)) * (1 - (p_w - p_lt_w))
#         weight_rb = (p_h - p_lt_h) * (p_w - p_lt_w)
#         weight_rt = (1 - (p_h - p_lt_h)) * (p_w - p_lt_w)
#         weight_lb = (p_h - p_lt_h) * (1 - (p_w - p_lt_w))
#
#         # (n, c, h_in, w_in), (n, h, w, k*k), (n, h, w, k*k) -> (n, c, h, w, k*k)
#         x_p_lt = get_feature_by_index(x, p_lt_h, p_lt_w)
#         x_p_rb = get_feature_by_index(x, p_rb_h, p_rb_w)
#         x_p_lb = get_feature_by_index(x, p_rb_h, p_lt_w)
#         x_p_rt = get_feature_by_index(x, p_lt_h, p_rb_w)
#
#         # (n, h, w, k*k) -> (n, 1, h, w, k*k) * (n, c, h, w, k*k) -> (n, c, h, w, k*k)
#         x_offset = (self.expand_dims(weight_lt, 1) * x_p_lt +
#                     self.expand_dims(weight_rb, 1) * x_p_rb +
#                     self.expand_dims(weight_lb, 1) * x_p_lb +
#                     self.expand_dims(weight_rt, 1) * x_p_rt)
#
#         if self.modulation:
#             # modulation (b, 1, h, w, N)
#             # (n, c, h, w) -> (n, k*k, h, w)
#             m = self.sigmoid(self.m_conv(x))
#             # (n, k*k, h, w) -> (n, h, w, k*k)
#             m = m.transpose(0, 2, 3, 1)
#             # (n, h, w, k*k) -> (n, 1, h, w, k*k)
#             m = self.expand_dims(m, 1)
#             # (n, 1, h, w, k*k) * (n, c, h, w, k*k) -> (n, c, h, w, k*k)
#             x_offset = x_offset * m
#         # (n, c, h, w, k*k) -> (n, c, h*k, w*k)
#         x_offset = regenerate_feature_map(x_offset)
#         # (n, c, h*k, w*k) -> (n, c, h, w)
#         out = self.conv(x_offset)
#         return out
#
#
# @ops.constexpr
# def _get_offset_base(offset_shape, stride):
#     """
#     get base position index from deformable shift of each kernel element.
#     """
#     # (n, 2*k*k, h, w)
#     k2, h, w = offset_shape[1] // 2, offset_shape[2], offset_shape[3]
#     k = int(k2**0.5)
#     # (k)
#     range_pn = np.arange(-(k - 1) // 2, (k - 1) // 2 + 1)
#     # (k, k), (k, k)
#     p_n_x, p_n_y = np.meshgrid(range_pn, range_pn, indexing='ij')
#     # (k*k), (k*k) -> (2*k, k)
#     p_n = np.concatenate((p_n_x, p_n_y), axis=0)
#     # (2*k, k) -> (1, 2*k*k, 1, 1)
#     _shape = (1, 2 * k2, 1, 1)
#     p_n = p_n.reshape(_shape)
#
#     range_h = np.arange(k // 2, h * stride + 1, stride)
#     range_w = np.arange(k // 2, w * stride + 1, stride)
#     # (h, w), (h, w)
#     p_0_x, p_0_y = np.meshgrid(range_h, range_w, indexing='ij')
#
#     # (h, w) -> (1, 1, h, w)
#     p_0_x = p_0_x.reshape(1, 1, h, w)
#     # (1, 1, h, w) -> (1, k*k, h, w)
#     p_0_x = np.tile(p_0_x, (1, k2, 1, 1))
#
#     # (h, w) -> (1, 1, h, w)
#     p_0_y = p_0_y.reshape(1, 1, h, w)
#     # (1, 1, h, w) -> (1, k*k, h, w)
#     p_0_y = np.tile(p_0_y, (1, k2, 1, 1))
#
#     # (1, k*k, h, w), (1, k*k, h, w) -> (1, 2*k*k, h, w)
#     p_0 = np.concatenate((p_0_x, p_0_y), axis=1)
#     # (1, 2*k*k, h, w) + (1, 2*k*k, 1, 1) -> (1, 2*k*k, h, w)
#     p = p_0 + p_n
#     return ms.Tensor(p, ms.float32)


class DeformConv(nn.Cell):
    def __init__(self, chi, cho,BN_MOMENTUM=0.9):
        super(DeformConv, self).__init__()
        self.actf = nn.SequentialCell(
            [nn.Conv2d(chi,cho,kernel_size=3,padding=1,pad_mode='pad'),
             # [
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU()]
        )
        # self.conv = DCN(chi, cho, kernel_size=(3,3), stride=1, padding=1, dilation=1, deformable_groups=1)
        # self.dcn = DeformConv2d(chi, cho, kernel_size=3, padding=1)
    def construct(self, x):
        # x = self.dcn(x)
        x = self.actf(x)
        return x


if __name__ == '__main__':
    model = DLASeg(BN_MOMENTUM=1,
                   batchsize=2,
                   pretrained=False,
                   down_ratio=4,
                   last_level=5)
    x = ms.Tensor(np.ones((2, 3, 384, 1280))).astype(ms.float32)
    print(x)
    for i in range(20):
        output=model(x)
        model.set_train()
        ops.print_('features.{}:'.format(i), output)
        # ops.print_('features.{} nan:'.format(i), ops.nonzero(ops.isnan(output)).shape[0])
