import numpy as np

from config import cfg
import mindspore as ms
from mindspore import nn
from mindspore import ops
from mindspore import Tensor


def box_iou(box1, box2):
	intersection = max((min(box1[2], box2[2]) - max(box1[0], box2[0])), 0) * max((min(box1[3], box2[3]) - max(box1[1], box2[1])), 0)
	union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - intersection

	return intersection / union

def project_image_to_rect(uv_depth,calib_dict):
    """ Input: nx3 first two channels are uv, 3rd channel
               is depth in rect camera coord.
        Output: nx3 points in rect camera coord.
    """
    n = uv_depth.shape[0]
    x = ((uv_depth[:, 0] - calib_dict['c_u']) * uv_depth[:, 2]) / calib_dict['f_u'] + calib_dict['b_x']
    y = ((uv_depth[:, 1] - calib_dict['c_u']) * uv_depth[:, 2]) / calib_dict['f_v'] + calib_dict['b_y']

    if isinstance(uv_depth, np.ndarray):
        pts_3d_rect = np.zeros((n, 3),ms.float32)
    else:
        pts_3d_rect = ops.zeros(uv_depth.shape,ms.float32)

    pts_3d_rect[:, 0] = x
    pts_3d_rect[:, 1] = y
    pts_3d_rect[:, 2] = uv_depth[:, 2]

    return pts_3d_rect


def select_point_of_interest(batch, index, feature_maps):
    '''
    Select POI(point of interest) on feature map
    Args:
        batch: batch size
        index: in point format or index format
        feature_maps: regression feature map in [N, C, H, W]

    Returns:

    '''
    w = feature_maps.shape[3]
    if len(index.shape) == 3:
        index = index[:, :, 1] * w + index[:, :, 0]
    index = index.view(batch, -1)
    # [N, C, H, W] -----> [N, H, W, C]
    feature_maps = ops.transpose(feature_maps, (0, 2, 3, 1))
    channel = feature_maps.shape[-1]
    # [N, H, W, C] -----> [N, H*W, C]
    feature_maps = feature_maps.view(batch, -1, channel)
    # expand index in channels
    index = ops.tile(ops.expand_dims(index, -1), (1, 1, channel))  # [1,80,72]
    # select specific features bases on POIs
    feature_maps = feature_maps.gather_elements(1, index)  # Left feature_maps shape: (B, num_objs, C)

    return feature_maps


class Converter_key2channel(object):
    def __init__(self, keys, channels):
        super(Converter_key2channel, self).__init__()

        # flatten keys and channels
        self.keys = [key for key_group in keys for key in key_group]
        self.channels = [channel for channel_groups in channels for channel in channel_groups]

    def __call__(self, key):
        # find the corresponding index
        index = self.keys.index(key)

        s = sum(self.channels[:index])
        e = s + self.channels[index]

        return slice(s, e, 1)   #s:0,e:4,


def group_norm(out_channels):
    num_groups = cfg.MODEL.GROUP_NORM.NUM_GROUPS
    if out_channels % 32 == 0:
        return nn.GroupNorm(num_groups, out_channels)
    else:
        return nn.GroupNorm(num_groups // 2, out_channels)

def get_feature_by_index(x, p_h, p_w):
    """gather feature by specified index"""
    # x (n, c, h_in, w_in)
    # p_h (n, h, w, k*k)
    # p_w (n, h, w, k*k)
    n, c, h_in, w_in = x.shape
    _, h, w, k2 = p_h.shape
    # (n, c, h_in, w_in) -> (n, h_in, w_in, c)
    x = x.transpose(0, 2, 3, 1)

    # the following is the opt for:
    # input(n, h_in, w_in, c), index_x/index_y(n, h, w, k*k) -> output(n, h, w, k*k, c)

    # (n, h_in, w_in, c) -> (n*h_in*w_in, c)
    x = x.reshape(-1, c)

    # (n)
    idx_0_n = ops.range(Tensor(0,ms.int32), Tensor(n,ms.int32), Tensor(1,ms.int32))
    # (n, h, w, k*k) + (n, h, w, k*k) + (n, 1, 1, 1) -> (n, h, w, k*k)
    index = p_w + p_h * w_in + idx_0_n.reshape(n, 1, 1, 1) * w_in * h_in

    # (n*h_in*w_in, c), (n, h, w, k*k) -> (n, h, w, k*k, c)
    x_offset = ops.gather(x, index, 0)
    # (n, h*w*k*k, c) -> (n, h*w, k*k, c)
    x_offset = x_offset.reshape(n, h * w, k2, c)
    # (n, h*w, k*k, c) -> (n, c, h*w, k*k)
    x_offset = x_offset.transpose(0, 3, 1, 2)
    # (n, c, h*w, k*k) -> (n, c, h, w, k*k)
    x_offset = x_offset.reshape(n, c, h, w, k2)
    return x_offset


def regenerate_feature_map(x_offset):
    """ get rescaled feature map which was enlarged by ks**2 times."""
    # offset (n, c, h, w, k*k)
    n, c, h, w, k2 = x_offset.shape
    k = ops.ScalarCast()(k2 ** 0.5, ms.int32)
    # (n, c, h, w, k*k) -> k * (n, c, h, w, k)
    splits = ops.Split(axis=-1, output_num=k)(x_offset)
    # k * (n, c, h, w, k) -> (n, c, h, k*w, k)
    x_offset = ops.concat(splits,axis=3)
    # (n, c, h, k*w, k) -> (n, c, h*k, w*k)
    x_offset = x_offset.reshape(n, c, h * k, w * k)
    return x_offset