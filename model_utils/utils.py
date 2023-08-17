# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# import csv
import os
# import pdb
from collections import defaultdict
from collections import deque

import numpy
import numpy as np
from mindspore import Tensor, ops, nn
import mindspore as ms
from shapely import Polygon


def rad_to_matrix(rotys, N):
    # device = rotys.device

    cos, sin = ops.cos(rotys), ops.sin(rotys)

    i_temp = ops.Tensor([[1, 0, 1],
                         [0, 1, 0],
                         [-1, 0, 1]], dtype=ms.float32)

    ry = i_temp.tile((N, 1)).view(N, -1, 3)

    ry[:, 0, 0] *= cos
    ry[:, 0, 2] *= sin
    ry[:, 2, 0] *= sin
    ry[:, 2, 2] *= cos

    return ry


def encode_box3d(rotys, dims, locs):
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
    ry = rad_to_matrix(rotys, N)

    # l, h, w
    dims_corners = ops.tile(dims.view((-1, 1)),(1, 8))
    dims_corners = dims_corners * 0.5
    dims_corners[:, 4:] = -dims_corners[:, 4:]
    index = ms.Tensor([[4, 5, 0, 1, 6, 7, 2, 3],
                        [0, 1, 2, 3, 4, 5, 6, 7],
                        [4, 0, 1, 5, 6, 2, 3, 7]], ms.int32).tile((N, 1))

    box_3d_object = ops.gather_elements(dims_corners, 1, index)
    b = box_3d_object.view((N, 3, -1))
    box_3d = ops.matmul(ry, b)  # ry:[11,3,3]   box_3d_object:[11,3,8]
    box_3d += ops.expand_dims(locs, -1).tile((1, 1, 8))

    return ops.transpose(box_3d, (0, 2, 1))

def project_image_to_rect(uv_depth,calib_dict):
    """ Input: nx3 first two channels are uv, 3rd channel
               is depth in rect camera coord.
        Output: nx3 points in rect camera coord.
    """
    n = uv_depth.shape[0]
    x = ((uv_depth[:, 0] - calib_dict['c_u']) * uv_depth[:, 2]) / calib_dict['f_u'] + calib_dict['b_x']
    y = ((uv_depth[:, 1] - calib_dict['c_u']) * uv_depth[:, 2]) / calib_dict['f_v'] + calib_dict['b_y']

    if isinstance(uv_depth, np.ndarray):
        pts_3d_rect = np.zeros((n, 3))
    else:
        pts_3d_rect = ops.zeros(uv_depth.shape,ms.float32)

    pts_3d_rect[:, 0] = x
    pts_3d_rect[:, 1] = y
    pts_3d_rect[:, 2] = uv_depth[:, 2]

    return pts_3d_rect
def decode_location_flatten(points, offsets, depths, calibs, pad_size, batch_idxs,down_ratio):
    batch_size = len(calibs)
    gts = ops.unique(batch_idxs)[0]
    locations = ops.zeros((points.shape[0], 3), ms.float32)
    points = (points + offsets) * down_ratio - pad_size[batch_idxs]  # Left points: The 3D centers in original images.

    for idx, gt in enumerate(gts):
        corr_pts_idx = ops.nonzero(batch_idxs == gt).squeeze(-1)
        calib = calibs[gt]
        # concatenate uv with depth
        corr_pts_depth = ops.concat((points[corr_pts_idx], depths[corr_pts_idx, None]),1)
        # locations = ops.tensor_scatter_add(locations, corr_pts_idx, corr_pts_depth)
        locations[corr_pts_idx] = project_image_to_rect(corr_pts_depth, calib)
    return locations


def prepare_targets(data,cfg):
    # print('prepare_targets')
    per_batch=cfg.SOLVER.IMS_PER_BATCH
    down_ratio = cfg.MODEL.BACKBONE.DOWN_RATIO
    corner_loss_depth = cfg.MODEL.HEAD.CORNER_LOSS_DEPTH
    edge_infor = [data[-3], data[-2]]
    calibs = []
    for i in range(per_batch):
        calibs.append(
            dict(P=data[22][i, :, :], R0=data[23][i, :, :], C2V=data[24][i, :, :], c_u=data[25][i], c_v=data[26][i],
                 f_u=data[27][i],
                 f_v=data[28][i], b_x=data[29][i], b_y=data[30][i]))
    reg_mask = ops.cast(data[9], ms.bool_)
    ori_imgs = ops.cast(data[14], ms.int32)
    trunc_mask = ops.cast(data[16], ms.int32)
    flatten_reg_mask_gt = reg_mask.view(-1).asnumpy().tolist()  # flatten_reg_mask_gt shape: (B * num_objs)

    # the corresponding image_index for each object, used for finding pad_size, calib and so on
    batch_idxs = ops.arange(per_batch,dtype=ms.int32).view(-1,1).expand_as(reg_mask).reshape(-1) # batch_idxs shape: (B * num_objs)
    batch_idxs = batch_idxs[flatten_reg_mask_gt]  # Only reserve the features of valid objects.
    valid_targets_bbox_points = data[4].view(-1, 2)[flatten_reg_mask_gt]  # valid_targets_bbox_points shape: (valid_objs, 2)

    # fcos-style targets for 2D
    target_bboxes_2D = data[12].view(-1, 4)[flatten_reg_mask_gt]  # target_bboxes_2D shape: (valid_objs, 4). 4 -> (x1, y1, x2, y2)
    target_bboxes_height = target_bboxes_2D[:, 3] - target_bboxes_2D[:, 1]  # target_bboxes_height shape: (valid_objs,)
    target_bboxes_width = target_bboxes_2D[:, 2] - target_bboxes_2D[:, 0]  # target_bboxes_width shape: (valid_objs,)

    target_regression_2D = ops.concat((valid_targets_bbox_points - target_bboxes_2D[:, :2], target_bboxes_2D[:,2:] - valid_targets_bbox_points),axis=1)  # offset to 2D bbox boundaries.
    mask_regression_2D = ops.logical_and(target_bboxes_height > 0,target_bboxes_width > 0)
    mask_regression_2D=mask_regression_2D.asnumpy().tolist()
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
                                          target_depths_3D,calibs,data[13],batch_idxs,down_ratio)  # target_locations_3D shape: (valid_objs, 3)
    target_corners_3D = encode_box3d(target_rotys_3D, target_dimensions_3D, target_locations_3D)  # target_corners_3D shape: (valid_objs, 8, 3)
    target_bboxes_3D = ops.concat((target_locations_3D, target_dimensions_3D, target_rotys_3D[:, None]),axis=1)  # target_bboxes_3D shape: (valid_objs, 7)

    target_trunc_mask = trunc_mask.view(-1)[flatten_reg_mask_gt]  # target_trunc_mask shape(valid_objs,)
    obj_weights = data[10].view(-1)[flatten_reg_mask_gt]  # obj_weights shape: (valid_objs,)
    target_corner_keypoints = data[5].view(len(flatten_reg_mask_gt), -1, 3)[flatten_reg_mask_gt]  # target_corner_keypoints shape: (val_objs, 10, 3)
    target_corner_depth_mask = data[6].view(-1, 3)[flatten_reg_mask_gt]

    keypoints_visible = data[21].view(-1, data[21].shape[-1])[flatten_reg_mask_gt]  # keypoints_visible shape: (valid_objs, 11)
    if corner_loss_depth == 'GRM':
        keypoints_visible = ops.tile(ops.expand_dims(keypoints_visible, 2), (1, 1, 2)).reshape((keypoints_visible.shape[0], -1))  # The effectness of first 22 GRM equations.
        GRM_valid_items = ops.concat((keypoints_visible, ops.ones((keypoints_visible.shape[0], 3), ms.bool_)),axis=1)  # GRM_valid_items shape: (valid_objs, 25)
    elif corner_loss_depth == 'soft_GRM':
        keypoints_visible = ops.tile(ops.expand_dims(keypoints_visible[:, 0:8], 2), (1, 1, 2)).reshape((keypoints_visible.shape[0], -1))  # The effectiveness of the first 16 equations. shape: (valid_objs, 16)
        direct_depth_visible = ops.ones((keypoints_visible.shape[0], 1), ms.bool_)
        veritical_group_visible = ops.cast(data[6].view(-1, 3)[flatten_reg_mask_gt],ms.bool_)  # veritical_group_visible shape: (valid_objs, 3)
        GRM_valid_items = ops.concat((ops.cast(keypoints_visible,ms.bool_), direct_depth_visible, veritical_group_visible),axis=1)  # GRM_valid_items shape: (val_objs, 20)
    else:
        GRM_valid_items = None
        # preparing outputs
    return_dict = {'cls_ids':data[3],'pad_size':data[13],'target_centers':data[4],'calib':calibs,'reg_2D': target_regression_2D, 'offset_3D': target_offset_3D, 'depth_3D': target_depths_3D,
               'orien_3D': target_orientation_3D,'valid_targets_bbox_points':valid_targets_bbox_points,
               'dims_3D': target_dimensions_3D, 'corners_3D': target_corners_3D, 'width_2D': target_bboxes_width,
               'rotys_3D': target_rotys_3D,'target_clses':target_clses,
               'cat_3D': target_bboxes_3D, 'trunc_mask_3D': target_trunc_mask, 'height_2D': target_bboxes_height,
               'GRM_valid_items': GRM_valid_items.asnumpy().tolist(),'target_corner_depth_mask':target_corner_depth_mask,
               'locations': target_locations_3D,'obj_weights':obj_weights,'target_corner_keypoints':target_corner_keypoints,'mask_regression_2D':mask_regression_2D,
                   'flatten_reg_mask_gt':flatten_reg_mask_gt,'batch_idxs':batch_idxs,'keypoints':data[5],'keypoints_depth_mask':data[6],
                   'ori_imgs':ori_imgs
               }

    return data[0], edge_infor, data[19], return_dict


class SmoothedValue():
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=10):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    @property
    def value(self):
        d = Tensor(list(self.deque))
        return d[-1].item()

    @property
    def median(self):
        d = Tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = Tensor(list(self.deque))
        return d.mean().asnumpy()

    @property
    def global_avg(self):
        return self.total / self.count


class MetricLogger():
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, Tensor) or isinstance(v, numpy.ndarray):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("{} object has no attribute {}".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            # loss_str.append(
            #     "{}: {:.4f} ({:.4f})".format(name, meter.median, meter.global_avg)
            # )
            loss_str.append(
                "{}: {:.4f}".format(name, meter.avg)
            )
        return self.delimiter.join(loss_str)


def get_device_id():
    device_id = os.getenv('DEVICE_ID', '0')
    return int(device_id)


class AllReduce(nn.Cell):
    def __init__(self):
        super(AllReduce, self).__init__()
        self.all_reduce = ops.AllReduce()

    def construct(self, x):
        return self.all_reduce(x)


def nms_hm(heat_map, kernel=3, reso=1):
    kernel = int(kernel / reso)
    if kernel % 2 == 0:
        kernel += 1

    pad = (kernel - 1) // 2
    maxpool2d = nn.MaxPool2d(kernel_size=kernel, stride=1, pad_mode='pad', padding=pad)
    hmax = maxpool2d(heat_map)

    eq_index = (hmax == heat_map)

    return heat_map * eq_index


# def get_iou_3d(pred_corners, target_corners):
#     """
#     :param corners3d: (N, 8, 3) in rect coords
#     :param query_corners3d: (N, 8, 3)
#     :return: IoU
#     """
#     min = ops.Minimum()
#     max = ops.Maximum()
#     zeros=ops.Zeros()
#
#     A, B = pred_corners, target_corners
#     N = A.shape[0]
#
#     # init output
#     iou3d = zeros((N,),ms.float32)
#
#     # for height overlap, since y face down, use the negative y
#     min_h_a = - A[:, 0:4, 1].sum(axis=1) / 4.0
#     max_h_a = - A[:, 4:8, 1].sum(axis=1) / 4.0
#     min_h_b = - B[:, 0:4, 1].sum(axis=1) / 4.0
#     max_h_b = - B[:, 4:8, 1].sum(axis=1) / 4.0
#
#     # overlap in height
#     h_max_of_min = max(min_h_a, min_h_b)
#     h_min_of_max = min(max_h_a, max_h_b)
#     h_overlap = max(zeros(h_min_of_max.shape,ms.float32),h_min_of_max - h_max_of_min)
#
#     # x-z plane overlap
#     A=A.asnumpy()
#     B=B.asnumpy()
#     for i in range(N):
#         if flatten_reg_mask_gt[i]==False:continue
#         # pdb.set_trace()
#         # print(A[i][0:4, [0, 2]])
#         # print(B[i][0:4, [0, 2]])
#         bottom_a, bottom_b =  Polygon(A[i][ 0:4, [0, 2]]), Polygon(B[i][ 0:4, [0, 2]])
#         if bottom_a.is_valid and bottom_b.is_valid:
#             # check is valid,  A valid Polygon may not possess any overlapping exterior or interior rings.
#             bottom_overlap = bottom_a.intersection(bottom_b)
#             bottom_overlap=bottom_overlap.area
#         else:
#             bottom_overlap =0
#
#         overlap3d = bottom_overlap * h_overlap[i]
#         union3d = bottom_a.area * (max_h_a[i] - min_h_a[i]) + bottom_b.area  * (max_h_b[i] - min_h_b[i]) - overlap3d
#
#         iou3d[i] = overlap3d / union3d
#
#     return iou3d


def get_iou_3d(pred_corners, target_corners):
    """
    :param corners3d: (N, 8, 3) in rect coords
    :param query_corners3d: (N, 8, 3)
    :return: IoU
    """
    min = ops.Minimum()
    max = ops.Maximum()
    zeros=ops.Zeros()

    A, B = pred_corners, target_corners
    N = A.shape[0]

    # init output
    iou3d = zeros((N,),ms.float32)

    # for height overlap, since y face down, use the negative y
    min_h_a = - A[:, 0:4, 1].sum(axis=1) / 4.0
    max_h_a = - A[:, 4:8, 1].sum(axis=1) / 4.0
    min_h_b = - B[:, 0:4, 1].sum(axis=1) / 4.0
    max_h_b = - B[:, 4:8, 1].sum(axis=1) / 4.0

    # overlap in height
    h_max_of_min = max(min_h_a, min_h_b)
    h_min_of_max = min(max_h_a, max_h_b)
    h_overlap = max(zeros(h_min_of_max.shape,ms.float32),h_min_of_max - h_max_of_min)

    # x-z plane overlap
    A=A.asnumpy()
    B=B.asnumpy()
    for i in range(N):
        bottom_a, bottom_b =  Polygon(A[i][0:4, [0, 2]]), Polygon(B[i][0:4, [0, 2]])
        if bottom_a.is_valid and bottom_b.is_valid:
            # check is valid,  A valid Polygon may not possess any overlapping exterior or interior rings.
            bottom_overlap = bottom_a.intersection(bottom_b)
            bottom_overlap=bottom_overlap.area
        else:
            bottom_overlap =0

        overlap3d = bottom_overlap * h_overlap[i]
        union3d = bottom_a.area * (max_h_a[i] - min_h_a[i]) + bottom_b.area  * (max_h_b[i] - min_h_b[i]) - overlap3d

        iou3d[i] = overlap3d / union3d

    return iou3d


def select_topk(heat_map, K=100):
    '''
    Args:
        heat_map: heat_map in [N, C, H, W]
        K: top k samples to be selected
        score: detection threshold

    Returns:

    '''
    batch, cls, height, width = heat_map.shape
    topk=ops.TopK()

    # First select topk scores in all classes and batchs
    # [N, C, H, W] -----> [N, C, H*W]
    heat_map = heat_map.view(batch, cls, -1)
    # Both in [N, C, K]
    topk_scores_all, topk_inds_all = topk(heat_map, K)[0]

    # topk_inds_all = topk_inds_all % (height * width) # todo: this seems redudant
    topk_ys = ops.cast(topk_inds_all / width,ms.float32)
    topk_xs = ops.cast(topk_inds_all % width,ms.float32)

    # assert isinstance(topk_xs, ops.cuda.FloatTensor)
    # assert isinstance(topk_ys, ops.cuda.FloatTensor)

    # Select topK examples across channel (classes)
    # [N, C, K] -----> [N, C*K]
    topk_scores_all = topk_scores_all.view(batch, -1)
    # Both in [N, K]
    topk_scores, topk_inds = topk(topk_scores_all, K)[0]
    topk_clses = ops.cast(topk_inds / K,ms.float32)

    # assert isinstance(topk_clses, ops.cuda.FloatTensor)

    # First expand it as 3 dimension
    topk_inds_all = _gather_feat(topk_inds_all.view(batch, -1, 1), topk_inds).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_inds).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_inds).view(batch, K)

    return topk_scores, topk_inds_all, topk_clses, topk_ys, topk_xs


def _gather_feat(feat, ind):
    '''
    Select specific indexs on feature map
    Args:
        feat: all results in 3 dimensions
        ind: positive index

    Returns:

    '''
    channel = feat.shape[-1]
    size=ms.Tensor(np.array([ind.shape[0], ind.shape[1], channel]),ms.int32)
    ind=ops.expand_dims(ind,-1).expand(size)
    feat = feat.gather_elements(1, ind)

    return feat


def get_iou3d(pred_bboxes, target_bboxes):
    num_query = target_bboxes.shape[0]

    # compute overlap along y axis
    min_h_a = - (pred_bboxes[:, 1] + pred_bboxes[:, 4] / 2)
    max_h_a = - (pred_bboxes[:, 1] - pred_bboxes[:, 4] / 2)
    min_h_b = - (target_bboxes[:, 1] + target_bboxes[:, 4] / 2)
    max_h_b = - (target_bboxes[:, 1] - target_bboxes[:, 4] / 2)

    # overlap in height
    h_max_of_min = ops.max(min_h_a, min_h_b)
    h_min_of_max = ops.min(max_h_a, max_h_b)
    h_overlap = (h_min_of_max - h_max_of_min).clamp_(min=0)

    # volumes of bboxes
    pred_volumes = pred_bboxes[:, 3] * pred_bboxes[:, 4] * pred_bboxes[:, 5]
    target_volumes = target_bboxes[:, 3] * target_bboxes[:, 4] * target_bboxes[:, 5]

    # derive x y l w alpha
    pred_bboxes = pred_bboxes[:, [0, 2, 3, 5, 6]]
    target_bboxes = target_bboxes[:, [0, 2, 3, 5, 6]]

    # convert bboxes to corners
    pred_corners = get_corners(pred_bboxes)
    target_corners = get_corners(target_bboxes)
    iou_3d = pred_bboxes.new_zeros(num_query)

    for i in range(num_query):
        ref_polygon = Polygon(pred_corners[i])
        target_polygon = Polygon(target_corners[i])
        overlap = ref_polygon.intersection(target_polygon).area
        # multiply bottom overlap and height overlap
        # for 3D IoU
        overlap3d = overlap * h_overlap[i]
        union3d = ref_polygon.area * (max_h_a[0] - min_h_a[0]) + target_polygon.area * (max_h_b[i] - min_h_b[i]) - overlap3d
        iou_3d[i] = overlap3d / union3d

    return iou_3d


def get_corners(bboxes):
    # bboxes: x, y, w, l, alpha; N x 5
    corners = ops.zeros((bboxes.shape[0], 4, 2), dtype=ms.float32)
    x, y, w, l = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    # compute cos and sin
    cos_alpha = ops.cos(bboxes[:, -1])
    sin_alpha = ops.sin(bboxes[:, -1])
    # front left
    corners[:, 0, 0] = x - w / 2 * cos_alpha - l / 2 * sin_alpha
    corners[:, 0, 1] = y - w / 2 * sin_alpha + l / 2 * cos_alpha

    # rear left
    corners[:, 1, 0] = x - w / 2 * cos_alpha + l / 2 * sin_alpha
    corners[:, 1, 1] = y - w / 2 * sin_alpha - l / 2 * cos_alpha

    # rear right
    corners[:, 2, 0] = x + w / 2 * cos_alpha + l / 2 * sin_alpha
    corners[:, 2, 1] = y + w / 2 * sin_alpha - l / 2 * cos_alpha

    # front right
    corners[:, 3, 0] = x + w / 2 * cos_alpha - l / 2 * sin_alpha
    corners[:, 3, 1] = y + w / 2 * sin_alpha + l / 2 * cos_alpha

    return corners


def uncertainty_guided_prune(separate_depths, GRM_uncern, cfg, depth_range=None, initial_use_uncern=True):
    '''
    Description:
        Prune the unresonable depth prediction results calculated by SoftGRM based on uncertainty.
    Input:
        separate_depths: The depths solved by SoftGRM. shape: (val_objs, 20).
        GRM_uncern: The estimated variance of SoftGRM equations. shape: (val_objs, 20).
        cfg: The config object.
        depth_range: The range of reasonable depth. Format: (depth_min, depth_max)
    Output:
        pred_depths: The solved depths. shape: (val_objs,)
        pred_uncerns: The solved uncertainty. shape: (val_objs,)
    '''
    objs_depth_list = []
    objs_uncern_list = []
    sigma_param = cfg.TEST.UNCERTAINTY_GUIDED_PARAM
    for obj_id in range(separate_depths.shape[0]):
        obj_depths = separate_depths[obj_id]
        obj_uncerns = GRM_uncern[obj_id]
        # Filter the depth estimations out of possible range.
        if depth_range != None:
            valid_depth_mask = ops.logical_and((obj_depths > depth_range[0]) , (obj_depths < depth_range[1]))
            obj_depths = ops.masked_select(obj_depths,valid_depth_mask)
            obj_uncerns = ops.masked_select(obj_uncerns,valid_depth_mask)
        # If all objects are filtered.
        if obj_depths.shape[0] == 0:
            objs_depth_list.append(ops.expand_dims(separate_depths[obj_id].mean(),0))
            objs_uncern_list.append(ops.expand_dims(GRM_uncern[obj_id].mean, 0))
            continue

        if initial_use_uncern:
            considered_index = obj_uncerns.argmin()
        else:
            considered_index = find_crowd_index(obj_depths)

        considered_mask = ops.zeros(obj_depths.shape, dtype=ms.bool_)
        considered_mask[considered_index] = True
        obj_depth_mean = obj_depths[considered_index]
        obj_depth_sigma = ops.sqrt(obj_uncerns[considered_index])
        flag = True
        search_cnt = 0
        while flag == True:
            search_cnt += 1
            flag = False
            new_considered_mask = (obj_depths > obj_depth_mean - sigma_param * obj_depth_sigma) & (
                        obj_depths < obj_depth_mean + sigma_param * obj_depth_sigma)
            if considered_mask.equal(new_considered_mask).sum()<considered_mask.shape[0] or search_cnt > 20:  # No new elements are considered.
                objs_depth_list.append(ops.expand_dims(obj_depth_mean,0))
                objs_uncern_list.append(ops.expand_dims((obj_depth_sigma * obj_depth_sigma),0))
                break
            else:
                considered_mask = new_considered_mask
                considered_depth = obj_depths[considered_mask]
                considered_uncern = obj_uncerns[considered_mask]
                considered_w = 1 / considered_uncern
                considered_w = considered_w / considered_w.sum()
                obj_depth_mean = (considered_w * considered_depth).sum()
                obj_depth_sigma = ops.sqrt((considered_w * considered_uncern).sum())
                flag = True

    pred_depths = ops.concat(objs_depth_list, axis=0)
    pred_uncerns = ops.concat(objs_uncern_list, axis=0)
    return pred_depths, pred_uncerns


def find_crowd_index(obj_depths):
    '''
    Description:
        Find the depth at the most crowded index for each objects.
    Input:
        obj_depths: The estimated depths of an object. shape: (num_depth,).
    Output:
        crowd_index: Int.
    '''
    num_depth = obj_depths.shape[0]

    depth_matrix = ops.expand_dims(obj_depths,0).expand(num_depth, num_depth)
    cost_matrix = (ops.expand_dims(obj_depths,1) - depth_matrix).abs()    # cost_matrix shape: (num_depth, num_depth)
    crowd_index = cost_matrix.sum(axis = 1).argmin()
    return crowd_index


def error_from_uncertainty(uncern):
    '''
    Description:
        Get the error derived from uncertainty.
    Input:
        uncern: uncertainty tensor. shape: (val_objs, 20)
    Output:
        error: The produced error. shape: (val_objs,)
    '''
    if uncern.ndim != 2:
        raise Exception("uncern must be a 2-dim tensor.")
    weights = 1 / uncern	# weights shape: (total_num_objs, 20)
    weights = weights / ops.ReduceSum(True)(weights, 1)
    error = ops.ReduceSum()(weights * uncern, 1)	# error shape: (valid_objs,)
    return error


def nms_3d(results, bboxes, scores, iou_threshold = 0.2):
    '''
    Description:
        Given the 3D bounding boxes of objects and confidence scores, remove the overlapped ones with low confidence.
    Input:
        results: The result tensor for KITTI. shape: (N, 14)
        bboxes: Vertex coordinates of 3D bounding boxes. shape: (N, 8, 3)
        scores: Confidence scores. shape: (N).
        iou_threshold: The IOU threshold for filering overlapped objects. Type: float.
    Output:
        preserved_results: results after NMS.
    '''
    descend_index = ops.flip(ops.Sort(axis=0)(scores), dims = (0,))
    results = results[descend_index, :]
    sorted_bboxes = bboxes[descend_index, :, :]

    box_indices = np.arange(0, sorted_bboxes.shape[0])
    suppressed_box_indices = []
    tmp_suppress = []

    while len(box_indices) > 0:

        if box_indices[0] not in suppressed_box_indices:
            selected_box = box_indices[0]
            tmp_suppress = []

            for i in range(len(box_indices)):
                if box_indices[i] != selected_box:
                    selected_iou = get_iou_3d(ops.expand_dims(sorted_bboxes[int(selected_box)],0), ops.expand_dims(sorted_bboxes[int(box_indices[i])],0))[0]
                    if selected_iou > iou_threshold:
                        suppressed_box_indices.append(box_indices[i])
                        tmp_suppress.append(i)

        box_indices = np.delete(box_indices, tmp_suppress, axis=0)
        box_indices = box_indices[1:]

    preserved_index = np.setdiff1d(np.arange(0, sorted_bboxes.shape[0]), np.array(suppressed_box_indices), assume_unique=True)
    preserved_results = results[preserved_index.tolist(), :]

    return preserved_results

