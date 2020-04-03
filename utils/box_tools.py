import torch
import math
import numpy as np
import torch.nn.functional as F
import torchvision
from config import cfg


def box_iou(box_a, box_b, eps=1e-5,is_tensor=True):
    # 计算 N个box与M个box的iou需要使用到torch与numpy的广播特性
    if is_tensor:
        # lt为交叉部分左上角坐标最大值, lt.shape -> (N,M,2), br为交叉部分右下角坐标最小值
        lt = torch.max(box_a[..., :2], box_b[..., :2])
        rb = torch.min(box_a[..., 2:], box_b[..., 2:])
        # 第一个axis是指定某一个box内宽高进行相乘,第二个axis是筛除那些没有交叉部分的box
        # 这个 < 和 all(axis=2) 是为了保证右下角的xy坐标必须大于左上角的xy坐标,否则最终没有重合部分的box公共面积为0
        area_overlap = torch.prod(rb - lt, dim=2) * (lt < rb).all(dim=2)
        # 分别计算bbox_a,bbox_b的面积,以及最后的iou
        area_a = torch.prod(box_a[..., 2:] - box_a[..., :2], dim=2)
        area_b = torch.prod(box_b[..., 2:] - box_b[..., :2], dim=2)
        iou = area_overlap / (area_a + area_b - area_overlap + eps)
    else:
        tl = np.maximum(box_a[..., :2], box_b[..., :2])
        br = np.minimum(box_a[..., 2:], box_b[..., 2:])
        area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
        area_a = np.prod(box_a[..., 2:] - box_a[..., :2], axis=2)
        area_b = np.prod(box_b[..., 2:] - box_b[..., :2], axis=2)
        iou = area_i / (area_a + area_b - area_i + eps)
    return iou


def loc2box(loc, box):
    # loc : dx,dy,dw,dh
    # box : x,y,w,h
    return torch.cat([
        loc[..., :2] * cfg.center_variance * box[..., 2:] + box[..., :2],
        torch.exp(loc[..., 2:] * cfg.size_variance) * box[..., 2:]
    ], dim=-1)


def box2loc(anchor_targets, anchors_xywh):
    # anchor_targets: x,y,w,h
    # anchors_xywh  : x,y,w,h
    return torch.cat([
        (anchor_targets[..., :2] - anchors_xywh[..., :2]) / anchors_xywh[..., 2:] / cfg.center_variance,
        torch.log(anchor_targets[..., 2:] / anchors_xywh[..., 2:]) / cfg.size_variance
    ], dim=-1)


def assign_anchors(target_boxes, target_labels, anchors_xyxy):
    """
    该方法的意义在于把所有的anchor都赋予相对应的拟合target_box与label(1正样本,0负样本,-1忽略)
    target_boxes:       [target_nums,4]         真实框的坐标
    target_labels:      [target_nums]           真实框标签
    anchors_xyxy:       [anchor_nums,4]         基础anchors的坐标
    anchor_targets:     [anchor_nums,4]         基础anchors要拟合的target_boxes
    anchor_labels:     [anchor_nums]            基础anchors要拟合的target_labels
    """
    # 顺便说一句,这里box_iou中前两个参数,谁在前谁在后是没有关系的.只要最终利用广播机制扩维到 [anchor_nums,target_nums,4]即可
    #               [1,target_nums,4]      [anchor_nums,1,4]
    ious = box_iou(target_boxes[None, :], anchors_xyxy[:, None], is_tensor=True)
    # 每个anchor与所有target的最大iou,最大iou对应的target索引
    anchor_maxious, anchor_argmaxious = ious.max(1)
    # 每个target与所有anchor的最大iou,最大iou对应的anchor索引
    target_maxious, target_argmaxious = ious.max(0)
    # anchor与target的匹配策略 下面这个for循环就是为了解决冲突问题而存在的
    # 1.每个anchor只能匹配一个target,但每个target可以匹配多个anchor
    # 2.anchor_maxious > iou_threshold的anchor就可以被视为正样本.该anchor的label也匹配为target的label
    # 3.和target IOU最高的那个anchor也被视为正样本,该anchor的label也匹配为target的label
    # 4.如果某个anchor在2与3中label匹配出现冲突的话,则该anchor的label以第3步的为准(主要是为了确保每个target都有一个anchor与其匹配)
    for target_index, anchor_index in enumerate(target_argmaxious):
        anchor_argmaxious[anchor_index] = target_index
    # 填充2是为了确保每个和target有最大IOU的anchor的IOU即使小于iou_threshold也算作正样本
    anchor_maxious.index_fill_(0, target_argmaxious, 2)
    anchor_labels = target_labels[anchor_argmaxious]
    # IOU值小于iou_threshold的被视为背景(和target有最大IOU的anchor除外)
    anchor_labels[anchor_maxious < 0.4] = 0
    # 原始论文中 如果anchor与target的IOU大于0.4小于0.5会被忽略,即最终不参与分类的loss计算.但是个人觉得这样会增加一些 hard examples
    anchor_labels[(anchor_maxious > 0.4) & (0.5 > anchor_maxious)] = -1
    anchor_targets = target_boxes[anchor_argmaxious]
    return anchor_targets, anchor_labels


# [x, y, w, h] -> [x_min, y_min, x_max, y_max]
def wh2xy(locations):
    return torch.cat([locations[..., :2] - locations[..., 2:] / 2, locations[..., :2] + locations[..., 2:] / 2], dim=-1)


# [x_min, y_min, x_max, y_max] -> [x, y, w, h]
def xy2wh(boxes):
    return torch.cat([ (boxes[..., :2] + boxes[..., 2:]) / 2, boxes[..., 2:] - boxes[..., :2]], dim=-1)


def create_anchors():
    # fpn输出的特征图大小(这里的值时匹配600*600的),注意如果网络输入不等于600*600,则features_maps也要相应修改
    # features_maps = [(75, 75), (38, 38), (19, 19), (10, 10), (5, 5)]
    # 至于为什么是从75*75的特征图开始计算,是因为一般小物体的长宽基本都在10px以上 75*75个特征点,每个特征点代表了8*8的像素区域
    # 这里写了一个根据网络输入尺寸而适应的features_maps
    features_width = cfg.height
    features_maps = []
    for i in range(7):
        features_width = math.ceil(features_width / 2)
        if i < 2:
            continue
        features_maps.append((features_width, features_width))
    # 这里是每个特征图对应的基础anchor长度
    anchor_sizes = [32, 64, 128, 256, 512]
    # ratios = np.array([0.5, 1, 2])  # 每个特征点上的三种长宽比 虽然没有使用上这个变量 但是最里面的两个for循环中有体现到
    scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])  # 每个特征点上的三种面积尺寸 约等于~ 1 1.25 1.6
    anchors = []
    for k, (feature_map_w, feature_map_h) in enumerate(features_maps):
        for i in range(feature_map_w):
            for j in range(feature_map_h):
                # 一行一行的创建中心坐标
                cx = (j + 0.5) / feature_map_w
                cy = (i + 0.5) / feature_map_h

                size = anchor_sizes[k] / cfg.height  # 将框体长宽转为 相对形式

                sides_square = scales * size  # 计算正方形检测框边长
                for side_square in sides_square:
                    anchors.append([cx, cy, side_square, side_square])  # 添加正方形检测框

                sides_long = sides_square * 2 ** (1 / 2)  # 计算长方形检测框长边
                for side_long in sides_long:
                    anchors.append([cx, cy, side_long, side_long / 2])  # 添加长方形检测框,宽为高两倍
                    anchors.append([cx, cy, side_long / 2, side_long])  # 添加长方形检测框,高为宽两倍

    anchors = torch.tensor(anchors)
    # 对超出图像范围的anchor进行截断,截断时先转为 [x_min, y_min, x_min, x_max]形式.然后再转回 [x, y, w, h]形式返回
    anchors = wh2xy(anchors)
    anchors.clamp_(max=1, min=0)
    anchors = xy2wh(anchors)
    return anchors


def NMS(pred_cls, pred_locs):
    """
    :param pred_cls:  torch.Size([batch_size, 67995, 19])
    :param pred_locs: torch.Size([batch_size, 67995, 4])
    :return:
    """
    anchors = create_anchors().cuda()
    batches_scores = F.softmax(pred_cls, dim=2)
    boxes = loc2box(pred_locs, anchors)
    batches_boxes = wh2xy(boxes)

    batch_size = batches_scores.size(0)
    results = []
    for batch_id in range(batch_size):
        processed_boxes = []
        processed_scores = []
        processed_labels = []

        per_img_scores, per_img_boxes = batches_scores[batch_id], batches_boxes[batch_id]  # (N, #CLS) (N, 4)
        for class_id in range(1, per_img_scores.size(1)):  # 跳过背景类
            scores = per_img_scores[:, class_id]
            mask = scores > cfg.score_threshold
            scores = scores[mask]
            # 如果某一类没有pred_box则跳过
            if scores.size(0) == 0:
                continue
            boxes = per_img_boxes[mask, :]

            keep = torchvision.ops.nms(boxes, scores, cfg.iou_nms)

            nmsed_boxes = boxes[keep, :]
            nmsed_labels = torch.tensor([class_id] * keep.size(0)).cuda()
            nmsed_scores = scores[keep]

            processed_boxes.append(nmsed_boxes)
            processed_scores.append(nmsed_scores)
            processed_labels.append(nmsed_labels)

        if len(processed_boxes) == 0:
            processed_boxes = torch.empty(0, 4)
            processed_labels = torch.empty(0)
            processed_scores = torch.empty(0)
        else:
            processed_boxes = torch.cat(processed_boxes, 0)
            processed_labels = torch.cat(processed_labels, 0)
            processed_scores = torch.cat(processed_scores, 0)
        # 如果最终满足条件的box超过100个,则只取score排名前100的box,这一步基本可以省略,只有在训练初期才会出现box超过100个的情况
        # if processed_boxes.size(0) > 100 > 0:
        #     processed_scores, keep = torch.topk(processed_scores, k=100)
        #     processed_boxes = processed_boxes[keep, :]
        #     processed_labels = processed_labels[keep]
        results.append([processed_boxes, processed_labels, processed_scores])
    return results