import torch.nn as nn
import torch
import torch.nn.functional as F
from config import cfg


class multiboxloss(nn.Module):
    def __init__(self):
        super(multiboxloss, self).__init__()
        self.neg_pos_ratio = cfg.neg_pos_ratio
        self.focal_loss = focal_loss()

    def forward(self, pred_scores, pred_locs, target_labels, target_locs):
        """
        计算cls损失和loc损失
        Args:
            pred_scores (batch_size, num_anchor, num_class): 预测框的类别
            pred_locs (batch_size, num_anchor, 4):           预测框的修正系数
            target_labels (batch_size, num_anchor):          目标框的类别
            target_locs (batch_size, num_anchor, 4):         目标框的修正系数
        """
        with torch.no_grad():
            # 注 原始RetinaNet中是没有hard_negative_mining的
            loss = -F.log_softmax(pred_scores, dim=2)[:, :, 0]
            mask = hard_negative_mining(loss, target_labels, self.neg_pos_ratio)
        # cls损失
        cls_loss = self.focal_loss(pred_scores[mask], target_labels[mask])

        pos_mask = target_labels > 0
        pred_locs = pred_locs[pos_mask, :].view(-1, 4)
        target_locs = target_locs[pos_mask, :].view(-1, 4)
        # loc损失
        loc_loss = F.smooth_l1_loss(pred_locs, target_locs, reduction='sum')
        num_pos = target_locs.size(0)
        return loc_loss / num_pos, cls_loss / (num_pos*4)


class focal_loss(nn.Module):
    def __init__(self, alpha=cfg.alpha, gamma=cfg.gamma, num_classes=cfg.num_class):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi) 可以单独拎出来用,替代cross_entropy
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,最终类别权重为[α, 1-α, 1-α, ....]
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        """
        super(focal_loss, self).__init__()
        if isinstance(alpha, list):
            assert len(alpha) == num_classes  # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            print("Focal_loss alpha = {}, 将对每一类权重进行精细化赋值".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
        self.gamma = gamma
        # print(" --- Multiboxloss : α={} γ={} num_classes={}".format(self.alpha, self.gamma, num_classes))

    def forward(self, pred_scores, target_labels):
        """
        focal_loss损失计算  这里的pred_scores是指经过hard_negative_mining的所有正样本,
        以及前num_neg(3倍正样本)个最大背景loss所在anchor预测的类别置信度,其中pos_num为一个batch中正样本总数
        :param pred_scores:    预测类别. size:[4*pos_num,num_class]
        :param target_labels:  实际类别. size:[4*pos_num]
        :return:
        """
        self.alpha = self.alpha.cuda()
        preds_softmax = F.softmax(pred_scores, dim=1)  # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果
        preds_logsoft = torch.log(preds_softmax)
        preds_softmax = preds_softmax.gather(1,target_labels.view(-1, 1))
        # 这一步等于 F.cross_entropy(pred_scores,target_labels) 只不过还没有算mean或者sum
        preds_logsoft = preds_logsoft.gather(1, target_labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, target_labels)
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)
        loss = torch.mul(self.alpha, loss.t()).sum()
        return loss


def hard_negative_mining(loss, labels, neg_pos_ratio=3):
    """
    loss其实是为所有batch_size中所有anchors的背景类损失
    先计算出所有正样本所在位置及其数量,并将正样本所在位置的背景损失设为负值,以及计算出负样本数量
    然后获取所有anchor的前num_neg个最大背景loss所在的anchor位置并与正样本所在anchor位置合并起来返回
    Args:
        loss (batch_size, num_anchors):   一个batch中所有anchor的背景类损失
        labels (batch_size, num_anchors): 已经赋予值的anchor的label
        neg_pos_ratio: 负例数量/正例数量
    """
    # 统计所有张图片中正样本所在位置和每张图片正样本有多少,以及计算出负样本的数量
    pos_mask = labels > 0
    ignore_mask = labels < 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_neg = num_pos * neg_pos_ratio
    # 这一步目的在于除了将正样本所在的anchor的loss设置为-1,还有一个隐藏信息表示
    # 如果其他任意anchor(非正样本)的loss越大说明,这个anchor被预测为背景的概率越小同时也表明被预测为某一其他类的概率越大,即FP
    # 而hard_negative_mining方法就是为了解决此问题而存在的,即尽可能多的降低FP的存在
    loss[pos_mask] = -1     # 因为某个anchor的与target的loss最小就是为0,所以-1可以排到最后
    loss[ignore_mask] = -2  # 对那些IOU在[0.4,0.5]区间的anchor忽略计算cls损失,-2确保可以在最后
    # 这里连续应用两次sort找出元素在降序之后的位置,可能比较难理解
    # 建议参考 https://blog.csdn.net/LXX516/article/details/78804884 对着图像化的数据来理解
    _, indexes = loss.sort(dim=1, descending=True)  # descending 降序 ,返回 value,index
    _, orders = indexes.sort(dim=1)
    # 获取那些背景类损失最大的前 num_neg个的位置mask,正样本及忽略样本除外
    neg_mask = orders < num_neg
    # 这里返回的mask中只有了正样本以及前num_neg(3倍正样本)最大背景loss所在anchor的位置才为True 即目标or背景
    return pos_mask | neg_mask