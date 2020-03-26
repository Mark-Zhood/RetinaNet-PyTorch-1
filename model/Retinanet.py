from .Resnet import build_resnet
from .Fpn import FPN
from .Predictor import predictor
from torch import nn


class RetinaNet(nn.Module):
    """
    :x 为批量输入的图片数据 -> torch.Size([B, 3, 600, 600])
    :return pred_score, torch.Size([B, 67995, num_classes]) RetinaNe网络预测所有anchor的修正系数
            pred_loc,  torch.Size([B, 67995, 4])            RetinaNe网络预测所有anchor的类别概率
    # 模型流程部分可以参考 https://lilianweng.github.io/lil-log/2018/12/27/object-detection-part-4.html 其中的RetinaNet部分
    """
    def __init__(self):
        super(RetinaNet,self).__init__()
        self.resnet = 'resnet50'
        self.num_classes = 19
        self.num_anchors = 9
        expansion_list={
            'resnet18': 1,
            'resnet34': 1,
            'resnet50': 4,
            'resnet101': 4,
            'resnet152': 4,
        }
        assert self.resnet in expansion_list
        # 初始化Resnet、FPN以、定位与分类卷积层
        self.backbone = build_resnet(self.resnet, pretrained=True)
        expansion = expansion_list[self.resnet]
        self.fpn = FPN(channels_of_fetures=[128*expansion, 256*expansion, 512*expansion])
        self.predictor = predictor()

    def forward(self, x):
        c3, c4, c5, p6, p7 = self.backbone(x)   # ResNet输出的五层特征图
        p3, p4, p5 = self.fpn([c3, c4, c5])     # 前三层特征图进FPN
        features = [p3, p4, p5, p6, p7]
        pred_score, pred_loc = self.predictor(features)
        return pred_score, pred_loc

