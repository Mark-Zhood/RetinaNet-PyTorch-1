from torch import nn
from torch.nn import functional as F


class FPN(nn.Module):
    def __init__(self,channels_of_fetures, channel_out=256):
        """
        fpn,特征金字塔
        :param channels_of_fetures: list,输入层的通道数,必须与输入特征图相对应
        :param channel_out:
        """
        super(FPN,self).__init__()
        self.channels_of_fetures = channels_of_fetures
        # 以下三个卷积只是起到降维的作用
        # lateral_conv1 -> 2048,256   lateral_conv2 -> 1024,256   lateral_conv3  -> 512,256
        self.lateral_conv1 = nn.Conv2d(channels_of_fetures[0], channel_out, kernel_size=1, stride=1, padding=0)
        self.lateral_conv2 = nn.Conv2d(channels_of_fetures[1], channel_out, kernel_size=1, stride=1, padding=0)
        self.lateral_conv3 = nn.Conv2d(channels_of_fetures[2], channel_out, kernel_size=1, stride=1, padding=0)

        self.top_down_conv1 = nn.Conv2d(channel_out, channel_out, kernel_size=3, stride=1, padding=1)
        self.top_down_conv2 = nn.Conv2d(channel_out, channel_out, kernel_size=3, stride=1, padding=1)
        self.top_down_conv3 = nn.Conv2d(channel_out, channel_out, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        c3, c4, c5 = features

        # 先降维到统一的维度,然后方便下面的特征融合(只是数值上的相加)
        p3 = self.lateral_conv3(c3)  # [B, 512, 75, 75]
        p4 = self.lateral_conv2(c4)  # [B, 1024, 38, 38]
        p5 = self.lateral_conv1(c5)  # [B, 2048, 19, 19]

        # PyTorch的上下采样函数 详情参考: https://www.jianshu.com/p/dc0d44911c6c
        p4 = F.interpolate(input=p5, size=(p4.size(2),p4.size(3)), mode="nearest") + p4
        p3 = F.interpolate(input=p4, size=(p3.size(2),p3.size(3)), mode="nearest") + p3

        # 3*3,stride=1, padding=1的卷积,对周围语义的整合
        p3 = self.top_down_conv1(p3)
        p4 = self.top_down_conv1(p4)
        p5 = self.top_down_conv1(p5)

        return p3, p4, p5
