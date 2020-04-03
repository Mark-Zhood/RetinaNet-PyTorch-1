import torch.nn as nn
from torch.nn import functional as F
import wget
import os
from config import cfg
import torch


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        # ResNet18及ResNet34中的Block是由 两个3*3的conv组成
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_c, out_c, stride=1, downsample=None):
        # ResNet50及ResNet101、ResNet152中的block是由 1*1的conv + 3*3的conv + 1*1的conv组成
        super(Bottleneck, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv1x1(in_c, out_c)
        self.bn1 = norm_layer(out_c)
        self.conv2 = conv3x3(out_c, out_c, stride)
        self.bn2 = norm_layer(out_c)
        self.conv3 = conv1x1(out_c, out_c * self.expansion)
        self.bn3 = norm_layer(out_c * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    # ResNet中主要靠一个conv一个maxpool+4个Block + average_pool 以及全连接层+softmax组成
    # 不同层数之间的ResNet主要是中间的4个Block中所包含的block类型以及数量不同,首尾都是一致的
    def __init__(self, res_name, zero_init_residual=False,):
        super(ResNet, self).__init__()
        # 不同层数的ResNet中Block里面的block种类以及对应的Block重复次数
        resnets = {
            'resnet18': [BasicBlock, [2, 2, 2, 2]],
            'resnet34': [BasicBlock, [3, 4, 6, 3]],
            'resnet50': [Bottleneck, [3, 4, 6, 3]],
            'resnet101': [Bottleneck, [3, 4, 23, 3]],
            'resnet152': [Bottleneck, [3, 8, 36, 3]],
        }
        # 获取res_name类型下的 block类型和各层重复的次数
        block = resnets[res_name][0]
        layers = resnets[res_name][1]

        self.res_name = res_name

        # 这个in_c代表的是每个Block中第一个conv的输入维度
        self.in_c = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 不管是多少层的ResNet其中每个Block的输入维度都是相同的 并且有规律翻倍 64 128 256 512
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # extra 额外层,用于在c5基础上输出p6,p7,这里开始就不属于ResNet了
        self.conv6 = nn.Conv2d(512*block.expansion, 256, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    # ResNet中Block的构建
    def _make_layer(self, block, out_c, num_block, stride=1):
        downsample = None
        # 这里面其实除了第一个Block的stride为1,其他都为2. or 后面的条件是为了确保Block内开始和结束的通道数不一致的情况下才会触发
        # 因为不管ResNet多少层,其中的Block的开始和结束通道都为 in_c 和 out_c*expansion 只不过18和34的expansion为1,其他的为4而已
        # 所以ResNet18、34 第一个Block都是没有downsample的,对于这两个网络来说stride != 1 是完全可以应付的
        # 而其他的ResNet网络则需要第二个条件判断,因为它们在第一个Block有维度的变化(*4)
        if stride != 1 or self.in_c != out_c * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_c, out_c * block.expansion, stride),
                nn.BatchNorm2d(out_c * block.expansion),
            )

        layers = []
        layers.append(block(self.in_c, out_c, stride, downsample))
        # 将每个Block的末尾conv输出通道数传给下一个Block的起始conv的输入维度
        self.in_c = out_c * block.expansion
        for _ in range(1, num_block):
            layers.append(block(self.in_c, out_c,))

        return nn.Sequential(*layers)

    def forward(self, x):
        c1 = self.conv1(x)
        c1 = self.bn1(c1)
        c1 = self.relu(c1)
        c1 = self.maxpool(c1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))
        return c3, c4, c5, p6, p7

    def load_weights(self):
        model_urls = {
            'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
            'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
            'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
            'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
            'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
        }

        url = model_urls[self.res_name]
        weight_name = url.split('/')[-1]
        weight_path = cfg.resnet_path

        if not os.path.exists(weight_path):

            print(' {} 不存在,下载中.....'.format(weight_name))
            wget.download(url=url, out=weight_path)

            print(' --- 权重文件已下载到 {} --- '.format(weight_path))
        self.load_state_dict(torch.load(weight_path), strict=False)
        print(' --- {} 权重加载成功 --- '.format(weight_name))


def build_resnet(res_name, pretrained=True):
    model = ResNet(res_name)
    if pretrained:
        model.load_weights()
    return model


if __name__ == '__main__':
    import torch
    net = build_resnet('resnet50',pretrained=False)
    print(net)
    exit()
    c3,c4,c5,p6,p7=net(torch.ones((1,3,600,600)))
    print(c3.size())
    print(c4.size())
    print(c5.size())
    print(p6.size())
    print(p7.size())
