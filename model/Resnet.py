import torch.nn as nn
from torch.nn import functional as F
import wget
import os
from config import cfg
import torch


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
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

    def __init__(self, arch, zero_init_residual=False,):
        super(ResNet, self).__init__()
        resnets = {
            'resnet18': [BasicBlock, [2, 2, 2, 2]],
            'resnet34': [BasicBlock, [3, 4, 6, 3]],
            'resnet50': [Bottleneck, [3, 4, 6, 3]],
            'resnet101': [Bottleneck, [3, 4, 23, 3]],
            'resnet152': [Bottleneck, [3, 8, 36, 3]],
        }
        block = resnets[arch][0]
        layers = resnets[arch][1]

        self.arch = arch

        self.dilation = 1
        self.base_width = 64
        self.in_c = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # extra 额外层,用于在c5基础上输出p6,p7
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

    def _make_layer(self, block, out_c, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_c != out_c * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_c, out_c * block.expansion, stride),
                nn.BatchNorm2d(out_c * block.expansion),
            )

        layers = []
        layers.append(block(self.in_c, out_c, stride, downsample))
        self.in_c = out_c * block.expansion
        for _ in range(1, blocks):
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

        url = model_urls[self.arch]
        weight_name = url.split('/')[-1]
        weight_path = cfg.res50_path

        if not os.path.exists(weight_path):

            print(' {} 不存在,下载中.....'.format(weight_name))
            wget.download(url=url, out=weight_path)

            print(' --- 权重文件已下载到 {} --- '.format(weight_path))
        self.load_state_dict(torch.load(weight_path), strict=False)
        print(' --- {} 权重加载成功 --- '.format(weight_name))


def build_resnet(arch, pretrained=True):
    assert arch in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    model = ResNet(arch)
    if pretrained:
        model.load_weights()
    return model


if __name__ == '__main__':
    import torch
    net = build_resnet('resnet50',pretrained=False)
    c3,c4,c5,p6,p7=net(torch.ones((1,3,600,600)))
    print(c3.size())
    print(c4.size())
    print(c5.size())
    print(p6.size())
    print(p7.size())
