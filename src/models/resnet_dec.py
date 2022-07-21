from torchvision.models import resnet18
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class ResizeConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 scale_factor,
                 mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride=1,
                              padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class BasicBlockDec(nn.Module):
    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes / stride)

        self.conv2 = nn.Conv2d(in_planes,
                               in_planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes,
                                   planes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes,
                                      planes,
                                      kernel_size=3,
                                      scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes,
                             planes,
                             kernel_size=3,
                             scale_factor=stride), nn.BatchNorm2d(planes))

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet18Dec(nn.Module):
    def __init__(self, num_Blocks=[2, 2, 2, 2], z_dim=512, nc=3):
        super().__init__()
        self.in_planes = 512

        self.linear = nn.Linear(z_dim, 512)

        self.layer4 = self._make_layer(BasicBlockDec,
                                       256,
                                       num_Blocks[3],
                                       stride=2)
        self.layer3 = self._make_layer(BasicBlockDec,
                                       128,
                                       num_Blocks[2],
                                       stride=2)
        self.layer2 = self._make_layer(BasicBlockDec,
                                       64,
                                       num_Blocks[1],
                                       stride=2)
        self.layer1 = self._make_layer(BasicBlockDec,
                                       64,
                                       num_Blocks[0],
                                       stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        # x = self.linear(z)
        # x = self.avgpool(z)
        # x = x.view(x.size(0), 512, 1, 1)
        # x = F.interpolate(x, scale_factor=7)
        # x = self.layer4(z)
        x = self.layer3(z)
        x = self.layer2(x)
        x = self.layer1(x)
        x = F.interpolate(x, size=(112, 112), mode='bilinear')
        x = self.conv1(x)
        x = x.view(x.size(0), 3, 224, 224)
        # x = F.sigmoid(x)
        return x
