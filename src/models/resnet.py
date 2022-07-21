from torchvision.models import resnet18, resnet50
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from .fourier import *
"""
ResNet adpated from Matsuura et al. 2020
(Code) https://github.com/mil-tokyo/dg_mmld/
(Paper) https://arxiv.org/pdf/1911.07661.pdf 
"""


def base_resnet(num_classes, model):
    if model == 'resnet18':
        model = resnet18(pretrained=True)
    elif model == 'resnet50':
        model = resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    nn.init.xavier_uniform_(model.fc.weight, .1)
    nn.init.constant_(model.fc.bias, 0.)
    return model


class Resnet(nn.Module):
    def __init__(self,
                 num_classes,
                 model='resnet18',
                 se_block=False,
                 sp_block=False,
                 r=16,
                 c=128):

        super(Resnet, self).__init__()
        base_model = base_resnet(num_classes, model)
        self.base_model = base_model

        self.conv_features = torch.nn.Sequential(
            base_model.conv1, base_model.bn1, base_model.relu,
            base_model.maxpool, base_model.layer1, base_model.layer2,
            base_model.layer3, base_model.layer4, base_model.avgpool,
            torch.nn.Flatten(start_dim=1))
        # No dense features really
        self.dense_features = torch.nn.Identity()

        self.features = torch.nn.Sequential(self.conv_features,
                                            self.dense_features)
        self.flatten = torch.nn.Flatten(start_dim=1)
        self.batch_norm = torch.nn.BatchNorm2d(128)

        self.sp_block = sp_block
        self.se_block = se_block
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        if se_block:
            "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
            self.amp_conv_1 = nn.Conv2d(c * 2,
                                        c,
                                        kernel_size=7,
                                        stride=1,
                                        padding=3,
                                        bias=False)
            self.amp_conv_2 = nn.Conv2d(c,
                                        c,
                                        kernel_size=7,
                                        stride=1,
                                        padding=3,
                                        bias=False)
            self.squeeze = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            self.orig_amp_fc = nn.Linear(c, (c) // r, bias=False)
            self.amp_mean_fc = nn.Linear(c, (c) // r, bias=False)
            self.shared_amp_fc = nn.Linear(((c) // r), c * 2, bias=False)

        if sp_block:
            kernel_size = 7
            self.orig_amp_conv = nn.Conv2d(2,
                                           2,
                                           kernel_size=kernel_size,
                                           padding=kernel_size // 2,
                                           bias=False)
            self.amp_mean_conv = nn.Conv2d(2,
                                           2,
                                           kernel_size=kernel_size,
                                           padding=kernel_size // 2,
                                           bias=False)
            self.shared_amp_conv = nn.Conv2d(2,
                                             1,
                                             kernel_size=kernel_size,
                                             padding=kernel_size // 2,
                                             bias=False)

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        #64
        x = self.base_model.layer1(x)  #64
        x = self.base_model.layer2(x)  #128
        x = self.base_model.layer3(x)  #256
        x = self.base_model.layer4(x)  #512

        x = self.base_model.avgpool(x)
        x = self.flatten(x)
        output_class = self.base_model.fc(x)
        return output_class

    def classifier(self, x):
        x = self.base_model.fc(x)
        return x

    def early_layer(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        #64
        return x

    def layer1(self, x):
        x = self.base_model.layer1(x)  #64
        return x

    def layer2(self, x):
        x = self.base_model.layer2(x)  #128
        return x

    def layer3(self, x):
        x = self.base_model.layer3(x)  #256
        return x

    def layer4(self, x):
        x = self.base_model.layer4(x)  #512
        return x

    def classifier_layer(self, x):
        x = self.base_model.avgpool(x)
        x = self.flatten(x)
        output_class = self.base_model.fc(x)
        return output_class

    def channel_attention_(self, orig_amp, amp_mean):
        bs, c, _, _ = orig_amp.shape
        oa = self.squeeze(orig_amp).view(bs, c)
        am = self.squeeze(amp_mean).view(bs, c)

        oa = self.relu(self.orig_amp_fc(oa))
        am = self.relu(self.amp_mean_fc(am))

        oa = self.shared_amp_fc(oa)
        am = self.shared_amp_fc(oa + am)

        oa = self.sigmoid(oa).view(bs, c, 1, 1)
        am = self.sigmoid(am).view(bs, c, 1, 1)

        orig_amp = orig_amp * oa.expand_as(orig_amp)
        amp_mean = amp_mean * am.expand_as(amp_mean)

        return orig_amp, amp_mean

    def channel_attention(self, orig_amp, amp_mean):
        oa_am = torch.cat([orig_amp, amp_mean], dim=1)
        oa_am = self.amp_conv_1(oa_am)
        oa_am = self.amp_conv_2(oa_am)
        return oa_am

    def spatial_attention(self, orig_amp, amp_mean):
        orig_amp_avg_out = torch.mean(orig_amp, dim=1, keepdim=True)
        amp_mean_avg_out = torch.mean(amp_mean, dim=1, keepdim=True)
        orig_amp_max_out, _ = torch.max(orig_amp, dim=1, keepdim=True)
        amp_mean_max_out, _ = torch.max(amp_mean, dim=1, keepdim=True)

        oa = torch.cat([orig_amp_avg_out, orig_amp_max_out], dim=1)
        am = torch.cat([amp_mean_avg_out, amp_mean_max_out], dim=1)

        oa = self.relu(self.orig_amp_conv(oa))
        am = self.relu(self.amp_mean_conv(am))

        oa = self.shared_amp_conv(oa)
        am = self.shared_amp_conv(am)

        oa = self.sigmoid(oa)
        am = self.sigmoid(am)

        orig_amp = orig_amp * oa
        amp_mean = amp_mean * am

        return orig_amp, amp_mean

    def attention_forward(self, orig_amp, amp_mean):

        if self.se_block:
            amp = self.channel_attention(orig_amp, amp_mean)

        if self.sp_block:
            orig_amp, amp_mean = self.spatial_attention(orig_amp, amp_mean)

        return amp
