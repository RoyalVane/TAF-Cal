from collections import OrderedDict

from torch import nn
import torch
import pathlib
import os
"""
CaffeNet adpated from Matsuura et al. 2020
(Code) https://github.com/mil-tokyo/dg_mmld/
(Paper) https://arxiv.org/pdf/1911.07661.pdf 
and Carlucci et al 2019
(Code) https://github.com/fmcarlucci/JigenDG
(Paper) https://arxiv.org/pdf/1903.06864.pdf
"""


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class AlexNetCaffe(nn.Module):
    def __init__(self, num_classes=100, dropout=True):
        super(AlexNetCaffe, self).__init__()
        self.features = nn.Sequential(
            OrderedDict([
                ("conversion", LambdaLayer(lambda x: 57.6 * x)),
                ("conv1", nn.Conv2d(3, 96, kernel_size=11, stride=4)),
                ("relu1", nn.ReLU(inplace=True)),
                ("pool1", nn.MaxPool2d(kernel_size=3, stride=2,
                                       ceil_mode=True)),
                ("norm1", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
                ("conv2", nn.Conv2d(96,
                                    256,
                                    kernel_size=5,
                                    padding=2,
                                    groups=2)),
                ("relu2", nn.ReLU(inplace=True)),
                ("pool2", nn.MaxPool2d(kernel_size=3, stride=2,
                                       ceil_mode=True)),
                ("norm2", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
                ("conv3", nn.Conv2d(256, 384, kernel_size=3, padding=1)),
                ("relu3", nn.ReLU(inplace=True)),
                ("conv4",
                 nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
                ("relu4", nn.ReLU(inplace=True)),
                ("conv5",
                 nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
                ("relu5", nn.ReLU(inplace=True)),
                ("pool5", nn.MaxPool2d(kernel_size=3, stride=2,
                                       ceil_mode=True)),
            ]))
        self.classifier = nn.Sequential(
            OrderedDict([("fc6", nn.Linear(256 * 6 * 6, 4096)),
                         ("relu6", nn.ReLU(inplace=True)),
                         ("drop6", nn.Dropout() if dropout else nn.Identity()),
                         ("fc7", nn.Linear(4096, 4096)),
                         ("relu7", nn.ReLU(inplace=True)),
                         ("drop7", nn.Dropout() if dropout else nn.Identity())
                         ]))

        self.class_classifier = nn.Linear(4096, num_classes)

    def forward(self, x):
        # moved to Lmabda
        # x = self.features(x*57.6)
        x = self.features(x)
        #57.6 is the magic number needed to bring torch data back to the range of caffe data, based on used std
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return self.class_classifier(x)


def caffenet(num_classes, pretrained=True):
    model = AlexNetCaffe(num_classes=num_classes)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, .1)
            nn.init.constant_(m.bias, 0.)

    if pretrained:
        current_path = pathlib.Path(__file__).parent.absolute()
        model_path = (os.path.join(current_path, "pretrained_alexnet.pth"))
        state_dict = torch.load(model_path)
        del state_dict["classifier.fc8.weight"]
        del state_dict["classifier.fc8.bias"]
        model.load_state_dict(state_dict, strict=False)
    return model