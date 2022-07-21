from collections import OrderedDict

import torch

from .caffenet.models import caffenet


class CaffeNet(torch.nn.Module):
    def __init__(self, num_classes):

        super().__init__()
        base_model = caffenet(num_classes, pretrained=True)
        self.conv_features = base_model.features
        self.dense_features = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1), base_model.classifier)
        self.features = torch.nn.Sequential(self.conv_features,
                                            self.dense_features)
        self.classifier = base_model.class_classifier

    def forward(self, x):
        return self.classifier(self.features(x))