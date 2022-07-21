from torch import nn
from torch.nn import functional as F
"""
Entropy Loss adpated from Matsuura et al. 2020
(Code) https://github.com/mil-tokyo/dg_mmld/
(Paper) https://arxiv.org/pdf/1911.07661.pdf 
"""


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(
            dim=1).mean()  #remove -1.0 if we want to maxmize the entropy
        return b


class Test_HLoss(nn.Module):
    def __init__(self):
        super(Test_HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x) * F.log_softmax(x)
        b = -1.0 * b.sum(dim=1).mean(
        )  #.mean()  #remove -1.0 if we want to maxmize the entropy
        return b
