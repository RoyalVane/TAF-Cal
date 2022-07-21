import torch
import numpy as np
from .fourier import *

#Adopted from https://github.com/MIPT-Oulu/RobustCartilageSegmentation/blob/master/rocaseg/components/mixup.py
"""
Example usage:
# Regular segmentation loss:
ys_pred_oai = self.models['segm'](xs_oai)
loss_segm = self.losses['segm'](input_=ys_pred_oai,
                                target=ys_true_arg_oai)
# Mixup
xs_mixup, ys_mixup_a, ys_mixup_b, lambda_mixup = mixup_data(
    x=xs_oai, y=ys_true_arg_oai,
    alpha=self.config['mixup_alpha'], device=maybe_gpu)
ys_pred_oai = self.models['segm'](xs_mixup)
loss_segm = mixup_criterion(criterion=self.losses['segm'],
                            pred=ys_pred_oai,
                            y_a=ys_mixup_a,
                            y_b=ys_mixup_b,
                            lam=lambda_mixup)
"""


def mixup_func(x, alpha=0.1):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(torch.get_device(x))
    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x


def mixup_func_se(x, model, alpha=0.1):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(torch.get_device(x))
    orig_x = x
    permed_x = x[index, :]
    orig_x, permed_x = model.se_forward(orig_x, permed_x)
    mixed_x = lam * orig_x + (1 - lam) * permed_x
    return mixed_x


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
