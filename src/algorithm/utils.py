import torch
import sys
import wandb
import numpy as np


def equal_batch_size(x1, y1, img1, x2, y2, img2):
    if x1.shape[0] > x2.shape[0]:
        x1_sub_x2_size = x1.shape[0] - x2.shape[0]
        x2_additional_indices = torch.randint(0, x2.shape[0] - 1,
                                              (x1_sub_x2_size, ))
        for i in range(x2_additional_indices.shape[0]):
            curr_idx = x2_additional_indices[i]
            x2 = torch.cat((x2, x2[curr_idx].unsqueeze(0)), 0)
            y2 = torch.cat((y2, y2[curr_idx].unsqueeze(0)), 0)
            img2 = torch.cat((img2, img2[curr_idx].unsqueeze(0)), 0)
    elif x1.shape[0] < x2.shape[0]:
        # x2_sub_x1_size = x2.shape[0] - x1.shape[0]
        # x1_additional_indices = torch.randint(0, x1.shape[0] - 1,
        #                                       (x2_sub_x1_size, ))
        # for i in range(x1_additional_indices.shape[0]):
        #     curr_idx = x1_additional_indices[i]
        #     x1 = torch.cat((x1, x1[curr_idx].unsqueeze(0)), 0)
        #     y1 = torch.cat((y1, y1[curr_idx].unsqueeze(0)), 0)
        x2 = x2[0:x1.shape[0]]
        y2 = y2[0:x1.shape[0]]
        img2 = img2[0:x1.shape[0]]

    return x1, y1, img1, x2, y2, img2


def get_same_batch_size(x1, x2):
    if x1.shape[0] > x2.shape[0]:
        x1_sub_x2_size = x1.shape[0] - x2.shape[0]
        x2_additional_indices = torch.randint(0, x2.shape[0],
                                              (x1_sub_x2_size, ))
        for i in range(x2_additional_indices.shape[0]):
            curr_idx = x2_additional_indices[i]
            x2 = torch.cat((x2, x2[curr_idx].unsqueeze(0)), 0)
    elif x1.shape[0] < x2.shape[0]:
        x2 = x2[0:x1.shape[0]]

    return x1, x2


def cosine_similarity(x1, x2):
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
    return cos(x1, x2)