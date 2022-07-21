from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import random
import copy


class LazyLoader:
    def __init__(self, initializer, *args, **kwargs):
        self.initializer = initializer
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return self.initializer(*self.args, **self.kwargs)


class ImageSet(torch.utils.data.Dataset):
    def __init__(self, path_f, parent_dir=''):

        lines = [p.strip().split() for p in open(path_f, 'r')]
        self.paths = [f'{parent_dir}/{path}' for path, _ in lines]
        self.labels = [int(label) - 1 for _, label in lines]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        label = self.labels[index]
        img = Image.open(path)
        return {'img': img, 'label': label, 'path': path}


class Mixed(torch.utils.data.Dataset):
    def __init__(self, *args):
        # assumes domains are lazy loaded for efficiency
        self.domains = [domain() for domain in args]
        self.lengths = [len(d) for d in self.domains]

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, index):

        for d, domain in enumerate(self.domains):
            if index >= len(domain):
                index -= len(domain)
            else:
                x = domain[index]
                x['domain'] = d
                return x


class Augmentation(torch.utils.data.Dataset):
    def __init__(self,
                 dataset,
                 baseline,
                 augmentation=lambda x: x,
                 augment_half=False,
                 use_rgb_convert=False,
                 ctr=False):
        self.dataset = dataset
        self.baseline = baseline
        self.augmentation = augmentation
        self.use_rgb_convert = use_rgb_convert
        self.augment_half = augment_half
        self.ctr = ctr

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x = self.dataset[index]
        if self.ctr:
            x['img_1'] = self.baseline(x['img'])
            x['img_2'] = self.baseline(x['img'])
            x['img'] = self.baseline(x['img'])
        else:
            if self.use_rgb_convert and x['img'].mode is not 'RGB':
                x['img'] = x['img'].convert('RGB')
            augment = True
            if self.augment_half and random.random() < 0.5:
                augment = False
            if augment:
                x['img'] = self.augmentation(x['img'])
            x['augmented'] = augment
            x['img'] = self.baseline(x['img'])
        return x