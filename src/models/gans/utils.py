import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import numpy as np
from torch.autograd import Variable
import torch.autograd as autograd


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def sample_data(dataset, Augmentation, data_transform, args, image_size=4):
    trainset = Augmentation(dataset,
                            baseline=data_transform(image_size),
                            augment_half=args.only_augment_half,
                            use_rgb_convert=args.use_rgb_convert)

    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=args.batch_size,
                                               drop_last=False,
                                               num_workers=6,
                                               shuffle=True)
    return train_loader


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples

    alpha = torch.tensor(np.random.random(
        (real_samples.size(0), 1, 1,
         1))).to(fake_samples.get_device()).float()
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples +
                    ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(torch.Tensor(real_samples.shape[0], 1).fill_(1.0),
                    requires_grad=False).to(fake_samples.get_device()).float()
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()
    return gradient_penalty