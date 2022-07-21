# import scipy.ndimage
import random
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import torch.nn as nn
from .fourier import *

from torchvision import transforms
from tqdm import tqdm
from .mixup import mixup_func, mixup_func_se


def inv_norm():
    return transforms.Compose([
        transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.255]),
        transforms.ToPILImage()
    ])


def init_conv_weights(m, activations='relu'):

    gain = torch.nn.init.calculate_gain(activations)

    if type(m) == torch.nn.Conv2d  \
            or type(m) == torch.nn.ConvTranspose2d:

        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        torch.nn.init.constant_(m.bias, 0.0)


def set_random_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Adapted from https://github.com/jik0730/VAT-pytorch/blob/a7424f2ff386ceb39f80053c4103f9cd505ea07c/vat.py
def kl_divergence_with_logit(q_logit, p_logit):
    q = F.softmax(q_logit, dim=1)
    qlogq = torch.mean(torch.sum(q * F.log_softmax(q_logit, dim=1), dim=1))
    qlogp = torch.mean(torch.sum(q * F.log_softmax(p_logit, dim=1), dim=1))
    return qlogq - qlogp


def evaluate(args, model, dset, device, c_loss_fn, _save_dir):
    num_correct = 0
    num_total = 0
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for batch in dset:
            x = batch['img'].to(device)
            y = batch['label'].to(device)
            yhat = model(x)
            total_loss += c_loss_fn(yhat, y)
            num_correct += (yhat.argmax(dim=1) == y).sum().item()
            num_total += x.size(0)
        loss = total_loss / len(dset)
        acc = num_correct / num_total
    return acc, loss


def evaluate_amp(args, model, dset, device, c_loss_fn, _save_dir):
    num_correct = 0
    num_total = 0
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for batch in dset:
            x = batch['img'].to(device)
            y = batch['label'].to(device)
            if args.amp_img:
                phase, amp = fft(x, device=device)
                x = amp
            model.train()
            early_features = model.early_layer(x)
            layer1_features = model.layer1(early_features)
            layer2_features = model.layer2(layer1_features)
            if args.amp_feat:
                phase, amp = fft(layer2_features, device=device)
                layer2_features = amp
            layer3_features = model.layer3(layer2_features)
            layer4_features = model.layer4(layer3_features)
            yhat = model.classifier_layer(layer4_features)
            total_loss += c_loss_fn(yhat, y)
            num_correct += (yhat.argmax(dim=1) == y).sum().item()
            num_total += x.size(0)
        loss = total_loss / len(dset)
        acc = num_correct / num_total
    return acc, loss


def evaluate_swap_amp(args, model, amp_features, dset, device, c_loss_fn):
    num_correct = 0
    num_total = 0
    total_loss = 0
    model.eval()
    test_amp_factor = args.test_amp_factor
    with torch.no_grad():
        for batch in dset:
            x = batch['img'].to(device)
            y = batch['label'].to(device)
            layer0_features = model.early_layer(x)
            if args.active_layers[0]:
                layer0_features = swap_amp(layer0_features, amp_features, 0,
                                           test_amp_factor, device, args,
                                           model)
            layer1_features = model.layer1(layer0_features)
            if args.active_layers[1]:
                layer1_features = swap_amp(layer1_features, amp_features, 1,
                                           test_amp_factor, device, args,
                                           model)
            layer2_features = model.layer2(layer1_features)
            if args.active_layers[2]:
                layer2_features = swap_amp(layer2_features, amp_features, 2,
                                           test_amp_factor, device, args,
                                           model)
            layer3_features = model.layer3(layer2_features)
            if args.active_layers[3]:
                layer3_features = swap_amp(layer3_features, amp_features, 3,
                                           test_amp_factor, device, args,
                                           model)
            layer4_features = model.layer4(layer3_features)
            yhat = model.classifier_layer(layer4_features)

            total_loss += c_loss_fn(yhat, y)
            num_correct += (yhat.argmax(dim=1) == y).sum().item()
            num_total += x.size(0)
        loss = total_loss / len(dset)
        acc = num_correct / num_total
    return acc, loss


def get_ampl_features(args, model, dset, device):
    amp_features = {
        "amp_layer0_all": None,
        "amp_layer1_all": None,
        "amp_layer2_all": None,
        "amp_layer3_all": None
    }
    model.train()
    with torch.no_grad():
        for i, batch in enumerate(dset):
            x = batch['img'].to(device)
            early_features = model.early_layer(x)
            layer1_features = model.layer1(early_features)
            layer2_features = model.layer2(layer1_features)
            layer3_features = model.layer3(layer2_features)
            layer4_features = model.layer4(layer3_features)
            yhat = model.classifier_layer(layer4_features)
            if args.active_layers[0]:
                _, layer0_amp = fft(early_features.cpu())
            if args.active_layers[1]:
                _, layer1_amp = fft(layer1_features.cpu())
            if args.active_layers[2]:
                _, layer2_amp = fft(layer2_features.cpu())
            if args.active_layers[3]:
                _, layer3_amp = fft(layer3_features.cpu())
            if i == 0:
                if args.active_layers[0]:
                    amp_features['amp_layer0_all'] = layer0_amp
                if args.active_layers[1]:
                    amp_features['amp_layer1_all'] = layer1_amp
                if args.active_layers[2]:
                    amp_features['amp_layer2_all'] = layer2_amp
                if args.active_layers[3]:
                    amp_features['amp_layer3_all'] = layer3_amp
            else:
                if args.active_layers[0]:
                    amp_features['amp_layer0_all'] = torch.cat(
                        [amp_features['amp_layer0_all'], layer0_amp])
                if args.active_layers[1]:
                    amp_features['amp_layer1_all'] = torch.cat(
                        [amp_features['amp_layer1_all'], layer1_amp])
                if args.active_layers[2]:
                    amp_features['amp_layer2_all'] = torch.cat(
                        [amp_features['amp_layer2_all'], layer2_amp])
                if args.active_layers[3]:
                    amp_features['amp_layer3_all'] = torch.cat(
                        [amp_features['amp_layer3_all'], layer3_amp])
    if args.active_layers[0]:
        amp_features['amp_layer0_mean'] = torch.mean(
            amp_features['amp_layer0_all'], dim=0).unsqueeze(0)
    if args.active_layers[1]:
        amp_features['amp_layer1_mean'] = torch.mean(
            amp_features['amp_layer1_all'], dim=0).unsqueeze(0)
    if args.active_layers[2]:
        amp_features['amp_layer2_mean'] = torch.mean(
            amp_features['amp_layer2_all'], dim=0).unsqueeze(0)
    if args.active_layers[3]:
        amp_features['amp_layer3_mean'] = torch.mean(
            amp_features['amp_layer3_all'], dim=0).unsqueeze(0)

    return amp_features


def further_train(args, model, amp_features, train, device, c_loss_fn,
                  optimizers, schedulers, test, val, _save_dir):
    ft_amp_factor = args.ft_amp_factor
    best_val_amp_acc = 0
    best_test_amp_acc = 0
    running_loss_class = 0.0
    amp_features = get_ampl_features(args, model, train, device)
    already_print = False
    print('--------------------------------------------------')
    for epoch in range(0, args.further_num_epochs):
        print(f'epoch: {epoch}')
        if epoch > 0 and args.keep_update_amp_mean:
            del amp_features
            amp_features = get_ampl_features(args, model, train, device)
        for i, batch in enumerate(tqdm(train)):
            x = batch['img'].to(device)
            y = batch['label'].to(device)
            if random.random() > (1 - args.swap_amp_ratio):
                do_swap_amp = True
                if args.calibrate_alpha != -1:
                    ft_amp_factor = np.random.beta(args.calibrate_alpha,
                                                   args.calibrate_alpha)
                    if not already_print:
                        print('using calibrate_alpha')
                        already_print = True
            else:
                do_swap_amp = False
            if (random.random() >
                (1 - args.mix_amp_ratio)) and args.mixup_amp_features:
                do_mix_amp = True
            else:
                do_mix_amp = False
            loss = 0
            model.train()
            layer0_features = model.early_layer(x)
            if do_swap_amp and args.active_layers[0]:
                layer0_features = swap_amp(layer0_features, amp_features, 0,
                                           ft_amp_factor, device, args, model)

            if do_mix_amp and args.mix_layers[0]:
                layer0_features = mixup_amp(args, layer0_features, device,
                                            model)
            layer1_features = model.layer1(layer0_features)

            if do_swap_amp and args.active_layers[1]:
                layer1_features = swap_amp(layer1_features, amp_features, 1,
                                           ft_amp_factor, device, args, model)
            if do_mix_amp and args.mix_layers[1]:
                layer1_features = mixup_amp(args, layer1_features, device,
                                            model)
            layer2_features = model.layer2(layer1_features)

            if do_swap_amp and args.active_layers[2]:
                layer2_features = swap_amp(layer2_features, amp_features, 2,
                                           ft_amp_factor, device, args, model)
            if do_mix_amp and args.mix_layers[2]:
                layer2_features = mixup_amp(args, layer2_features, device,
                                            model)
            layer3_features = model.layer3(layer2_features)
            if do_swap_amp and args.active_layers[3]:
                layer3_features = swap_amp(layer3_features, amp_features, 3,
                                           ft_amp_factor, device, args, model)
            if do_mix_amp and args.mix_layers[3]:
                layer3_features = mixup_amp(args, layer3_features, device,
                                            model)
            layer4_features = model.layer4(layer3_features)
            yhat = model.classifier_layer(layer4_features)

            c_loss = c_loss_fn(yhat, y)
            running_loss_class += c_loss

            loss += c_loss

            for optim in optimizers:
                optim.zero_grad()
            loss.backward()
            # if do_mix_amp:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            for optim in optimizers:
                optim.step()
        for scheduler in schedulers:
            scheduler.step()
        test_acc, test_loss = evaluate(args,
                                       model,
                                       test,
                                       device,
                                       c_loss_fn,
                                       _save_dir='')
        amp_test_acc, amp_test_loss = evaluate_swap_amp(
            args, model, amp_features, test, device, c_loss_fn)
        if amp_test_acc >= best_test_amp_acc:
            best_test_amp_acc = amp_test_acc
        print(
            f'further train epoch {epoch}: test_acc={test_acc:.4f} | test_loss={test_loss:.4f}'
        )
        print(
            f'calibrate_test_acc={amp_test_acc:.4f} | calibrate_test_loss={amp_test_loss:.4f}'
        )
        if args.add_val:
            amp_val_acc, amp_val_loss = evaluate_swap_amp(
                args, model, amp_features, val, device, c_loss_fn)
            if amp_val_acc >= best_val_amp_acc:
                torch.save(model.state_dict(), f'{_save_dir}/amp_val_model.pt')
                best_val_amp_acc = amp_val_acc
            print(
                f'amp_val_acc={amp_val_acc:.4f} | amp_val_loss={amp_val_loss:.4f}'
            )
        epoch_loss_class = running_loss_class / i
        print(f'class_loss={epoch_loss_class:.4f}')
        running_loss_class = 0
    return model, amp_features, best_test_amp_acc


# def mixup_amp(args, features, device):
#     phase, mixed_amp = fft(features, device)
#     if args.mix_options[0]:
#         if random.random() > 0.5:
#             mixed_amp = mixup_func(mixed_amp, args.mixup_alpha)
#     # else:
#     if args.mix_options[1]:
#         # if random.random() > 0.5:
#         amp_idx = torch.randperm(mixed_amp.shape[0])
#         mixed_amp = mixed_amp[amp_idx].view(mixed_amp.size())
#     features = ifft(phase, mixed_amp, device)
#     return features


def swap_amp(features, amp_features, layer, orig_amp_factor, device, args,
             model):
    phase, amp = fft(features, device=device)
    amp_mean = amp_features[f'amp_layer{layer}_mean'].to(device).repeat(
        features.shape[0], 1, 1, 1)
    orig_amp = amp
    if args.se_block or args.sp_block:
        fused_amp = model.attention_forward(
            amp, amp_mean
        )  #TODO: Find a way to normalize amplitude by amp mean | add computed amp with orig amp. | use a domain discriminator to constrain style info, and a class classifier to constrain phase.
        amp = (orig_amp_factor * orig_amp) + (1 - orig_amp_factor) * fused_amp
    else:
        amp = (orig_amp_factor * amp) + (1 - orig_amp_factor) * amp_mean
    features = ifft(phase, amp, device)
    return features


def mixup_amp(args, features, device, model):
    phase, mixed_amp = fft(features, device)
    if args.mix_options[0]:
        if random.random() > args.mix_func_ratio:
            mixed_amp = mixup_func(mixed_amp, args.mixup_alpha)
    # else:
    if args.mix_options[1]:
        amp_idx = torch.randperm(mixed_amp.shape[0])
        mixed_amp = mixed_amp[amp_idx].view(mixed_amp.size())

    features = ifft(phase, mixed_amp, device)
    return features
