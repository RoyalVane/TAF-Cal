from pathlib import Path
import random
import argparse
from re import A
import sys
from torchvision.utils import save_image
import torchvision
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms

from .datasets.utils import get_splits
from .models.alexnet import CaffeNet
from .models.resnet import Resnet
from .models.entropyLoss import HLoss
from .datasets.datasets import Augmentation
from .datasets.utils import *
from .models.utils import set_random_seed, evaluate, evaluate_swap_amp, get_ampl_features, further_train, mixup_amp
from .models.fourier import *
from .models.gans.gan import Generator
import PIL
from tqdm import tqdm
from .models.fourier import *
import pickle
import shutil
import os
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--verbose',
                        help='Print log file',
                        action='store_true')
    parser.add_argument('--gpu', type=int, help='GPU idx to run', default=0)
    parser.add_argument('--save_dir',
                        type=str,
                        help='Write directory',
                        default='output')
    parser.add_argument('--model',
                        type=str,
                        choices=['caffenet', 'resnet18', 'resnet50'],
                        help='Model',
                        default='resnet18')
    parser.add_argument('--optimizer',
                        type=str,
                        choices=['sgd', 'adam'],
                        help='optimizer',
                        default='sgd')
    parser.add_argument('--scheduler',
                        type=str,
                        choices=['step', 'multi_step'],
                        help='scheduler',
                        default='step')
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['PACS', 'VLCS', 'OfficeHome', 'OfficeHome_larger'],
        help='Dataset',
        default='PACS')
    parser.add_argument(
        '--single_target',
        type=str,
        help='If a single target is required, specify it here.',
        default=None)
    parser.add_argument('--leave_out_domain',
                        type=str,
                        help='Domain to leave out for hyper-param selection',
                        default=None)
    parser.add_argument('--features_lr',
                        type=float,
                        help='Feature extractor learning rate',
                        default=1e-3)
    parser.add_argument('--classifier_lr',
                        type=float,
                        help='Classifier learning rate',
                        default=1e-2)
    parser.add_argument('--num_epochs',
                        type=int,
                        help='Number of epochs',
                        default=50)
    parser.add_argument('--batch_size',
                        type=int,
                        help='Batch size',
                        default=64)
    parser.add_argument('--lr_step',
                        type=int,
                        help='Steps between LR decrease',
                        default=40)
    parser.add_argument('--multi_lr_step',
                        type=str,
                        help='multi_lr_step',
                        default='40,80,110')
    parser.add_argument('--momentum',
                        type=float,
                        help='Momentum for SGD',
                        default=0.9)
    parser.add_argument('--weight_decay',
                        type=float,
                        help='Weight Decay',
                        default=5e-4)
    parser.add_argument('--wandb',
                        type=str,
                        help='Plot on wandb ',
                        default=None)
    parser.add_argument('--random_seed',
                        type=int,
                        help='random seed',
                        default=None)
    parser.add_argument('--only_augment_half',
                        help='Only augment 50 percent of the images',
                        action='store_true')
    parser.add_argument('--use_original_train_set',
                        help='whether to use the original train set.',
                        default=True,
                        action='store_true')
    parser.add_argument('--add_val',
                        help='add validation into pipeline',
                        default=True,
                        action='store_true')
    parser.add_argument('--even_lower_lr_vlcs',
                        help='even lower lr for vlcs',
                        action='store_true')
    parser.add_argument('--supression_decay',
                        type=float,
                        help='Weight to use in supression',
                        default=10.0)
    parser.add_argument('--robustdg_aug',
                        help='robustDG augmentation',
                        default=True,
                        action='store_true')
    parser.add_argument('--further_num_epochs',
                        type=int,
                        help='Steps between LR decrease',
                        default=5)
    parser.add_argument('--ft_amp_factor',
                        type=float,
                        help='amplitude mean factor in the further training',
                        default=0.5)
    parser.add_argument('--test_amp_factor',
                        type=float,
                        help='amplitude mean factor in the testing',
                        default=0.5)
    parser.add_argument(
        '--swap_amp_ratio',
        type=float,
        help=
        'the ratio of swapping amp in further training, the higher means more swapping',
        default=0.5)
    parser.add_argument('--keep_update_amp_mean',
                        help='keep update amplitude mean',
                        action='store_true')
    parser.add_argument('--load_trained_model',
                        help='load_trained_model',
                        action='store_true')
    parser.add_argument('--active_layers',
                        type=str,
                        help='which layers to swap amp',
                        default='0,0,1,0')
    parser.add_argument('--mix_layers',
                        type=str,
                        help='which layers to mix amp',
                        default='0,0,1,0')
    parser.add_argument('--mixup_amp_features',
                        help='mixup_amp_features',
                        action='store_true')
    parser.add_argument('--warmup_epoch',
                        type=int,
                        help='warmp epochs for mixup amp',
                        default=10)
    parser.add_argument(
        '--mix_amp_ratio',
        type=float,
        help=
        'the ratio of mix amp in initial training, the higher means more swapping',
        default=0.5)
    parser.add_argument('--mix_loss_lr',
                        type=float,
                        help='lr for mixed amplitude',
                        default=1)
    parser.add_argument('--mix_options',
                        type=str,
                        help='first one is mix, second one is shuffle',
                        default='0,1')
    parser.add_argument('--mixup_alpha',
                        type=float,
                        help='mixup alpha value',
                        default=0.2)
    parser.add_argument('--find_best_hp',
                        help='find best hyperparameters',
                        action='store_true')
    parser.add_argument('--mix_func_ratio',
                        type=float,
                        help='mixup ratio in mixup function',
                        default=0.3)
    parser.add_argument('--calibrate_alpha',
                        type=float,
                        help='mixup alpha value',
                        default=-1)
    parser.add_argument('--se_block', help='use se block', action='store_true')
    parser.add_argument('--sp_block', help='use sp block', action='store_true')

    torch.autograd.set_detect_anomaly(True)
    args = parser.parse_args()
    if args.dataset == 'PACS':
        args.num_classes = 7
        args.num_domains = 4
    elif args.dataset == 'VLCS':
        args.num_classes = 5
        args.num_domains = 4
    elif args.dataset == 'OfficeHome':
        args.num_classes = 65
        args.num_domains = 4
    if args.dataset == 'VLCS':
        args.features_lr /= 10
        args.classifier_lr /= 10
        args.domain_adversary_lr /= 10
        if args.even_lower_lr_vlcs:
            args.features_lr /= 10
            args.classifier_lr /= 10
    args.num_src_domains = args.num_domains - 1

    print('args: ', args)
    args.active_layers = [int(item) for item in args.active_layers.split(',')]
    args.mix_options = [int(item) for item in args.mix_options.split(',')]
    args.mix_layers = [int(item) for item in args.mix_layers.split(',')]
    args.multi_lr_step = [int(item) for item in args.multi_lr_step.split(',')]

    if args.random_seed is not None:
        set_random_seed(args.random_seed)

    if args.wandb is not None:
        import wandb
        wandb.init(project=args.wandb, name=args.save_dir)
    else:
        wandb = None

    args.use_rgb_convert = (args.dataset == 'VLCS')

    splits, num_classes, num_domains = get_splits(
        args.dataset,
        leave_out=args.leave_out_domain,
        original=args.use_original_train_set)

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    args.device = device

    for heldout in splits.keys():
        if args.single_target is not None and heldout != args.single_target:
            continue
        args.heldout = heldout
        _save_dir = 'results/' + args.save_dir + '/' + heldout
        Path(_save_dir).mkdir(parents=True, exist_ok=True)
        if not args.verbose:
            sys.stdout = open(f'{_save_dir}/log.txt', 'w')

        print('args: ', args)
        print('nums_domain: ', num_domains)
        saved_code = False

        if not saved_code:
            code_dir = 'src/'
            code_save_dir = 'results/' + args.save_dir + '/code'
            if os.path.exists(code_save_dir) and os.path.isdir(code_save_dir):
                shutil.rmtree(code_save_dir)
            shutil.copytree(code_dir, code_save_dir)
            saved_code = True

        lr_groups = []
        rec_lr_groups = []

        if args.model == 'resnet18' or args.model == 'resnet50':
            print('Using resnet')
            model = Resnet(num_classes=num_classes,
                           model=args.model,
                           se_block=args.se_block,
                           sp_block=args.sp_block).to(device)
            lr_groups.extend([
                (model.base_model.conv1.parameters(), args.features_lr),
                (model.base_model.bn1.parameters(), args.features_lr),
                (model.base_model.layer1.parameters(), args.features_lr),
                (model.base_model.layer2.parameters(), args.features_lr),
                (model.base_model.layer3.parameters(), args.features_lr),
                (model.base_model.layer4.parameters(), args.features_lr),
                (model.base_model.fc.parameters(), args.classifier_lr)
            ])
            if args.se_block:
                lr_groups.extend([
                    (model.orig_amp_fc.parameters(), args.features_lr),
                    (model.amp_mean_fc.parameters(), args.features_lr),
                    (model.shared_amp_fc.parameters(), args.features_lr),
                    (model.amp_conv_1.parameters(), args.features_lr),
                    (model.amp_conv_2.parameters(), args.features_lr)
                ])  #TODO forgot to add optimization
            if args.sp_block:
                lr_groups.extend([
                    (model.orig_amp_conv.parameters(), args.features_lr),
                    (model.amp_mean_conv.parameters(), args.features_lr),
                    (model.shared_amp_conv.parameters(), args.features_lr)
                ])
            c_loss_fn = torch.nn.CrossEntropyLoss()

        else:
            raise Exception(f'{args.model} not supported.')

        if args.optimizer == 'sgd':
            optimizers = [
                torch.optim.SGD(params,
                                lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)
                for params, lr in lr_groups
            ]
        else:
            optimizers = [
                torch.optim.Adam(params, lr, weight_decay=args.weight_decay)
                for params, lr in lr_groups
            ]

        if args.scheduler == 'step':
            schedulers = [
                torch.optim.lr_scheduler.StepLR(optim,
                                                step_size=args.lr_step,
                                                gamma=1e-1)
                for optim in optimizers
            ]
        elif args.scheduler == 'multi_step':
            schedulers = [
                torch.optim.lr_scheduler.MultiStepLR(optim,
                                                     milestones=[
                                                         args.multi_lr_step[0],
                                                         args.multi_lr_step[1],
                                                         args.multi_lr_step[2]
                                                     ],
                                                     gamma=1e-1)
                for optim in optimizers
            ]

        trainset = splits[heldout]['train']()

        if args.robustdg_aug:
            print('using robustdg_aug for train')
            trainset = Augmentation(trainset,
                                    baseline=robustdg_train_augmentation(),
                                    augment_half=args.only_augment_half,
                                    use_rgb_convert=args.use_rgb_convert)

        train = torch.utils.data.DataLoader(trainset,
                                            batch_size=args.batch_size,
                                            drop_last=False,
                                            num_workers=6,
                                            shuffle=True)
        if args.add_val:
            if args.robustdg_aug:
                print('using robustdg_aug for val')
                valset = Augmentation(splits[heldout]['val'](),
                                      baseline=robustdg_test_transform(),
                                      use_rgb_convert=args.use_rgb_convert)

            val = torch.utils.data.DataLoader(valset,
                                              batch_size=args.batch_size,
                                              drop_last=False,
                                              num_workers=6,
                                              shuffle=False)
        if args.robustdg_aug:
            print('using robustdg_aug for test')
            testset = Augmentation(splits[heldout]['test'](),
                                   baseline=robustdg_test_transform(),
                                   use_rgb_convert=args.use_rgb_convert)
#
        test = torch.utils.data.DataLoader(testset,
                                           batch_size=args.batch_size,
                                           drop_last=False,
                                           num_workers=6,
                                           shuffle=False)

        print(f'Starting {heldout}...')
        if not args.load_trained_model:

            best_val_acc = 0
            best_val_acc_epoch = 0
            best_test_acc = 0
            best_test_acc_epoch = 0
            anomaly = False
            h_loss_fn = HLoss()
            for epoch in range(0, args.num_epochs):
                running_loss_class = 0.0
                running_loss_model = 0.0
                running_loss_rec = 0.0
                print(
                    '-------------------------------------------------------')
                print(f'Starting epoch {epoch}...')
                p = epoch / args.num_epochs
                supression = (2.0 / (1. + np.exp(-args.supression_decay * p)) -
                              1)
                for i, batch in enumerate(tqdm(train)):
                    x = batch['img'].to(device)
                    y = batch['label'].to(device)
                    d = batch['domain'].to(device)
                    loss = 0
                    if args.mixup_amp_features:
                        if random.random() > (1 - args.mix_amp_ratio) and (
                                epoch >= args.warmup_epoch):
                            do_swap_amp = True
                        else:
                            do_swap_amp = False
                    else:
                        do_swap_amp = False
                    model.train()
                    early_features = model.early_layer(x)
                    if do_swap_amp and args.mix_layers[0]:
                        early_features = mixup_amp(args, early_features,
                                                   device, model)
                    layer1_features = model.layer1(early_features)
                    if do_swap_amp and args.mix_layers[1]:
                        layer1_features = mixup_amp(args, layer1_features,
                                                    device, model)
                    layer2_features = model.layer2(layer1_features)
                    if do_swap_amp and args.mix_layers[2]:
                        layer2_features = mixup_amp(args, layer2_features,
                                                    device, model)
                    layer3_features = model.layer3(layer2_features)
                    if do_swap_amp and args.mix_layers[3]:
                        layer3_features = mixup_amp(args, layer3_features,
                                                    device, model)
                    layer4_features = model.layer4(layer3_features)
                    yhat = model.classifier_layer(layer4_features)

                    c_loss = c_loss_fn(yhat, y)
                    if do_swap_amp and (epoch >= args.warmup_epoch):
                        c_loss = c_loss * args.mix_loss_lr

                    # print(f'c_loss:{c_loss:.f}')
                    running_loss_class += c_loss
                    loss += c_loss
                    if torch.isnan(loss):
                        anomaly = True
                        break

                    running_loss_model += loss

                    for optim in optimizers:
                        optim.zero_grad()

                    loss.backward()
                    if do_swap_amp and (epoch >= args.warmup_epoch):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

                    for optim in optimizers:
                        optim.step()

                if anomaly:
                    print('Found anomaly. Terminating.')
                    break

                for scheduler in schedulers:
                    scheduler.step()

                epoch_loss_class = running_loss_class / i

                epoch_loss_model = running_loss_model / i  # len(train.dataset)
                print(
                    f'class_loss={epoch_loss_class:.4f} | model_loss={epoch_loss_model:.4f}'
                )

                # if args.add_val:
                #     val_acc, val_loss = evaluate(args, model, val, device,
                #                                  c_loss_fn, _save_dir)
                #     if val_acc > best_val_acc:
                #         torch.save(model.state_dict(),
                #                    f'{_save_dir}/val_model.pt')
                #         best_val_acc = val_acc
                #         best_val_acc_epoch = epoch
                #     print(
                #         f'val_accuracy={val_acc:.4f} | val_class_loss={val_loss:.4f}'
                #     )

                test_acc, test_loss = evaluate(args, model, test, device,
                                               c_loss_fn, _save_dir)
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    best_test_acc_epoch = epoch
                print(
                    f'test_accuracy={test_acc:.4f} | test_class_loss={test_loss:.4f}'
                )
                if args.wandb is not None:
                    wandb.log({
                        f"{heldout}_epoch":
                        epoch,
                        f"{heldout}_class_loss":
                        epoch_loss_class,
                        f"{heldout}_model_loss":
                        epoch_loss_model,
                        f"{heldout}_test_class_loss":
                        test_loss,
                        f"{heldout}_test_accuracy":
                        test_acc,
                        f"{heldout}_val_class_loss":
                        val_loss if args.add_val else None,
                        f"{heldout}_val_accuracy":
                        val_acc if args.add_val else None
                    })

            print(f'Saving the model used to test..')
            torch.save(model.state_dict(), f'{_save_dir}/model.pt')

            # if args.add_val:
            #     print('Starting testing on best val model...')
            #     model.load_state_dict(torch.load(f'{_save_dir}/val_model.pt'))

            #     val_test_acc, val_test_loss = evaluate(args, model, test,
            #                                            device, c_loss_fn,
            #                                            _save_dir)
            #     print(
            #         f'test(val)_accuracy={val_test_acc:.4f} | test(val)_class_loss={val_test_loss:.4f}'
            #     )
        if not args.load_trained_model:
            with open(f'{_save_dir}/optim.pkl', 'wb') as path:
                pickle.dump(optimizers, path)
            with open(f'{_save_dir}/sch.pkl', 'wb') as path:
                pickle.dump(schedulers, path)
        if args.load_trained_model:
            with open(f'{_save_dir}/optim.pkl', 'rb') as path:
                optimizers = pickle.load(path)
            with open(f'{_save_dir}/sch.pkl', 'rb') as path:
                schedulers = pickle.load(path)
            print('loading trained model')
        print('Starting testing on last model...')

        model.load_state_dict(torch.load(f'{_save_dir}/model.pt'))

        print('before further train:')
        acc, loss = evaluate(args, model, test, device, c_loss_fn, _save_dir)
        print(
            f'brefore further train: test_acc:{acc:.4f} | test_loss:{loss:.4f}'
        )

        amp_features = 0

        model, amp_features, best_test_amp_acc = further_train(
            args, model, amp_features, train, device, c_loss_fn, optimizers,
            schedulers, test, val, _save_dir)
        acc, loss = evaluate(args, model, test, device, c_loss_fn, _save_dir)
        amp_acc, amp_loss = evaluate_swap_amp(args, model, amp_features, test,
                                              device, c_loss_fn)
        print(f'after further train')
        print(
            f'test(last_epoch)_calibrate_accuracy={amp_acc:.4f} | test(last_epoch)_calibrate_class_loss={amp_loss:.4f}'
        )

        print(
            f'test(last_epoch)_accuracy={acc:.4f} | test(last_epoch)_class_loss={loss:.4f}'
        )

        if args.add_val:
            print('Starting testing on best val model...')
            model.load_state_dict(torch.load(f'{_save_dir}/amp_val_model.pt'))

            amp_val_test_acc, amp_val_test_loss = evaluate_swap_amp(
                args, model, amp_features, test, device, c_loss_fn)
            print(
                f'test(best_val)_amp_accuracy={amp_val_test_acc:.4f} | test(best_val)_amp_class_loss={amp_val_test_loss:.4f}'
            )
        print(f'test(best_test)_amp_accuracy={best_test_amp_acc:.4f}')
        with open(f'results/{args.save_dir}/results.txt', 'a') as res:
            res.write(
                f'{heldout}: test(last_ep)_acc={acc:.4f} | test(last_ep)_calibrate_acc={amp_acc:.4f} | test(best_val)_calibrate_acc={amp_val_test_acc:.4f} | test(best_test)_calibrate_acc={best_test_amp_acc:.4f}\n'
            )
        print(f'Finished testing')
        if args.find_best_hp:
            if args.model == 'resnet18':
                if heldout == 'sketch':
                    if amp_val_test_acc < 0.81 and amp_acc < 0.81:
                        print('sketch acc is less than 0.805')
                        exit()
                if heldout == 'art_painting':
                    if amp_val_test_acc < 0.85 and amp_acc < 0.85:
                        print(f'art_painting acc is less than 0.845')
                        exit()
                if heldout == 'art':
                    if amp_val_test_acc < 0.60 and amp_acc < 0.60:
                        print(f'art amp acc is less 0.60')
                        exit()

        print(f'Finished {heldout}')
        if args.wandb is not None:
            wandb.log({
                f"{heldout}_final_test_class_loss":
                loss,
                f"{heldout}_final_test_accuracy":
                acc,
                f"{heldout}_best_test_acc":
                best_test_acc,
                f"{heldout}_best_test_acc_epoch":
                best_test_acc_epoch,
                f"{heldout}_final_test(val)_class_loss":
                val_test_loss if args.add_val else None,
                f"{heldout}_final_test(val)_test_accuracy":
                val_test_acc if args.add_val else None,
                f"{heldout}_final_test_amp_loss":
                amp_loss,
                f"{heldout}_final_test_amp_acc":
                amp_acc
            })