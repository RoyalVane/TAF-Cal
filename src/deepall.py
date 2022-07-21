from pathlib import Path
import random
import argparse
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
from .models.utils import set_random_seed, evaluate
from .models.fourier import *
from .models.gans.gan import Generator
import PIL
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
                        default=128)
    parser.add_argument('--lr_step',
                        type=int,
                        help='Steps between LR decrease',
                        default=40)
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

        lr_groups = []

        if args.model == 'resnet18' or args.model == 'resnet50':
            print('Using resnet')
            model = Resnet(num_classes=num_classes,
                           model=args.model,
                           se_block=False,
                           sp_block=False).to(device)
            lr_groups.extend([
                (model.base_model.conv1.parameters(), args.features_lr),
                (model.base_model.bn1.parameters(), args.features_lr),
                (model.base_model.layer1.parameters(), args.features_lr),
                (model.base_model.layer2.parameters(), args.features_lr),
                (model.base_model.layer3.parameters(), args.features_lr),
                (model.base_model.layer4.parameters(), args.features_lr),
                (model.base_model.fc.parameters(), args.classifier_lr)
            ])
            c_loss_fn = torch.nn.CrossEntropyLoss()

        else:
            raise Exception(f'{args.model} not supported.')

        optimizers = [
            torch.optim.SGD(params,
                            lr,
                            momentum=args.momentum,
                            nesterov=True,
                            weight_decay=args.weight_decay)
            for params, lr in lr_groups
        ]

        schedulers = [
            torch.optim.lr_scheduler.StepLR(optim,
                                            step_size=args.lr_step,
                                            gamma=1e-1) for optim in optimizers
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
                                              shuffle=True)
        if args.robustdg_aug:
            print('using robustdg_aug for test')
            testset = Augmentation(splits[heldout]['test'](),
                                   baseline=robustdg_test_transform(),
                                   use_rgb_convert=args.use_rgb_convert)
#
        test = torch.utils.data.DataLoader(testset,
                                           batch_size=1,
                                           drop_last=False,
                                           num_workers=6,
                                           shuffle=True)

        print(f'Starting {heldout}...')
        #Do the contrastive learning pretraining
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
            print('-------------------------------------------------------')
            print(f'Starting epoch {epoch}...')
            p = epoch / args.num_epochs
            supression = (2.0 / (1. + np.exp(-args.supression_decay * p)) - 1)
            for i, batch in enumerate(train):
                x = batch['img'].to(device)
                x_clone = x.clone()
                y = batch['label'].to(device)
                d = batch['domain'].to(device)
                loss = 0

                model.train()
                z_conv = model.conv_features(x)
                z = model.dense_features(z_conv)
                yhat = model.classifier(z)
                c_loss = c_loss_fn(yhat, y)

                # print(f'c_loss:{c_loss:.4f}')
                running_loss_class += c_loss
                loss += c_loss
                if torch.isnan(loss):
                    anomaly = True
                    break

                running_loss_model += loss

                for optim in optimizers:
                    optim.zero_grad()

                loss.backward()

                for optim in optimizers:
                    optim.step()

            if anomaly:
                print('Found anomaly. Terminating.')
                break

            for scheduler in schedulers:
                scheduler.step()

            epoch_loss_class = running_loss_class / i
            print(f'epoch{epoch}_class_loss={epoch_loss_class:.4f}')

            epoch_loss_model = running_loss_model / i  # len(train.dataset)
            print(f'epoch{epoch}_model_loss={epoch_loss_model:.4f}')

            if args.add_val:

                val_acc, val_loss = evaluate(args, model, val, device,
                                             c_loss_fn, _save_dir)
                if val_acc > best_val_acc:
                    torch.save(model.state_dict(), f'{_save_dir}/val_model.pt')
                    best_val_acc = val_acc
                    best_val_acc_epoch = epoch
                print(f'val_class_loss={val_loss:.4f}')
                print(f'val_accuracy={val_acc:.4f}')

            test_acc, test_loss = evaluate(args, model, test, device,
                                           c_loss_fn, _save_dir)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_acc_epoch = epoch
            print(f'test_class_loss={test_loss:.4f}')
            print(f'test_accuracy={test_acc:.4f}')

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

        if args.add_val:
            print('Starting testing on best val model...')
            model.load_state_dict(torch.load(f'{_save_dir}/val_model.pt'))

            val_test_acc, val_test_loss = evaluate(args, model, test, device,
                                                   c_loss_fn, _save_dir)
            print(f'test(val)_class_loss={val_test_loss:.4f}')
            print(f'test(val)_accuracy={val_test_acc:.4f}')

        print('Starting testing on last model...')
        model.load_state_dict(torch.load(f'{_save_dir}/model.pt'))

        acc, loss = evaluate(args, model, test, device, c_loss_fn, _save_dir)

        with open(f'results/{args.save_dir}/results.txt', 'a') as res:
            res.write(f'{heldout}: acc={acc:.4f} loss={loss:.4f}\n')
        print(f'test_class_loss={loss:.4f}')
        print(f'test_accuracy={acc:.4f}')
        print(f'Finished testing')

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
                val_test_acc if args.add_val else None
            })
