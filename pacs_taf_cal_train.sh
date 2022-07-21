#!/bin/bash

random_seed=0


python3 -m src.taf_cal_train \
    --active_layers=0,0,1,0 \
    --add_val \
    --batch_size=64 \
    --calibrate_alpha=-1 \
    --classifier_lr=0.01 \
    --dataset=PACS \
    --gpu=0 \
    --features_lr=0.001 \
    --ft_amp_factor=0.5 \
    --further_num_epochs=15 \
    --keep_update_amp_mean \
    --lr_step=43 \
    --mix_amp_ratio=0.5 \
    --mix_func_ratio=0.3 \
    --mix_layers=0,0,1,0 \
    --mix_loss_lr=1 \
    --mix_options=1,1 \
    --mixup_alpha=0.2 \
    --mixup_amp_features \
    --model=resnet50 \
    --momentum=0.9 \
    --num_epochs=35 \
    --random_seed=0 \
    --robustdg_aug \
    --save_dir=pacs/resnet50 \
    --scheduler=step \
    --swap_amp_ratio=0.3 \
    --test_amp_factor=0.5 \
    --use_original_train_set \
    --warmup_epoch=20