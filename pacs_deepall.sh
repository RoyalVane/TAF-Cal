#!/bin/bash



random_seed=0


python3 -m src.deepall \
    --save_dir=pacs/deepall/resnet50/rs$((random_seed)) \
    --gpu=1 \
    --random_seed=$random_seed \
    --num_epochs=50 \
    --lr_step=43 \
    --batch_size=64 \
    --add_val \
    --model=resnet50

