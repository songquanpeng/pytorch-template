#!/bin/bash
filename=$(basename "$0");exp_id="${filename%.*}"
CUDA_VISIBLE_DEVICES=0 python main.py \
--exp_id "$exp_id" --about "" \
--mode train \
--start_iter 0 --end_iter 100000 \
--use_tensorboard true --save_loss true \
--dataset CelebA \
--batch_size 8 --img_size 256 \
--train_path ./archive/celeba_hq/train \
--test_path ./archive/celeba_hq/test \
--compare_path ./archive/celeba_hq/train