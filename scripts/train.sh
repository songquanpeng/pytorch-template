#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python main.py \
--about "" \
--mode train --dataset CelebA --dataset_path ../celeba/images \
--use_tensorboard true \
--batch_size 8 --img_size 128 \
--exp_id "exp_main"
