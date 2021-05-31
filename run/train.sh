#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python main.py \
--mode train --dataset CelebA --img_size 128 \
--cluster_npz_path data/celeba/clusters.npz \
--use_tensorboard true --dataset_path ../celeba/images \
--batch_size 8 --val_batch_size 32 \
--exp_id "exp_main" --img_size 128
