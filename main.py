import os
import argparse
import random
import torch
import numpy as np
from torch.backends import cudnn

from utils.misc import get_datetime, str2bool, save_json


def main(args):
    exp_path = os.path.join(args.exp_dir, args.exp_id)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    save_json(exp_path, args)
    print(args)

    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # About this experiment.
    parser.add_argument('--about', type=str, default="")
    parser.add_argument('--exp_id', type=str, default=get_datetime(), help='Sub-folder for this expr.')
    parser.add_argument('--exp_dir', type=str, default='expr')

    # Meta arguments.
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'sample'])
    parser.add_argument('--device', type=str, default='cuda')

    # Model related arguments.
    parser.add_argument('--img_size', type=int, default=128)

    # Dataset related arguments.
    parser.add_argument('--dataset', type=str, choices=['CUB2011', 'CelebA'])
    parser.add_argument('--dataset_path', type=str, required=True)

    # Optimizing related arguments.
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate.")
    parser.add_argument('--beta1', type=float, default=0.0)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--batch_size', type=int, default=32)

    # Loss hyper arguments.
    parser.add_argument('--lambda_adv', type=float, default=1)

    # Step related arguments.
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--sample_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=5000)

    # Log related arguments.
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # Others
    parser.add_argument('--seed', type=int, default=0, help='Seed for random number generator.')
    parser.add_argument('--cudnn_benchmark', type=str2bool, default=True)

    main(parser.parse_args())
