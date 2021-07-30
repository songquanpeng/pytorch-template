import datetime
import os
import subprocess
import requests
import random
import torch
import numpy as np
from torch.backends import cudnn
from utils.file import prepare_dirs, list_sub_folders


def send_message(message, exp_id=""):
    url = os.environ.get('MESSAGE_PUSH_URL')
    if url:
        try:
            url = f"{url}?type=corp&title={exp_id}&description={message}"
            res = requests.get(url)
            if res.status_code != 200:
                print('Failed to send message.')
        except:
            print('Failed to send message.')


def get_datetime(short=False):
    format_str = '%Y%m%d%H%M%S' if short else '%Y-%m-%d_%H-%M-%S'
    return datetime.datetime.now().strftime(format_str)


def str2bool(v):
    return v.lower() in ['true']


def setup(args):
    cudnn.benchmark = args.cudnn_benchmark
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    args.log_dir = os.path.join(args.exp_dir, args.exp_id, "logs")
    args.sample_dir = os.path.join(args.exp_dir, args.exp_id, "samples")
    args.model_dir = os.path.join(args.exp_dir, args.exp_id, "models")
    args.eval_dir = os.path.join(args.exp_dir, args.exp_id, "eval")
    prepare_dirs([args.log_dir, args.sample_dir, args.model_dir, args.eval_dir])
    args.record_file = os.path.join(args.exp_dir, args.exp_id, "records.txt")
    args.loss_file = os.path.join(args.exp_dir, args.exp_id, "losses.csv")

    args.domains = list_sub_folders(args.train_path, full_path=False)
    args.num_domains = len(args.domains)


def validate(args):
    assert args.eval_every % args.save_every == 0
    assert args.num_domains == len(list_sub_folders(args.test_path, full_path=False))
    if args.cache_dataset:
        assert args.preload_dataset, "Use cached dataset requires you enable preloading dataset!"


def get_commit_hash():
    process = subprocess.Popen(['git', 'log', '-n', '1'], stdout=subprocess.PIPE)
    output = process.communicate()[0]
    output = output.decode('utf-8')
    return output[7:13]
