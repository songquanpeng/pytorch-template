import datetime
import os
import subprocess

import requests


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
    else:
        print(message)


def get_datetime(short=False):
    format_str = '%Y%m%d%H%M%S' if short else '%Y-%m-%d_%H-%M-%S'
    return datetime.datetime.now().strftime(format_str)


def get_commit_hash():
    process = subprocess.Popen(['git', 'log', '-n', '1'], stdout=subprocess.PIPE)
    output = process.communicate()[0]
    output = output.decode('utf-8')
    return output[7:13]


def start_tensorboard(working_dir, logdir='logs'):
    try:
        process = subprocess.Popen(['tensorboard', '--logdir', logdir, '--bind_all'], cwd=working_dir)
    except FileNotFoundError as e:
        print(f"Error: failed to start Tensorboard -- {e}")


def str2bool(v):
    return v.lower() in ['true']


def str2list(string, separator='-', target_type=int):
    return list(map(target_type, string.split(separator)))
