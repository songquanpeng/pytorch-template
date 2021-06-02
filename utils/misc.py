import datetime
import os
import requests
import json


def send_message(message):
    url = os.environ.get('MESSAGE_PUSH_URL')
    if url:
        url = f"{url}{message}"
        res = requests.get(url)
        if res.status_code != 200:
            print('Failed to send message.')


def get_datetime():
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def save_json(target_path, config, filename='config'):
    with open(os.path.join(target_path, f"{filename}.json"), 'w') as f:
        print(json.dumps(config.__dict__, sort_keys=True, indent=4), file=f)


def str2bool(v):
    return v.lower() in ['true']
