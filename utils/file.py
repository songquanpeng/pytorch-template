import os
import glob
import json


def list_all_images(path, full_path=True):
    image_types = ('png', 'jpg', 'jpeg')
    image_list = []
    for image_type in image_types:
        image_list.extend(glob.glob(os.path.join(path, f"**/*.{image_type}"), recursive=True))
    if not full_path:
        image_list = [os.path.relpath(image, path) for image in image_list]
    image_list = [p.replace("\\", '/') for p in image_list]
    return image_list


def list_sub_folders(path, full_path=True):
    folders = []
    for name in os.listdir(path):
        if os.path.isdir(os.path.join(path, name)):
            x = os.path.join(path, name) if full_path else name
            folders.append(x)
    folders.sort()
    return folders


def make_path(paths):
    if type(paths) != list:
        paths = [paths]
    for path in paths:
        os.makedirs(path, exist_ok=True)


def prepare_dirs(dirs):
    for dir_ in dirs:
        if not os.path.exists(dir_):
            os.makedirs(dir_)


def save_json(target_path, config, filename='config'):
    with open(os.path.join(target_path, f"{filename}.json"), 'w') as f:
        print(json.dumps(config.__dict__, sort_keys=True, indent=4), file=f)


def write_record(record, file_path, print_screen=True):
    if print_screen:
        print(record)
    with open(file_path, 'a') as f:
        f.write(record + "\n")
