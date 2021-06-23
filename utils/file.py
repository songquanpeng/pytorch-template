import os
import glob


def list_all_images(path):
    image_types = ('png', 'jpg', 'jpeg')
    image_list = []
    for image_type in image_types:
        image_list.extend(glob.glob(os.path.join(path, f"**/*.{image_type}"), recursive=True))
    image_list = [os.path.relpath(image, path) for image in image_list]
    image_list = [p.replace("\\", '/') for p in image_list]
    return image_list


def list_sub_folders(path, full_path=True):
    folders = []
    for f in os.listdir(path):
        if os.path.isdir(f):
            folders.append(os.path.join(path, f))
    return folders


def make_paths(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)


def prepare_dirs(dirs):
    for dir_ in dirs:
        if not os.path.exists(dir_):
            os.makedirs(dir_)
