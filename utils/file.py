import glob
import json
import os
import pickle
import shutil


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
        print(json.dumps(config.__dict__, sort_keys=True, indent=4, ensure_ascii=False), file=f)


def write_record(record, file_path, print_screen=True):
    if print_screen:
        print(record)
    with open(file_path, 'a') as f:
        f.write(record + "\n")


def delete_dir(path):
    if path is None:
        return
    try:
        shutil.rmtree(path)
    except:
        print(f"Failed to delete dir:{path}")


def copy(file, src, dst):
    shutil.copyfile(os.path.join(src, file), os.path.join(dst, file))


def delete_model(model_dir, step):
    if step == 0:
        return
    files = glob.glob(os.path.join(model_dir, f"{step:06d}*.ckpt"))
    try:
        for file in files:
            os.remove(file)
    except:
        print("Failed to delete old models.")


def get_sample_path(sample_dir, sample_id):
    return os.path.join(sample_dir, f"sample_{str(sample_id)}")


def delete_sample(sample_dir, eval_id):
    if not eval_id:
        return
    sample_path = get_sample_path(sample_dir, eval_id)
    try:
        shutil.rmtree(sample_path)
    except:
        print(f"Failed to delete dir: {sample_path}")


cache_dir = 'archive/cache'


def save_cache(data, name):
    os.makedirs(cache_dir, exist_ok=True)
    try:
        with open(os.path.join(cache_dir, name), 'wb') as f:
            pickle.dump(data, f)
    except:
        print(f"Failed to save cache: {name}")


def load_cache(name):
    with open(os.path.join(cache_dir, name), 'rb') as f:
        return pickle.load(f)


def exist_cache(name):
    return os.path.exists(os.path.join(cache_dir, name))


def safe_filename(unsafe, mark=''):
    if mark:
        unsafe = mark + "__" + unsafe
    unsafe = unsafe.replace('\\', '_')
    unsafe = unsafe.replace('/', '_')
    safe = unsafe.replace(':', '_')
    return safe
