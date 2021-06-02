import os
import glob


def list_folder(path):
    image_types = ('png', 'jpg', 'jpeg')
    image_list = []
    for image_type in image_types:
        image_list.extend(glob.glob(os.path.join(path, f"**/*.{image_type}"), recursive=True))
    image_list = [os.path.relpath(image, path) for image in image_list]
    image_list = [p.replace("\\", '/') for p in image_list]
    return image_list


def make_paths(paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
