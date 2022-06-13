import os

import numpy as np
from PIL import Image
from torch.utils import data

from utils.file import list_all_images, list_sub_folders, exist_cache, load_cache, save_cache, safe_filename


class DefaultDataset(data.Dataset):
    """ No label. """

    def __init__(self, root, transform=None):
        self.samples = list_all_images(root)
        self.samples.sort()
        self.transform = transform

    def load_image(self, path):
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __getitem__(self, index):
        return self.load_image(self.samples[index])

    def __len__(self):
        return len(self.samples)


class FolderDataset(data.Dataset):
    """ Deprecated, use torchvision.datasets.ImageFolder instead. """

    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples = []
        self.targets = []
        self.classes = list_sub_folders(root)

        for i, class_ in enumerate(self.classes):
            filenames = list_all_images(class_)
            class_samples = filenames
            self.targets.extend([i] * len(class_samples))
            self.samples.extend(class_samples)

    def load_image(self, path):
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __getitem__(self, index):
        return self.load_image(self.samples[index]), self.targets[index]

    def __len__(self):
        return len(self.samples)


class NpzDataset(data.Dataset):
    """
    Deprecated, reference purpose only now.
    Sometimes we need more information, not only image and its corresponding label.
    We can use a npz file to offer this information.
    This file should have the following keys:
    1. samples: the path array of images, should be relative path (image_root).
    2. labels: corresponding labels.

    Notice: please make sure the order is correct among those attributes.
    """

    def __init__(self, npz_path, npz_image_root, transform=None):
        self.image_root = npz_image_root
        self.transform = transform
        npz_data = np.load(npz_path, allow_pickle=True)
        self.samples = npz_data["samples"]
        self.targets = npz_data["labels"]

    def __getitem__(self, index):
        sub_path = self.samples[index]
        path = os.path.join(self.image_root, sub_path)
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[index]

    def __len__(self):
        return len(self.samples)
