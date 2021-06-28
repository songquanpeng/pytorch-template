import os
import numpy as np
from torch.utils import data
from PIL import Image
from utils.file import list_all_images, list_sub_folders


class DefaultDataset(data.Dataset):
    """ No label. """

    def __init__(self, root, transform=None):
        self.samples = list_all_images(root)
        self.samples.sort()
        self.transform = transform

    def __getitem__(self, index):
        fname = self.samples[index]
        img = Image.open(fname).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)


class NpzDataset(data.Dataset):
    """
    Sometimes we need more information, not only image and its corresponding label.
    We can use a npz file to offer those information.
    This file should has following keys:
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
