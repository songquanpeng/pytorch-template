import os
import numpy as np
from torch.utils import data
from PIL import Image
from utils.file import list_all_images, list_sub_folders, exist_cache, load_cache, save_cache, safe_filename


class DefaultDataset(data.Dataset):
    """ No label. """

    def __init__(self, root, transform=None, in_memory=False, use_cache=False):
        self.samples = list_all_images(root)
        self.samples.sort()
        self.transform = transform
        self.in_memory = in_memory
        if in_memory:
            print("Loading dataset into memory...")
            cache_name = safe_filename(root, 'DefaultDataset')
            cache_available = use_cache and exist_cache(cache_name)
            if cache_available:
                print('Loading cache...')
                cache = load_cache(cache_name)
                self.samples = cache['samples']
            else:
                for i, path in enumerate(self.samples):
                    self.samples[i] = self.load_image(path)
                if use_cache and not exist_cache(cache_name):
                    print('Saving cache...')
                    cache = {'samples': self.samples}
                    save_cache(cache, cache_name)
            print('Done.')

    def load_image(self, path):
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __getitem__(self, index):
        if self.in_memory:
            return self.samples[index]
        else:
            return self.load_image(self.samples[index])

    def __len__(self):
        return len(self.samples)


class FolderDataset(data.Dataset):
    """ torchvision.datasets.ImageFolder with in memory option """

    def __init__(self, root, transform=None, in_memory=False, use_cache=False):
        self.transform = transform
        self.in_memory = in_memory
        self.samples = []
        self.targets = []
        self.classes = list_sub_folders(root)
        if in_memory:
            print("Loading dataset into memory...")
        cache_name = safe_filename(root, 'FolderDataset')
        cache_available = use_cache and exist_cache(cache_name)
        if cache_available:
            print('Loading cache...')
            cache = load_cache(cache_name)
            self.samples = cache['samples']
            self.targets = cache['targets']
        else:
            for i, class_ in enumerate(self.classes):
                filenames = list_all_images(class_)
                if in_memory:
                    class_samples = []
                    for filename in filenames:
                        class_samples.append(self.load_image(filename))
                else:
                    class_samples = filenames
                self.targets.extend([i] * len(class_samples))
                self.samples.extend(class_samples)
            if use_cache and not exist_cache(cache_name):
                print('Saving cache...')
                cache = {'samples': self.samples, 'targets': self.targets}
                save_cache(cache, cache_name)

        if in_memory:
            print("Dataset loading done.")

    def load_image(self, path):
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __getitem__(self, index):
        if self.in_memory:
            return self.samples[index], self.targets[index]
        else:
            return self.load_image(self.samples[index]), self.targets[index]

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
