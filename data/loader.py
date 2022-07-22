import numpy as np
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

from data.dataset import DefaultDataset
from data.dataset import FolderDataset as ImageFolder
from utils.file import list_all_images


def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))


def get_train_loader(train_path, img_size, batch_size, dataset, num_workers=4, **kwargs):
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    if dataset == 'CelebA':
        transform = transforms.Compose([
            transforms.Resize([img_size, img_size]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])
    elif dataset == 'CUB2011':
        transform = transforms.Compose([
            transforms.Resize(int(img_size * 76 / 64)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)])
    else:
        assert False, f"Unsupported dataset: {dataset}"

    dataset = ImageFolder(root=train_path, transform=transform)
    sampler = _make_balanced_sampler(dataset.targets)

    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=True,
                           sampler=sampler,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=True)


def get_test_loader(test_path, img_size, batch_size, dataset=None, num_workers=4, **kwargs):
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])
    dataset = ImageFolder(root=test_path, transform=transform)

    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers,
                           pin_memory=True)


def get_eval_loader(path, img_size, batch_size, dataset=None, num_workers=4, **kwargs):
    # Path should be an image folder without sub folders.
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])
    dataset = DefaultDataset(root=path, transform=transform)

    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=False)


def get_selected_loader(selected_path, img_size, dataset=None, num_workers=0, **kwargs):
    # Path should be an image folder without sub folders.
    batch_size = len(list_all_images(selected_path))
    assert batch_size < 64, "too many selected images!"
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])
    dataset = DefaultDataset(root=selected_path, transform=transform)

    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=num_workers,
                           pin_memory=True)
