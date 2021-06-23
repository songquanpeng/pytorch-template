import os
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

import numpy as np

from data.dataset import DefaultDataset, ClassFolderDataset


def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))


def get_train_loader(args):
    img_size = args.img_size
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    if args.dataset == 'CelebA':
        transform = transforms.Compose([
            transforms.Resize([img_size, img_size]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])
    elif args.dataset == 'CUB2011':
        transform = transforms.Compose([
            transforms.Resize(int(img_size * 76 / 64)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)])
    else:
        assert False, f"Unsupported dataset: {args.dataset}"

    dataset = ImageFolder(root=os.path.join(args.dataset_path, 'train'), transform=transform)
    sampler = _make_balanced_sampler(dataset.targets)

    return data.DataLoader(dataset=dataset,
                           batch_size=args.batch_size,
                           sampler=sampler,
                           num_workers=args.num_workers,
                           pin_memory=True,
                           drop_last=True)


def get_test_loader(args):
    img_size = args.img_size
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])
    dataset = ImageFolder(root=os.path.join(args.dataset_path, 'val'), transform=transform)

    return data.DataLoader(dataset=dataset,
                           batch_size=args.batch_size,
                           shuffle=True,
                           num_workers=args.num_workers,
                           pin_memory=True)
