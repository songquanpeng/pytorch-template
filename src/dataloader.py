import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms

import numpy as np

from src.dataset import DefaultDataset


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
        crop = transforms.RandomResizedCrop(
            img_size, scale=[0.8, 1.0], ratio=[0.9, 1.1])
        rand_crop = transforms.Lambda(lambda x: crop(x) if random.random() < args.randcrop_prob else x)

        transform = transforms.Compose([
            rand_crop,
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

    dataset = DefaultDataset(args.dataset_path, transform)
    # sampler = _make_balanced_sampler(dataset.train_labels)

    return data.DataLoader(dataset=dataset,
                           batch_size=args.batch_size,
                           # sampler=sampler,
                           num_workers=args.num_workers,
                           pin_memory=True,
                           drop_last=True)


def get_test_loader(args):
    pass
