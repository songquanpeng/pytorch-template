from torch.utils import data
from PIL import Image
from torchvision.datasets import ImageFolder

from utils.file import list_all_images, list_sub_folders


class DefaultDataset(data.Dataset):
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


class ClassFolderDataset(data.Dataset):
    def __init__(self, args, transform=None):
        class_folders = list_sub_folders(args.dataset_path)
        assert len(class_folders) == args.num_domains
        self.samples = []
        self.labels = []
        for i, class_folder in enumerate(class_folders):
            images = list_all_images(class_folder)
            self.samples.append(images)
            self.labels.append([i] * len(images))

        self.transform = transform

    def __getitem__(self, index):
        fname = self.samples[index]
        img = Image.open(fname).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[index]

    def __len__(self):
        return len(self.samples)
