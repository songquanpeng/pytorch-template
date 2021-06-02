from torch.utils import data
from PIL import Image

from utils.file import list_folder


class DefaultDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples = list_folder(root)
        self.samples.sort()
        self.transform = transform
        self.targets = None

    def __getitem__(self, index):
        fname = self.samples[index]
        img = Image.open(fname).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)
