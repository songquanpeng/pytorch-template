import torch
from munch import Munch


class Fetcher:
    def __init__(self, loader, args):
        self.loader = loader
        self.device = torch.device(args.device)
        self.latent_dim = args.latent_dim
        self.generate_noise = args.mode == 'train'

    def __iter__(self):
        return self

    def __next__(self):
        try:
            x, y = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y = next(self.iter)
        if self.generate_noise:
            z = torch.randn(x.size(0), self.latent_dim)
            inputs = Munch(x=x, y=y, z=z)
        else:
            inputs = Munch(x=x, y=y)

        return Munch({k: v.to(self.device) for k, v in inputs.items()})
