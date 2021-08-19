import numpy as np
import torch.nn as nn

from models.layers import ResBlk, AdainResBlk


class Generator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        img_size = args.img_size
        dim_in = 2 ** 14 // img_size
        max_conv_dim = 512
        dim_out = None
        self.img_size = img_size
        self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 3, 1, 1, 0))

        repeat_num = int(np.log2(img_size)) - 4
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            self.encode.append(ResBlk(dim_in, dim_out, normalize=True, down_sample=True))
            self.decode.insert(0, AdainResBlk(dim_out, dim_in, style_dim=args.style_dim, up_sample=True))
            dim_in = dim_out

        for _ in range(2):
            self.encode.append(ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(0, AdainResBlk(dim_out, dim_out, style_dim=args.style_dim))

    def forward(self, x, s):
        x = self.from_rgb(x)
        for block in self.encode:
            x = block(x)
        for block in self.decode:
            x = block(x, s)
        x = self.to_rgb(x)
        return x
