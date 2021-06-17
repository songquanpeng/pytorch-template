import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import ResBlk


class Generator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        img_size = args.img_size
        dim_in = 2 ** 14 // img_size
        max_conv_dim = 512
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
            self.decode.append(ResBlk(dim_out, dim_in, normalize=True, down_sample=False))

    def forward(self, x):
        x = self.from_rgb(x)
        for block in self.encode:
            x = block(x)
        for block in self.decode:
            x = block(x)
        x = self.to_rgb(x)
        return x
