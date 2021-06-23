import torch
import torch.nn as nn


class MappingNetwork(nn.Module):
    def __init__(self, args):
        super().__init__()
        latent_dim, style_dim, num_domains = args.latent_dim, args.style_dim, args.num_domains
        layers = []
        layers += [nn.Linear(latent_dim, style_dim)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(style_dim, style_dim)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Sequential(nn.Linear(style_dim, style_dim), nn.ReLU(),
                                            nn.Linear(style_dim, style_dim), nn.ReLU(),
                                            nn.Linear(style_dim, style_dim), nn.ReLU(),
                                            nn.Linear(style_dim, style_dim))]

    def forward(self, z, y):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s
