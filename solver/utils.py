import torch
from torch import nn
from utils.image import save_image


def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def moving_average(model, model_ema, beta):
    for param, param_ema in zip(model.parameters(), model_ema.parameters()):
        param_ema.data = torch.lerp(param.data, param_ema.data, beta)


def requires_grad(model, flag=True):
    for parameter in model.parameters():
        parameter.requires_grad = flag


@torch.no_grad()
def translate_using_latent(nets, args, x_src, y_trg_list, z_trg_list, filename):
    x_concat = [x_src]
    for y_trg in y_trg_list:
        for z_trg in z_trg_list:
            s_trg = nets.mapping_network(z_trg, y_trg)
            x_fake = nets.generator(x_src, s_trg)
            x_concat += [x_fake]
    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, x_src.size()[0], filename)
