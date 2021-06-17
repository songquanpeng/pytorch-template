import torch
from munch import Munch
import torch.nn.functional as F


def compute_d_loss(nets, args, sample):
    # Real images
    sample.x.requires_grad_()
    out = nets.discriminator(sample.x)
    loss_real = adv_loss(out, 1)

    # Fake images
    with torch.no_grad():
        x_fake = nets.generator(sample.x)
    out = nets.discriminator(x_fake)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake
    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item())


def compute_g_loss(nets, args, sample):
    x_fake = nets.generator(sample.x)
    out = nets.discriminator(x_fake)

    loss_adv = adv_loss(out, 1)

    loss = loss_adv
    return loss, Munch(adv=loss_adv.item())


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss
