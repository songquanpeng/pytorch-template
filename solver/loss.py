import torch
from munch import Munch
import torch.nn.functional as F


def compute_d_loss(nets, args, sample_org, sample_ref):
    # Real images
    sample_org.x.requires_grad_()
    out = nets.discriminator(sample_org.x, sample_org.y)
    loss_real = adv_loss(out, 1)

    # Fake images
    with torch.no_grad():
        s_ref = nets.mapping_network(sample_ref.z, sample_ref.y)
        x_fake = nets.generator(sample_org.x, s_ref)
    out = nets.discriminator(x_fake, sample_ref.y)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake
    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item())


def compute_g_loss(nets, args, sample_org, sample_ref):
    s_ref = nets.mapping_network(sample_ref.z, sample_ref.y)
    x_fake = nets.generator(sample_org.x, s_ref)
    out = nets.discriminator(x_fake, sample_ref.y)

    loss_adv = adv_loss(out, 1)

    loss = loss_adv
    return loss, Munch(adv=loss_adv.item())


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss
