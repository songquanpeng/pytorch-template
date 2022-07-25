import torch
import torch.nn.functional as F
from munch import Munch

lpips_loss_fn = None


def compute_d_loss(nets, args, sample_org, sample_ref):
    # Real images
    sample_org.x.requires_grad_()
    out = nets.discriminator(sample_org.x, sample_org.y)
    loss_real = adv_loss(out, 1)
    if args.lambda_r1:
        loss_r1 = r1_reg(out, sample_org.x)

    # Fake images
    with torch.no_grad():
        s_ref = nets.mapping_network(sample_ref.z, sample_ref.y)
        x_fake = nets.generator(sample_org.x, s_ref)
    out = nets.discriminator(x_fake, sample_ref.y)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake
    loss_item = Munch(real=loss_real.item(), fake=loss_fake.item())

    if args.lambda_r1:
        loss += args.lambda_r1 * loss_r1
        loss_item.r1 = loss_r1.item()

    return loss, loss_item


def compute_g_loss(nets, args, sample_org, sample_ref):
    s_ref = nets.mapping_network(sample_ref.z, sample_ref.y)
    x_fake = nets.generator(sample_org.x, s_ref)
    out = nets.discriminator(x_fake, sample_ref.y)

    loss_adv = adv_loss(out, 1)

    loss = loss_adv
    loss_item = Munch(adv=loss_adv.item())
    return loss, loss_item


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg


def calc_lpips_loss(args, x1, x2):
    global lpips_loss_fn
    if not lpips_loss_fn:
        import lpips
        lpips_loss_fn = lpips.LPIPS(net=args.which_lpips).cuda()
    return lpips_loss_fn(x1, x2).mean()
