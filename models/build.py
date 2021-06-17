import copy
from munch import Munch
from models.generator import Generator
from models.discriminator import Discriminator


def build_model(args):
    generator = Generator(args)
    discriminator = Discriminator(args)
    generator_ema = copy.deepcopy(generator)

    nets = Munch(generator=generator, discriminator=discriminator)
    nets_ema = Munch(generator=generator_ema)

    return nets, nets_ema
