import torch
import os
import time
import datetime

from munch import Munch
from utils.file import prepare_dirs
from utils.checkpoint import CheckpointIO
from utils.logger import Logger
from models.build import build_model
from solver.utils import he_init, moving_average, requires_grad
from solver.loss import compute_g_loss, compute_d_loss
from data.fetcher import Fetcher


class Solver:
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device(args.device)

        # Directories
        self.log_dir = os.path.join(args.exp_dir, args.exp_id, "logs")
        self.sample_dir = os.path.join(args.exp_dir, args.exp_id, "samples")
        self.model_save_dir = os.path.join(args.exp_dir, args.exp_id, "models")
        self.result_dir = os.path.join(args.exp_dir, args.exp_id, "results")
        prepare_dirs([self.log_dir, self.sample_dir, self.model_save_dir, self.result_dir])

        self.nets, self.nets_ema = build_model(args)
        # self.to(self.device)
        for net in self.nets.values():
            net.to(self.device)
        for net in self.nets_ema.values():
            net.to(self.device)

        if args.mode == 'train':
            # Setup optimizers for all nets to learn.
            self.optims = Munch()
            for net in self.nets.keys():
                self.optims[net] = torch.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr=args.d_lr if net == 'discriminator' else args.lr,
                    betas=(args.beta1, args.beta2),
                    weight_decay=args.weight_decay)
            self.ckptios = [
                CheckpointIO(self.model_save_dir + '/{:06d}_nets.ckpt', **self.nets),
                CheckpointIO(self.model_save_dir + '/{:06d}_nets_ema.ckpt', **self.nets_ema),
                CheckpointIO(self.model_save_dir + '/{:06d}_optims.ckpt', **self.optims)]
        else:
            self.ckptios = [CheckpointIO(self.model_save_dir + '/{:06d}_nets_ema.ckpt', **self.nets_ema)]

        self.use_tensorboard = args.use_tensorboard
        if self.use_tensorboard:
            self.logger = Logger(self.log_dir)

    def initialize_parameters(self):
        if self.args.parameter_init == 'he':
            for name, network in self.nets.items():
                if 'ema' not in name:
                    print('Initializing %s...' % name)
                    network.apply(he_init)
        elif self.args.parameter_init == 'default':
            # Do nothing because the parameters has been initialized in this manner.
            pass

    def save_model(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)

    def load_model(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    def load_model_from_path(self, path):
        for ckptio in self.ckptios:
            ckptio.load_from_path(path)

    def zero_grad(self):
        for optimizer in self.optims:
            optimizer.zero_grad()

    def train(self, loaders):
        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims

        train_fetcher = Fetcher(loaders.train, args.dataset, args.device)
        test_fetcher = Fetcher(loaders.test, args.dataset, args.device)

        # Those fixed samples are used to show the trend.
        fixed_train_sample = next(train_fetcher)
        fixed_test_sample = next(test_fetcher)

        # Load or initialize the model parameters.
        if args.start_iter > 0:
            self.load_model(args.start_iter)
        else:
            self.initialize_parameters()

        print('Start training...')
        start_time = time.time()
        for i in range(args.start_iter, args.end_iter):
            sample = next(train_fetcher)

            # Train the discriminator
            d_loss, d_loss_ref = compute_d_loss(nets, args, sample)
            self.zero_grad()
            d_loss.backward()
            optims.discriminator.step()

            # Train the generator
            g_loss, g_loss_ref = compute_g_loss(nets, args, sample)
            self.zero_grad()
            g_loss.backward()
            optims.generator.step()

            # Update generator_ema
            moving_average(nets.generator, nets_ema.generator, beta=args.ema_beta)

            if (i + 1) % args.log_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "[%s][%i/%i]: " % (elapsed, i + 1, args.total_iters)
                all_losses = dict()
                for loss, prefix in zip([d_loss_ref, g_loss_ref], ['D/', 'G/']):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                print(log)
                if self.use_tensorboard:
                    for tag, value in all_losses.items():
                        self.logger.scalar_summary(tag, value, i + 1)

            if (i + 1) % args.sample_every == 0:
                # TODO: Save the sampled images to self.sample_dir
                pass

            if (i + 1) % args.save_every == 0:
                self.save_model(i + 1)

            if (i + 1) % args.eval_every == 0:
                # TODO: Evaluate the model
                pass

    @torch.no_grad()
    def sample(self, loaders):
        args = self.args
        nets_ema = self.nets_ema
        # TODO: Use the trained model to sample
        pass

    @torch.no_grad()
    def evaluate(self):
        assert self.args.use_iter != 0
        self.load_model(self.args.use_iter)
        # TODO: Evaluate the model
        pass
