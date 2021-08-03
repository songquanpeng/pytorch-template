import torch
import os
import time
import datetime

from munch import Munch
from utils.checkpoint import CheckpointIO
from utils.misc import get_datetime, send_message
from utils.model import print_network
from utils.file import delete_dir, write_record, delete_model
from models.build import build_model
from solver.utils import he_init, moving_average
from solver.misc import translate_using_latent, generate_samples
from solver.loss import compute_g_loss, compute_d_loss
from data.fetcher import Fetcher
from metrics.eval import calculate_metrics, calculate_total_fid
from metrics.fid import calculate_fid_given_paths


class Solver:
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device(args.device)
        self.nets, self.nets_ema = build_model(args)
        for name, module in self.nets.items():
            print_network(module, name)
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
                CheckpointIO(args.model_dir + '/{:06d}_nets.ckpt', **self.nets),
                CheckpointIO(args.model_dir + '/{:06d}_nets_ema.ckpt', **self.nets_ema),
                CheckpointIO(args.model_dir + '/{:06d}_optims.ckpt', **self.optims)]
        else:
            self.ckptios = [CheckpointIO(args.model_dir + '/{:06d}_nets_ema.ckpt', **self.nets_ema)]

        self.use_tensorboard = args.use_tensorboard
        if self.use_tensorboard:
            from utils.logger import Logger
            self.logger = Logger(args.log_dir)

    def initialize_parameters(self):
        if self.args.parameter_init == 'he':
            for name, network in self.nets.items():
                if 'ema' not in name:
                    print('Initializing %s...' % name, end=' ')
                    network.apply(he_init)
                    print('Done.')
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
        for optimizer in self.optims.values():
            optimizer.zero_grad()

    def train(self, loaders):
        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims

        train_fetcher = Fetcher(loaders.train, args)
        test_fetcher = Fetcher(loaders.test, args)

        # Those fixed samples are used to show the trend.
        fixed_train_sample = next(train_fetcher)
        fixed_test_sample = next(test_fetcher)
        if args.selected_path:
            # actually we don't care y
            fixed_selected_samples = next(iter(loaders.selected))
            fixed_selected_samples = fixed_selected_samples.to(self.device)

        # Load or initialize the model parameters.
        if args.start_iter > 0:
            self.load_model(args.start_iter)
        else:
            self.initialize_parameters()

        best_fid = 10000
        best_step = 0
        print('Start training...')
        start_time = time.time()
        for step in range(args.start_iter + 1, args.end_iter + 1):
            sample_org = next(train_fetcher)  # sample that to be translated
            sample_ref = next(train_fetcher)  # reference samples

            # Train the discriminator
            d_loss, d_loss_ref = compute_d_loss(nets, args, sample_org, sample_ref)
            self.zero_grad()
            d_loss.backward()
            optims.discriminator.step()

            # Train the generator
            g_loss, g_loss_ref = compute_g_loss(nets, args, sample_org, sample_ref)
            self.zero_grad()
            g_loss.backward()
            optims.generator.step()
            optims.mapping_network.step()

            # Update generator_ema
            moving_average(nets.generator, nets_ema.generator, beta=args.ema_beta)
            moving_average(nets.mapping_network, nets_ema.mapping_network, beta=args.ema_beta)

            if step % args.log_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "[%s]-[%i/%i]: " % (elapsed, step, args.end_iter)
                all_losses = dict()
                for loss, prefix in zip([d_loss_ref, g_loss_ref], ['D/', 'G/']):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                print(log)
                if args.save_loss:
                    if step == args.log_every:
                        header = ','.join(['iter'] + [str(loss) for loss in all_losses.keys()])
                        write_record(header, args.loss_file, False)
                    log = ','.join([str(step)] + [str(loss) for loss in all_losses.values()])
                    write_record(log, args.loss_file, False)
                if self.use_tensorboard:
                    for tag, value in all_losses.items():
                        self.logger.scalar_summary(tag, value, step)

            if step % args.sample_every == 0:
                repeat_num = 2
                N = args.batch_size
                y_trg_list = [torch.tensor(y).repeat(N).to(self.device) for y in range(min(args.num_domains, 5))]
                z_trg_list = torch.randn(repeat_num, 1, args.latent_dim).repeat(1, N, 1).to(self.device)
                translate_using_latent(nets, args, fixed_test_sample.x, y_trg_list, z_trg_list,
                                       os.path.join(args.sample_dir, f"latent_test_{step}.jpg"))
                translate_using_latent(nets, args, fixed_train_sample.x, y_trg_list, z_trg_list,
                                       os.path.join(args.sample_dir, f"latent_train_{step}.jpg"))
                if args.selected_path:
                    N = fixed_selected_samples.shape[0]
                    y_trg_list = [torch.tensor(y).repeat(N).to(self.device) for y in range(min(args.num_domains, 5))]
                    z_trg_list = torch.randn(repeat_num, 1, args.latent_dim).repeat(1, N, 1).to(self.device)
                    translate_using_latent(nets, args, fixed_selected_samples, y_trg_list, z_trg_list,
                                           os.path.join(args.sample_dir, f"latent_selected_{step}.jpg"))

            if step % args.save_every == 0:
                self.save_model(step)
                last_step = step - args.save_every
                if last_step != best_step and not args.keep_all_models:
                    delete_model(args.model_dir, last_step)

            if step % args.eval_every == 0:
                fid = calculate_total_fid(nets_ema, args, f"step_{step}")
                if fid < best_fid:
                    if not args.keep_all_models:
                        delete_model(args.model_dir, best_step)
                    best_fid = fid
                    best_step = step
                info = f"step: {step:.2f} current fid: {fid:.2f} history best fid: {best_fid:.2f}"
                send_message(info, args.exp_id)
                write_record(info, args.record_file)
        send_message("Model training completed.")

    @torch.no_grad()
    def sample(self):
        args = self.args
        assert args.eval_iter != 0
        self.load_model(args.eval_iter)
        nets_ema = self.nets_ema
        if not args.sample_id:
            args.sample_id = get_datetime()
        sample_path = os.path.join(args.sample_dir, args.sample_id)
        generate_samples(nets_ema, args, sample_path)
        return sample_path

    @torch.no_grad()
    def evaluate(self):
        args = self.args
        assert args.eval_path != "", "eval_path shouldn't be empty"
        target_path = args.eval_path
        sample_path = self.sample()
        fid = calculate_fid_given_paths(paths=[target_path, sample_path], img_size=args.img_size,
                                        batch_size=args.eval_batch_size)
        print(f"FID is: {fid}")
        send_message(f"Sample {args.sample_id}'s FID is {fid}")
        if not args.keep_eval_files:
            delete_dir(sample_path)
