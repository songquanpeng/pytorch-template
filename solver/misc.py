import torch
import os
import copy
from utils.image import save_image
from data.loader import get_eval_loader
from tqdm import tqdm
from utils.file import make_path


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


@torch.no_grad()
def translate_using_label(nets, args, x_src, y_trg_list, filename):
    x_concat = [x_src]
    for y_trg in y_trg_list:
        x_fake = nets.generator(x_src, y_trg)
        x_concat += [x_fake]
    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, x_src.size()[0], filename)


@torch.no_grad()
def generate_samples(nets, args, path):
    args = copy.deepcopy(args)
    args.batch_size = args.eval_batch_size
    for src_idx, src_domain in enumerate(args.domains):
        loader = get_eval_loader(path=os.path.join(args.test_path, src_domain), **args)
        target_domains = [domain for domain in args.domains]
        for trg_idx, trg_domain in enumerate(target_domains):
            if trg_domain == src_domain:
                continue
            save_path = os.path.join(path, f"{src_domain}2{trg_domain}")
            make_path(save_path)
            for i, query_image in enumerate(tqdm(loader, total=len(loader))):
                N = query_image.size(0)
                query_image = query_image.to(args.device)
                class_label = torch.tensor([trg_idx] * N).to(args.device)
                images = []
                for j in range(args.eval_repeat_num):
                    latent_code = torch.randn(N, args.latent_dim).to(args.device)
                    style_code = nets.mapping_network(latent_code, class_label)
                    generated_image = nets.generator(query_image, style_code)
                    images.append(generated_image)
                    for k in range(N):
                        filename = os.path.join(save_path, f"{i * args.eval_batch_size + k}_{j}.png")
                        save_image(generated_image[k], col_num=1, filename=filename)
