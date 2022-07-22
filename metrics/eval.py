import os

import torch

from metrics.fid import calculate_fid_given_paths
from solver.misc import generate_samples
from utils.file import write_record, delete_dir, get_sample_path


@torch.no_grad()
def calculate_metrics(nets, args, step, keep_samples=False):
    write_record(f"Calculating metrics for step {step}...", args.record_file)
    sample_path = get_sample_path(args.eval_dir, step)
    generate_samples(nets, args, sample_path)
    fid = calculate_fid(args, sample_path)
    if not keep_samples:
        delete_dir(sample_path)
    return fid


@torch.no_grad()
def calculate_fid(args, sample_path):
    fid_list = []
    for src_domain in args.domains:
        target_domains = [domain for domain in args.domains if domain != src_domain]
        for trg_domain in target_domains:
            task = f"{src_domain}2{trg_domain}"
            path_real = os.path.join(args.compare_path, src_domain)
            path_fake = os.path.join(sample_path, task)
            print(f'Calculating FID for {task}...')
            fid = calculate_fid_given_paths(paths=[path_real, path_fake], img_size=args.img_size,
                                            batch_size=args.eval_batch_size, use_cache=args.eval_cache)
            fid_list.append(fid)
            write_record(f"FID for {task}: {fid}", args.record_file)
    fid_mean = sum(fid_list) / len(fid_list)
    write_record(f"FID mean: {fid_mean}", args.record_file)
    return fid_mean


@torch.no_grad()
def calculate_total_fid(nets_ema, args, step, keep_samples=False):
    target_path = args.compare_path
    sample_path = get_sample_path(args.eval_dir, step)
    generate_samples(nets_ema, args, sample_path)
    fid = calculate_fid_given_paths(paths=[target_path, sample_path],
                                    img_size=args.img_size,
                                    batch_size=args.eval_batch_size,
                                    use_cache=args.eval_cache)
    if not keep_samples:
        delete_dir(sample_path)
    return fid
