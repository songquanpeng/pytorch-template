import torch
import os
from solver.misc import generate_samples
from metrics.fid import calculate_fid_given_paths
from utils.file import write_record


@torch.no_grad()
def calculate_metrics(nets, args, step):
    write_record(f"Calculating metrics for step {step}...", args.record_file)
    sample_path = os.path.join(args.eval_dir, f"step_{step}")
    generate_samples(nets, args, sample_path)
    calculate_fid(args, sample_path)


@torch.no_grad()
def calculate_fid(args, sample_path):
    fid_list = []
    for src_domain in args.domains:
        target_domains = [domain for domain in args.domains if domain != src_domain]
        for trg_domain in target_domains:
            task = f"{src_domain}2{trg_domain}"
            path_real = os.path.join(args.test_path, src_domain)
            path_fake = os.path.join(sample_path, task)
            print(f'Calculating FID for {task}...')
            fid = calculate_fid_given_paths(paths=[path_real, path_fake], img_size=args.img_size,
                                            batch_size=args.eval_batch_size)
            fid_list.append(fid)
            write_record(f"FID for {task}: {fid}", args.record_file)
    fid_mean = sum(fid_list) / len(fid_list)
    write_record(f"FID mean: {fid_mean}", args.record_file)
