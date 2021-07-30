import torch
import os
from solver.misc import generate_samples
from metrics.fid import calculate_fid_given_paths
from utils.file import write_record, delete_dir


@torch.no_grad()
def calculate_metrics(nets, args, step):
    write_record(f"Calculating metrics for step {step}...", args.record_file)
    sample_path = os.path.join(args.eval_dir, f"step_{step}")
    generate_samples(nets, args, sample_path)
    fid = calculate_fid(args, sample_path)
    if not args.keep_eval_files:
        delete_dir(sample_path)
    return fid


@torch.no_grad()
def calculate_fid(args, sample_path):
    fid_list = []
    for src_domain in args.domains:
        target_domains = [domain for domain in args.domains if domain != src_domain]
        for trg_domain in target_domains:
            task = f"{src_domain}2{trg_domain}"
            path_real = os.path.join(args.eval_path, src_domain)
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
def calculate_total_fid(nets_ema, args, eval_id):
    target_path = args.eval_path
    sample_path = os.path.join(args.sample_dir, str(eval_id))
    generate_samples(nets_ema, args, sample_path)
    fid = calculate_fid_given_paths(paths=[target_path, sample_path],
                                    img_size=args.img_size,
                                    batch_size=args.eval_batch_size,
                                    use_cache=args.eval_cache)
    if not args.keep_eval_files:
        delete_dir(sample_path)
    return fid
