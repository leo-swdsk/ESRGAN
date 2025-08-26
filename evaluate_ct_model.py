import os
import csv
import json
import random
import argparse
import math
from collections import defaultdict

import torch

from rrdb_ct_model import RRDBNet_CT
from ct_dataset_loader import CT_Dataset_SR
from ct_sr_evaluation import compare_methods


def get_patient_dirs(root_folder):
    dirs = [os.path.join(root_folder, d) for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]
    return sorted(dirs)


def split_patients(root_folder, seed=42):
    patient_dirs = get_patient_dirs(root_folder)
    if len(patient_dirs) == 0:
        raise RuntimeError(f"No patient directories found under {root_folder}")
    random.seed(seed)
    random.shuffle(patient_dirs)
    n = len(patient_dirs)
    train_cut = int(0.70 * n)
    val_cut = int(0.85 * n)
    return {
        'train': patient_dirs[:train_cut],
        'val': patient_dirs[train_cut:val_cut],
        'test': patient_dirs[val_cut:]
    }


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def evaluate_split(root_folder, split_name, model_path, output_dir, device='cuda', scale=2, preset='soft_tissue',
                   max_patients=None, max_slices_per_patient=None, slice_sampling='random', seed=42):
    ensure_dir(output_dir)

    # Prepare model
    device_t = torch.device(device if (device == 'cuda' and torch.cuda.is_available()) else 'cpu')
    model = RRDBNet_CT(scale=scale).to(device_t)
    model.load_state_dict(torch.load(model_path, map_location=device_t))
    model.eval()

    # Determine patients for the requested split
    splits = split_patients(root_folder)
    patient_dirs = splits.get(split_name)
    if patient_dirs is None:
        raise ValueError(f"split_name must be one of train|val|test, got: {split_name}")
    if max_patients is not None:
        patient_dirs = patient_dirs[:max_patients]

    # Output files
    csv_path = os.path.join(output_dir, f"metrics_{split_name}.csv")
    json_path = os.path.join(output_dir, f"summary_{split_name}.json")

    # Collect per-slice metrics and per-patient aggregations
    fieldnames = ['patient_id', 'slice_index', 'method', 'MSE', 'RMSE', 'MAE', 'PSNR', 'SSIM']  # MAE hinzugefügt
    rows = []
    patient_to_method_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    rng = random.Random(seed)

    with torch.no_grad():
        for p_idx, patient_dir in enumerate(patient_dirs):
            patient_id = os.path.basename(patient_dir)

            # Full-slice dataset for evaluation (no random crop)
            ds = CT_Dataset_SR(patient_dir, scale_factor=scale, do_random_crop=False, normalization='global',)
            num_slices = len(ds)
            limit_slices = min(num_slices, max_slices_per_patient) if max_slices_per_patient else num_slices

            # choose slice indices according to sampling strategy
            if limit_slices >= num_slices:
                indices = list(range(num_slices))
            else:
                if slice_sampling == 'first':
                    indices = list(range(limit_slices))
                else:  # 'random'
                    indices = rng.sample(range(num_slices), k=limit_slices)
                    indices.sort()

            for s_idx in indices:
                lr, hr = ds[s_idx]
                results = compare_methods(lr, hr, model)

                for method_name, metrics in results.items():
                    rows.append({
                        'patient_id': patient_id,
                        'slice_index': s_idx,
                        'method': method_name,
                        'MSE': float(metrics['MSE']),
                        'RMSE': float(metrics['RMSE']),
                        'PSNR': float(metrics['PSNR']),
                        'SSIM': float(metrics['SSIM'])
                    })
                    for metric_name, metric_value in metrics.items():
                        patient_to_method_metrics[patient_id][method_name][metric_name].append(float(metric_value))

    # Write per-slice CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Aggregate statistics
    def mean_std(values):
        finite_vals = [v for v in values if math.isfinite(v)]
        if len(finite_vals) == 0:
            return 0.0, 0.0
        m = sum(finite_vals) / len(finite_vals)
        var = sum((v - m) ** 2 for v in finite_vals) / max(1, (len(finite_vals) - 1))
        return m, var ** 0.5

    # Per-slice global aggregation
    global_by_method = defaultdict(lambda: defaultdict(list))
    for r in rows:
        method_name = r['method']
        for metric_name in ['MSE', 'RMSE', 'MAE', 'PSNR', 'SSIM']:  # MAE hinzugefügt
            global_by_method[method_name][metric_name].append(float(r[metric_name]))

    global_summary = {}
    for method_name, metric_map in global_by_method.items():
        global_summary[method_name] = {}
        for metric_name, values in metric_map.items():
            m, s = mean_std(values)
            global_summary[method_name][metric_name] = {'mean': m, 'std': s, 'n': len(values)}

    # Per-patient aggregation (each patient contributes one mean per metric)
    per_patient_summary = {}
    for patient_id, method_map in patient_to_method_metrics.items():
        per_patient_summary[patient_id] = {}
        for method_name, metric_map in method_map.items():
            per_patient_summary[patient_id][method_name] = {}
            for metric_name, values in metric_map.items():
                m, s = mean_std(values)
                per_patient_summary[patient_id][method_name][metric_name] = {'mean': m, 'std': s, 'n': len(values)}

    # Aggregate across patients (each patient weighted equally)
    patient_level_agg = {}
    methods = set()
    for patient_id in per_patient_summary:
        methods.update(per_patient_summary[patient_id].keys())
    for method_name in methods:
        patient_level_agg[method_name] = {}
        for metric_name in ['MSE', 'RMSE', 'MAE', 'PSNR', 'SSIM']:  # MAE hinzugefügt
            patient_means = []
            for patient_id in per_patient_summary:
                if method_name in per_patient_summary[patient_id]:
                    if metric_name in per_patient_summary[patient_id][method_name]:
                        patient_means.append(per_patient_summary[patient_id][method_name][metric_name]['mean'])
            m, s = mean_std(patient_means)
            patient_level_agg[method_name][metric_name] = {'mean_of_patient_means': m, 'std_of_patient_means': s, 'num_patients': len(patient_means)}

    # Write JSON summary
    summary = {
        'split': split_name,
        'num_patients': len(patient_dirs),
        'paths': {
            'csv': csv_path,
            'json': json_path
        },
        'global_per_slice': global_summary,
        'per_patient': per_patient_summary,
        'patient_level_aggregate': patient_level_agg
    }
    # Sanitize JSON (replace inf/nan with strings) for strict JSON compatibility
    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        if isinstance(obj, float):
            if math.isfinite(obj):
                return obj
            if math.isinf(obj):
                return 'inf'
            return 'nan'
        return obj

    with open(json_path, 'w') as f:
        json.dump(sanitize(summary), f, indent=2, allow_nan=False)

    print(f"[Eval] Wrote per-slice CSV to: {csv_path}")
    print(f"[Eval] Wrote summary JSON to: {json_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate CT SR model with aggregated metrics and summaries')
    parser.add_argument('--root', type=str, required=True, help='Root folder containing patient subfolders (same as training root)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help='Which split to evaluate')
    parser.add_argument('--model_path', type=str, default='rrdb_ct_best.pth', help='Path to trained model weights')
    parser.add_argument('--output_dir', type=str, default='eval_results', help='Directory to write CSV/JSON outputs')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--scale', type=int, default=2, help='Upsampling scale (must match model)')
    parser.add_argument('--preset', type=str, default='soft_tissue', help='Window preset (for completeness)')
    parser.add_argument('--max_patients', type=int, default=None, help='Optional limit of patients for a quick run')
    parser.add_argument('--max_slices_per_patient', type=int, default=None, help='Optional limit of slices per patient for a quick run')
    parser.add_argument('--slice_sampling', type=str, default='random', choices=['first', 'random'], help='How to select slices when limited')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible sampling')
    args = parser.parse_args()

    evaluate_split(
        root_folder=args.root,
        split_name=args.split,
        model_path=args.model_path,
        output_dir=args.output_dir,
        device=args.device,
        scale=args.scale,
        preset=args.preset,
        max_patients=args.max_patients,
        max_slices_per_patient=args.max_slices_per_patient,
        slice_sampling=args.slice_sampling,
        seed=args.seed
    )


if __name__ == '__main__':
    main()


