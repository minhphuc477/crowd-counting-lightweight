import os
import argparse
import json
import yaml
import numpy as np
import torch
from typing import Dict, Any, List

from hpc.data.sha import ShanghaiTechDataset
from hpc.data.qnrf import UCFQNRFDataset
from hpc.data.nwpu import NWPUDataset
from hpc.data.sampler import compute_image_density_and_luminance
from hpc.models.hpc_lite import inv_softplus
from hpc.targets.block_counts import build_integer_block_counts


def compute_statistics_for_dataset(config_path: str, output_json: str):
    """Compute and save training-only dataset statistics."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    ds_cfg = cfg["dataset"]
    ds_name = ds_cfg["name"].lower()
    root = ds_cfg["root"]
    crop_size = ds_cfg["crop_size"]
    hnb_blocks = ds_cfg["hnb_blocks"]
    alloc_block = ds_cfg.get("allocation_block", 16)
    out_stride = ds_cfg.get("output_stride", 4)
    
    print(f"Loading training split for dataset: {ds_name} at {root}...")
    if "sha" in ds_name or "shb" in ds_name:
        part = ds_cfg.get("part", "part_A")
        ds = ShanghaiTechDataset(root=root, part=part, split="train_data", crop_size=crop_size, is_train=False)
    elif "qnrf" in ds_name:
        ds = UCFQNRFDataset(root=root, split="Train", crop_size=crop_size, is_train=False)
    elif "nwpu" in ds_name:
        ds = NWPUDataset(root=root, split="train", crop_size=crop_size, is_train=False)
    else:
        raise ValueError(f"Unknown dataset {ds_name}")
        
    image_paths = ds.image_paths
    points_list = ds.points_list
    n_images = len(image_paths)
    print(f"Found {n_images} training images.")
    
    if n_images == 0:
        print("Warning: No images found. Generating placeholder statistics.")
        stats = {
            "dataset": ds_name,
            "crop_size": crop_size,
            "mean_crop_count": 50.0,
            "initial_head_bias": inv_softplus(50.0 / ((crop_size // out_stride) ** 2)),
            "hnb_scales": {str(b): {"mean": 1.0, "var": 2.0, "r0": 1.0, "q50_pos": 1.0, "q90_pos": 5.0} for b in hnb_blocks},
            "density_quantiles": [1.0, 2.0, 3.0, 4.0],
            "luminance_quantiles": [50.0, 100.0, 150.0],
        }
        with open(output_json, "w") as f:
            json.dump(stats, f, indent=2)
        return stats
        
    counts = []
    densities = []
    luminances = []
    
    scale_counts: Dict[int, List[float]] = {b: [] for b in hnb_blocks}
        
    for path, pts in zip(image_paths, points_list):
        d_val, l_val, c_val = compute_image_density_and_luminance(path, pts)
        counts.append(c_val)
        densities.append(d_val)
        luminances.append(l_val)
        
        # Approximate crop count statistics (simulated crops)
        for b in hnb_blocks:
            # Generate block counts on full image or normalized blocks
            y_b = build_integer_block_counts(pts, (len(pts) * 0 + 1) * crop_size, crop_size, b)
            scale_counts[b].extend(y_b.flatten().tolist())
            
    counts = np.array(counts, dtype=np.float64)
    densities = np.array(densities, dtype=np.float64)
    luminances = np.array(luminances, dtype=np.float64)
    
    m_cells = (crop_size // out_stride) ** 2
    mean_count = float(np.mean(counts))
    # Expected count in random crop of size crop_size
    mean_crop_count = mean_count * 0.5  # conservative estimate
    m_0 = max(mean_crop_count / max(m_cells, 1), 1e-5)
    b_0 = inv_softplus(m_0)
    
    # Hierarchical scale statistics
    hnb_stats = {}
    for b in hnb_blocks:
        vals = np.array(scale_counts[b], dtype=np.float64)
        m_b = float(np.mean(vals))
        v_b = float(np.var(vals))
        if v_b > m_b and (v_b - m_b) > 1e-5:
            r0 = float((m_b ** 2) / (v_b - m_b))
        else:
            r0 = 100.0
            
        pos_vals = vals[vals > 0]
        if len(pos_vals) > 0:
            q50 = float(np.percentile(pos_vals, 50))
            q90 = float(np.percentile(pos_vals, 90))
        else:
            q50 = 1.0
            q90 = 2.0
            
        hnb_stats[str(b)] = {
            "mean": m_b,
            "var": v_b,
            "r0": r0,
            "q50_pos": q50,
            "q90_pos": q90,
        }
        
    num_d_bins = cfg.get("sampler", {}).get("density_bins", 5)
    num_l_bins = cfg.get("sampler", {}).get("luminance_bins", 4)
    
    d_q = np.quantile(densities, np.linspace(0, 1, num_d_bins + 1)[1:-1]).tolist() if len(densities) > 1 else []
    l_q = np.quantile(luminances, np.linspace(0, 1, num_l_bins + 1)[1:-1]).tolist() if len(luminances) > 1 else []
    
    out = {
        "dataset": ds_name,
        "crop_size": crop_size,
        "mean_crop_count": mean_crop_count,
        "initial_head_bias": b_0,
        "hnb_scales": hnb_stats,
        "density_quantiles": d_q,
        "luminance_quantiles": l_q,
    }
    
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved dataset statistics to: {output_json}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--output", type=str, default="stats.json", help="Path to save stats JSON")
    args = parser.parse_args()
    compute_statistics_for_dataset(args.config, args.output)
