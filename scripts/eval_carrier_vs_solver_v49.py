from __future__ import annotations

import os
import sys
from pathlib import Path
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from rmr_core.data import CrowdManifestDataset, collate_eval
from rmr_v3.eval import load_model_from_ckpt

runs = [
    ('v49_m1_m04_anchor', 'M1 (m04 Anchor)'),
    ('v49_m2_repdw_neck_on_m04', 'M2 (RepDW Neck)'),
    ('v49_m3_repdw_deficit_seeding', 'M3 (+Deficit Seed)'),
    ('v49_m4_track_b_nextgen_w64', 'M4 (Track B W64)')
]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Dataset
test_manifest = "data/sha_a_test.jsonl"
val_ds = CrowdManifestDataset(test_manifest, train=False, output_stride=4)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_eval)
print(f"Total test images: {len(val_ds)}")

results = {}
detailed_per_image = {}

for r_name, label in runs:
    run_dir = Path("runs/sha_a") / r_name
    ckpt_path = run_dir / "best_val_mae.pt"
    cfg_path = run_dir / "resolved_config.yaml"
    
    if not ckpt_path.exists():
        print(f"Checkpoint not found for {r_name}")
        continue
        
    print(f"Loading {label} from {ckpt_path}...")
    model, _, _, _ = load_model_from_ckpt(ckpt_path, device=device, use_ema=True, config_path=cfg_path)
    model.eval()
    
    y0_preds = []
    ystar_preds = []
    gt_counts = []
    img_ids = []
    
    with torch.no_grad():
        for batch in val_loader:
            item = batch[0]
            img = item["image"].unsqueeze(0).to(device)
            target_y = item["target_y"].to(device)
            gt_count = float(target_y.sum().item())
            img_id = item["id"]
            
            out = model(img)
            
            y_final_count = float(out.y.sum().item())
            y0_count = float(out.y0.sum().item())
            
            y0_preds.append(y0_count)
            ystar_preds.append(y_final_count)
            gt_counts.append(gt_count)
            img_ids.append(img_id)
            
    y0_arr = np.array(y0_preds)
    ystar_arr = np.array(ystar_preds)
    gt_arr = np.array(gt_counts)
    
    err_0 = np.abs(y0_arr - gt_arr)
    err_star = np.abs(ystar_arr - gt_arr)
    
    sparse_mask = gt_arr <= 100
    mod_mask = (gt_arr > 100) & (gt_arr <= 500)
    dense_mask = gt_arr > 500
    extreme_mask = gt_arr > 1000
    
    results[label] = {
        'mae_0': np.mean(err_0), 'mae_star': np.mean(err_star),
        'rmse_0': np.sqrt(np.mean((y0_arr - gt_arr)**2)), 'rmse_star': np.sqrt(np.mean((ystar_arr - gt_arr)**2)),
        'sparse_0': np.mean(err_0[sparse_mask]), 'sparse_star': np.mean(err_star[sparse_mask]),
        'mod_0': np.mean(err_0[mod_mask]), 'mod_star': np.mean(err_star[mod_mask]),
        'dense_0': np.mean(err_0[dense_mask]), 'dense_star': np.mean(err_star[dense_mask]),
        'extreme_0': np.mean(err_0[extreme_mask]), 'extreme_star': np.mean(err_star[extreme_mask]),
        'help_count': int(np.sum(err_star < err_0)),
        'harm_count': int(np.sum(err_star > err_0)),
        'tie_count': int(np.sum(err_star == err_0)),
        'delta_mae': np.mean(err_star) - np.mean(err_0),
        'bias_0': np.mean(y0_arr - gt_arr),
        'bias_star': np.mean(ystar_arr - gt_arr),
    }
    
    detailed_per_image[label] = {
        'img_id': img_ids,
        'gt': gt_counts,
        'y0': y0_preds,
        'ystar': ystar_preds,
        'err_0': err_0.tolist(),
        'err_star': err_star.tolist()
    }

print("\n" + "=" * 120)
print(f"{'Model':<20} | {'Type':<6} | {'Overall':<7} | {'RMSE':<7} | {'Sparse':<7} | {'Mod':<7} | {'Dense':<7} | {'Extr>1k':<7} | {'Bias':<6} | {'Help/Harm'}")
print("=" * 120)

for label, d in results.items():
    help_ratio = f"{d['help_count']}/{d['harm_count']}"
    delta = d['delta_mae']
    print(f"{label:<20} | {'y0':<6} | {d['mae_0']:<7.2f} | {d['rmse_0']:<7.2f} | {d['sparse_0']:<7.2f} | {d['mod_0']:<7.2f} | {d['dense_0']:<7.2f} | {d['extreme_0']:<7.2f} | {d['bias_0']:<6.1f} | -")
    print(f"{'':<20} | {'y*':<6} | {d['mae_star']:<7.2f} | {d['rmse_star']:<7.2f} | {d['sparse_star']:<7.2f} | {d['mod_star']:<7.2f} | {d['dense_star']:<7.2f} | {d['extreme_star']:<7.2f} | {d['bias_star']:<6.1f} | {help_ratio} (Δ={delta:+.2f})")
    print("-" * 120)

# Save per image predictions
with open("runs/v49_per_image_predictions.json", "w") as f:
    json.dump(detailed_per_image, f)
print("Saved per image predictions to runs/v49_per_image_predictions.json")
