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

test_manifest = "data/sha_a_test.jsonl"
val_ds = CrowdManifestDataset(test_manifest, train=False, output_stride=4)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_eval)

print("=" * 110)
print(f"{'Model':<25} | {'No-TTA MAE':<12} | {'No-TTA RMSE':<12} | {'HFlip-TTA MAE':<14} | {'HFlip-TTA RMSE':<14} | {'TTA Delta'}")
print("=" * 110)

for r_name, label in runs:
    run_dir = Path("runs/sha_a") / r_name
    ckpt_path = run_dir / "best_val_mae.pt"
    cfg_path = run_dir / "resolved_config.yaml"
    
    if not ckpt_path.exists():
        continue
        
    model, _, _, _ = load_model_from_ckpt(ckpt_path, device=device, use_ema=True, config_path=cfg_path)
    model.eval()
    
    standard_preds = []
    tta_preds = []
    gt_counts = []
    
    with torch.no_grad():
        for batch in val_loader:
            item = batch[0]
            img = item["image"].unsqueeze(0).to(device)
            target_y = item["target_y"].to(device)
            gt_count = float(target_y.sum().item())
            
            # Standard forward
            out = model(img)
            y_std = float(out.y.sum().item())
            
            # Horizontal flip forward
            img_hflip = torch.flip(img, dims=[-1])
            out_hflip = model(img_hflip)
            y_hflip = float(out_hflip.y.sum().item())
            
            # Average prediction
            y_tta = (y_std + y_hflip) / 2.0
            
            standard_preds.append(y_std)
            tta_preds.append(y_tta)
            gt_counts.append(gt_count)
            
    std_arr = np.array(standard_preds)
    tta_arr = np.array(tta_preds)
    gt_arr = np.array(gt_counts)
    
    mae_std = np.mean(np.abs(std_arr - gt_arr))
    rmse_std = np.sqrt(np.mean((std_arr - gt_arr)**2))
    
    mae_tta = np.mean(np.abs(tta_arr - gt_arr))
    rmse_tta = np.sqrt(np.mean((tta_arr - gt_arr)**2))
    
    delta = mae_tta - mae_std
    print(f"{label:<25} | {mae_std:<12.2f} | {rmse_std:<12.2f} | {mae_tta:<14.2f} | {rmse_tta:<14.2f} | {delta:+.2f}")

print("=" * 110)
