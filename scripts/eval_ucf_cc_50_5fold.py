"""Evaluate and aggregate 5-Fold Cross Validation results for UCF-CC-50 benchmark.

Computes MAE and RMSE for each of the 5 folds and reports the official publication-grade average:
    MAE_avg = mean(MAE_1, ..., MAE_5)
    RMSE_avg = mean(RMSE_1, ..., RMSE_5)
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import torch
import numpy as np

from rmr_v3.config import load_config
from rmr_v3.engine import make_model
from rmr_core.data import CrowdManifestDataset, collate_eval
from rmr_core.evaluation import evaluate_dataset
from torch.utils.data import DataLoader


def evaluate_5folds(base_dir: str | Path, is_scratch: bool = False, output_stride: int = 2) -> None:
    base_dir = Path(base_dir)
    fold_maes = []
    fold_rmses = []
    
    prefix = "exp_ucf_cc_50_scratch_fold" if is_scratch else "exp_ucf_cc_50_fold"
    
    print("\n" + "="*85)
    print(f"UCF-CC-50 5-FOLD CROSS VALIDATION EVALUATION ({'SCRATCH' if is_scratch else 'PRETRAINED'})")
    print("="*85)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for fold in range(1, 6):
        fold_name = f"{prefix}{fold}"
        run_dir = base_dir / fold_name
        ckpt_path = run_dir / "best_val_mae.pt"
        if not ckpt_path.exists():
            ckpt_path = run_dir / "best_checkpoint.pth"
            
        test_manifest = f"data/ucf_cc_50_fold{fold}_test.jsonl"
        cfg_path = Path("configs/benchmarks") / f"{fold_name}.yaml"
        
        if not ckpt_path.exists():
            print(f"Fold {fold}: Checkpoint not found at {ckpt_path} (Run not finished yet)")
            continue
            
        cfg = load_config(str(cfg_path))
        model, _ = make_model(cfg)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        
        test_ds = CrowdManifestDataset(
            manifest=test_manifest,
            train=False,
            data_root="data",
            output_stride=output_stride,
        )
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_eval)
        
        rows, summary = evaluate_dataset(
            model=model,
            loader=test_loader,
            device=device,
            output_stride=output_stride,
            run_tiling=False,
            density_bins=(500.0, 1500.0),
        )
        
        mae = float(summary["MAE"])
        rmse = float(summary["RMSE"])
        fold_maes.append(mae)
        fold_rmses.append(rmse)
        
        print(f"Fold {fold} ({fold_name}): MAE = {mae:.2f} | RMSE = {rmse:.2f} (10 test images)")
        
    print("="*85)
    if len(fold_maes) == 5:
        mean_mae = np.mean(fold_maes)
        mean_rmse = np.mean(fold_rmses)
        print(f">>> OFFICIAL 5-FOLD AVERAGE UCF-CC-50: MAE = {mean_mae:.2f} | RMSE = {mean_rmse:.2f}")
    else:
        print(f"Evaluated {len(fold_maes)}/5 folds. Complete all 5 runs to get the official 5-fold average.")
    print("="*85 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate UCF-CC-50 5-Fold Cross Validation")
    parser.add_argument("--base-dir", type=str, default="runs/benchmarks", help="Base directory containing runs")
    parser.add_argument("--scratch", action="store_true", help="Evaluate scratch folds instead of pretrained")
    parser.add_argument("--output-stride", type=int, default=2, help="Output stride of model")
    args = parser.parse_args()
    
    evaluate_5folds(args.base_dir, is_scratch=args.scratch, output_stride=args.output_stride)
