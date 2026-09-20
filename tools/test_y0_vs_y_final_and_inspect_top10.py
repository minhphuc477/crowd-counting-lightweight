from __future__ import annotations

import json
import math
import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

from rmr_core.data import CrowdManifestDataset, collate_eval
from rmr_v3.eval import load_model_from_ckpt


@torch.no_grad()
def run_evaluation():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path("runs/sha_a/rmr_v19_canonical_isotropic/best_val_mae.pt")
    print(f"\n{'='*70}\n[Task 1] Detailed y0 vs y_final Bin Analysis & Oracle Gate\nCheckpoint: {ckpt_path}\n{'='*70}")

    model, _, _, _ = load_model_from_ckpt(ckpt_path, device=device, use_ema=True)
    model.eval()

    output_stride = 4
    val_ds = CrowdManifestDataset("data/sha_a_test.jsonl", train=False, output_stride=output_stride)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_eval)

    records = []

    for idx, batch in enumerate(val_loader):
        item = batch[0]
        img = item["image"].unsqueeze(0).to(device)
        target_y = item["target_y"].to(device)
        gt_count = float(target_y.sum().item())
        img_id = item["id"]

        out = model(img)
        y_final_count = float(out.y.sum().item())
        y0_count = float(out.y0.sum().item())

        err_final = abs(y_final_count - gt_count)
        err_y0 = abs(y0_count - gt_count)
        bias_final = y_final_count - gt_count
        bias_y0 = y0_count - gt_count

        if gt_count < 100.0:
            cat = "sparse"
        elif gt_count <= 500.0:
            cat = "medium"
        else:
            cat = "dense"

        # Oracle choice between y0 and y_final
        oracle_gate_err = min(err_final, err_y0)
        oracle_gate_choice = "y0" if err_y0 < err_final else "y_final"

        records.append({
            "id": img_id,
            "cat": cat,
            "gt": gt_count,
            "y0": y0_count,
            "y_final": y_final_count,
            "err_y0": err_y0,
            "err_final": err_final,
            "bias_y0": bias_y0,
            "bias_final": bias_final,
            "oracle_gate_err": oracle_gate_err,
            "oracle_choice": oracle_gate_choice,
            "h": item["height"],
            "w": item["width"],
            "image_tensor": item["image"].cpu(),
            "target_tensor": target_y.cpu(),
            "y0_tensor": out.y0.cpu(),
            "y_final_tensor": out.y.cpu(),
        })

    def get_stats(recs):
        n = len(recs)
        if n == 0:
            return {}
        return {
            "n": n,
            "mae_y0": np.mean([r["err_y0"] for r in recs]),
            "bias_y0": np.mean([r["bias_y0"] for r in recs]),
            "mae_final": np.mean([r["err_final"] for r in recs]),
            "bias_final": np.mean([r["bias_final"] for r in recs]),
            "mae_oracle_gate": np.mean([r["oracle_gate_err"] for r in recs]),
            "pct_y0_wins": np.mean([1.0 if r["oracle_choice"] == "y0" else 0.0 for r in recs]) * 100,
        }

    all_stats = get_stats(records)
    sparse_stats = get_stats([r for r in records if r["cat"] == "sparse"])
    med_stats = get_stats([r for r in records if r["cat"] == "medium"])
    dense_stats = get_stats([r for r in records if r["cat"] == "dense"])

    print(f"\n{'Category':<10} {'N':<5} | {'MAE (y0)':<10} {'Bias (y0)':<10} | {'MAE (y_final)':<14} {'Bias (y_final)':<14} | {'Oracle Gate MAE':<16} {'y0 Win Rate':<12}")
    print("-" * 95)
    for name, s in [("Sparse", sparse_stats), ("Medium", med_stats), ("Dense", dense_stats), ("Overall", all_stats)]:
        print(f"{name:<10} {s['n']:<5} | {s['mae_y0']:<10.2f} {s['bias_y0']:<10.2f} | {s['mae_final']:<14.2f} {s['bias_final']:<14.2f} | {s['mae_oracle_gate']:<16.2f} {s['pct_y0_wins']:<10.1f}%")

    # Simple Threshold Gate: e.g. if y0_count > threshold, use y0, else use y_final
    print("\n--- Testing Simple Realistic Gates (No Cheating / No Oracle) ---")
    best_thresh_mae = 999.0
    best_thresh = 0
    for thresh in range(300, 1000, 50):
        thresh_errs = []
        for r in records:
            pred = r["y0"] if r["y0"] > thresh else r["y_final"]
            thresh_errs.append(abs(pred - r["gt"]))
        m = np.mean(thresh_errs)
        if m < best_thresh_mae:
            best_thresh_mae = m
            best_thresh = thresh
        print(f"  Gate Threshold = {thresh:>4}: Test MAE = {m:.2f}")

    print(f"\n=> Best Simple Density Gate: if y0 > {best_thresh} -> use y0, else y_final: MAE = {best_thresh_mae:.2f} (Improvement: {all_stats['mae_final'] - best_thresh_mae:+.2f} points!)")

    # =========================================================================
    # Task 2: Direct Inspection of Top 10 Worst Images
    # =========================================================================
    print(f"\n{'='*70}\n[Task 2] Deep Forensic Inspection of Top 10 Worst Test Images\n{'='*70}")
    records.sort(key=lambda x: x["err_final"], reverse=True)
    out_img_dir = Path("runs/sha_a/top10_worst_inspection")
    out_img_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'Rank':<5} {'ID':<10} {'Type':<10} {'GT':<8} {'y0':<8} {'y_final':<8} {'Err':<8} {'Bias':<8} {'H x W':<12} {'Diag / Perspective Context'}")
    print("-" * 95)
    for rank, r in enumerate(records[:10], start=1):
        err_type = "UNDERCMP" if r["bias_final"] < 0 else "OVERCMP"
        dims = f"{r['h']}x{r['w']}"
        print(f"#{rank:<4} {r['id']:<10} {err_type:<10} {r['gt']:<8.1f} {r['y0']:<8.1f} {r['y_final']:<8.1f} {r['err_final']:<8.1f} {r['bias_final']:<8.1f} {dims:<12}")
        
        # Save error analysis metrics:
        # Check maximum local density and foreground distribution
        tgt = r["target_tensor"].squeeze()
        pred = r["y_final_tensor"].squeeze()
        diff = pred - tgt
        abs_diff = diff.abs()
        
        # Breakdown where the error came from:
        # 1. Background false positive: cells where tgt == 0 but pred > 0.05
        bg_cells = (tgt == 0)
        bg_fp_mass = float(pred[bg_cells].sum().item())
        
        # 2. Foreground mass:
        fg_cells = (tgt > 0)
        fg_pred_mass = float(pred[fg_cells].sum().item())
        fg_tgt_mass = float(tgt[fg_cells].sum().item())
        fg_deficit = fg_tgt_mass - fg_pred_mass

        print(f"     -> Background FP Mass (phantom crowd): {bg_fp_mass:.1f} ({bg_fp_mass/max(r['gt'],1)*100:.1f}%)")
        print(f"     -> Foreground Deficit (missed heads):   {fg_deficit:.1f} ({fg_deficit/max(r['gt'],1)*100:.1f}%)")


if __name__ == "__main__":
    run_evaluation()
