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
import matplotlib.pyplot as plt

from rmr_core.data import CrowdManifestDataset, collate_eval
from rmr_core.operators import regional_sum
from rmr_v3.eval import load_model_from_ckpt


@torch.no_grad()
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path("runs/sha_a/rmr_v19_canonical_isotropic/best_val_mae.pt")
    print(f"\n{'='*70}\n[STEP 1 & 2 CRITICAL INVESTIGATION]\nCheckpoint: {ckpt_path}\nDevice: {device}\n{'='*70}")

    model, _, _, _ = load_model_from_ckpt(ckpt_path, device=device, use_ema=True)
    model.eval()

    output_stride = 4

    # =========================================================================
    # PART 1: EMPTY REGION ANALYSIS & TOP-10 MONTAGES ON TEST SET (182 IMAGES)
    # =========================================================================
    print("\n>>> [PART 1] Evaluating Empty Regions on Test Set (182 images)...")
    test_ds = CrowdManifestDataset("data/sha_a_test.jsonl", train=False, output_stride=output_stride)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_eval)

    test_records = []

    for idx, batch in enumerate(test_loader):
        item = batch[0]
        img = item["image"].unsqueeze(0).to(device)
        target_y = item["target_y"].to(device)
        gt_count = float(target_y.sum().item())
        img_id = item["id"]

        out = model(img)
        pred_count = float(out.y.sum().item())
        y0_count = float(out.y0.sum().item())

        grid_h, grid_w = target_y.shape[-2:]
        reg = model._regions(grid_h, grid_w, device, stride=output_stride)
        b_gt = regional_sum(target_y.unsqueeze(0).float(), reg.boxes, out_dtype=torch.float32)
        b_pred = out.b_region.float()

        # 32px regions (scale_id == 0)
        s0_mask = (reg.scale_id == 0)
        gt_s0 = b_gt[0, 0, s0_mask].cpu().numpy()
        pred_s0 = b_pred[0, 0, s0_mask].cpu().numpy()

        empty_mask = (gt_s0 == 0.0)
        n_total_reg = len(gt_s0)
        n_empty_reg = int(empty_mask.sum())
        empty_frac = n_empty_reg / max(n_total_reg, 1)

        empty_pred_mean = float(np.mean(pred_s0[empty_mask])) if n_empty_reg > 0 else 0.0
        occupied_pred_mean = float(np.mean(pred_s0[~empty_mask])) if (n_total_reg - n_empty_reg) > 0 else 0.0
        empty_pred_sum = float(np.sum(pred_s0[empty_mask])) if n_empty_reg > 0 else 0.0

        signed_err = pred_count - gt_count
        abs_err = abs(signed_err)

        test_records.append({
            "id": img_id,
            "gt": gt_count,
            "pred": pred_count,
            "y0": y0_count,
            "signed_err": signed_err,
            "abs_err": abs_err,
            "n_total_reg": n_total_reg,
            "n_empty_reg": n_empty_reg,
            "empty_frac": empty_frac,
            "empty_pred_mean": empty_pred_mean,
            "occupied_pred_mean": occupied_pred_mean,
            "empty_pred_sum": empty_pred_sum,
            "h": item["height"],
            "w": item["width"],
            "raw_image": item["image"].cpu(),
            "target_map": target_y.cpu().squeeze().numpy(),
            "pred_map": out.y.cpu().squeeze().numpy(),
        })

    # Correlation between signed error and empty fraction across all 182 test images
    signed_errs = np.array([r["signed_err"] for r in test_records])
    empty_fracs = np.array([r["empty_frac"] for r in test_records])
    corr_empty_signed = float(np.corrcoef(signed_errs, empty_fracs)[0, 1])

    abs_errs = np.array([r["abs_err"] for r in test_records])
    corr_empty_abs = float(np.corrcoef(abs_errs, empty_fracs)[0, 1])

    print(f"\n--- [1A] Correlation Analysis Across All 182 Test Images ---")
    print(f"Correlation (Signed Error vs Empty Region Fraction): {corr_empty_signed:+.4f}")
    print(f"Correlation (Absolute Error vs Empty Region Fraction): {corr_empty_abs:+.4f}")

    # Compare 10 Worst vs 10 Best
    test_records.sort(key=lambda x: x["abs_err"], reverse=True)
    worst10 = test_records[:10]
    best10 = test_records[-10:]

    def summarize_group(recs, group_name):
        print(f"\n--- [1B] {group_name} (N={len(recs)}) ---")
        print(f"  Mean GT Count:               {np.mean([r['gt'] for r in recs]):.1f}")
        print(f"  Mean Pred Count:             {np.mean([r['pred'] for r in recs]):.1f}")
        print(f"  Mean Signed Bias:            {np.mean([r['signed_err'] for r in recs]):+.1f}")
        print(f"  Mean Absolute Error:         {np.mean([r['abs_err'] for r in recs]):.1f}")
        print(f"  Mean Empty Region Fraction:  {np.mean([r['empty_frac'] for r in recs])*100:.1f}%")
        print(f"  Mean Pred in Empty Box (32px): {np.mean([r['empty_pred_mean'] for r in recs]):.4f} people/box")
        print(f"  Mean Pred in Occupied Box:     {np.mean([r['occupied_pred_mean'] for r in recs]):.4f} people/box")

    summarize_group(worst10, "Top 10 WORST Images (Highest Error)")
    summarize_group(best10, "Top 10 BEST Images (Lowest Error - Control Group)")

    # Save Montages for Top 10 Worst Images
    montage_dir = Path("runs/sha_a/top10_worst_montages")
    montage_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- [1C] Saving Montages to {montage_dir}... ---")

    for rank, r in enumerate(worst10, start=1):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Panel 1: Original Image
        # Denormalize image: image was normalized with standard ImageNet mean/std
        raw_img = r["raw_image"].clone()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        raw_img = raw_img * std + mean
        raw_img = raw_img.clamp(0, 1).permute(1, 2, 0).numpy()
        axes[0].imshow(raw_img)
        axes[0].set_title(f"Image: {r['id']}\n{r['h']}x{r['w']}")
        axes[0].axis("off")

        # Panel 2: GT Density Map
        im1 = axes[1].imshow(r["target_map"], cmap="jet")
        axes[1].set_title(f"GT Density\nCount: {r['gt']:.1f}")
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Panel 3: Pred Density Map
        im2 = axes[2].imshow(r["pred_map"], cmap="jet")
        axes[2].set_title(f"Pred Density (y_final)\nCount: {r['pred']:.1f} (Bias: {r['signed_err']:+.1f})")
        axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        # Panel 4: Error Map
        err_map = np.abs(r["pred_map"] - r["target_map"])
        im3 = axes[3].imshow(err_map, cmap="hot")
        axes[3].set_title(f"Absolute Error Map\nMAE: {r['abs_err']:.1f}")
        axes[3].axis("off")
        plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

        plt.tight_layout()
        out_path = montage_dir / f"rank{rank:02d}_{r['id']}_err{r['abs_err']:.0f}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"  Saved: {out_path.name}")

    # =========================================================================
    # PART 2: Y0 VS Y_FINAL ON TRAINING SET (300 TRAIN IMAGES)
    # =========================================================================
    print(f"\n{'='*70}\n>>> [PART 2] Evaluating y0 vs y_final on 300 Training Images...\n{'='*70}")
    train_ds = CrowdManifestDataset("data/sha_a_train_all.jsonl", train=False, output_stride=output_stride)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_eval)

    train_records = []

    for idx, batch in enumerate(train_loader):
        item = batch[0]
        img = item["image"].unsqueeze(0).to(device)
        target_y = item["target_y"].to(device)
        gt_count = float(target_y.sum().item())
        img_id = item["id"]

        out = model(img)
        pred_count = float(out.y.sum().item())
        y0_count = float(out.y0.sum().item())

        err_final = abs(pred_count - gt_count)
        err_y0 = abs(y0_count - gt_count)
        bias_final = pred_count - gt_count
        bias_y0 = y0_count - gt_count

        if gt_count < 100.0:
            cat = "sparse"
        elif gt_count <= 500.0:
            cat = "medium"
        else:
            cat = "dense"

        oracle_gate_err = min(err_final, err_y0)
        oracle_choice = "y0" if err_y0 < err_final else "y_final"

        train_records.append({
            "id": img_id,
            "cat": cat,
            "gt": gt_count,
            "y0": y0_count,
            "y_final": pred_count,
            "err_y0": err_y0,
            "err_final": err_final,
            "bias_y0": bias_y0,
            "bias_final": bias_final,
            "oracle_gate_err": oracle_gate_err,
            "oracle_choice": oracle_choice,
        })

    def get_train_stats(recs):
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

    t_all = get_train_stats(train_records)
    t_sparse = get_train_stats([r for r in train_records if r["cat"] == "sparse"])
    t_med = get_train_stats([r for r in train_records if r["cat"] == "medium"])
    t_dense = get_train_stats([r for r in train_records if r["cat"] == "dense"])

    print(f"\n{'TRAIN SET':<10} {'N':<5} | {'MAE (y0)':<10} {'Bias (y0)':<10} | {'MAE (y_final)':<14} {'Bias (y_final)':<14} | {'Oracle Gate MAE':<16} {'y0 Win Rate':<12}")
    print("-" * 95)
    for name, s in [("Sparse", t_sparse), ("Medium", t_med), ("Dense", t_dense), ("Overall", t_all)]:
        print(f"{name:<10} {s['n']:<5} | {s['mae_y0']:<10.2f} {s['bias_y0']:<10.2f} | {s['mae_final']:<14.2f} {s['bias_final']:<14.2f} | {s['mae_oracle_gate']:<16.2f} {s['pct_y0_wins']:<10.1f}%")


if __name__ == "__main__":
    main()
