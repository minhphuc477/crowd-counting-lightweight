from __future__ import annotations

import csv
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
from rmr_v3.solver import unrolled_sirt_solver


@torch.no_grad()
def evaluate_dataset(model, manifest_path: str, device: torch.device, output_stride: int = 4):
    ds = CrowdManifestDataset(manifest_path, train=False, output_stride=output_stride)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_eval)

    records = []
    for idx, batch in enumerate(loader):
        item = batch[0]
        img = item["image"].unsqueeze(0).to(device)
        target_y = item["target_y"].to(device)
        gt_count = float(target_y.sum().item())
        img_id = item["id"]
        h, w = item["height"], item["width"]

        out = model(img)
        pred_count = float(out.y.sum().item())
        y0_count = float(out.y0.sum().item())

        grid_h, grid_w = target_y.shape[-2:]
        reg = model._regions(grid_h, grid_w, device, stride=output_stride)
        b_gt = regional_sum(target_y.unsqueeze(0).float(), reg.boxes, out_dtype=torch.float32)

        oracle_out = unrolled_sirt_solver(
            y0=out.y0,
            b_solver=b_gt,
            weight_solver=out.solver_region_weight,
            regions=reg,
            iterations=model.cfg.iterations,
            omega=model.cfg.omega,
            solver_strength=1.0,
            residual_clip=model.cfg.residual_clip,
            eps=model.cfg.eps,
            solver_mode=model.cfg.solver_mode,
            density_gate_rho=model.cfg.density_gate_rho,
            density_gate_floor=model.cfg.density_gate_floor,
            proximal_tau=model.cfg.proximal_tau,
            proximal_mode=model.cfg.proximal_mode,
            proximal_mu=model.cfg.proximal_mu,
            tv_lambda=model.cfg.tv_lambda,
            tv_type=model.cfg.tv_type,
            tv_eps_c=model.cfg.tv_eps_c,
            laplace_kernel=model._laplace_kernel,
            adjoint_mode=model.cfg.adjoint_mode,
            b_variance=None,
            morozov_gamma=model.cfg.morozov_gamma,
            use_barzilai_borwein=model.cfg.use_barzilai_borwein,
            bb_clamp_min=model.cfg.bb_clamp_min,
            bb_clamp_max=model.cfg.bb_clamp_max,
            output_stride=output_stride,
            use_anscombe=model.cfg.use_anscombe_sirt,
            anscombe_c=model.cfg.anscombe_c,
            adaptive_tau=model.cfg.adaptive_tau,
            adaptive_tau_rho0=model.cfg.adaptive_tau_rho0,
        )
        oracle_count = float(oracle_out["y"].sum().item())

        if gt_count < 100.0:
            cat = "sparse"
        elif gt_count <= 500.0:
            cat = "medium"
        else:
            cat = "dense"

        err_y0 = abs(y0_count - gt_count)
        bias_y0 = y0_count - gt_count
        err_final = abs(pred_count - gt_count)
        bias_final = pred_count - gt_count
        err_oracle = abs(oracle_count - gt_count)
        bias_oracle = oracle_count - gt_count

        records.append({
            "id": img_id,
            "height": h,
            "width": w,
            "cat": cat,
            "gt": gt_count,
            "y0": y0_count,
            "y_final": pred_count,
            "oracle": oracle_count,
            "abs_err_y0": err_y0,
            "bias_y0": bias_y0,
            "abs_err_final": err_final,
            "bias_final": bias_final,
            "abs_err_oracle": err_oracle,
            "bias_oracle": bias_oracle,
            "raw_image": item["image"].cpu(),
            "target_map": target_y.cpu().squeeze().numpy(),
            "pred_map": out.y.cpu().squeeze().numpy(),
        })

    return records


def save_csv(records: list[dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id", "height", "width", "cat", "gt", "y0", "y_final", "oracle",
        "abs_err_y0", "bias_y0", "abs_err_final", "bias_final",
        "abs_err_oracle", "bias_oracle"
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            row = {k: f"{r[k]:.4f}" if isinstance(r[k], float) else r[k] for k in fieldnames}
            writer.writerow(row)
    print(f"Saved: {out_path} ({len(records)} rows)")


def print_stats_table(records: list[dict], name: str):
    print(f"\n{'='*95}\n{name} (N={len(records)})\n{'='*95}")
    header = f"{'Category':<10} {'N':<5} {'Mean GT':<10} | {'MAE y0':<10} {'Bias y0':<10} | {'MAE y_fin':<10} {'Bias y_fin':<10} | {'MAE Orac':<10} {'Bias Orac':<10}"
    print(header)
    print("-" * 95)

    for cat in ["sparse", "medium", "dense", "overall"]:
        if cat == "overall":
            sub = records
        else:
            sub = [r for r in records if r["cat"] == cat]
        n = len(sub)
        if n == 0:
            continue
        mean_gt = np.mean([r["gt"] for r in sub])
        mae_y0 = np.mean([r["abs_err_y0"] for r in sub])
        bias_y0 = np.mean([r["bias_y0"] for r in sub])
        mae_fin = np.mean([r["abs_err_final"] for r in sub])
        bias_fin = np.mean([r["bias_final"] for r in sub])
        mae_orc = np.mean([r["abs_err_oracle"] for r in sub])
        bias_orc = np.mean([r["bias_oracle"] for r in sub])
        print(f"{cat.capitalize():<10} {n:<5} {mean_gt:<10.1f} | {mae_y0:<10.2f} {bias_y0:<10.2f} | {mae_fin:<10.2f} {bias_fin:<10.2f} | {mae_orc:<10.2f} {bias_orc:<10.2f}")


def generate_verified_montages(test_records: list[dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Sort strictly by abs_err_final descending
    sorted_recs = sorted(test_records, key=lambda x: x["abs_err_final"], reverse=True)
    worst10 = sorted_recs[:10]

    print(f"\n--- Saving Verified Montages for Top 10 Worst Test Images to {out_dir} ---")
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for rank, r in enumerate(worst10, start=1):
        fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))

        raw_img = (r["raw_image"].clone() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        axes[0].imshow(raw_img)
        axes[0].set_title(f"Rank #{rank}: {r['id']}\n{r['height']}x{r['width']} | Cat: {r['cat']}", fontsize=11)
        axes[0].axis("off")

        im1 = axes[1].imshow(r["target_map"], cmap="jet")
        axes[1].set_title(f"GT Density Map\nCount: {r['gt']:.1f}", fontsize=11)
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(r["pred_map"], cmap="jet")
        axes[2].set_title(f"Pred Density (y_final)\nCount: {r['y_final']:.1f} (Bias: {r['bias_final']:+.1f})", fontsize=11)
        axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        err_map = np.abs(r["pred_map"] - r["target_map"])
        im3 = axes[3].imshow(err_map, cmap="hot")
        axes[3].set_title(f"Absolute Error Map\nMAE: {r['abs_err_final']:.1f} (y0: {r['y0']:.1f})", fontsize=11)
        axes[3].axis("off")
        plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

        plt.tight_layout()
        out_file = out_dir / f"rank{rank:02d}_{r['id']}_err{r['abs_err_final']:.0f}.png"
        fig.savefig(out_file, dpi=120)
        plt.close(fig)
        print(f"  Rank #{rank:02d}: {r['id']:<15} GT={r['gt']:<7.1f} Pred={r['y_final']:<7.1f} Bias={r['bias_final']:<+7.1f} MAE={r['abs_err_final']:<7.1f} -> {out_file.name}")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path("runs/sha_a/rmr_v19_canonical_isotropic/best_val_mae.pt")
    print(f"Loading checkpoint: {ckpt_path} on {device}")
    model, _, _, _ = load_model_from_ckpt(ckpt_path, device=device, use_ema=True)
    model.eval()

    # 1. Test Set Evaluation (182 images)
    print("\n>>> Evaluating Test Set (182 images)...")
    test_records = evaluate_dataset(model, "data/sha_a_test.jsonl", device=device)
    test_csv_path = Path("runs/sha_a/authoritative_test_predictions_v19.csv")
    save_csv(test_records, test_csv_path)
    print_stats_table(test_records, "TEST SET EVALUATION")

    # 2. Train Set Evaluation (300 images)
    print("\n>>> Evaluating Train Set (300 images)...")
    train_records = evaluate_dataset(model, "data/sha_a_train_all.jsonl", device=device)
    train_csv_path = Path("runs/sha_a/authoritative_train_predictions_v19.csv")
    save_csv(train_records, train_csv_path)
    print_stats_table(train_records, "TRAIN SET EVALUATION")

    # 3. Verified Montages
    montage_dir = Path("runs/sha_a/top10_worst_montages_verified")
    generate_verified_montages(test_records, montage_dir)


if __name__ == "__main__":
    main()
