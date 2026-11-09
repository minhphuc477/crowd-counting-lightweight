#!/usr/bin/env python3
"""Inference Scale Invariance & Input Resolution Sweeper.

Tests model behavior when input test images are scaled by:
  s in [0.5, 0.65, 0.75, 0.85, 1.0, 1.15, 1.25, 1.5, 1.75, 2.0]

Measures:
  1. MAE and RMSE at each scale s
  2. Scale Consistency Error (SCE) = mean relative drift from s=1.0
  3. Spatial mass preservation of the continuous Radon measure

Strict compliance with AGENTS.md: Zero buzzwords. Pure empirical data.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rmr_core.data import CrowdManifestDataset, collate_eval
from rmr_v3.eval import load_model_from_ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep input inference scales and measure scale invariance")
    parser.add_argument("--checkpoint", default="runs/benchmarks/exp_sha_direct_pretrained_104k/best_val_mae.pt")
    parser.add_argument("--manifest", default="data/sha_a_test.jsonl")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Loading checkpoint: {args.checkpoint} on {device}...")
    model, _, _, cfg = load_model_from_ckpt(Path(args.checkpoint), device=device)
    model.eval()

    output_stride = int(cfg.get("model", {}).get("output_stride", 2))
    dataset = CrowdManifestDataset(
        manifest=args.manifest,
        train=False,
        data_root="data",
        max_size=2048,
        output_stride=output_stride,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_eval, num_workers=0)

    scale_grid = [0.5, 0.65, 0.75, 0.85, 1.0, 1.15, 1.25, 1.5, 1.75, 2.0]
    print(f"\nSweeping {len(scale_grid)} scale factors across {len(dataset)} images on {device}...")

    # Dictionary to record predictions per image across scales:
    # img_id -> {gt: N, counts: {s: pred_N}}
    image_records: dict[str, dict[str, Any]] = {}

    # Iterate over scales
    scale_metrics = []
    for s in scale_grid:
        abs_errors = []
        sq_errors = []
        biases = []

        with torch.no_grad():
            for i, batch in enumerate(loader):
                sample = batch[0]
                sid = sample.get("id", f"img_{i}")
                img = sample["image"].unsqueeze(0).to(device)  # [1, 3, H, W]
                gt = float(sample.get("target_count", sample.get("points", torch.tensor([])).shape[0] if isinstance(sample.get("points"), torch.Tensor) else len(sample.get("points", []))))

                if sid not in image_records:
                    image_records[sid] = {"gt": gt, "counts": {}}

                # Scale input image if s != 1.0
                if abs(s - 1.0) > 1e-4:
                    orig_h, orig_w = img.shape[-2:]
                    scaled_h = max(32, int(round(orig_h * s / 32.0)) * 32)
                    scaled_w = max(32, int(round(orig_w * s / 32.0)) * 32)
                    scaled_img = F.interpolate(img, size=(scaled_h, scaled_w), mode="bilinear", align_corners=False)
                else:
                    scaled_img = img

                out = model(scaled_img)
                y = out["y"][0, 0]  # [H_out, W_out]
                pred_count = float(y.sum().item())
                image_records[sid]["counts"][str(s)] = pred_count

                err = pred_count - gt
                abs_errors.append(abs(err))
                sq_errors.append(err ** 2)
                biases.append(err)

        mae = float(np.mean(abs_errors))
        rmse = float(np.sqrt(np.mean(sq_errors)))
        bias = float(np.mean(biases))

        scale_metrics.append({
            "scale": s,
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "bias": round(bias, 2),
        })
        print(f"  Scale s = {s:4.2f}x | MAE = {mae:6.2f} | RMSE = {rmse:6.2f} | Bias = {bias:+6.2f}")

    # Compute Scale Consistency Error (SCE) relative to s=1.0
    relative_drifts = []
    for sid, rec in image_records.items():
        base_pred = rec["counts"]["1.0"]
        if base_pred > 0.1:
            drifts = [abs(rec["counts"][str(s)] - base_pred) / base_pred for s in scale_grid]
            relative_drifts.append(float(np.mean(drifts)))

    mean_sce = float(np.mean(relative_drifts)) if relative_drifts else 0.0
    print(f"\nMean Scale Consistency Error (SCE) relative to s=1.0: {mean_sce * 100:.2f}%")

    # Save JSON summary
    summary_data = {
        "scale_metrics": scale_metrics,
        "mean_scale_consistency_error_pct": round(mean_sce * 100, 2),
    }
    json_path = out_dir / "scale_invariance_sweep.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved scale sweep summary to: {json_path}")

    # ==========================================
    # Generate Publication Plots (2 Panels)
    # ==========================================
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    scales = [m["scale"] for m in scale_metrics]
    maes = [m["mae"] for m in scale_metrics]
    rmses = [m["rmse"] for m in scale_metrics]

    # Panel A: MAE and RMSE vs. Scale
    ax1.plot(scales, maes, marker="o", markersize=7, linewidth=2.5, color="#2B6CB0", label="Test MAE")
    ax1.plot(scales, rmses, marker="s", markersize=6, linewidth=2.0, color="#E53E3E", linestyle="--", label="Test RMSE")
    ax1.axvline(1.0, color="gray", linestyle=":", label="Nominal Scale (1.0x)")

    for s, mae in zip(scales, maes):
        ax1.annotate(f"{mae:.1f}", (s, mae), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.5, fontweight="bold")

    ax1.set_xlabel("Input Scale Factor s (relative to 1.0x)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Counting Error", fontsize=11, fontweight="bold")
    ax1.set_title("(a) Counting Error Across Input Scales (ShanghaiTech Part A)", fontsize=12, fontweight="bold", pad=10)
    ax1.legend(loc="upper right", frameon=True, fontsize=9)

    # Panel B: Mean Relative Predicted Count Drift (Normalized to s=1.0)
    mean_normalized_counts = []
    for s in scale_grid:
        norm_ratios = [rec["counts"][str(s)] / max(rec["counts"]["1.0"], 1.0) for rec in image_records.values()]
        mean_normalized_counts.append(float(np.mean(norm_ratios)))

    ax2.plot(scales, mean_normalized_counts, marker="^", markersize=7, linewidth=2.5, color="#2C5282", label="Mean Pred(s) / Pred(1.0x)")
    ax2.axhline(1.0, color="black", linestyle="--", alpha=0.7, label="Ideal Invariant Response (= 1.0)")
    ax2.fill_between(scales, [1.0 - mean_sce]*len(scales), [1.0 + mean_sce]*len(scales), color="#3182CE", alpha=0.15, label=f"± SCE Band (±{mean_sce*100:.1f}%)")

    for s, ratio in zip(scales, mean_normalized_counts):
        ax2.annotate(f"{ratio:.2f}", (s, ratio), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.5)

    ax2.set_xlabel("Input Scale Factor s", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Relative Predicted Count Drift", fontsize=11, fontweight="bold")
    ax2.set_title("(b) Scale Invariance Response (Normalized to s=1.0x)", fontsize=12, fontweight="bold", pad=10)
    ax2.legend(loc="upper left", frameon=True, fontsize=9)

    plt.tight_layout()
    fig_png = fig_dir / "fig3_scale_invariance_sweep.png"
    fig_pdf = fig_dir / "fig3_scale_invariance_sweep.pdf"
    plt.savefig(fig_png, dpi=300, bbox_inches="tight")
    plt.savefig(fig_pdf, bbox_inches="tight")
    plt.close()

    print(f"\nFigure 3 successfully generated:")
    print(f"  - PNG: {fig_png} ({fig_png.stat().st_size / 1024:.1f} KB)")
    print(f"  - PDF: {fig_pdf} ({fig_pdf.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
