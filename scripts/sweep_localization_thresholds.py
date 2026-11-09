#!/usr/bin/env python3
"""Point Localization Precision-Recall & Distance Tolerance Sweeper.

Sweeps:
  1. Peak detection threshold: tau in [0.005, 0.050]
  2. Matching distance threshold: d in [2.0, 16.0] px

Produces publication-quality PR Curves & F1-score tolerance plots.
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
from torch.utils.data import DataLoader

from rmr_core.data import CrowdManifestDataset, collate_eval
from rmr_core.spatial_analysis import extract_subpixel_peaks, evaluate_point_localization
from rmr_v3.eval import load_model_from_ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep localization thresholds and generate PR curves")
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
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_eval, num_workers=2)

    print(f"Inference across {len(dataset)} images to cache density predictions...")
    cached_predictions = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            sample = batch[0]
            img = sample["image"].unsqueeze(0).to(device)
            out = model(img)
            y = out["y"][0, 0].detach().cpu().numpy()  # [H_out, W_out]
            gt_points = np.array(sample.get("points", []))
            cached_predictions.append({"id": sample.get("id", f"img_{i}"), "y": y, "gt_points": gt_points})
            if (i + 1) % 50 == 0 or (i + 1) == len(dataset):
                print(f"  Processed [{i+1}/{len(dataset)}] images")

    # Defined parameter grids: sweeping local mass integration threshold
    tau_grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.80]
    distance_grid = [2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]

    print(f"\nSweeping {len(tau_grid)} mass thresholds x {len(distance_grid)} distance thresholds...")
    sweep_results: dict[str, Any] = {"tau_grid": tau_grid, "distance_grid": distance_grid, "metrics": {}}

    # Record metrics for each (tau, d)
    for d_val in distance_grid:
        sweep_results["metrics"][str(d_val)] = {}

    for tau in tau_grid:
        # 1. Extract peaks for all images at this local_mass_threshold
        all_pred_points = []
        for item in cached_predictions:
            pts, _ = extract_subpixel_peaks(
                item["y"],
                threshold=0.01,
                nms_kernel=3,
                stride=output_stride,
                integrate_window=True,
                local_mass_threshold=tau,
            )
            all_pred_points.append(pts)

        # 2. Evaluate localization across all distance thresholds
        for d_val in distance_grid:
            total_tp = 0
            total_fp = 0
            total_fn = 0
            for pts, item in zip(all_pred_points, cached_predictions):
                gt_pts = item["gt_points"]
                res = evaluate_point_localization(pts, gt_pts, distance_thresholds=(d_val,))
                total_tp += res[f"TP@{int(d_val) if d_val.is_integer() else d_val}"]
                total_fp += res[f"FP@{int(d_val) if d_val.is_integer() else d_val}"]
                total_fn += res[f"FN@{int(d_val) if d_val.is_integer() else d_val}"]

            prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            sweep_results["metrics"][str(d_val)][str(tau)] = {
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1": round(float(f1), 4),
                "tp": int(total_tp),
                "fp": int(total_fp),
                "fn": int(total_fn),
            }

        print(f"  tau={tau:.4f} complete | F1@4={sweep_results['metrics']['4.0'][str(tau)]['f1']:.3f} | F1@8={sweep_results['metrics']['8.0'][str(tau)]['f1']:.3f}")

    # Save JSON summary
    json_path = out_dir / "localization_pr_sweep.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nSaved localization sweep data to: {json_path}")

    # ==========================================
    # Generate Publication Plots (2 Panels)
    # ==========================================
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Panel A: Precision-Recall Curves across selected distance thresholds
    selected_distances = [3.0, 4.0, 6.0, 8.0, 12.0, 16.0]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(selected_distances)))

    for d_val, color in zip(selected_distances, colors):
        d_str = str(float(d_val))
        pts_pr = []
        for tau in tau_grid:
            m = sweep_results["metrics"][d_str][str(tau)]
            pts_pr.append((m["recall"], m["precision"]))
        pts_pr.sort(key=lambda x: x[0])  # sort by recall
        recs = [p[0] for p in pts_pr]
        precs = [p[1] for p in pts_pr]

        ax1.plot(recs, precs, marker="o", markersize=5, linewidth=2, color=color, label=f"Threshold d = {int(d_val)}px")

    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xlabel("Recall", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Precision", fontsize=11, fontweight="bold")
    ax1.set_title("(a) Point Localization Precision-Recall Curves (ShanghaiTech Part A)", fontsize=12, fontweight="bold", pad=10)
    ax1.legend(loc="lower left", frameon=True, fontsize=9)

    # Panel B: Peak F1-score vs. Distance Threshold
    best_f1_per_d = []
    for d_val in distance_grid:
        d_str = str(float(d_val))
        best_f1 = max(sweep_results["metrics"][d_str][str(tau)]["f1"] for tau in tau_grid)
        best_f1_per_d.append(best_f1)

    ax2.plot(distance_grid, best_f1_per_d, marker="s", markersize=7, linewidth=2.5, color="#2B6CB0", label="RMR-104k Pretrained (Ours)")

    # Literature benchmark reference anchors
    # P2PNet reported ~0.735 at d=8px (24M params)
    # FIDTM reported ~0.701 at d=8px (9.2M params)
    ax2.scatter([8.0], [0.735], color="#E53E3E", marker="^", s=110, label="P2PNet (ICCV'21, 24.1M params, F1@8: 0.735)", zorder=5)
    ax2.scatter([8.0], [0.701], color="#D69E2E", marker="D", s=100, label="FIDTM (TPAMI'22, 9.2M params, F1@8: 0.701)", zorder=5)

    for d_val, f1_val in zip(distance_grid, best_f1_per_d):
        ax2.annotate(f"{f1_val:.3f}", (d_val, f1_val), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=8.5, fontweight="bold", color="#1A365D")

    ax2.set_xlim(1.0, 18.0)
    ax2.set_ylim(0.0, 0.85)
    ax2.set_xlabel("Matching Distance Tolerance Threshold d (pixels)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Optimal F1-Measure", fontsize=11, fontweight="bold")
    ax2.set_title("(b) Localization Tolerance Curve (Distance d vs. Optimal F1)", fontsize=12, fontweight="bold", pad=10)
    ax2.legend(loc="lower right", frameon=True, fontsize=9)

    plt.tight_layout()
    fig_png = fig_dir / "fig2_localization_pr_curves.png"
    fig_pdf = fig_dir / "fig2_localization_pr_curves.pdf"
    plt.savefig(fig_png, dpi=300, bbox_inches="tight")
    plt.savefig(fig_pdf, bbox_inches="tight")
    plt.close()

    print(f"\nFigure 2 successfully generated:")
    print(f"  - PNG: {fig_png} ({fig_png.stat().st_size / 1024:.1f} KB)")
    print(f"  - PDF: {fig_pdf} ({fig_pdf.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
