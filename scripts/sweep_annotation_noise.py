#!/usr/bin/env python3
"""Annotation Noise & Missing Label Stress-Tester.

Simulates human annotation imperfections:
  1. Positional Gaussian Jitter: sigma in [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0] px
     Reflects click inaccuracy of human crowd annotators.
  2. Missing Label Dropout: rho in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
     Reflects annotator fatigue / missed heads in dense crowds.

Strict compliance with AGENTS.md: Zero buzzwords. Pure empirical stress-testing.
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
    parser = argparse.ArgumentParser(description="Sweep annotation noise and label dropout")
    parser.add_argument("--checkpoint", default="runs/benchmarks/exp_sha_direct_pretrained_104k/best_val_mae.pt")
    parser.add_argument("--manifest", default="data/sha_a_test.jsonl")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--num-trials", type=int, default=5)
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

    print(f"Extracting subpixel peaks for {len(dataset)} images at optimal mass threshold tau=0.25...")
    cached_predictions = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            sample = batch[0]
            img = sample["image"].unsqueeze(0).to(device)
            out = model(img)
            y = out["y"][0, 0].detach().cpu().numpy()
            pts, _ = extract_subpixel_peaks(
                y,
                threshold=0.01,
                nms_kernel=3,
                stride=output_stride,
                integrate_window=True,
                local_mass_threshold=0.25,
            )
            raw_gt = sample.get("points", [])
            gt_points = np.array(raw_gt) if not isinstance(raw_gt, np.ndarray) else raw_gt
            cached_predictions.append({
                "id": sample.get("id", f"img_{i}"),
                "pred_pts": pts,
                "gt_pts": gt_points,
            })

    # ========================================================
    # 1. Positional Gaussian Jitter Sweep: sigma in [0 .. 12] px
    # ========================================================
    jitter_sigmas = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    print(f"\n--- 1. Evaluating Positional Gaussian Jitter ({len(jitter_sigmas)} sigma levels, {args.num_trials} trials each) ---")
    jitter_results = []

    for sigma in jitter_sigmas:
        trial_f1_4 = []
        trial_f1_8 = []
        trial_prec_8 = []
        trial_rec_8 = []

        for trial in range(args.num_trials):
            rng = np.random.default_rng(seed=42 + trial * 100 + int(sigma * 10))
            tot_tp4, tot_fp4, tot_fn4 = 0, 0, 0
            tot_tp8, tot_fp8, tot_fn8 = 0, 0, 0

            for item in cached_predictions:
                pts = item["pred_pts"]
                gt = item["gt_pts"]
                if len(gt) > 0 and sigma > 0.0:
                    noise = rng.normal(loc=0.0, scale=sigma, size=gt.shape)
                    noisy_gt = gt + noise
                else:
                    noisy_gt = gt

                res4 = evaluate_point_localization(pts, noisy_gt, distance_thresholds=(4.0,))
                tot_tp4 += res4["TP@4"]
                tot_fp4 += res4["FP@4"]
                tot_fn4 += res4["FN@4"]

                res8 = evaluate_point_localization(pts, noisy_gt, distance_thresholds=(8.0,))
                tot_tp8 += res8["TP@8"]
                tot_fp8 += res8["FP@8"]
                tot_fn8 += res8["FN@8"]

            p4 = tot_tp4 / (tot_tp4 + tot_fp4) if (tot_tp4 + tot_fp4) > 0 else 0.0
            r4 = tot_tp4 / (tot_tp4 + tot_fn4) if (tot_tp4 + tot_fn4) > 0 else 0.0
            f1_4 = 2 * p4 * r4 / (p4 + r4) if (p4 + r4) > 0 else 0.0

            p8 = tot_tp8 / (tot_tp8 + tot_fp8) if (tot_tp8 + tot_fp8) > 0 else 0.0
            r8 = tot_tp8 / (tot_tp8 + tot_fn8) if (tot_tp8 + tot_fn8) > 0 else 0.0
            f1_8 = 2 * p8 * r8 / (p8 + r8) if (p8 + r8) > 0 else 0.0

            trial_f1_4.append(f1_4)
            trial_f1_8.append(f1_8)
            trial_prec_8.append(p8)
            trial_rec_8.append(r8)

        m_f1_4 = float(np.mean(trial_f1_4))
        m_f1_8 = float(np.mean(trial_f1_8))
        s_f1_8 = float(np.std(trial_f1_8))
        m_p8 = float(np.mean(trial_prec_8))
        m_r8 = float(np.mean(trial_rec_8))

        jitter_results.append({
            "sigma_px": sigma,
            "f1_at_4px": round(m_f1_4, 4),
            "f1_at_8px": round(m_f1_8, 4),
            "f1_at_8px_std": round(s_f1_8, 4),
            "prec_at_8px": round(m_p8, 4),
            "rec_at_8px": round(m_r8, 4),
        })
        print(f"  sigma={sigma:4.1f}px | F1@8px={m_f1_8:.4f} ± {s_f1_8:.4f} | Prec@8={m_p8:.4f} | Rec@8={m_r8:.4f} | F1@4px={m_f1_4:.4f}")

    # ========================================================
    # 2. Missing Annotation Dropout Sweep: rho in [0 .. 0.30]
    # ========================================================
    drop_rates = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    print(f"\n--- 2. Evaluating Missing Annotation Dropout ({len(drop_rates)} drop rates, {args.num_trials} trials each) ---")
    drop_results = []

    for rho in drop_rates:
        trial_f1_8 = []
        trial_prec_8 = []
        trial_rec_8 = []

        for trial in range(args.num_trials):
            rng = np.random.default_rng(seed=1337 + trial * 100 + int(rho * 100))
            tot_tp8, tot_fp8, tot_fn8 = 0, 0, 0

            for item in cached_predictions:
                pts = item["pred_pts"]
                gt = item["gt_pts"]
                if len(gt) > 0 and rho > 0.0:
                    keep_mask = rng.uniform(size=len(gt)) >= rho
                    dropped_gt = gt[keep_mask]
                else:
                    dropped_gt = gt

                res8 = evaluate_point_localization(pts, dropped_gt, distance_thresholds=(8.0,))
                tot_tp8 += res8["TP@8"]
                tot_fp8 += res8["FP@8"]
                tot_fn8 += res8["FN@8"]

            p8 = tot_tp8 / (tot_tp8 + tot_fp8) if (tot_tp8 + tot_fp8) > 0 else 0.0
            r8 = tot_tp8 / (tot_tp8 + tot_fn8) if (tot_tp8 + tot_fn8) > 0 else 0.0
            f1_8 = 2 * p8 * r8 / (p8 + r8) if (p8 + r8) > 0 else 0.0

            trial_f1_8.append(f1_8)
            trial_prec_8.append(p8)
            trial_rec_8.append(r8)

        m_f1_8 = float(np.mean(trial_f1_8))
        s_f1_8 = float(np.std(trial_f1_8))
        m_p8 = float(np.mean(trial_prec_8))
        m_r8 = float(np.mean(trial_rec_8))

        drop_results.append({
            "drop_rate": rho,
            "f1_at_8px": round(m_f1_8, 4),
            "f1_at_8px_std": round(s_f1_8, 4),
            "prec_at_8px": round(m_p8, 4),
            "rec_at_8px": round(m_r8, 4),
        })
        print(f"  drop_rate={rho:4.2f} | F1@8px={m_f1_8:.4f} ± {s_f1_8:.4f} | Prec@8={m_p8:.4f} | Rec@8={m_r8:.4f}")

    # Save JSON summary
    summary_data = {
        "positional_jitter": jitter_results,
        "label_dropout": drop_results,
    }
    json_path = out_dir / "annotation_noise_sweep.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSaved annotation noise summary to: {json_path}")

    # ==========================================
    # Generate Publication Plots (2 Panels)
    # ==========================================
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Panel A: F1 vs. Positional Jitter sigma
    sigmas = [r["sigma_px"] for r in jitter_results]
    f1s_8 = [r["f1_at_8px"] for r in jitter_results]
    f1s_4 = [r["f1_at_4px"] for r in jitter_results]
    precs_8 = [r["prec_at_8px"] for r in jitter_results]
    recs_8 = [r["rec_at_8px"] for r in jitter_results]

    ax1.plot(sigmas, f1s_8, marker="o", markersize=7, linewidth=2.5, color="#2B6CB0", label="F1-score @ 8px")
    ax1.plot(sigmas, precs_8, marker="^", markersize=6, linewidth=1.8, color="#38A169", linestyle="--", label="Precision @ 8px")
    ax1.plot(sigmas, recs_8, marker="v", markersize=6, linewidth=1.8, color="#DD6B20", linestyle="--", label="Recall @ 8px")
    ax1.plot(sigmas, f1s_4, marker="s", markersize=6, linewidth=1.8, color="#805AD5", linestyle=":", label="F1-score @ 4px")

    for s, f1 in zip(sigmas, f1s_8):
        ax1.annotate(f"{f1:.3f}", (s, f1), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.5, fontweight="bold")

    ax1.set_xlabel("Annotation Positional Jitter Std Dev $\\sigma$ (pixels)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Localization Performance", fontsize=11, fontweight="bold")
    ax1.set_title("(a) Robustness to Positional Annotation Jitter (ShanghaiTech Part A)", fontsize=12, fontweight="bold", pad=10)
    ax1.legend(loc="upper right", frameon=True, fontsize=9)
    ax1.set_ylim(0.0, 0.75)

    # Panel B: F1, Precision, Recall vs. Label Dropout rate rho
    rhos = [r["drop_rate"] for r in drop_results]
    f1s_drop = [r["f1_at_8px"] for r in drop_results]
    precs_drop = [r["prec_at_8px"] for r in drop_results]
    recs_drop = [r["rec_at_8px"] for r in drop_results]

    ax2.plot(rhos, f1s_drop, marker="o", markersize=7, linewidth=2.5, color="#2B6CB0", label="F1-score @ 8px")
    ax2.plot(rhos, precs_drop, marker="^", markersize=6, linewidth=2.0, color="#E53E3E", linestyle="--", label="Precision @ 8px (drops as FP rises)")
    ax2.plot(rhos, recs_drop, marker="v", markersize=6, linewidth=2.0, color="#38A169", linestyle="-.", label="Recall @ 8px (stable on kept GT)")

    for r, f1 in zip(rhos, f1s_drop):
        ax2.annotate(f"{f1:.3f}", (r, f1), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.5, fontweight="bold")

    ax2.set_xlabel("Missing Annotation Dropout Rate $\\rho$ (0.0 = full GT, 0.30 = 30% missing)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Localization Performance @ 8px", fontsize=11, fontweight="bold")
    ax2.set_title("(b) Localization Behavior under Annotation Dropout (Missed Heads)", fontsize=12, fontweight="bold", pad=10)
    ax2.legend(loc="lower left", frameon=True, fontsize=9)
    ax2.set_ylim(0.3, 0.7)

    plt.tight_layout()
    fig_png = fig_dir / "fig6_annotation_noise_stress_test.png"
    fig_pdf = fig_dir / "fig6_annotation_noise_stress_test.pdf"
    plt.savefig(fig_png, dpi=300, bbox_inches="tight")
    plt.savefig(fig_pdf, bbox_inches="tight")
    plt.close()

    print(f"\nFigure 6 successfully generated:")
    print(f"  - PNG: {fig_png} ({fig_png.stat().st_size / 1024:.1f} KB)")
    print(f"  - PDF: {fig_pdf} ({fig_pdf.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
