#!/usr/bin/env python3
"""Unified Failure Attribution Audit v2: Flat-DM16 (R2) vs Neural DTM Tree (R4).

Adjudicates among competing hypotheses for the R2-R4 performance gap:
1. Tail support & local crop support mismatch (6,119 sliding training crops).
2. Inference context shift: Full vs Tile-256 vs Tile-448.
3. Cell occupancy error accounting (occupied cell deficit vs empty cell mass).
4. Local multiplicity calibration with image-cluster bootstrap (B=1000, 95% CIs).
5. Multivariate attribution regression with MacKinnon-White (1985) HC3 robust standard errors.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hpc.data.factory import build_evaluation_dataset
from hpc.data.point_counts import build_exact_count_pyramid
from hpc.diagnostics.fg_bg_decomposition import decompose_cell_occupancy_errors
from hpc.diagnostics.multiplicity_calibration import MultiplicityAccumulator
from hpc.diagnostics.tail_support import (
    compute_crop_percentile,
    compute_dataset_support_profile,
    compute_relative_percentiles,
    profile_crop_support_distribution,
)
from tools.eval_localization import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Failure Attribution Audit v2")
    parser.add_argument("--config-r2", default="configs/factorial_a_crop256_c16.yaml")
    parser.add_argument("--checkpoint-r2", default="runs/factorial_a_crop256_c16/best.pt")
    parser.add_argument("--config-r4", default="configs/ntpc_sha.yaml")
    parser.add_argument("--checkpoint-r4", default="runs/ntpc_sha/best.pt")
    parser.add_argument("--audit-v2-json", default="runs/objective_audit/audit_v2_full_test.json")
    parser.add_argument("--output", default="runs/failure_attribution_audit/attribution_report.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def fit_ols_hc3(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
) -> Dict[str, Any]:
    """Fit OLS regression with MacKinnon-White (1985) HC3 robust covariance matrix."""
    n, p = X.shape
    X_d = np.column_stack([np.ones(n), X])
    q, r_mat = np.linalg.qr(X_d)
    beta = np.linalg.solve(r_mat, q.T @ y)

    y_hat = X_d @ beta
    residuals = y - y_hat
    rss = float(np.sum(residuals**2))
    tss = float(np.sum((y - np.mean(y))**2))
    r_squared = 1.0 - (rss / tss) if tss > 0 else 0.0

    # Leverage h_ii = diag(X_d (X_d^T X_d)^{-1} X_d^T)
    XtX_inv = np.linalg.inv(X_d.T @ X_d)
    h = np.sum((X_d @ XtX_inv) * X_d, axis=1)
    h = np.clip(h, 0.0, 0.999)

    # HC3 weight: omega_i = e_i^2 / (1 - h_ii)^2
    omega = (residuals / (1.0 - h))**2
    cov_hc3 = XtX_inv @ (X_d.T @ (omega[:, None] * X_d)) @ XtX_inv
    se_hc3 = np.sqrt(np.maximum(0.0, np.diag(cov_hc3)))

    df_resid = max(1, n - p - 1)
    t_stats = beta / (se_hc3 + 1e-12)
    p_values = [float(2.0 * (1.0 - stats.t.cdf(abs(t), df=df_resid))) for t in t_stats]

    results = {
        "r_squared": r_squared,
        "n_samples": n,
        "features": {},
    }
    all_names = ["intercept"] + feature_names
    for i, name in enumerate(all_names):
        results["features"][name] = {
            "coef": float(beta[i]),
            "hc3_std_err": float(se_hc3[i]),
            "t_stat": float(t_stats[i]),
            "p_value": float(p_values[i]),
        }
    return results


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 80)
    print("STARTING FAILURE ATTRIBUTION AUDIT V2 (SCIENTIFIC STANDARD)")
    print("=" * 80, flush=True)

    with open(args.config_r2, "r", encoding="utf-8") as f:
        cfg_r2 = yaml.safe_load(f)
    with open(args.config_r4, "r", encoding="utf-8") as f:
        cfg_r4 = yaml.safe_load(f)

    # -----------------------------------------------------------------------
    # PHASE 1: TAIL SUPPORT & 6,119 TRAINING CROP PROFILING
    # -----------------------------------------------------------------------
    print("\n[Phase 1/5] Profiling Training (N=300) and Test (N=182) Distributions ...", flush=True)
    ds_train, _ = build_evaluation_dataset(cfg_r2, split="train_data")
    ds_test, _ = build_evaluation_dataset(cfg_r2, split="test_data")

    t0 = time.time()
    train_profiles = compute_dataset_support_profile(ds_train)
    test_profiles = compute_dataset_support_profile(ds_test)
    test_pctls = compute_relative_percentiles(test_profiles, train_profiles)

    # Profile sliding 256x256 crops across train set
    print("  Profiling sliding crop count distribution across 300 training images ...", flush=True)
    train_crop_counts = profile_crop_support_distribution(ds_train, crop_size=256, step=128)
    print(
        f"  Profiled {len(train_crop_counts)} training crops in {time.time() - t0:.1f}s. "
        f"Median={np.median(train_crop_counts):.1f}, p90={np.percentile(train_crop_counts, 90):.1f}, "
        f"p99={np.percentile(train_crop_counts, 99):.1f}, Max={np.max(train_crop_counts)}",
        flush=True,
    )

    # Outliers in test set
    outlier_73 = test_pctls[73]
    outlier_174 = test_pctls[174]
    print(f"  IMG_165 (idx 73)  : Count Pctl={outlier_73['gt_count_pctl']:.1f}%, Density Pctl={outlier_73['density_10k_pctl']:.1f}%, Max Y16 Pctl={outlier_73['max_y_16_pctl']:.1f}%")
    print(f"  IMG_92  (idx 174) : Count Pctl={outlier_174['gt_count_pctl']:.1f}%, Density Pctl={outlier_174['density_10k_pctl']:.1f}%, Max Y16 Pctl={outlier_174['max_y_16_pctl']:.1f}%")

    # -----------------------------------------------------------------------
    # LOAD MODELS FOR INFERENCE PHASES
    # -----------------------------------------------------------------------
    print("\nLoading R2 and R4 models to device ...", flush=True)
    m2 = build_model(cfg_r2, args.checkpoint_r2, device)
    m4 = build_model(cfg_r4, args.checkpoint_r4, device)

    # -----------------------------------------------------------------------
    # PHASE 2 & 3 & 4: TEST SET EVALUATION
    # -----------------------------------------------------------------------
    print("\n[Phase 2-4/5] Running Full vs Tiled Inference, Cell Occupancy Accounting, Multiplicity Accumulation ...", flush=True)

    audit_v2_data: Dict[int, float] = {}
    if os.path.isfile(args.audit_v2_json):
        with open(args.audit_v2_json, "r", encoding="utf-8") as f:
            v2_json = json.load(f)
        for r in v2_json.get("records_r4", []):
            idx = r["image_index"]
            i_dest = r["r4"].get("pairwise_destructive_interferences", {}).get("64_to_32_vs_32_to_16", 0.0)
            audit_v2_data[idx] = float(i_dest)

    acc_r2 = MultiplicityAccumulator(strides=(4, 8, 16), max_k=8)
    acc_r4 = MultiplicityAccumulator(strides=(4, 8, 16), max_k=8)

    records: List[Dict[str, Any]] = []
    t_start = time.time()

    for idx in range(len(ds_test)):
        sample = ds_test[idx]
        img = sample["image"].unsqueeze(0).to(device)
        gt_count = float(sample["gt_count"])
        pts = np.asarray(sample["gt_points"], dtype=np.float32).reshape(-1, 2).copy()
        h, w = int(img.shape[-2]), int(img.shape[-1])
        pts[:, 0] = np.clip(pts[:, 0], 0.0, max(0.0, float(w - 1)))
        pts[:, 1] = np.clip(pts[:, 1], 0.0, max(0.0, float(h - 1)))

        # Exact ground truth pyramids
        target_pyramid = build_exact_count_pyramid(
            [torch.from_numpy(pts).float()],
            height=h,
            width=w,
            block_sizes=(4, 8, 16),
            pad_multiple=64,
            device=device,
        )

        with torch.no_grad():
            # Phase 2: Full-image, Tile-256, Tile-448
            c2_full, _ = m2.predict(img)
            c4_full, _ = m4.predict(img)

            c2_tile256, _ = m2.predict_tiled(img, tile_size=256, halo=64)
            c4_tile256, _ = m4.predict_tiled(img, tile_size=256, halo=64)

            c2_tile448, _ = m2.predict_tiled(img, tile_size=448, halo=64)
            c4_tile448, _ = m4.predict_tiled(img, tile_size=448, halo=64)

            # Padded forward for exact grid decomposition and multiplicity alignment
            pad_h = (64 - (h % 64)) % 64
            pad_w = (64 - (w % 64)) % 64
            img_pad = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h))
            mass_pad2 = m2.forward_mass(img_pad)
            mass_pad4 = m4.forward_mass(img_pad)

            m2_valid = {}
            m4_valid = {}
            tgt_valid = {}
            for s in (4, 8, 16):
                k = s // 4
                m2_s = torch.nn.functional.avg_pool2d(mass_pad2, kernel_size=k, stride=k) * (k * k)
                m4_s = torch.nn.functional.avg_pool2d(mass_pad4, kernel_size=k, stride=k) * (k * k)
                out_h = math.ceil(h / s)
                out_w = math.ceil(w / s)
                m2_valid[s] = m2_s[..., :out_h, :out_w]
                m4_valid[s] = m4_s[..., :out_h, :out_w]
                tgt_valid[s] = target_pyramid[s][..., :out_h, :out_w]

            # Phase 3: Cell Occupancy Decomposition at stride 16
            occ2 = decompose_cell_occupancy_errors(m2_valid[16], tgt_valid[16], stride=16)
            occ4 = decompose_cell_occupancy_errors(m4_valid[16], tgt_valid[16], stride=16)

            # Phase 4: Accumulate Multiplicity pairs at stride 4, 8, 16
            for s in (4, 8, 16):
                acc_r2.add_image(m2_valid[s], {s: tgt_valid[s]})
                acc_r4.add_image(m4_valid[s], {s: tgt_valid[s]})

        # Context shift
        shift2 = abs(c2_full.item() - c2_tile256.item())
        shift4 = abs(c4_full.item() - c4_tile256.item())

        e2_abs = abs(c2_full.item() - gt_count)
        e4_abs = abs(c4_full.item() - gt_count)
        delta_abs = e4_abs - e2_abs  # positive when R4 is worse than R2

        # Max crop count in this image
        max_crop_in_img = 0
        if len(pts) > 0:
            for cy in range(0, max(1, h - 256 + 1), 128):
                for cx in range(0, max(1, w - 256 + 1), 128):
                    inc = (pts[:, 0] >= cx) & (pts[:, 0] < cx + 256) & (pts[:, 1] >= cy) & (pts[:, 1] < cy + 256)
                    cnt = int(inc.sum())
                    if cnt > max_crop_in_img:
                        max_crop_in_img = cnt
        crop_pctl = compute_crop_percentile(max_crop_in_img, train_crop_counts)

        rec = {
            "index": idx,
            "img_path": sample.get("img_path", f"img_{idx}"),
            "gt_count": gt_count,
            "tail_stats": test_profiles[idx],
            "tail_pctls": test_pctls[idx],
            "max_crop_in_img": max_crop_in_img,
            "crop_pctl": crop_pctl,
            "inference": {
                "r2_full": float(c2_full.item()),
                "r4_full": float(c4_full.item()),
                "r2_tile256": float(c2_tile256.item()),
                "r4_tile256": float(c4_tile256.item()),
                "r2_tile448": float(c2_tile448.item()),
                "r4_tile448": float(c4_tile448.item()),
                "e2_full_abs": float(e2_abs),
                "e4_full_abs": float(e4_abs),
                "delta_abs": float(delta_abs),
                "context_shift_r2": float(shift2),
                "context_shift_r4": float(shift4),
            },
            "occupancy_r2": occ2,
            "occupancy_r4": occ4,
            "tree_interference": audit_v2_data.get(idx, 0.0),
        }
        records.append(rec)

        if (idx + 1) % 45 == 0 or (idx + 1) == len(ds_test):
            print(f"Evaluated {idx + 1}/{len(ds_test)} images ...", flush=True)

    print(f"Full test evaluation completed in {time.time() - t_start:.1f}s.", flush=True)

    # -----------------------------------------------------------------------
    # SUMMARIES
    # -----------------------------------------------------------------------
    def _mae(preds: List[float], gts: List[float]) -> float:
        return float(np.mean(np.abs(np.array(preds) - np.array(gts))))

    def _bias(preds: List[float], gts: List[float]) -> float:
        return float(np.mean(np.array(preds) - np.array(gts)))

    gts = [r["gt_count"] for r in records]
    p2_f = [r["inference"]["r2_full"] for r in records]
    p4_f = [r["inference"]["r4_full"] for r in records]
    p2_t256 = [r["inference"]["r2_tile256"] for r in records]
    p4_t256 = [r["inference"]["r4_tile256"] for r in records]
    p2_t448 = [r["inference"]["r2_tile448"] for r in records]
    p4_t448 = [r["inference"]["r4_tile448"] for r in records]

    inference_summary = {
        "r2_full": {"mae": _mae(p2_f, gts), "bias": _bias(p2_f, gts)},
        "r4_full": {"mae": _mae(p4_f, gts), "bias": _bias(p4_f, gts)},
        "r2_tile256": {"mae": _mae(p2_t256, gts), "bias": _bias(p2_t256, gts)},
        "r4_tile256": {"mae": _mae(p4_t256, gts), "bias": _bias(p4_t256, gts)},
        "r2_tile448": {"mae": _mae(p2_t448, gts), "bias": _bias(p2_t448, gts)},
        "r4_tile448": {"mae": _mae(p4_t448, gts), "bias": _bias(p4_t448, gts)},
    }

    # Descriptive Cell Occupancy Accounting (Stride 16)
    occupancy_summary = {
        "r2": {
            "occupied_deficit_mean": float(np.mean([r["occupancy_r2"]["occupied_deficit"] for r in records])),
            "occupied_surplus_mean": float(np.mean([r["occupancy_r2"]["occupied_surplus"] for r in records])),
            "empty_cell_mass_mean": float(np.mean([r["occupancy_r2"]["empty_cell_mass"] for r in records])),
            "empty_cell_compensation_mean": float(np.mean([r["occupancy_r2"]["empty_cell_compensation"] for r in records])),
            "empty_cell_mass_fraction_mean": float(np.mean([r["occupancy_r2"]["empty_cell_mass_fraction"] for r in records])),
            "compensation_ratio_mean": float(np.mean([r["occupancy_r2"]["compensation_ratio"] for r in records])),
        },
        "r4": {
            "occupied_deficit_mean": float(np.mean([r["occupancy_r4"]["occupied_deficit"] for r in records])),
            "occupied_surplus_mean": float(np.mean([r["occupancy_r4"]["occupied_surplus"] for r in records])),
            "empty_cell_mass_mean": float(np.mean([r["occupancy_r4"]["empty_cell_mass"] for r in records])),
            "empty_cell_compensation_mean": float(np.mean([r["occupancy_r4"]["empty_cell_compensation"] for r in records])),
            "empty_cell_mass_fraction_mean": float(np.mean([r["occupancy_r4"]["empty_cell_mass_fraction"] for r in records])),
            "compensation_ratio_mean": float(np.mean([r["occupancy_r4"]["compensation_ratio"] for r in records])),
        },
    }

    # Image-Cluster Bootstrap Multiplicity Calibration
    print("\nRunning Image-Cluster Bootstrap (B=1000) for Multiplicity Calibration ...", flush=True)
    boot_r2 = acc_r2.cluster_bootstrap(stride=16, n_boot=1000, seed=args.seed)
    boot_r4 = acc_r4.cluster_bootstrap(stride=16, n_boot=1000, seed=args.seed)
    paired_boot = MultiplicityAccumulator.cluster_bootstrap_paired_diff(acc_r4, acc_r2, stride=16, n_boot=1000, seed=args.seed)
    pooled_r2 = acc_r2.summarize()
    pooled_r4 = acc_r4.summarize()

    # -----------------------------------------------------------------------
    # PHASE 5: HC3 ROBUST MULTIVARIATE REGRESSION (EXOGENOUS PREDICTORS ONLY)
    # -----------------------------------------------------------------------
    print("\n[Phase 5/5] Fitting HC3 Robust OLS Model for Delta = |e_R4| - |e_R2| ...", flush=True)
    delta_y = np.array([r["inference"]["delta_abs"] for r in records], dtype=np.float64)

    # Exogenous predictors only (No circular compensation metric!)
    pctl_count = np.array([r["tail_pctls"]["gt_count_pctl"] for r in records], dtype=np.float64)
    pctl_density = np.array([r["tail_pctls"]["density_10k_pctl"] for r in records], dtype=np.float64)
    pctl_crop_max = np.array([r["crop_pctl"] for r in records], dtype=np.float64)
    pctl_nn_p10 = np.array([r["tail_pctls"]["nn_p10_pctl"] for r in records], dtype=np.float64)
    ctx_shift_diff = np.array([r["inference"]["context_shift_r4"] - r["inference"]["context_shift_r2"] for r in records], dtype=np.float64)
    tree_interf = np.array([r["tree_interference"] for r in records], dtype=np.float64)

    feature_matrix_raw = np.column_stack([
        pctl_count,
        pctl_density,
        pctl_crop_max,
        pctl_nn_p10,
        ctx_shift_diff,
        tree_interf,
    ])
    feature_names = [
        "tail_count_pctl",
        "tail_density_pctl",
        "tail_crop_max_pctl",
        "tail_nn_p10_pctl",
        "context_shift_gap",
        "tree_interference",
    ]

    # Standardize predictors (z-scores)
    mean_X = np.mean(feature_matrix_raw, axis=0)
    std_X = np.std(feature_matrix_raw, axis=0)
    std_X[std_X == 0.0] = 1.0
    X_std = (feature_matrix_raw - mean_X) / std_X

    # Fit HC3 robust regression
    hc3_model = fit_ols_hc3(X_std, delta_y, feature_names)

    # -----------------------------------------------------------------------
    # SAVE REPORT
    # -----------------------------------------------------------------------
    report_data = {
        "metadata": {
            "checkpoint_r2": args.checkpoint_r2,
            "checkpoint_r4": args.checkpoint_r4,
            "test_images": len(records),
            "train_images": len(train_profiles),
            "train_crops_profiled": len(train_crop_counts),
            "elapsed_seconds": time.time() - t0,
        },
        "inference_summary": inference_summary,
        "occupancy_accounting_summary": occupancy_summary,
        "multiplicity_calibration": {
            "pooled_r2": pooled_r2,
            "pooled_r4": pooled_r4,
            "cluster_bootstrap_r2": boot_r2,
            "cluster_bootstrap_r4": boot_r4,
            "cluster_bootstrap_paired_diff": paired_boot,
        },
        "regression_attribution_hc3": hc3_model,
        "records": records,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, allow_nan=True)
    print(f"Wrote audit report to {out_path}", flush=True)

    # -----------------------------------------------------------------------
    # PRINT REPORT
    # -----------------------------------------------------------------------
    print("\n" + "=" * 105)
    print("FAILURE ATTRIBUTION AUDIT V2 (SCIENTIFIC STANDARD REPORT)")
    print("=" * 105)

    # Table 1: Context Shift
    print("\n--- 1. Inference Context Shift: Full-Image vs Patch/Tiled Evaluation ---")
    h1 = f"{'Inference Mode':<22} | {'R2 MAE':>10} {'R2 Bias':>10} | {'R4 MAE':>10} {'R4 Bias':>10} | {'Gap (R4 - R2)':>14}"
    print(h1)
    print("-" * len(h1))
    for m_key, m_label in [("full", "Full-image (native)"), ("tile256", "Tiled 256 (train size)"), ("tile448", "Tiled 448")]:
        m2_m = inference_summary[f"r2_{m_key}"]
        m4_m = inference_summary[f"r4_{m_key}"]
        gap = m4_m["mae"] - m2_m["mae"]
        print(f"{m_label:<22} | {m2_m['mae']:>10.2f} {m2_m['bias']:>10.2f} | {m4_m['mae']:>10.2f} {m4_m['bias']:>10.2f} | {gap:>+14.2f}")

    # Table 2: Cell Occupancy Accounting
    print("\n--- 2. Descriptive Cell Occupancy Accounting (Grid Stride 16) ---")
    print("     [Note: Measures spatial cell sparsity, not semantic FG/BG segmentation]")
    h2 = f"{'Accounting Metric':<32} | {'R2 Flat-DM16':>18} | {'R4 DTM Tree':>18} | {'Difference (R4 - R2)':>22}"
    print(h2)
    print("-" * len(h2))
    s2 = occupancy_summary["r2"]
    s4 = occupancy_summary["r4"]
    print(f"{'Occupied Deficit (missed)':<32} | {s2['occupied_deficit_mean']:>18.2f} | {s4['occupied_deficit_mean']:>18.2f} | {s4['occupied_deficit_mean'] - s2['occupied_deficit_mean']:>+22.2f}")
    print(f"{'Occupied Surplus (excess)':<32} | {s2['occupied_surplus_mean']:>18.2f} | {s4['occupied_surplus_mean']:>18.2f} | {s4['occupied_surplus_mean'] - s2['occupied_surplus_mean']:>+22.2f}")
    print(f"{'Empty Cell Mass (sparse mass)':<32} | {s2['empty_cell_mass_mean']:>18.2f} | {s4['empty_cell_mass_mean']:>18.2f} | {s4['empty_cell_mass_mean'] - s2['empty_cell_mass_mean']:>+22.2f}")
    print(f"{'Empty Cell Compensation':<32} | {s2['empty_cell_compensation_mean']:>18.2f} | {s4['empty_cell_compensation_mean']:>18.2f} | {s4['empty_cell_compensation_mean'] - s2['empty_cell_compensation_mean']:>+22.2f}")
    print(f"{'Empty Cell Mass Fraction':<32} | {s2['empty_cell_mass_fraction_mean']*100:>17.1f}% | {s4['empty_cell_mass_fraction_mean']*100:>17.1f}% | {s4['empty_cell_mass_fraction_mean']*100 - s2['empty_cell_mass_fraction_mean']*100:>+21.1f}%")

    # Table 3: Multiplicity Calibration with Cluster Bootstrap
    print("\n--- 3. Local Multiplicity Calibration with Image-Cluster Bootstrap (Stride 16, B=1000) ---")
    h3 = f"{'k':<5} | {'Cells':>8} {'Imgs':>6} | {'R2 Mean [95% CI]':>26} | {'R4 Mean [95% CI]':>26} | {'Diff (R4 - R2) [95% CI]':>26} {'Sig?':>5}"
    print(h3)
    print("-" * len(h3))
    s16_r2 = pooled_r2.get(16, {})
    for k in range(9):
        k_key = f"k_{k}"
        if k_key not in s16_r2:
            continue
        n_c = int(s16_r2[k_key]["n_cells"])
        n_img = int(boot_r2.get(k_key, {}).get("n_contributing_images", 0))
        r2_b = boot_r2.get(k_key, {})
        r4_b = boot_r4.get(k_key, {})
        p_b = paired_boot.get(k_key, {})

        r2_str = f"{r2_b.get('mean', float('nan')):.3f} [{r2_b.get('ci_lower', float('nan')):.3f}, {r2_b.get('ci_upper', float('nan')):.3f}]"
        r4_str = f"{r4_b.get('mean', float('nan')):.3f} [{r4_b.get('ci_lower', float('nan')):.3f}, {r4_b.get('ci_upper', float('nan')):.3f}]"
        diff_str = f"{p_b.get('diff_mean', float('nan')):+0.3f} [{p_b.get('diff_ci_lower', float('nan')):+0.3f}, {p_b.get('diff_ci_upper', float('nan')):+0.3f}]"
        sig_str = "YES*" if p_b.get("significant", False) else "no"
        print(f"{k:<5} | {n_c:>8} {n_img:>6} | {r2_str:>26} | {r4_str:>26} | {diff_str:>26} {sig_str:>5}")

    # Overflow >8
    if "k_gt_8" in boot_r2:
        k_key = "k_gt_8"
        n_c = int(pooled_r2.get(16, {}).get(k_key, {}).get("n_cells", 0))
        n_img = int(boot_r2.get(k_key, {}).get("n_contributing_images", 0))
        r2_b = boot_r2.get(k_key, {})
        r4_b = boot_r4.get(k_key, {})
        p_b = paired_boot.get(k_key, {})
        r2_str = f"{r2_b.get('mean', float('nan')):.3f} [{r2_b.get('ci_lower', float('nan')):.3f}, {r2_b.get('ci_upper', float('nan')):.3f}]"
        r4_str = f"{r4_b.get('mean', float('nan')):.3f} [{r4_b.get('ci_lower', float('nan')):.3f}, {r4_b.get('ci_upper', float('nan')):.3f}]"
        diff_str = f"{p_b.get('diff_mean', float('nan')):+0.3f} [{p_b.get('diff_ci_lower', float('nan')):+0.3f}, {p_b.get('diff_ci_upper', float('nan')):+0.3f}]"
        sig_str = "YES*" if p_b.get("significant", False) else "no"
        print(f"{'>8':<5} | {n_c:>8} {n_img:>6} | {r2_str:>26} | {r4_str:>26} | {diff_str:>26} {sig_str:>5}")

    # Table 4: HC3 Robust Attribution Regression
    print("\n--- 4. Heteroskedasticity-Robust Attribution Regression (HC3 Covariance) ---")
    print(f"Target: Delta = |e_R4| - |e_R2| | R^2 = {hc3_model['r_squared']:.4f} (N = {hc3_model['n_samples']} images)")
    h4 = f"{'Exogenous Predictor':<28} | {'Beta (Std Coef)':>16} | {'HC3 Robust SE':>14} | {'t-stat':>10} | {'p-value':>12}"
    print(h4)
    print("-" * len(h4))
    for name in feature_names:
        f_info = hc3_model["features"][name]
        sig = "***" if f_info["p_value"] < 0.001 else "**" if f_info["p_value"] < 0.01 else "*" if f_info["p_value"] < 0.05 else ""
        print(f"{name:<28} | {f_info['coef']:>+16.4f} | {f_info['hc3_std_err']:>14.4f} | {f_info['t_stat']:>10.2f} | {f_info['p_value']:>11.4f} {sig}")

    tree_p = hc3_model["features"]["tree_interference"]["p_value"]
    print("-" * len(h4))
    print(f"Tree Interference (HC3 robust test): t = {hc3_model['features']['tree_interference']['t_stat']:.2f}, p = {tree_p:.4f}")
    if tree_p >= 0.05:
        print(">> VERDICT: The measured tree-interference statistic provides no additional linear explanatory value in this attribution model.")
    else:
        print(">> VERDICT: The measured tree-interference statistic exhibits a statistically significant association with Delta.")
    print("=" * 105 + "\n")


if __name__ == "__main__":
    main()
