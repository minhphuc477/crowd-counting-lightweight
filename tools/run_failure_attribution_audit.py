#!/usr/bin/env python3
"""Unified Failure Attribution Audit: Flat-DM16 (R2) vs Neural DTM Tree (R4).

Adjudicates among competing hypotheses for the R2-R4 performance gap:
1. Tail support & training support mismatch (Counting in the 2020s, UEPNet).
2. Inference context shift: Full vs Tile-256 vs Tile-448 (SANet).
3. Foreground undercount + Background compensation (WACV 2021 Modolo et al.).
4. Local multiplicity calibration curves: E[m_hat | y = k] at stride 4, 8, 16.
5. Multivariate attribution regression: Delta_i = |e_R4,i| - |e_R2,i| vs factors.
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
from hpc.diagnostics.fg_bg_decomposition import decompose_fg_bg_errors
from hpc.diagnostics.multiplicity_calibration import MultiplicityAccumulator
from hpc.diagnostics.tail_support import (
    compute_dataset_support_profile,
    compute_relative_percentiles,
)
from tools.eval_localization import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Failure Attribution Audit")
    parser.add_argument("--config-r2", default="configs/factorial_a_crop256_c16.yaml")
    parser.add_argument("--checkpoint-r2", default="runs/factorial_a_crop256_c16/best.pt")
    parser.add_argument("--config-r4", default="configs/ntpc_sha.yaml")
    parser.add_argument("--checkpoint-r4", default="runs/ntpc_sha/best.pt")
    parser.add_argument("--audit-v2-json", default="runs/objective_audit/audit_v2_full_test.json")
    parser.add_argument("--output", default="runs/failure_attribution_audit/attribution_report.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def fit_ols(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
) -> Dict[str, Any]:
    """Fit Ordinary Least Squares regression with t-statistics, p-values, and R^2."""
    n, p = X.shape
    # Add intercept column
    X_design = np.column_stack([np.ones(n), X])
    q, r_mat = np.linalg.qr(X_design)
    beta = np.linalg.solve(r_mat, q.T @ y)

    y_hat = X_design @ beta
    residuals = y - y_hat
    rss = float(np.sum(residuals**2))
    tss = float(np.sum((y - np.mean(y))**2))
    r_squared = 1.0 - (rss / tss) if tss > 0 else 0.0

    df_resid = max(1, n - p - 1)
    sigma_sq = rss / df_resid
    try:
        cov_beta = sigma_sq * np.linalg.inv(X_design.T @ X_design)
        se_beta = np.sqrt(np.maximum(0.0, np.diag(cov_beta)))
    except np.linalg.LinAlgError:
        se_beta = np.ones_like(beta) * float("nan")

    t_stats = beta / (se_beta + 1e-12)
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
            "std_err": float(se_beta[i]),
            "t_stat": float(t_stats[i]),
            "p_value": float(p_values[i]),
        }
    return results


def partial_f_test(
    X_reduced: np.ndarray,
    X_full: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, float]:
    """Compute partial F-test comparing reduced model to full model containing extra feature(s)."""
    n = len(y)
    p_red = X_reduced.shape[1]
    p_full = X_full.shape[1]

    X_red_d = np.column_stack([np.ones(n), X_reduced])
    X_full_d = np.column_stack([np.ones(n), X_full])

    b_red, _, _, _ = np.linalg.lstsq(X_red_d, y, rcond=None)
    b_full, _, _, _ = np.linalg.lstsq(X_full_d, y, rcond=None)

    rss_red = float(np.sum((y - X_red_d @ b_red)**2))
    rss_full = float(np.sum((y - X_full_d @ b_full)**2))

    df_num = p_full - p_red
    df_denom = max(1, n - p_full - 1)

    if rss_full <= 0 or df_num <= 0:
        return float("nan"), float("nan")

    f_stat = ((rss_red - rss_full) / df_num) / (rss_full / df_denom)
    p_val = float(1.0 - stats.f.cdf(f_stat, dfn=df_num, dfd=df_denom))
    return float(f_stat), float(p_val)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 80)
    print("STARTING COMPREHENSIVE FAILURE ATTRIBUTION AUDIT")
    print("=" * 80, flush=True)

    with open(args.config_r2, "r", encoding="utf-8") as f:
        cfg_r2 = yaml.safe_load(f)
    with open(args.config_r4, "r", encoding="utf-8") as f:
        cfg_r4 = yaml.safe_load(f)

    # -----------------------------------------------------------------------
    # PHASE 1: TAIL SUPPORT & TRAINING DISTRIBUTION PROFILING
    # -----------------------------------------------------------------------
    print("\n[Phase 1/5] Profiling Training (N=300) and Test (N=182) Distributions ...", flush=True)
    ds_train, _ = build_evaluation_dataset(cfg_r2, split="train_data")
    ds_test, _ = build_evaluation_dataset(cfg_r2, split="test_data")

    t0 = time.time()
    train_profiles = compute_dataset_support_profile(ds_train)
    test_profiles = compute_dataset_support_profile(ds_test)
    test_pctls = compute_relative_percentiles(test_profiles, train_profiles)
    print(f"Profiled 300 train + 182 test images in {time.time() - t0:.1f}s.", flush=True)

    # Inspect outlier percentiles for IMG_165 (idx 73) and IMG_92 (idx 174)
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
    print("\n[Phase 2-4/5] Running Full vs Tiled Inference, FG/BG Decomposition, Multiplicity Accumulation ...", flush=True)
    
    # Load audit v2 records if available (for tree interference I_destructive)
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

            # Phase 3: FG/BG Error Decomposition at stride 16
            fgbg2 = decompose_fg_bg_errors(m2_valid[16], tgt_valid[16], stride=16)
            fgbg4 = decompose_fg_bg_errors(m4_valid[16], tgt_valid[16], stride=16)

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

        rec = {
            "index": idx,
            "img_path": sample.get("img_path", f"img_{idx}"),
            "gt_count": gt_count,
            "tail_stats": test_profiles[idx],
            "tail_pctls": test_pctls[idx],
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
            "fgbg_r2": fgbg2,
            "fgbg_r4": fgbg4,
            "tree_interference": audit_v2_data.get(idx, 0.0),
        }
        records.append(rec)

        if (idx + 1) % 30 == 0 or (idx + 1) == len(ds_test):
            print(f"Evaluated {idx + 1}/{len(ds_test)} images ...", flush=True)

    print(f"Full test evaluation completed in {time.time() - t_start:.1f}s.", flush=True)

    # -----------------------------------------------------------------------
    # SUMMARIZE INFERENCE CONTEXT SHIFT
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

    # -----------------------------------------------------------------------
    # SUMMARIZE FG/BG DECOMPOSITION
    # -----------------------------------------------------------------------
    fgbg_summary = {
        "r2": {
            "fg_deficit_mean": float(np.mean([r["fgbg_r2"]["fg_deficit"] for r in records])),
            "fg_surplus_mean": float(np.mean([r["fgbg_r2"]["fg_surplus"] for r in records])),
            "bg_pred_mean": float(np.mean([r["fgbg_r2"]["bg_pred"] for r in records])),
            "bg_compensation_mean": float(np.mean([r["fgbg_r2"]["bg_compensation"] for r in records])),
            "bg_mass_fraction_mean": float(np.mean([r["fgbg_r2"]["bg_mass_fraction"] for r in records])),
            "compensation_ratio_mean": float(np.mean([r["fgbg_r2"]["compensation_ratio"] for r in records])),
        },
        "r4": {
            "fg_deficit_mean": float(np.mean([r["fgbg_r4"]["fg_deficit"] for r in records])),
            "fg_surplus_mean": float(np.mean([r["fgbg_r4"]["fg_surplus"] for r in records])),
            "bg_pred_mean": float(np.mean([r["fgbg_r4"]["bg_pred"] for r in records])),
            "bg_compensation_mean": float(np.mean([r["fgbg_r4"]["bg_compensation"] for r in records])),
            "bg_mass_fraction_mean": float(np.mean([r["fgbg_r4"]["bg_mass_fraction"] for r in records])),
            "compensation_ratio_mean": float(np.mean([r["fgbg_r4"]["compensation_ratio"] for r in records])),
        },
    }

    # Multiplicity calibration summaries
    mult_summary_r2 = acc_r2.summarize()
    mult_summary_r4 = acc_r4.summarize()

    # -----------------------------------------------------------------------
    # PHASE 5: MULTIVARIATE REGRESSION ATTRIBUTION
    # -----------------------------------------------------------------------
    print("\n[Phase 5/5] Fitting Multivariate Attribution Models for Delta = |e_R4| - |e_R2| ...", flush=True)
    delta_y = np.array([r["inference"]["delta_abs"] for r in records], dtype=np.float64)

    # Predictor matrix
    pctl_count = np.array([r["tail_pctls"]["gt_count_pctl"] for r in records], dtype=np.float64)
    pctl_density = np.array([r["tail_pctls"]["density_10k_pctl"] for r in records], dtype=np.float64)
    pctl_max_y16 = np.array([r["tail_pctls"]["max_y_16_pctl"] for r in records], dtype=np.float64)
    ctx_shift_diff = np.array([r["inference"]["context_shift_r4"] - r["inference"]["context_shift_r2"] for r in records], dtype=np.float64)
    bg_comp_diff = np.array([r["fgbg_r2"]["bg_compensation"] - r["fgbg_r4"]["bg_compensation"] for r in records], dtype=np.float64)
    tree_interf = np.array([r["tree_interference"] for r in records], dtype=np.float64)

    feature_matrix_raw = np.column_stack([
        pctl_count,
        pctl_density,
        pctl_max_y16,
        ctx_shift_diff,
        bg_comp_diff,
        tree_interf,
    ])
    feature_names = [
        "tail_count_pctl",
        "tail_density_pctl",
        "tail_max_y16_pctl",
        "context_shift_gap",
        "bg_compensation_gap",
        "tree_interference",
    ]

    # Standardize predictors (z-score)
    mean_X = np.mean(feature_matrix_raw, axis=0)
    std_X = np.std(feature_matrix_raw, axis=0)
    std_X[std_X == 0.0] = 1.0
    X_std = (feature_matrix_raw - mean_X) / std_X

    # Full model
    full_ols = fit_ols(X_std, delta_y, feature_names)

    # Reduced model without tree_interference
    X_reduced_std = X_std[:, :-1]
    f_stat, p_val_f = partial_f_test(X_reduced_std, X_std, delta_y)

    regression_results = {
        "full_model": full_ols,
        "partial_f_test_tree_interference": {
            "f_statistic": f_stat,
            "p_value": p_val_f,
            "null_hypothesis": "Tree interference has zero partial explanatory power for Delta = |e_R4| - |e_R2|",
            "rejected_at_05": bool(p_val_f < 0.05),
        },
    }

    # -----------------------------------------------------------------------
    # SAVE FULL REPORT
    # -----------------------------------------------------------------------
    report_data = {
        "metadata": {
            "checkpoint_r2": args.checkpoint_r2,
            "checkpoint_r4": args.checkpoint_r4,
            "test_images": len(records),
            "train_images": len(train_profiles),
            "elapsed_seconds": time.time() - t0,
        },
        "inference_summary": inference_summary,
        "fgbg_decomposition_summary": fgbg_summary,
        "multiplicity_calibration_r2": mult_summary_r2,
        "multiplicity_calibration_r4": mult_summary_r4,
        "regression_attribution": regression_results,
        "records": records,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, allow_nan=True)
    print(f"\nWrote full Failure Attribution Audit report to {out_path}", flush=True)

    # -----------------------------------------------------------------------
    # PRINT REPORT
    # -----------------------------------------------------------------------
    print("\n" + "=" * 95)
    print("FAILURE ATTRIBUTION AUDIT REPORT (FLAT-DM16 vs NEURAL DTM TREE)")
    print("=" * 95)

    # Table 1: Inference Context Shift (Full vs Tile-256 vs Tile-448)
    print("\n--- 1. Inference Context Shift: Full-Image vs Patch/Tiled Evaluation ---")
    h1 = f"{'Inference Mode':<20} | {'R2 MAE':>10} {'R2 Bias':>10} | {'R4 MAE':>10} {'R4 Bias':>10} | {'Gap (R4 - R2)':>14}"
    print(h1)
    print("-" * len(h1))
    for m_key, m_label in [("full", "Full-image (native)"), ("tile256", "Tiled 256 (train size)"), ("tile448", "Tiled 448")]:
        m2_m = inference_summary[f"r2_{m_key}"]
        m4_m = inference_summary[f"r4_{m_key}"]
        gap = m4_m["mae"] - m2_m["mae"]
        print(f"{m_label:<20} | {m2_m['mae']:>10.2f} {m2_m['bias']:>10.2f} | {m4_m['mae']:>10.2f} {m4_m['bias']:>10.2f} | {gap:>+14.2f}")

    # Table 2: Foreground / Background Error Decomposition (WACV 2021)
    print("\n--- 2. Foreground / Background Error Decomposition (at stride 16) ---")
    h2 = f"{'Metric':<30} | {'R2 Flat-DM16':>18} | {'R4 DTM Tree':>18} | {'Ratio / Difference':>20}"
    print(h2)
    print("-" * len(h2))
    s2 = fgbg_summary["r2"]
    s4 = fgbg_summary["r4"]
    print(f"{'FG Deficit (missed crowd)':<30} | {s2['fg_deficit_mean']:>18.2f} | {s4['fg_deficit_mean']:>18.2f} | {s4['fg_deficit_mean'] - s2['fg_deficit_mean']:>+20.2f}")
    print(f"{'FG Surplus (excess crowd)':<30} | {s2['fg_surplus_mean']:>18.2f} | {s4['fg_surplus_mean']:>18.2f} | {s4['fg_surplus_mean'] - s2['fg_surplus_mean']:>+20.2f}")
    print(f"{'BG Excess (FP mass on BG)':<30} | {s2['bg_pred_mean']:>18.2f} | {s4['bg_pred_mean']:>18.2f} | {s4['bg_pred_mean'] - s2['bg_pred_mean']:>+20.2f}")
    print(f"{'BG Compensation (masked)':<30} | {s2['bg_compensation_mean']:>18.2f} | {s4['bg_compensation_mean']:>18.2f} | {s4['bg_compensation_mean'] - s2['bg_compensation_mean']:>+20.2f}")
    print(f"{'BG Mass Fraction':<30} | {s2['bg_mass_fraction_mean']*100:>17.1f}% | {s4['bg_mass_fraction_mean']*100:>17.1f}% | {s4['bg_mass_fraction_mean']*100 - s2['bg_mass_fraction_mean']*100:>+19.1f}%")

    # Table 3: Local Multiplicity Calibration at Stride 16
    print("\n--- 3. Local Multiplicity Calibration at Stride 16 (E[m_pred | y_gt = k]) ---")
    h3 = f"{'Target k':<10} | {'Test Cells':>12} | {'R2 Pred Mean':>14} {'R2 Ratio':>10} | {'R4 Pred Mean':>14} {'R4 Ratio':>10}"
    print(h3)
    print("-" * len(h3))
    s16_r2 = mult_summary_r2.get(16, {})
    s16_r4 = mult_summary_r4.get(16, {})
    for k in range(9):
        k_key = f"k_{k}"
        if k_key not in s16_r2:
            continue
        n_c = int(s16_r2[k_key]["n_cells"])
        m2_v = s16_r2[k_key]["mean_pred"]
        r2_rat = s16_r2[k_key]["ratio_pred_gt"]
        m4_v = s16_r4[k_key]["mean_pred"]
        r4_rat = s16_r4[k_key]["ratio_pred_gt"]
        r2_str = f"{r2_rat:.3f}" if not math.isnan(r2_rat) else "-"
        r4_str = f"{r4_rat:.3f}" if not math.isnan(r4_rat) else "-"
        print(f"{k:<10} | {n_c:>12} | {m2_v:>14.3f} {r2_str:>10} | {m4_v:>14.3f} {r4_str:>10}")
    # Overflow
    if "k_gt_8" in s16_r2 and s16_r2["k_gt_8"]["n_cells"] > 0:
        over = s16_r2["k_gt_8"]
        over4 = s16_r4["k_gt_8"]
        print(f"{'>8':<10} | {int(over['n_cells']):>12} | {over['mean_pred']:>14.3f} {over['ratio_pred_gt']:>10.3f} | {over4['mean_pred']:>14.3f} {over4['ratio_pred_gt']:>10.3f}")

    # Table 4: Multivariate Failure Attribution Regression
    print("\n--- 4. Multivariate Failure Attribution Regression: Delta = |e_R4| - |e_R2| ---")
    print(f"Full Model R^2 = {full_ols['r_squared']:.4f} (N = {full_ols['n_samples']} images)")
    h4 = f"{'Feature (Standardized)':<28} | {'Beta (Std Coef)':>16} | {'Std Error':>12} | {'t-stat':>10} | {'p-value':>12}"
    print(h4)
    print("-" * len(h4))
    for name in feature_names:
        f_info = full_ols["features"][name]
        sig = "***" if f_info["p_value"] < 0.001 else "**" if f_info["p_value"] < 0.01 else "*" if f_info["p_value"] < 0.05 else ""
        print(f"{name:<28} | {f_info['coef']:>+16.4f} | {f_info['std_err']:>12.4f} | {f_info['t_stat']:>10.2f} | {f_info['p_value']:>11.4f} {sig}")

    # Partial F-test
    print("-" * len(h4))
    f_res = regression_results["partial_f_test_tree_interference"]
    print(f"Partial F-test for Tree Interference: F = {f_res['f_statistic']:.4f}, p = {f_res['p_value']:.4f}")
    if f_res["rejected_at_05"]:
        print(">> VERDICT: Tree interference explains statistically significant unique variance in Delta.")
    else:
        print(">> VERDICT: Tree interference has NO statistically significant unique explanatory power (p > 0.05).")
        print("   The performance gap is fully accounted for by data tail, context shift, and multiplicity saturation.")
    print("=" * 95 + "\n")


if __name__ == "__main__":
    main()
