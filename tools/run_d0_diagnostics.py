#!/usr/bin/env python3
"""Run D0 Pre-Model Diagnostic Suite (D-R, D-K, D-L, D-M).

Evaluates candidate bottlenecks on trained checkpoints or baseline models:
- D-R / G-R: +/-1, +/-2 px translation / sampling-phase instability.
- D-K / G-K: Inter-person separability collapse across encoder depth.
- D-L / G-L: Normalized effective representation rank collapse.
- D-M / G-M: Foreground crowd vs background gradient allocation.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from hpc.data import ShanghaiTechDataset, ntpc_collate_fn
from hpc.diagnostics import (
    evaluate_effective_rank_single_image,
    evaluate_gradient_allocation_single_batch,
    evaluate_phase_shift_single_image,
    evaluate_separability_single_image,
)
from hpc.losses.ntpc import NTPCConfig, NTPCLoss
from hpc.models.factory import build_model_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run D0 Diagnostic Suite")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint .pt")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--split", default="test_data", help="Dataset split (train_data, val_data, test_data)")
    parser.add_argument("--max-samples", type=int, default=None, help="Max images to evaluate")
    parser.add_argument(
        "--diagnostics",
        default="all",
        help="Comma-separated diagnostics to run: dr,dk,dl,dm or all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running D0 Diagnostic Suite on device: {device}", flush=True)

    # Load model
    model = build_model_from_config(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint} (epoch={ckpt.get('epoch', 'N/A')})", flush=True)

    # Build dataset
    ds_cfg = cfg["dataset"]
    dataset = ShanghaiTechDataset(
        root=ds_cfg["root"],
        part=ds_cfg.get("part", "part_A"),
        split=args.split,
        crop_size=int(ds_cfg.get("crop_size", 256)),
        is_train=False,
        coordinate_base=int(ds_cfg.get("coordinate_base", 0)),
        image_mean=tuple(ds_cfg.get("image_mean", [0.5, 0.5, 0.5])),
        image_std=tuple(ds_cfg.get("image_std", [0.5, 0.5, 0.5])),
    )
    n_total = len(dataset)
    n_eval = min(n_total, args.max_samples) if args.max_samples is not None else n_total
    part_name = ds_cfg.get("part", "part_A")
    print(f"Loaded dataset: {part_name}/{args.split} with {n_total} images (evaluating {n_eval})", flush=True)

    # Prepare criterion for D-M
    loss_cfg = cfg.get("loss", {})
    criterion = NTPCLoss(NTPCConfig(
        mode=loss_cfg.get("mode", "r2_flat_dm"),
        root_loss=loss_cfg.get("root_loss", "nb"),
        root_dispersion=float(cfg.get("statistics", {}).get("root_dispersion", 50.0)),
        kappa_flat16=float(loss_cfg.get("kappa_flat16", 20.0)),
    ))

    run_dr = args.diagnostics == "all" or "dr" in args.diagnostics.lower()
    run_dk = args.diagnostics == "all" or "dk" in args.diagnostics.lower()
    run_dl = args.diagnostics == "all" or "dl" in args.diagnostics.lower()
    run_dm = args.diagnostics == "all" or "dm" in args.diagnostics.lower()

    dr_results: List[Dict[str, float]] = []
    dk_results: List[Dict[str, Any]] = []
    dl_results: List[Dict[str, Any]] = []
    dm_results: List[Dict[str, float]] = []

    from hpc.data.point_counts import build_exact_count_pyramid

    start_time = time.time()
    for idx in range(n_eval):
        sample = dataset[idx]
        img_tensor = sample["image"]
        pts = sample["gt_points"]
        img_b = img_tensor.unsqueeze(0).to(device)

        if run_dr:
            res_dr = evaluate_phase_shift_single_image(model, img_b, device=device)
            res_dr["gt_count"] = float(len(pts))
            dr_results.append(res_dr)

        if run_dk:
            res_dk = evaluate_separability_single_image(model, img_b, pts, device=device)
            res_dk["gt_count"] = float(len(pts))
            dk_results.append(res_dk)

        if run_dl:
            res_dl = evaluate_effective_rank_single_image(model, img_b, pts, device=device)
            res_dl["gt_count"] = float(len(pts))
            dl_results.append(res_dl)

        if run_dm:
            _, _, ih, iw = img_b.shape
            pts_clamped = pts.copy() if len(pts) > 0 else pts
            if len(pts_clamped) > 0:
                pts_clamped[:, 0] = np.clip(pts_clamped[:, 0], -0.5, iw - 0.5)
                pts_clamped[:, 1] = np.clip(pts_clamped[:, 1], -0.5, ih - 0.5)
            hp = ((ih + 63) // 64) * 64
            wp = ((iw + 63) // 64) * 64
            img_padded = F.pad(img_b, (0, wp - iw, 0, hp - ih), mode="replicate")
            tree = build_exact_count_pyramid(
                [torch.from_numpy(pts_clamped).float()],
                height=ih,
                width=iw,
                block_sizes=(4, 8, 16, 32, 64),
                pad_multiple=64,
            )
            targets_b = {k: tree[k].to(device) for k in (4, 8, 16, 32, 64)}
            targets_b["N"] = tree["N"].to(device)
            res_dm = evaluate_gradient_allocation_single_batch(
                model, criterion, img_padded, targets_b, points_list=[pts_clamped], device=device
            )
            res_dm["gt_count"] = float(len(pts))
            dm_results.append(res_dm)

        if (idx + 1) % 10 == 0 or (idx + 1) == n_eval:
            print(f"Evaluated [{idx + 1}/{n_eval}] images ({time.time() - start_time:.1f}s)...", flush=True)

    # Aggregations
    summary: Dict[str, Any] = {
        "metadata": {
            "config": os.path.abspath(args.config),
            "checkpoint": os.path.abspath(args.checkpoint),
            "split": args.split,
            "samples_evaluated": n_eval,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    }

    # Aggregate D-R
    if run_dr and dr_results:
        summary["D-R_phase_shift"] = {
            "mean_count_relative_std": float(np.mean([r["count_relative_std"] for r in dr_results])),
            "median_count_relative_std": float(np.median([r["count_relative_std"] for r in dr_results])),
            "p90_count_relative_std": float(np.percentile([r["count_relative_std"] for r in dr_results], 90)),
            "mean_interior_mass_mae": float(np.mean([r["interior_mass_mae_mean"] for r in dr_results])),
            "mean_c4_cos_sim": float(np.mean([r["feature_c4_cos_sim"] for r in dr_results])),
            "mean_c8_cos_sim": float(np.mean([r["feature_c8_cos_sim"] for r in dr_results])),
            "mean_c16_cos_sim": float(np.mean([r["feature_c16_cos_sim"] for r in dr_results])),
        }

    # Aggregate D-K
    if run_dk and dk_results:
        dk_agg: Dict[str, Any] = {"mean_knn_spacing": float(np.mean([r["knn_spacing_mean"] for r in dk_results]))}
        for bname in ["le8", "8_16", "16_32", "gt32"]:
            ptrs = [
                r["bins"][bname]["P4_mass_peak_to_trough_ratio"]
                for r in dk_results
                if bname in r.get("bins", {}) and "P4_mass_peak_to_trough_ratio" in r["bins"][bname]
            ]
            merged_fracs = [
                r["bins"][bname]["P4_mass_merged_fraction"]
                for r in dk_results
                if bname in r.get("bins", {}) and "P4_mass_merged_fraction" in r["bins"][bname]
            ]
            c16_sims = [
                r["bins"][bname]["C16_cos_sim_to_midpoint"]
                for r in dk_results
                if bname in r.get("bins", {}) and "C16_cos_sim_to_midpoint" in r["bins"][bname]
            ]
            if ptrs:
                dk_agg[f"bin_{bname}"] = {
                    "median_peak_to_trough_ratio": float(np.median(ptrs)),
                    "mean_merged_fraction": float(np.mean(merged_fracs)),
                    "mean_c16_midpoint_sim": float(np.mean(c16_sims)) if c16_sims else float("nan"),
                    "sample_count": len(ptrs),
                }
        summary["D-K_separability"] = dk_agg

    # Aggregate D-L
    if run_dl and dl_results:
        summary["D-L_effective_rank"] = {
            "mean_depth_decay_c16_to_c4": float(np.mean([r["depth_decay_c16_to_c4"] for r in dl_results])),
            "c4_norm_participation_ratio": float(np.mean([r["stages"]["C4"]["normalized_participation_ratio"] for r in dl_results])),
            "c8_norm_participation_ratio": float(np.mean([r["stages"]["C8"]["normalized_participation_ratio"] for r in dl_results])),
            "c16_norm_participation_ratio": float(np.mean([r["stages"]["C16"]["normalized_participation_ratio"] for r in dl_results])),
            "c4_spectral_entropy_rank": float(np.mean([r["stages"]["C4"]["spectral_entropy_rank"] for r in dl_results])),
            "c16_spectral_entropy_rank": float(np.mean([r["stages"]["C16"]["spectral_entropy_rank"] for r in dl_results])),
        }

    # Aggregate D-M
    if run_dm and dm_results:
        summary["D-M_gradient_allocation"] = {
            "mean_c4_fg_energy_fraction": float(np.mean([r.get("C4_fg_energy_fraction", float("nan")) for r in dm_results])),
            "mean_c8_fg_energy_fraction": float(np.mean([r.get("C8_fg_energy_fraction", float("nan")) for r in dm_results])),
            "mean_c16_fg_energy_fraction": float(np.mean([r.get("C16_fg_energy_fraction", float("nan")) for r in dm_results])),
            "mean_c16_fg_to_bg_density_ratio": float(np.mean([r.get("C16_fg_to_bg_density_ratio", float("nan")) for r in dm_results])),
        }

    # Falsification Gate Matrix Verdicts
    gates: Dict[str, Dict[str, Any]] = {}
    if run_dr and "D-R_phase_shift" in summary:
        rel_std = summary["D-R_phase_shift"]["mean_count_relative_std"]
        c16_cos = summary["D-R_phase_shift"]["mean_c16_cos_sim"]
        # Gate G-R: Significant phase shift instability if relative count std > 2.0% or c16 cos sim < 0.95
        gates["G-R (Phase Shift Instability)"] = {
            "metric_value": f"RelStd={rel_std*100:.2f}%, C16_Cos={c16_cos:.3f}",
            "threshold": "RelStd > 2.0% or C16_Cos < 0.95",
            "verdict": "ACTIVE BOTTLENECK" if (rel_std > 0.02 or c16_cos < 0.95) else "REJECTED (ROBUST)",
        }

    if run_dk and "D-K_separability" in summary:
        bin_le8 = summary["D-K_separability"].get("bin_le8", {})
        merged_frac = bin_le8.get("mean_merged_fraction", 0.0)
        ptr = bin_le8.get("median_peak_to_trough_ratio", 1.0)
        gates["G-K (Separability Collapse)"] = {
            "metric_value": f"MergedFrac@<=8px={merged_frac*100:.1f}%, PTR={ptr:.2f}",
            "threshold": "MergedFrac > 50% or PTR <= 1.0",
            "verdict": "ACTIVE BOTTLENECK" if (merged_frac > 0.50 or ptr <= 1.0) else "REJECTED (SEPARABLE)",
        }

    if run_dl and "D-L_effective_rank" in summary:
        c16_pr = summary["D-L_effective_rank"]["c16_norm_participation_ratio"]
        decay = summary["D-L_effective_rank"]["mean_depth_decay_c16_to_c4"]
        gates["G-L (Effective Rank Collapse)"] = {
            "metric_value": f"C16_NormPR={c16_pr:.3f}, DepthDecay={decay:.3f}",
            "threshold": "C16_NormPR < 0.20 or DepthDecay < 0.60",
            "verdict": "ACTIVE BOTTLENECK" if (c16_pr < 0.20 or decay < 0.60) else "REJECTED (HIGH_RANK)",
        }

    if run_dm and "D-M_gradient_allocation" in summary:
        c16_fg = summary["D-M_gradient_allocation"]["mean_c16_fg_energy_fraction"]
        gates["G-M (Gradient Dilution)"] = {
            "metric_value": f"C16_FG_Energy={c16_fg*100:.1f}%",
            "threshold": "C16_FG_Energy < 60%",
            "verdict": "ACTIVE BOTTLENECK" if c16_fg < 0.60 else "REJECTED (CONCENTRATED)",
        }

    summary["falsification_gates"] = gates

    # Save output
    out_path = args.output or os.path.join("runs", f"d0_diagnostics_{os.path.splitext(os.path.basename(args.config))[0]}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n" + "=" * 70, flush=True)
    print("           D0 DIAGNOSTIC SUITE SUMMARY & FALSIFICATION GATES          ", flush=True)
    print("=" * 70, flush=True)
    print(json.dumps(gates, indent=2), flush=True)
    print(f"\nDetailed diagnostics saved to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
