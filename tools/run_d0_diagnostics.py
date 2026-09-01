#!/usr/bin/env python3
"""Run D0 Pre-Model Diagnostic Suite (D-R, D-K, D-L, D-M).

Evaluates candidate bottlenecks on trained checkpoints or baseline models:
- D-R / G-R: Sampling-phase and translation instability with exact inverse alignment.
- D-K / G-K: Inter-person separability collapse normalized by local head scale.
- D-L / G-L: Normalized effective representation rank with sample-matched SVD.
- D-M / G-M: Area-normalized foreground crowd vs background gradient allocation.
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

from hpc.data import ShanghaiTechDataset, ntpc_collate_fn
from hpc.data.point_counts import build_exact_count_pyramid
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
            "mean_c4_cos_sim_aligned": float(np.mean([r["feature_c4_cos_sim_aligned"] for r in dr_results])),
            "mean_c8_cos_sim_aligned": float(np.mean([r["feature_c8_cos_sim_aligned"] for r in dr_results])),
            "mean_c16_cos_sim_aligned": float(np.mean([r["feature_c16_cos_sim_aligned"] for r in dr_results])),
        }
        if "feature_c32_cos_sim_aligned" in dr_results[0]:
            summary["D-R_phase_shift"]["mean_c32_cos_sim_aligned"] = float(
                np.mean([r["feature_c32_cos_sim_aligned"] for r in dr_results])
            )

    # Aggregate D-K
    if run_dk and dk_results:
        dk_agg: Dict[str, Any] = {"mean_head_scale_proxy": float(np.mean([r["head_scale_proxy_mean"] for r in dk_results]))}
        for bname in ["le_0p5", "0p5_1p0", "1p0_2p0", "gt_2p0"]:
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
            c4_sims = [
                r["bins"][bname]["C4_cos_sim_to_midpoint"]
                for r in dk_results
                if bname in r.get("bins", {}) and "C4_cos_sim_to_midpoint" in r["bins"][bname]
            ]
            c16_sims = [
                r["bins"][bname]["C16_cos_sim_to_midpoint"]
                for r in dk_results
                if bname in r.get("bins", {}) and "C16_cos_sim_to_midpoint" in r["bins"][bname]
            ]
            c32_sims = [
                r["bins"][bname]["C32_cos_sim_to_midpoint"]
                for r in dk_results
                if bname in r.get("bins", {}) and "C32_cos_sim_to_midpoint" in r["bins"][bname]
            ]
            if ptrs:
                b_info: Dict[str, Any] = {
                    "median_peak_to_trough_ratio": float(np.median(ptrs)),
                    "mean_merged_fraction": float(np.mean(merged_fracs)),
                    "mean_c4_midpoint_sim": float(np.mean(c4_sims)) if c4_sims else float("nan"),
                    "mean_c16_midpoint_sim": float(np.mean(c16_sims)) if c16_sims else float("nan"),
                    "sample_count": len(ptrs),
                }
                if c32_sims:
                    b_info["mean_c32_midpoint_sim"] = float(np.mean(c32_sims))
                dk_agg[f"bin_{bname}"] = b_info
        summary["D-K_separability"] = dk_agg

    # Aggregate D-L
    if run_dl and dl_results:
        dl_agg: Dict[str, Any] = {
            "mean_depth_decay_c16_to_c4": float(np.mean([r["depth_decay_c16_to_c4"] for r in dl_results])),
            "c4_norm_participation_ratio": float(np.mean([r["stages"]["C4"]["normalized_participation_ratio"] for r in dl_results])),
            "c8_norm_participation_ratio": float(np.mean([r["stages"]["C8"]["normalized_participation_ratio"] for r in dl_results])),
            "c16_norm_participation_ratio": float(np.mean([r["stages"]["C16"]["normalized_participation_ratio"] for r in dl_results])),
            "c4_spectral_entropy_rank": float(np.mean([r["stages"]["C4"]["spectral_entropy_rank"] for r in dl_results])),
            "c16_spectral_entropy_rank": float(np.mean([r["stages"]["C16"]["spectral_entropy_rank"] for r in dl_results])),
        }
        if "C32" in dl_results[0]["stages"]:
            dl_agg["c32_norm_participation_ratio"] = float(np.mean([r["stages"]["C32"]["normalized_participation_ratio"] for r in dl_results]))
            dl_agg["c32_spectral_entropy_rank"] = float(np.mean([r["stages"]["C32"]["spectral_entropy_rank"] for r in dl_results]))
            dl_agg["mean_depth_decay_c32_to_c16"] = float(np.mean([r.get("depth_decay_c32_to_c16", float("nan")) for r in dl_results]))
            dl_agg["mean_depth_decay_c32_to_c4"] = float(np.mean([r.get("depth_decay_c32_to_c4", float("nan")) for r in dl_results]))
        summary["D-L_effective_rank"] = dl_agg

    # Aggregate D-M
    if run_dm and dm_results:
        dm_agg: Dict[str, Any] = {
            "c4_fg_energy_fraction": float(np.mean([r.get("C4_fg_energy_fraction", float("nan")) for r in dm_results])),
            "c8_fg_energy_fraction": float(np.mean([r.get("C8_fg_energy_fraction", float("nan")) for r in dm_results])),
            "c16_fg_energy_fraction": float(np.mean([r.get("C16_fg_energy_fraction", float("nan")) for r in dm_results])),
            "c16_gradient_enrichment": float(np.mean([r.get("C16_gradient_enrichment", float("nan")) for r in dm_results])),
            "c16_gradient_density_ratio": float(np.mean([r.get("C16_gradient_density_ratio", float("nan")) for r in dm_results])),
        }
        if "C32_fg_energy_fraction" in dm_results[0]:
            dm_agg["c32_fg_energy_fraction"] = float(np.mean([r.get("C32_fg_energy_fraction", float("nan")) for r in dm_results]))
            dm_agg["c32_gradient_enrichment"] = float(np.mean([r.get("C32_gradient_enrichment", float("nan")) for r in dm_results]))
            dm_agg["c32_gradient_density_ratio"] = float(np.mean([r.get("C32_gradient_density_ratio", float("nan")) for r in dm_results]))
        summary["D-M_gradient_allocation"] = dm_agg

    # Objective Diagnostic Synthesis
    diag_synthesis: Dict[str, Any] = {}
    if "D-R_phase_shift" in summary:
        dr = summary["D-R_phase_shift"]
        diag_synthesis["D-R (Sampling Phase Shift)"] = {
            "count_relative_std": f"{dr['mean_count_relative_std']*100:.2f}% (p90={dr['p90_count_relative_std']*100:.2f}%)",
            "aligned_feature_c16_cos_sim": f"{dr['mean_c16_cos_sim_aligned']:.4f}",
            "aligned_feature_c32_cos_sim": f"{dr.get('mean_c32_cos_sim_aligned', float('nan')):.4f}" if "mean_c32_cos_sim_aligned" in dr else "N/A",
            "interior_mass_mae": f"{dr['mean_interior_mass_mae']:.5f}",
        }
    if "D-K_separability" in summary:
        dk = summary["D-K_separability"]
        le_0p5 = dk.get("bin_le_0p5", {})
        diag_synthesis["D-K (Separability by Normalized Spacing r=d/s_head)"] = {
            "crowded_r<=0.5_merged_frac": f"{le_0p5.get('mean_merged_fraction', float('nan'))*100:.1f}%",
            "crowded_r<=0.5_ptr": f"{le_0p5.get('median_peak_to_trough_ratio', float('nan')):.3f}",
            "crowded_r<=0.5_c16_midpoint_sim": f"{le_0p5.get('mean_c16_midpoint_sim', float('nan')):.4f}",
            "crowded_r<=0.5_c32_midpoint_sim": f"{le_0p5.get('mean_c32_midpoint_sim', float('nan')):.4f}" if "mean_c32_midpoint_sim" in le_0p5 else "N/A",
        }
    if "D-L_effective_rank" in summary:
        dl = summary["D-L_effective_rank"]
        diag_synthesis["D-L (Effective Representation Rank)"] = {
            "c4_norm_pr": f"{dl['c4_norm_participation_ratio']:.3f}",
            "c16_norm_pr": f"{dl['c16_norm_participation_ratio']:.3f}",
            "c32_norm_pr": f"{dl.get('c32_norm_participation_ratio', float('nan')):.3f}" if "c32_norm_participation_ratio" in dl else "N/A",
            "depth_decay_c16_to_c4": f"{dl['mean_depth_decay_c16_to_c4']:.3f}",
            "depth_decay_c32_to_c16": f"{dl.get('mean_depth_decay_c32_to_c16', float('nan')):.3f}" if "mean_depth_decay_c32_to_c16" in dl else "N/A",
        }
    if "D-M_gradient_allocation" in summary:
        dm = summary["D-M_gradient_allocation"]
        diag_synthesis["D-M (Gradient Allocation)"] = {
            "c16_fg_energy": f"{dm['c16_fg_energy_fraction']*100:.1f}%",
            "c16_gradient_enrichment": f"{dm['c16_gradient_enrichment']:.2f}x",
            "c16_gradient_density_ratio": f"{dm['c16_gradient_density_ratio']:.2f}x",
            "c32_gradient_enrichment": f"{dm.get('c32_gradient_enrichment', float('nan')):.2f}x" if "c32_gradient_enrichment" in dm else "N/A",
        }
    summary["diagnostic_synthesis"] = diag_synthesis

    out_path = args.output or os.path.join("runs", f"d0_diagnostics_{os.path.splitext(os.path.basename(args.config))[0]}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n" + "=" * 70, flush=True)
    print("                 D0 DIAGNOSTIC SUITE SUMMARY REPORT                   ", flush=True)
    print("=" * 70, flush=True)
    print(json.dumps(diag_synthesis, indent=2), flush=True)
    print(f"\nDetailed diagnostics saved to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
