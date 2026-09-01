#!/usr/bin/env python3
"""Run D0 Pre-Model Diagnostic Suite (D-R, D-K, D-L, D-M).

Evaluates candidate bottlenecks on trained checkpoints or baseline models:
- D-R / G-R: Sampling-phase and translation instability via natural shifted views.
- D-K / G-K: Same-scene far-pair normalized inter-person separability retention across depth.
- D-L / G-L: Normalized scale-invariant effective rank on lexicographically sorted matched crowd points.
- D-M / G-M: Area-normalized foreground crowd vs background gradient allocation on natural unpadded crops.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from hpc.data.factory import build_evaluation_dataset
from hpc.data.point_counts import build_exact_count_pyramid
from hpc.diagnostics import (
    evaluate_effective_rank_single_image,
    evaluate_gradient_allocation_single_batch,
    evaluate_phase_shift_single_image,
    evaluate_separability_single_image,
)
from hpc.losses.factory import build_ntpc_criterion_from_config
from hpc.models.factory import assert_checkpoint_compatible, build_model_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run D0 Diagnostic Suite")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint .pt")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--split", default=None, help="Dataset split override (test_data, Train, Test, val)")
    parser.add_argument("--max-samples", type=int, default=None, help="Max images to evaluate")
    parser.add_argument(
        "--dr-margin-px",
        type=int,
        default=96,
        help="Border margin for D-R phase-shift diagnostic",
    )
    parser.add_argument(
        "--diagnostics",
        default="all",
        help="Comma-separated diagnostics to run: dr,dk,dl,dm or all",
    )
    return parser.parse_args()


def make_natural_dm_crop(
    image: torch.Tensor,
    points: np.ndarray,
    max_crop: int = 256,
) -> Optional[Tuple[torch.Tensor, np.ndarray]]:
    """Extract a centered, natural multiple-of-64 crop without artificial canvas padding."""
    _, _, h, w = image.shape
    crop_h = min(h, max_crop)
    crop_w = min(w, max_crop)
    crop_h = (crop_h // 64) * 64
    crop_w = (crop_w // 64) * 64
    if crop_h < 64 or crop_w < 64:
        return None

    y0 = (h - crop_h) // 2
    x0 = (w - crop_w) // 2
    crop = image[..., y0:y0 + crop_h, x0:x0 + crop_w]

    pts = np.asarray(points, dtype=np.float32).copy()
    if len(pts):
        pts[:, 0] -= x0
        pts[:, 1] -= y0
        keep = (
            (pts[:, 0] >= -0.5)
            & (pts[:, 0] <= crop_w - 0.5)
            & (pts[:, 1] >= -0.5)
            & (pts[:, 1] <= crop_h - 0.5)
        )
        pts = pts[keep]

    return crop, pts


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running D0 Diagnostic Suite on device: {device}", flush=True)

    # Load model strictly from checkpoint without re-downloading ImageNet pretraining
    ckpt = torch.load(args.checkpoint, map_location=device)
    assert_checkpoint_compatible(ckpt, cfg)
    model = build_model_from_config(cfg, load_pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint} (epoch={ckpt.get('epoch', 'N/A')})", flush=True)

    # Build dataset generic factory
    dataset, resolved_split = build_evaluation_dataset(cfg, split=args.split)
    n_total = len(dataset)
    n_eval = min(n_total, args.max_samples) if args.max_samples is not None else n_total
    print(f"Loaded {resolved_split}: {n_total} images; evaluating {n_eval}", flush=True)

    # Build criterion strictly from checkpoint metadata to avoid configuration drift
    criterion_cfg = ckpt.get("config", cfg)
    criterion_crop_stats = ckpt.get("resolved_crop_statistics")
    criterion = build_ntpc_criterion_from_config(
        criterion_cfg,
        crop_statistics=criterion_crop_stats,
    ).to(device)

    # Parse diagnostic tokens safely
    tokens = {t.strip().lower() for t in args.diagnostics.split(",") if t.strip()}
    if "all" in tokens:
        tokens = {"dr", "dk", "dl", "dm"}
    unknown = tokens - {"dr", "dk", "dl", "dm"}
    if unknown:
        raise ValueError(f"Unknown diagnostics: {sorted(unknown)}")

    run_dr = "dr" in tokens
    run_dk = "dk" in tokens
    run_dl = "dl" in tokens
    run_dm = "dm" in tokens

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
            res_dr = evaluate_phase_shift_single_image(model, img_b, device=device, border_margin_px=args.dr_margin_px)
            res_dr["gt_count"] = float(len(pts))
            dr_results.append(res_dr)

        if run_dk:
            res_dk = evaluate_separability_single_image(model, img_b, pts, device=device)
            res_dk["gt_count"] = float(len(pts))
            dk_results.append(res_dk)

        if run_dl:
            res_dl = evaluate_effective_rank_single_image(model, img_b, pts, device=device)
            if res_dl.get("valid", False):
                res_dl["gt_count"] = float(len(pts))
                dl_results.append(res_dl)

        if run_dm:
            dm_sample = make_natural_dm_crop(img_b, pts, max_crop=int(cfg["dataset"].get("crop_size", 256)))
            if dm_sample is not None:
                dm_image, dm_points = dm_sample
                _, _, dh, dw = dm_image.shape
                tree = build_exact_count_pyramid(
                    [torch.from_numpy(dm_points).float()],
                    height=dh,
                    width=dw,
                    block_sizes=(4, 8, 16, 32, 64),
                    pad_multiple=64,
                )
                targets_b = {k: tree[k].to(device) for k in (4, 8, 16, 32, 64)}
                targets_b["N"] = tree["N"].to(device)
                res_dm = evaluate_gradient_allocation_single_batch(
                    model, criterion, dm_image, targets_b, points_list=[dm_points], valid_hw=None, device=device
                )
                res_dm["gt_count"] = float(len(dm_points))
                dm_results.append(res_dm)

        if (idx + 1) % 10 == 0 or (idx + 1) == n_eval:
            print(f"Evaluated [{idx + 1}/{n_eval}] images ({time.time() - start_time:.1f}s)...", flush=True)

    summary: Dict[str, Any] = {
        "metadata": {
            "config": os.path.abspath(args.config),
            "checkpoint": os.path.abspath(args.checkpoint),
            "split": resolved_split,
            "samples_evaluated": n_eval,
            "dr_margin_px": args.dr_margin_px,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    }

    # Aggregate D-R
    if run_dr and dr_results:
        summary["D-R_phase_shift"] = {
            "mean_count_relative_std": float(np.mean([r["count_relative_std"] for r in dr_results])),
            "median_count_relative_std": float(np.median([r["count_relative_std"] for r in dr_results])),
            "p90_count_relative_std": float(np.percentile([r["count_relative_std"] for r in dr_results], 90)),
            "mean_mass_mae": float(np.mean([r["mass_mae_mean"] for r in dr_results])),
            "mean_c4_cos_sim_aligned": float(np.mean([r["feature_c4_cos_sim_aligned"] for r in dr_results])),
            "mean_c8_cos_sim_aligned": float(np.mean([r["feature_c8_cos_sim_aligned"] for r in dr_results])),
            "mean_c16_cos_sim_aligned": float(np.mean([r["feature_c16_cos_sim_aligned"] for r in dr_results])),
        }
        if "feature_c32_cos_sim_aligned" in dr_results[0]:
            summary["D-R_phase_shift"]["mean_c32_cos_sim_aligned"] = float(
                np.mean([r["feature_c32_cos_sim_aligned"] for r in dr_results])
            )

    # Aggregate D-K (Primary: Same-Scene Far-Pair Normalized Separability)
    if run_dk and dk_results:
        dk_agg: Dict[str, Any] = {
            "mean_knn_spacing_px": float(np.mean([r["mean_knn_spacing_px"] for r in dk_results]))
        }
        for bname in ["le8", "8_16", "16_32", "gt32"]:
            valid = [r["bins"][bname] for r in dk_results if bname in r.get("bins", {})]
            if not valid:
                continue

            def pair_weighted_metric(stage: str) -> float:
                total = 0.0
                n = 0
                for x in valid:
                    key = f"{stage}_inter_person_dissimilarity_sum"
                    if key in x:
                        total += float(x[key])
                        n += int(x["num_pairs"])
                return float(total / n) if n > 0 else float("nan")

            total_pairs = sum(int(x["num_pairs"]) for x in valid)
            ptr_values = [val for x in valid for val in x.get("P4_mass_peak_to_trough_values", [])]
            merged_count = sum(int(x.get("P4_mass_merged_count", 0)) for x in valid)

            info: Dict[str, Any] = {
                "sample_count": total_pairs,
                "median_peak_to_trough_ratio": float(np.median(ptr_values)) if ptr_values else float("nan"),
                "mean_merged_fraction": float(merged_count / total_pairs) if total_pairs else float("nan"),
                "c4_inter_person_dissimilarity": pair_weighted_metric("C4"),
                "c8_inter_person_dissimilarity": pair_weighted_metric("C8"),
                "c16_inter_person_dissimilarity": pair_weighted_metric("C16"),
            }
            if any("C32_inter_person_dissimilarity_sum" in x for x in valid):
                info["c32_inter_person_dissimilarity"] = pair_weighted_metric("C32")

            dk_agg[f"bin_{bname}"] = info

        # Same-scene normalization (each image's near pair normalized by its own gt32 far pair)
        def pair_mean(info_d: dict, stage: str) -> float:
            n = int(info_d.get("num_pairs", 0))
            total = info_d.get(f"{stage}_inter_person_dissimilarity_sum")
            if n <= 0 or total is None:
                return float("nan")
            return float(total) / float(n)

        for bname in ["le8", "8_16", "16_32"]:
            same_scene: Dict[str, List[Tuple[float, int]]] = {
                "C4": [], "C8": [], "C16": [], "C32": []
            }
            matched_images = 0
            for r in dk_results:
                bins = r.get("bins", {})
                near = bins.get(bname)
                far = bins.get("far_control_gt32")
                if near is None or far is None:
                    continue
                w = min(int(near.get("num_pairs", 0)), int(far.get("num_pairs", 0)))
                if w <= 0:
                    continue
                matched_images += 1
                for stage in same_scene:
                    near_v = pair_mean(near, stage)
                    far_v = pair_mean(far, stage)
                    if np.isfinite(near_v) and np.isfinite(far_v) and far_v > 1e-8:
                        same_scene[stage].append((near_v / far_v, w))

            info = dk_agg.get(f"bin_{bname}")
            if info is not None:
                for stage, values in same_scene.items():
                    if not values:
                        continue
                    num = sum(v * w for v, w in values)
                    den = sum(w for _, w in values)
                    info[f"{stage.lower()}_same_scene_normalized_separability"] = float(num / den)
                info["same_scene_matched_images"] = matched_images
                
                # Retention ratios
                c4_ss = info.get("c4_same_scene_normalized_separability")
                c16_ss = info.get("c16_same_scene_normalized_separability")
                c32_ss = info.get("c32_same_scene_normalized_separability")
                if c4_ss is not None and c16_ss is not None:
                    info["same_scene_retention_c16_over_c4"] = float(c16_ss / max(c4_ss, 1e-8))
                if c16_ss is not None and c32_ss is not None:
                    info["same_scene_retention_c32_over_c16"] = float(c32_ss / max(c16_ss, 1e-8))
                if c4_ss is not None and c32_ss is not None:
                    info["same_scene_retention_c32_over_c4"] = float(c32_ss / max(c4_ss, 1e-8))

        summary["D-K_separability"] = dk_agg

    # Aggregate D-L (Strictly crowd-point matched samples)
    if run_dl and dl_results:
        dl_agg: Dict[str, Any] = {
            "crowd_images_evaluated": len(dl_results),
            "mean_depth_decay_c16_to_c4": float(np.mean([r["depth_decay_c16_to_c4"] for r in dl_results])),
            "c4_norm_participation_ratio": float(np.mean([r["stages"]["C4"]["normalized_participation_ratio"] for r in dl_results])),
            "c8_norm_participation_ratio": float(np.mean([r["stages"]["C8"]["normalized_participation_ratio"] for r in dl_results])),
            "c16_norm_participation_ratio": float(np.mean([r["stages"]["C16"]["normalized_participation_ratio"] for r in dl_results])),
            "c4_spectral_entropy_rank": float(np.mean([r["stages"]["C4"]["spectral_entropy_rank"] for r in dl_results])),
            "c16_spectral_entropy_rank": float(np.mean([r["stages"]["C16"]["spectral_entropy_rank"] for r in dl_results])),
        }
        if dl_results and "C32" in dl_results[0]["stages"]:
            dl_agg["c32_norm_participation_ratio"] = float(np.mean([r["stages"]["C32"]["normalized_participation_ratio"] for r in dl_results]))
            dl_agg["c32_spectral_entropy_rank"] = float(np.mean([r["stages"]["C32"]["spectral_entropy_rank"] for r in dl_results]))
            dl_agg["mean_depth_decay_c32_to_c16"] = float(np.mean([r.get("depth_decay_c32_to_c16", float("nan")) for r in dl_results]))
            dl_agg["mean_depth_decay_c32_to_c4"] = float(np.mean([r.get("depth_decay_c32_to_c4", float("nan")) for r in dl_results]))
        summary["D-L_effective_rank"] = dl_agg

    # Aggregate D-M
    if run_dm and dm_results:
        def finite_mean(values: List[Any]) -> float:
            arr = np.asarray(values, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            return float(arr.mean()) if len(arr) else float("nan")

        dm_agg: Dict[str, Any] = {
            "c4_fg_energy_fraction": finite_mean([r.get("C4_fg_energy_fraction") for r in dm_results]),
            "c8_fg_energy_fraction": finite_mean([r.get("C8_fg_energy_fraction") for r in dm_results]),
            "c16_fg_energy_fraction": finite_mean([r.get("C16_fg_energy_fraction") for r in dm_results]),
            "c16_gradient_enrichment": finite_mean([r.get("C16_gradient_enrichment") for r in dm_results]),
            "c16_gradient_density_ratio": finite_mean([r.get("C16_gradient_density_ratio") for r in dm_results]),
            "c16_valid_samples": sum(bool(r.get("C16_valid_fg_bg", False)) for r in dm_results),
        }
        if any("C32_fg_energy_fraction" in r for r in dm_results):
            dm_agg["c32_fg_energy_fraction"] = finite_mean([r.get("C32_fg_energy_fraction") for r in dm_results])
            dm_agg["c32_gradient_enrichment"] = finite_mean([r.get("C32_gradient_enrichment") for r in dm_results])
            dm_agg["c32_gradient_density_ratio"] = finite_mean([r.get("C32_gradient_density_ratio") for r in dm_results])
            dm_agg["c32_valid_samples"] = sum(bool(r.get("C32_valid_fg_bg", False)) for r in dm_results)
        summary["D-M_gradient_allocation"] = dm_agg

    # Objective Diagnostic Synthesis
    diag_synthesis: Dict[str, Any] = {}
    if "D-R_phase_shift" in summary:
        dr = summary["D-R_phase_shift"]
        diag_synthesis["D-R (Sampling Phase Shift)"] = {
            "count_relative_std": f"{dr['mean_count_relative_std']*100:.2f}% (p90={dr['p90_count_relative_std']*100:.2f}%)",
            "aligned_feature_c16_cos_sim": f"{dr['mean_c16_cos_sim_aligned']:.4f}",
            "aligned_feature_c32_cos_sim": f"{dr.get('mean_c32_cos_sim_aligned', float('nan')):.4f}" if "mean_c32_cos_sim_aligned" in dr else "N/A",
            "mass_mae": f"{dr['mean_mass_mae']:.5f}",
        }
    if "D-K_separability" in summary:
        dk = summary["D-K_separability"]
        le8 = dk.get("bin_le8", {})
        diag_synthesis["D-K (Same-Scene Separability Retention @ d<=8px)"] = {
            "c4_same_scene_norm_separability": f"{le8.get('c4_same_scene_normalized_separability', float('nan'))*100:.1f}%",
            "c8_same_scene_norm_separability": f"{le8.get('c8_same_scene_normalized_separability', float('nan'))*100:.1f}%",
            "c16_same_scene_norm_separability": f"{le8.get('c16_same_scene_normalized_separability', float('nan'))*100:.1f}%",
            "c32_same_scene_norm_separability": f"{le8.get('c32_same_scene_normalized_separability', float('nan'))*100:.1f}%" if "c32_same_scene_normalized_separability" in le8 else "N/A",
            "same_scene_retention_c16_over_c4": f"{le8.get('same_scene_retention_c16_over_c4', float('nan'))*100:.1f}%",
            "same_scene_retention_c32_over_c16": f"{le8.get('same_scene_retention_c32_over_c16', float('nan'))*100:.1f}%" if "same_scene_retention_c32_over_c16" in le8 else "N/A",
            "same_scene_retention_c32_over_c4": f"{le8.get('same_scene_retention_c32_over_c4', float('nan'))*100:.1f}%" if "same_scene_retention_c32_over_c4" in le8 else "N/A",
            "same_scene_matched_images": le8.get("same_scene_matched_images", 0),
            "output_mass_merged_frac": f"{le8.get('mean_merged_fraction', float('nan'))*100:.1f}%",
        }
    if "D-L_effective_rank" in summary:
        dl = summary["D-L_effective_rank"]
        diag_synthesis["D-L (Effective Representation Rank on Crowd Points)"] = {
            "c4_norm_pr": f"{dl['c4_norm_participation_ratio']:.3f}",
            "c16_norm_pr": f"{dl['c16_norm_participation_ratio']:.3f}",
            "c32_norm_pr": f"{dl.get('c32_norm_participation_ratio', float('nan')):.3f}" if "c32_norm_participation_ratio" in dl else "N/A",
            "depth_decay_c16_to_c4": f"{dl['mean_depth_decay_c16_to_c4']:.3f}",
            "depth_decay_c32_to_c16": f"{dl.get('mean_depth_decay_c32_to_c16', float('nan')):.3f}" if "mean_depth_decay_c32_to_c16" in dl else "N/A",
        }
    if "D-M_gradient_allocation" in summary:
        dm = summary["D-M_gradient_allocation"]
        diag_synthesis["D-M (Gradient Allocation - Natural Unpadded Crops)"] = {
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
