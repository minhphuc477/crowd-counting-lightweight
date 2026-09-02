#!/usr/bin/env python3
"""CLI runner for Objective Mechanism Audit: Flat-DM16 vs Hierarchical DTM Tree.

Executes on frozen model predictions to answer:
Why does Flat-DM16 (R2) beat Hierarchical DTM Tree (R4/R5) in dense crowd counting?
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
import torch
import torch.nn as nn
import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hpc.data.factory import build_evaluation_dataset
from hpc.data.point_counts import build_exact_count_pyramid
from hpc.diagnostics.objective_mechanism_audit import (
    compute_audit_for_mode,
    stratify_by_density,
    summarize_audit_group,
    sweep_kappa_on_crop,
)
from hpc.losses.ntpc import NTPCConfig, NTPCLoss
from hpc.models.factory import assert_checkpoint_compatible, build_model_from_config
from hpc.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Objective Mechanism (R2 Flat-DM vs R4 DTM Tree)")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint .pt")
    parser.add_argument("--output", default="runs/objective_audit/audit_results.json", help="Output JSON path")
    parser.add_argument("--crop-size", type=int, default=256, help="Crop size for standardized spatial support")
    parser.add_argument("--max-samples", type=int, default=None, help="Max test images to evaluate (default: all)")
    parser.add_argument("--kappas", type=str, default="2,5,10,20,50,100", help="Comma-separated kappa values to sweep")
    parser.add_argument("--sweep-samples", type=int, default=30, help="Number of crops to run full kappa sweep on")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_centered_crop(
    image: torch.Tensor,
    points: np.ndarray,
    crop_size: int = 256,
) -> Optional[Tuple[torch.Tensor, np.ndarray]]:
    """Extract a centered crop of exact size (crop_size x crop_size) without canvas padding."""
    _, _, h, w = image.shape
    if h < crop_size or w < crop_size:
        # If image is smaller than crop_size, return None
        return None
    y0 = (h - crop_size) // 2
    x0 = (w - crop_size) // 2
    crop = image[..., y0:y0 + crop_size, x0:x0 + crop_size]

    pts = np.asarray(points, dtype=np.float32).copy()
    if len(pts):
        pts[:, 0] -= x0
        pts[:, 1] -= y0
        keep = (
            (pts[:, 0] >= -0.5)
            & (pts[:, 0] <= crop_size - 0.5)
            & (pts[:, 1] >= -0.5)
            & (pts[:, 1] <= crop_size - 0.5)
        )
        pts = pts[keep]

    return crop, pts


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print(f"Loading checkpoint {args.checkpoint} ...", flush=True)
    model = build_model_from_config(cfg, load_pretrained=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    assert_checkpoint_compatible(ckpt, cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    dataset, split = build_evaluation_dataset(cfg)
    total_images = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)
    print(f"Dataset: {len(dataset)} images in split '{split}', evaluating on {total_images} images (crop={args.crop_size}x{args.crop_size})", flush=True)

    kappas = [float(k.strip()) for k in args.kappas.split(",") if k.strip()]

    # Standard criteria for R2 and R4 at default kappa=20
    crit_r2_default = NTPCLoss(NTPCConfig(mode="r2_flat_dm", root_loss="nb", kappa_flat16=20.0)).to(device)
    crit_r4_default = NTPCLoss(NTPCConfig(
        mode="r4_dtm_tree16",
        root_loss="nb",
        kappa_root64=20.0,
        kappa_64_32=20.0,
        kappa_32_16=20.0,
    )).to(device)

    records: List[Dict[str, Any]] = []
    sweep_records: List[Dict[str, Any]] = []
    skipped = 0

    t0 = time.time()
    for idx in range(total_images):
        sample = dataset[idx]
        image_full = sample["image"].unsqueeze(0)  # [1, 3, H, W]
        points_full = np.asarray(sample["gt_points"], dtype=np.float32).reshape(-1, 2)

        crop_res = make_centered_crop(image_full, points_full, crop_size=args.crop_size)
        if crop_res is None:
            skipped += 1
            continue
        crop_img, crop_pts = crop_res

        # Build exact target pyramid on crop
        target_pyramid = build_exact_count_pyramid(
            [torch.from_numpy(crop_pts).float()],
            height=args.crop_size,
            width=args.crop_size,
            block_sizes=(4, 8, 16, 32, 64),
            pad_multiple=64,
            device=device,
        )

        gt_count = float(len(crop_pts))
        crop_img = crop_img.to(device)

        # Forward pass on frozen model to obtain predicted mass map
        with torch.no_grad():
            mass = model.forward_mass(crop_img)  # [1, 1, H/4, W/4]
            pred_count = float(mass.sum().item())

        signed_error = pred_count - gt_count

        # Compute R2 audit
        audit_r2 = compute_audit_for_mode(
            mass, target_pyramid, crit_r2_default,
            active_components=("root_magnitude", "flat_16"),
        )

        # Compute R4 audit
        audit_r4 = compute_audit_for_mode(
            mass, target_pyramid, crit_r4_default,
            active_components=("root_magnitude", "root_to_64", "64_to_32", "32_to_16"),
        )

        record = {
            "image_index": idx,
            "img_path": sample.get("img_path", f"img_{idx}"),
            "gt_count": gt_count,
            "pred_count": pred_count,
            "signed_error": signed_error,
            "abs_error": abs(signed_error),
            "r2": {
                "component_metrics": audit_r2["component_metrics"],
                "pairwise_cosines": audit_r2["pairwise_cosines"],
            },
            "r4": {
                "component_metrics": audit_r4["component_metrics"],
                "pairwise_cosines": audit_r4["pairwise_cosines"],
            },
        }
        records.append(record)

        # Offline kappa sweep on selected samples (or all if sweep_samples >= count)
        if len(sweep_records) < args.sweep_samples:
            sweep_res = sweep_kappa_on_crop(mass, target_pyramid, kappas=kappas, device=device)
            sweep_records.append({
                "image_index": idx,
                "gt_count": gt_count,
                "pred_count": pred_count,
                "sweep": sweep_res,
            })

        if (idx + 1) % 30 == 0 or (idx + 1) == total_images:
            print(f"Processed {idx + 1}/{total_images} images ({len(records)} valid crops, {skipped} skipped) ...", flush=True)

    elapsed = time.time() - t0
    print(f"Audit computation completed in {elapsed:.1f}s.", flush=True)

    # Density stratification
    density_bins = stratify_by_density(records)
    summary_by_density = {
        bin_name: summarize_audit_group(bin_records)
        for bin_name, bin_records in density_bins.items()
    }

    # Summarize kappa sweep across crops
    kappa_summary = {}
    for kappa in kappas:
        k_str = f"k_{int(kappa) if kappa == int(kappa) else kappa}"
        r2_mag_cos = [r["sweep"]["r2_flat"][k_str]["flat_16"]["magnitude_cosine"] for r in sweep_records]
        r2_norm = [r["sweep"]["r2_flat"][k_str]["flat_16"]["norm"] for r in sweep_records]
        r4_mag_cos = [r["sweep"]["r4_tree"][k_str]["32_to_16"]["magnitude_cosine"] for r in sweep_records]
        r4_norm = [r["sweep"]["r4_tree"][k_str]["32_to_16"]["norm"] for r in sweep_records]
        r4_cos_root_32 = [r["sweep"]["r4_tree"][k_str]["cos_root_vs_32_16"] for r in sweep_records if math.isfinite(r["sweep"]["r4_tree"][k_str]["cos_root_vs_32_16"])]

        kappa_summary[k_str] = {
            "kappa": kappa,
            "r2_flat16_mag_cos_mean": float(np.mean(r2_mag_cos)) if r2_mag_cos else float("nan"),
            "r2_flat16_norm_mean": float(np.mean(r2_norm)) if r2_norm else float("nan"),
            "r4_32_to_16_mag_cos_mean": float(np.mean(r4_mag_cos)) if r4_mag_cos else float("nan"),
            "r4_32_to_16_norm_mean": float(np.mean(r4_norm)) if r4_norm else float("nan"),
            "r4_cos_root_vs_32_16_mean": float(np.mean(r4_cos_root_32)) if r4_cos_root_32 else float("nan"),
        }

    output_data = {
        "metadata": {
            "checkpoint": args.checkpoint,
            "config": args.config,
            "crop_size": args.crop_size,
            "valid_crops": len(records),
            "skipped_images": skipped,
            "kappas_swept": kappas,
            "elapsed_seconds": elapsed,
        },
        "density_stratified_summary": summary_by_density,
        "kappa_sweep_summary": kappa_summary,
        "per_crop_records": records,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, allow_nan=True)
    print(f"\nWrote full audit results to {out_path}", flush=True)

    # -----------------------------------------------------------------------
    # PRINT TERMINAL REPORT
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("OBJECTIVE MECHANISM AUDIT: FLAT-DM16 (R2) vs NEURAL DTM TREE (R4)")
    print("=" * 90)

    # 1. Density Stratification Table
    print("\n--- 1. Density-Stratified Magnitude Cosine rho_magnitude = cos(grad, 1) ---")
    print("(rho > 0: loss pushes count DOWN / undercount; rho < 0: loss pushes count UP)")
    header = f"{'Density Bin':<12} {'Crops':>6} {'Mean GT':>9} {'Bias':>8} | {'R2 Flat16':>10} {'R2 Total':>9} | {'R4 32->16':>10} {'R4 Total':>9}"
    print(header)
    print("-" * len(header))
    for bname in ["all", "sparse", "medium", "dense"]:
        s = summary_by_density[bname]
        cnt = s["count"]
        gt = s.get("mean_gt_count", float("nan"))
        err = s.get("mean_count_error", float("nan"))
        r2_f16 = s.get("r2", {}).get("flat16_mag_cos", float("nan"))
        r2_tot = s.get("r2", {}).get("total_mag_cos", float("nan"))
        r4_32 = s.get("r4", {}).get("32_to_16_mag_cos", float("nan"))
        r4_tot = s.get("r4", {}).get("total_mag_cos", float("nan"))
        print(f"{bname.capitalize():<12} {cnt:>6} {gt:>9.1f} {err:>8.1f} | {r2_f16:>10.4f} {r2_tot:>9.4f} | {r4_32:>10.4f} {r4_tot:>9.4f}")

    # 2. Gradient Conflicts Table
    print("\n--- 2. Pairwise Component Gradient Cosine & Conflict Rate ---")
    print("(Conflict rate = fraction of crops where cos(g_a, g_b) < 0, i.e., antagonistic gradients)")
    header2 = f"{'Density Bin':<12} | {'R2: Root vs Flat16':^24} | {'R4: Root vs 32->16':^24} | {'R4: 64->32 vs 32->16':^24}"
    print(header2)
    print(f"{'':<12} | {'Mean Cos':>10} {'Conflict%':>12} | {'Mean Cos':>10} {'Conflict%':>12} | {'Mean Cos':>10} {'Conflict%':>12}")
    print("-" * len(header2))
    for bname in ["all", "sparse", "medium", "dense"]:
        s = summary_by_density[bname]
        r2_cos = s.get("r2", {}).get("cos_root_vs_flat16", float("nan"))
        r2_cr = s.get("r2", {}).get("conflict_root_vs_flat16", float("nan")) * 100
        r4_cos_r32 = s.get("r4", {}).get("cos_root_vs_32_16", float("nan"))
        r4_cr_r32 = s.get("r4", {}).get("conflict_root_vs_32_16", float("nan")) * 100
        r4_cos_tree = s.get("r4", {}).get("cos_64_32_vs_32_16", float("nan"))
        r4_cr_tree = s.get("r4", {}).get("conflict_64_32_vs_32_16", float("nan")) * 100
        print(f"{bname.capitalize():<12} | {r2_cos:>10.4f} {r2_cr:>11.1f}% | {r4_cos_r32:>10.4f} {r4_cr_r32:>11.1f}% | {r4_cos_tree:>10.4f} {r4_cr_tree:>11.1f}%")

    # 3. Kappa Sweep Table
    print("\n--- 3. Offline Kappa Sweep (N=30 crops) ---")
    header3 = f"{'Kappa':<8} | {'R2 Flat16 MagCos':>17} {'R2 Flat16 Norm':>15} | {'R4 32->16 MagCos':>17} {'R4 32->16 Norm':>15} {'R4 Root vs 32->16 Cos':>22}"
    print(header3)
    print("-" * len(header3))
    for k_str, ks in kappa_summary.items():
        k_val = ks["kappa"]
        r2_mc = ks["r2_flat16_mag_cos_mean"]
        r2_nm = ks["r2_flat16_norm_mean"]
        r4_mc = ks["r4_32_to_16_mag_cos_mean"]
        r4_nm = ks["r4_32_to_16_norm_mean"]
        r4_c = ks["r4_cos_root_vs_32_16_mean"]
        print(f"{k_val:<8} | {r2_mc:>17.4f} {r2_nm:>15.2f} | {r4_mc:>17.4f} {r4_nm:>15.2f} {r4_c:>22.4f}")

    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
