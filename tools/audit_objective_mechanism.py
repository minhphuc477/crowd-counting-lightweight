#!/usr/bin/env python3
"""CLI runner for Objective Mechanism Audit v2: Flat-DM16 vs Hierarchical DTM Tree.

Evaluates on actual trained checkpoints:
- R2 Flat-DM16 checkpoint: runs/factorial_a_crop256_c16/best.pt
- R4 Neural DTM Tree checkpoint: runs/ntpc_sha/best.pt

Evaluates:
1. Exact component mass gradients d(L_k)/d(mass) and true total gradient (fixed bug: no root double-counting).
2. Euler scale projection <g, m> and cos(g, m) (scale invariance test).
3. Parameter-space count direction:
   cos(grad_theta(N), grad_theta(L_k)) = <grad_theta(N), grad_theta(L_k)> / (||grad_theta(N)|| ||grad_theta(L_k)||)
   - cos > 0 => parameter update along -grad_theta(L_k) decreases predicted count in model weight space!
   - cos < 0 => parameter update increases count.
4. Cancellation ratio:
   C(g_a, g_b) = 1 - ||g_a + g_b|| / (||g_a|| + ||g_b||)
5. Stratified by local crop count: Low (<50), Medium (50-150), High (>=150).
6. Offline kappa sweep tracking tree cancellation C and cos(64->32, 32->16).
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
    cancellation_ratio,
    compute_audit_for_mode_v2,
    stratify_by_local_crop_count,
    summarize_audit_group_v2,
    sweep_kappa_on_crop_v2,
)
from hpc.losses.ntpc import NTPCConfig, NTPCLoss
from hpc.models.factory import assert_checkpoint_compatible, build_model_from_config
from hpc.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Objective Mechanism Audit v2")
    parser.add_argument("--config-r2", default="configs/factorial_a_crop256_c16.yaml", help="Path to R2 config YAML")
    parser.add_argument("--checkpoint-r2", default="runs/factorial_a_crop256_c16/best.pt", help="Path to R2 checkpoint .pt")
    parser.add_argument("--config-r4", default="configs/ntpc_sha.yaml", help="Path to R4 config YAML")
    parser.add_argument("--checkpoint-r4", default="runs/ntpc_sha/best.pt", help="Path to R4 checkpoint .pt")
    parser.add_argument("--output", default="runs/objective_audit/audit_v2_results.json", help="Output JSON path")
    parser.add_argument("--crop-size", type=int, default=256, help="Crop size for standardized spatial support")
    parser.add_argument("--max-samples", type=int, default=None, help="Max test images to evaluate (default: all)")
    parser.add_argument("--kappas", type=str, default="2,5,10,20,50,100", help="Comma-separated kappa values to sweep")
    parser.add_argument("--sweep-samples", type=int, default=40, help="Number of crops to run kappa sweep on")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_centered_crop(
    image: torch.Tensor,
    points: np.ndarray,
    crop_size: int = 256,
) -> Optional[Tuple[torch.Tensor, np.ndarray]]:
    _, _, h, w = image.shape
    if h < crop_size or w < crop_size:
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


def load_model(cfg_path: str, ckpt_path: str, device: torch.device) -> Tuple[nn.Module, dict]:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model = build_model_from_config(cfg, load_pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    assert_checkpoint_compatible(ckpt, cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, cfg


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)

    print("=" * 80)
    print("STARTING OBJECTIVE MECHANISM AUDIT V2 (R2 vs R4)")
    print("=" * 80, flush=True)

    print(f"Loading R2 model: {args.checkpoint_r2} ...", flush=True)
    model_r2, cfg_r2 = load_model(args.config_r2, args.checkpoint_r2, device)

    print(f"Loading R4 model: {args.checkpoint_r4} ...", flush=True)
    model_r4, cfg_r4 = load_model(args.config_r4, args.checkpoint_r4, device)

    dataset, split = build_evaluation_dataset(cfg_r2)
    total_images = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)
    print(f"Dataset: {len(dataset)} images in split '{split}', evaluating on {total_images} images (crop={args.crop_size}x{args.crop_size})", flush=True)

    kappas = [float(k.strip()) for k in args.kappas.split(",") if k.strip()]

    # Losses
    crit_r2 = NTPCLoss(NTPCConfig(mode="r2_flat_dm", root_loss="nb", kappa_flat16=20.0)).to(device)
    crit_r4 = NTPCLoss(NTPCConfig(
        mode="r4_dtm_tree16",
        root_loss="nb",
        kappa_root64=20.0,
        kappa_64_32=20.0,
        kappa_32_16=20.0,
    )).to(device)

    records_r2_native: List[Dict[str, Any]] = []
    records_r4_native: List[Dict[str, Any]] = []
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
        crop_img = crop_img.to(device)
        gt_count = float(len(crop_pts))

        target_pyramid = build_exact_count_pyramid(
            [torch.from_numpy(crop_pts).float()],
            height=args.crop_size,
            width=args.crop_size,
            block_sizes=(4, 8, 16, 32, 64),
            pad_multiple=64,
            device=device,
        )

        # -------------------------------------------------------------------
        # 1. Evaluate R2 Model Native State (Flat-DM16 trained)
        # -------------------------------------------------------------------
        with torch.no_grad():
            mass_r2 = model_r2.forward_mass(crop_img)
            pred_count_r2 = float(mass_r2.sum().item())

        audit_r2_native = compute_audit_for_mode_v2(
            model=model_r2,
            crop_img=crop_img,
            mass=mass_r2,
            targets=target_pyramid,
            criterion=crit_r2,
            active_components=("root_magnitude", "flat_16"),
        )
        rec_r2 = {
            "image_index": idx,
            "img_path": sample.get("img_path", f"img_{idx}"),
            "gt_count": gt_count,
            "pred_count": pred_count_r2,
            "signed_error": pred_count_r2 - gt_count,
            "r2": {
                "component_metrics": audit_r2_native["component_metrics"],
                "pairwise_cosines": audit_r2_native["pairwise_cosines"],
                "pairwise_cancellations": audit_r2_native["pairwise_cancellations"],
                "param_metrics": audit_r2_native["param_metrics"],
            },
        }
        records_r2_native.append(rec_r2)

        # -------------------------------------------------------------------
        # 2. Evaluate R4 Model Native State (DTM Tree trained)
        # -------------------------------------------------------------------
        with torch.no_grad():
            mass_r4 = model_r4.forward_mass(crop_img)
            pred_count_r4 = float(mass_r4.sum().item())

        audit_r4_native = compute_audit_for_mode_v2(
            model=model_r4,
            crop_img=crop_img,
            mass=mass_r4,
            targets=target_pyramid,
            criterion=crit_r4,
            active_components=("root_magnitude", "root_to_64", "64_to_32", "32_to_16"),
        )
        rec_r4 = {
            "image_index": idx,
            "img_path": sample.get("img_path", f"img_{idx}"),
            "gt_count": gt_count,
            "pred_count": pred_count_r4,
            "signed_error": pred_count_r4 - gt_count,
            "r4": {
                "component_metrics": audit_r4_native["component_metrics"],
                "pairwise_cosines": audit_r4_native["pairwise_cosines"],
                "pairwise_cancellations": audit_r4_native["pairwise_cancellations"],
                "param_metrics": audit_r4_native["param_metrics"],
            },
        }
        records_r4_native.append(rec_r4)

        # -------------------------------------------------------------------
        # 3. Kappa Sweep on R4 Native Mass Maps (tracking cancellation C)
        # -------------------------------------------------------------------
        if len(sweep_records) < args.sweep_samples:
            sweep_res = sweep_kappa_on_crop_v2(mass_r4, target_pyramid, kappas=kappas, device=device)
            sweep_records.append({
                "image_index": idx,
                "gt_count": gt_count,
                "sweep": sweep_res,
            })

        if (idx + 1) % 30 == 0 or (idx + 1) == total_images:
            print(f"Processed {idx + 1}/{total_images} images ({len(records_r2_native)} valid crops, {skipped} skipped) ...", flush=True)

    elapsed = time.time() - t0
    print(f"Audit computation completed in {elapsed:.1f}s.", flush=True)

    # Density stratification (by local crop count)
    bins_r2 = stratify_by_local_crop_count(records_r2_native)
    summary_r2 = {b: summarize_audit_group_v2(r) for b, r in bins_r2.items()}

    bins_r4 = stratify_by_local_crop_count(records_r4_native)
    summary_r4 = {b: summarize_audit_group_v2(r) for b, r in bins_r4.items()}

    # Kappa sweep summary (on R4 native mass maps)
    kappa_summary = {}
    for kappa in kappas:
        k_str = f"k_{int(kappa) if kappa == int(kappa) else kappa}"
        cos_tree = [r["sweep"]["r4_tree"][k_str]["cos_64_32_vs_32_16"] for r in sweep_records if math.isfinite(r["sweep"]["r4_tree"][k_str]["cos_64_32_vs_32_16"])]
        canc_tree = [r["sweep"]["r4_tree"][k_str]["cancellation_64_32_vs_32_16"] for r in sweep_records if math.isfinite(r["sweep"]["r4_tree"][k_str]["cancellation_64_32_vs_32_16"])]
        cos_r32 = [r["sweep"]["r4_tree"][k_str]["cos_root_vs_32_16"] for r in sweep_records if math.isfinite(r["sweep"]["r4_tree"][k_str]["cos_root_vs_32_16"])]
        norm_32 = [r["sweep"]["r4_tree"][k_str]["32_to_16"]["norm"] for r in sweep_records]
        norm_flat = [r["sweep"]["r2_flat"][k_str]["flat_16"]["norm"] for r in sweep_records]

        kappa_summary[k_str] = {
            "kappa": kappa,
            "cos_64_32_vs_32_16_mean": float(np.mean(cos_tree)) if cos_tree else float("nan"),
            "cancellation_64_32_vs_32_16_mean": float(np.mean(canc_tree)) if canc_tree else float("nan"),
            "conflict_rate_64_32_vs_32_16": float(np.mean([1.0 if c < 0.0 else 0.0 for c in cos_tree])) if cos_tree else float("nan"),
            "cos_root_vs_32_16_mean": float(np.mean(cos_r32)) if cos_r32 else float("nan"),
            "r4_32_to_16_norm_mean": float(np.mean(norm_32)) if norm_32 else float("nan"),
            "r2_flat16_norm_mean": float(np.mean(norm_flat)) if norm_flat else float("nan"),
        }

    output_data = {
        "metadata": {
            "checkpoint_r2": args.checkpoint_r2,
            "checkpoint_r4": args.checkpoint_r4,
            "crop_size": args.crop_size,
            "valid_crops": len(records_r2_native),
            "skipped_images": skipped,
            "kappas_swept": kappas,
            "elapsed_seconds": elapsed,
        },
        "r2_native_summary": summary_r2,
        "r4_native_summary": summary_r4,
        "kappa_sweep_summary": kappa_summary,
        "records_r2": records_r2_native,
        "records_r4": records_r4_native,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, allow_nan=True)
    print(f"\nWrote full audit v2 results to {out_path}", flush=True)

    # -----------------------------------------------------------------------
    # PRINT TERMINAL REPORT
    # -----------------------------------------------------------------------
    print("\n" + "=" * 95)
    print("OBJECTIVE MECHANISM AUDIT V2 REPORT (EVALUATED ON BOTH R2 & R4 CHECKPOINTS)")
    print("=" * 95)

    # Table 1: Model Prediction & Parameter-Space Count Direction
    print("\n--- 1. Parameter-Space Count Direction cos(grad_theta(N), grad_theta(L_k)) ---")
    print("cos > 0: optimizer step (-grad_theta(L)) DECREASES predicted count in model weight space (undercount push)")
    print("cos < 0: optimizer step INCREASES predicted count in model weight space")
    print("cos ~ 0: count-neutral spatial allocation")
    h1 = f"{'Model & Density':<22} {'Count':>6} {'Mean GT':>8} {'Bias':>8} | {'Root ParamCos':>14} {'Alloc ParamCos':>15} {'Total ParamCos':>15}"
    print(h1)
    print("-" * len(h1))

    for bname in ["all", "low (<50)", "medium (50-150)", "high (>=150)"]:
        # R2
        s2 = summary_r2[bname]
        cnt2 = s2["count"]
        gt2 = s2.get("mean_gt_count", float("nan"))
        err2 = s2.get("mean_count_error", float("nan"))
        r2_rt_pc = s2.get("r2", {}).get("root_param_cos", float("nan"))
        r2_f16_pc = s2.get("r2", {}).get("flat16_param_cos", float("nan"))
        r2_tot_pc = s2.get("r2", {}).get("total_param_cos", float("nan"))
        print(f"R2 {bname:<19} {cnt2:>6} {gt2:>8.1f} {err2:>8.2f} | {r2_rt_pc:>14.4f} {r2_f16_pc:>15.4f} {r2_tot_pc:>15.4f}")

        # R4
        s4 = summary_r4[bname]
        err4 = s4.get("mean_count_error", float("nan"))
        r4_rt_pc = s4.get("r4", {}).get("root_param_cos", float("nan"))
        r4_32_pc = s4.get("r4", {}).get("32_16_param_cos", float("nan"))
        r4_tot_pc = s4.get("r4", {}).get("total_param_cos", float("nan"))
        print(f"R4 {bname:<19} {cnt2:>6} {gt2:>8.1f} {err4:>8.2f} | {r4_rt_pc:>14.4f} {r4_32_pc:>15.4f} {r4_tot_pc:>15.4f}")
        print("-" * len(h1))

    # Table 2: Tree Level Conflict & Actual Cancellation Ratio on R4 Native Checkpoint
    print("\n--- 2. Tree Level Gradient Conflict & Cancellation Ratio on R4 Native Checkpoint ---")
    print("Conflict% = fraction of crops where cos(g_64, g_32) < 0 (antagonistic)")
    print("Cancellation C = 1 - ||g_a + g_b|| / (||g_a|| + ||g_b||) in [0, 1] (0 = no cancellation, 1 = 100% destroyed)")
    h2 = f"{'Crop Density Bin':<22} | {'Mean Cos(64->32, 32->16)':^28} | {'Conflict%':^12} | {'Cancellation Ratio C':^22}"
    print(h2)
    print("-" * len(h2))
    for bname in ["all", "low (<50)", "medium (50-150)", "high (>=150)"]:
        s4 = summary_r4[bname]
        cos_tr = s4.get("r4", {}).get("cos_64_32_vs_32_16", float("nan"))
        cr_tr = s4.get("r4", {}).get("conflict_64_32_vs_32_16", float("nan")) * 100
        c_tr = s4.get("r4", {}).get("cancellation_64_32_vs_32_16", float("nan"))
        print(f"{bname:<22} | {cos_tr:^28.4f} | {cr_tr:^11.1f}% | {c_tr:^22.4f}")

    # Table 3: Kappa Sweep on R4 Native Checkpoint
    print("\n--- 3. Offline Kappa Sweep on R4 Native Checkpoint (N=40 crops) ---")
    h3 = f"{'Kappa':<8} | {'Cos(64->32, 32->16)':>20} {'Conflict%':>12} {'Cancellation C':>16} | {'R4 32->16 Norm':>15} {'R2 Flat16 Norm':>15}"
    print(h3)
    print("-" * len(h3))
    for k_str, ks in kappa_summary.items():
        k_val = ks["kappa"]
        c_tr = ks["cos_64_32_vs_32_16_mean"]
        cr_tr = ks["conflict_rate_64_32_vs_32_16"] * 100
        canc = ks["cancellation_64_32_vs_32_16_mean"]
        n32 = ks["r4_32_to_16_norm_mean"]
        nf = ks["r2_flat16_norm_mean"]
        print(f"{k_val:<8} | {c_tr:>20.4f} {cr_tr:>11.1f}% {canc:>16.4f} | {n32:>15.2f} {nf:>15.2f}")

    print("=" * 95 + "\n")


if __name__ == "__main__":
    main()
