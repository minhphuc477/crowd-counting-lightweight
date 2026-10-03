#!/usr/bin/env python3
"""Deep Architectural Diagnostics & Mathematical Error Analysis Toolkit.

Performs first-principles forensic diagnostics on crowd counting architectures:
1. Spatial Cell Collision & Information Loss Audit (Stride 4 vs 2 vs 8).
2. Rayleigh Resolution Limit Violation Fraction (Delta < 1.26 / f_c).
3. Latent Feature SVD Rank & Representation Collapse Analysis.
4. Softplus Carrier Logit Dynamic Range & Saturation Barrier Audit.
5. ASPP-Lite Dilated Sampling Coverage & Gridding Artifacts.

Usage:
    python scripts/diagnose_architecture_and_errors.py --manifest data/sha_a_test.jsonl
    python scripts/diagnose_architecture_and_errors.py --manifest data/sha_a_test.jsonl \
        --config configs/rmr_v34/rmr_v34_diag_canonical.yaml \
        --checkpoint runs/sha_a/sub60_e26_scale_seeded_carrier/best_val_mae.pt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import yaml


def audit_dataset_collisions(
    manifest_path: str | Path,
    strides: tuple[int, ...] = (2, 4, 8),
    rayleigh_cutoff_px: float = 5.04,  # 1.26 / (1/4) = 5.04 px for Stride 4
) -> dict[str, Any]:
    """Audit ground truth point coordinates for spatial cell collisions."""
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_file}")

    total_images = 0
    total_heads = 0
    dense_images = 0
    sparse_images = 0
    mod_images = 0

    rayleigh_violations = 0
    total_pairs_evaluated = 0

    # Collision stats per stride
    stride_stats: dict[int, dict[str, int]] = {
        s: {
            "total_heads": 0,
            "occupied_cells": 0,
            "single_cells": 0,
            "double_cells": 0,
            "triple_plus_cells": 0,
            "max_cell_count": 0,
            "collided_heads": 0,
        }
        for s in strides
    }

    with open(manifest_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            raw_pts = item.get("points", [])
            if not raw_pts:
                continue

            pts = np.asarray(raw_pts, dtype=np.float32)
            n_heads = len(pts)
            total_images += 1
            total_heads += n_heads

            if n_heads <= 100:
                sparse_images += 1
            elif n_heads <= 500:
                mod_images += 1
            else:
                dense_images += 1

            # Inter-head distance check for Rayleigh limit (sampled if n_heads is very large)
            if n_heads > 1:
                pts_t = torch.from_numpy(pts)
                if n_heads > 1000:
                    idx = torch.randperm(n_heads)[:1000]
                    sample_pts = pts_t[idx]
                else:
                    sample_pts = pts_t

                dists = torch.cdist(sample_pts, sample_pts)
                dists.fill_diagonal_(float("inf"))
                min_dists, _ = dists.min(dim=1)
                sub_rayleigh = (min_dists < rayleigh_cutoff_px).sum().item()
                rayleigh_violations += sub_rayleigh
                total_pairs_evaluated += len(sample_pts)

            # Evaluate cell collision per stride
            for s in strides:
                st = stride_stats[s]
                st["total_heads"] += n_heads
                # Discrete cell indices
                cell_x = (pts[:, 0] // s).astype(np.int64)
                cell_y = (pts[:, 1] // s).astype(np.int64)
                cell_coords = np.stack([cell_x, cell_y], axis=1)

                _, counts = np.unique(cell_coords, axis=0, return_counts=True)
                st["occupied_cells"] += len(counts)
                st["single_cells"] += int((counts == 1).sum())
                st["double_cells"] += int((counts == 2).sum())
                st["triple_plus_cells"] += int((counts >= 3).sum())
                st["max_cell_count"] = max(st["max_cell_count"], int(counts.max()))
                st["collided_heads"] += int(counts[counts > 1].sum())

    report: dict[str, Any] = {
        "dataset_summary": {
            "manifest": str(manifest_file),
            "total_images": total_images,
            "total_heads": total_heads,
            "sparse_images (<=100)": sparse_images,
            "moderate_images (100-500)": mod_images,
            "dense_images (>500)": dense_images,
            "avg_heads_per_image": round(total_heads / max(total_images, 1), 2),
        },
        "rayleigh_resolution_audit": {
            "rayleigh_cutoff_px": rayleigh_cutoff_px,
            "evaluated_heads": total_pairs_evaluated,
            "rayleigh_violating_heads": rayleigh_violations,
            "sub_rayleigh_fraction_pct": round(
                100.0 * rayleigh_violations / max(total_pairs_evaluated, 1), 2
            ),
        },
        "stride_collision_audit": {},
    }

    for s, st in stride_stats.items():
        occ = max(st["occupied_cells"], 1)
        tot = max(st["total_heads"], 1)
        info_loss_pct = round(100.0 * (tot - occ) / tot, 2)
        collided_heads_pct = round(100.0 * st["collided_heads"] / tot, 2)
        report["stride_collision_audit"][f"stride_{s}"] = {
            "occupied_cells": st["occupied_cells"],
            "single_head_cells_pct": round(100.0 * st["single_cells"] / occ, 2),
            "double_head_cells_pct": round(100.0 * st["double_cells"] / occ, 2),
            "triple_plus_cells_pct": round(100.0 * st["triple_plus_cells"] / occ, 2),
            "max_heads_in_single_cell": st["max_cell_count"],
            "heads_in_multi_occupancy_cells_pct": collided_heads_pct,
            "discrete_information_loss_pct": info_loss_pct,
        }

    return report


def audit_feature_svd(
    model: torch.nn.Module,
    manifest_path: str | Path,
    device: torch.device,
    max_images: int = 50,
) -> dict[str, Any]:
    """Compute SVD singular spectrum and effective rank on backbone & neck features."""
    from rmr_core.data import resolve_manifest_path
    from PIL import Image
    from torchvision import transforms

    model.eval()
    model.to(device)

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    p4_feats_list: list[torch.Tensor] = []
    z0_vals_list: list[torch.Tensor] = []
    y0_vals_list: list[torch.Tensor] = []

    count = 0
    manifest_file = Path(manifest_path)
    data_root = manifest_file.parent if manifest_file.parent.name == "data" else ROOT / "data"

    with open(manifest_file, "r", encoding="utf-8") as f:
        for line in f:
            if count >= max_images:
                break
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            img_rel = item.get("image", "")
            img_path = resolve_manifest_path(img_rel, data_root=data_root)
            if img_path is None or not img_path.exists():
                continue

            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
                img_t = tf(img_rgb).unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(img_t)
                # Sample 200 spatial points from p4 to conserve RAM
                p4 = out.get("p4", None)
                if p4 is None and hasattr(model, "_extract_carrier_features"):
                    p4, _, _ = model._extract_carrier_features(img_t)
                if p4 is not None:
                    # Flatten spatial: [C, H*W] -> [H*W, C]
                    c = p4.shape[1]
                    flat_p4 = p4.squeeze(0).permute(1, 2, 0).reshape(-1, c)
                    sample_idx = torch.randperm(flat_p4.shape[0])[:200]
                    p4_feats_list.append(flat_p4[sample_idx].cpu())

                if "z0" in out:
                    z0 = out["z0"].squeeze(0).reshape(-1)
                    sample_idx = torch.randperm(z0.shape[0])[:500]
                    z0_vals_list.append(z0[sample_idx].cpu())
                if "y0" in out:
                    y0 = out["y0"].squeeze(0).reshape(-1)
                    sample_idx = torch.randperm(y0.shape[0])[:500]
                    y0_vals_list.append(y0[sample_idx].cpu())

            count += 1

    if not p4_feats_list:
        return {"error": "No features collected"}

    # Aggregate spatial features for SVD
    X = torch.cat(p4_feats_list, dim=0).float()  # [N, C]
    X_centered = X - X.mean(dim=0, keepdim=True)
    # Singular values of X_centered
    _, S, _ = torch.linalg.svd(X_centered, full_matrices=False)
    s_vals = S.numpy()
    s_norm = s_vals / np.sum(s_vals)

    # Effective rank (Shannon entropy): exp(-sum p_i ln p_i)
    eff_rank = float(np.exp(-np.sum(s_norm * np.log(np.maximum(s_norm, 1e-12)))))
    # Stable rank: ||S||_F^2 / ||S||_inf^2
    stable_rank = float(np.sum(s_vals**2) / max(s_vals[0] ** 2, 1e-12))

    top1_energy = float(s_norm[0])
    top2_energy = float(np.sum(s_norm[:2]))
    top4_energy = float(np.sum(s_norm[:4])) if len(s_norm) >= 4 else 1.0

    z0_stats = {}
    if z0_vals_list:
        all_z0 = torch.cat(z0_vals_list, dim=0).numpy()
        # Logit required for count >= 2.0 under Softplus(z): z = ln(exp(2) - 1) approx 1.854
        # Logit required for count >= 3.0: z = ln(exp(3) - 1) approx 2.949
        z0_stats = {
            "mean": round(float(np.mean(all_z0)), 4),
            "std": round(float(np.std(all_z0)), 4),
            "min": round(float(np.min(all_z0)), 4),
            "p50": round(float(np.percentile(all_z0, 50)), 4),
            "p95": round(float(np.percentile(all_z0, 95)), 4),
            "p99": round(float(np.percentile(all_z0, 99)), 4),
            "max": round(float(np.max(all_z0)), 4),
            "pct_cells_z0_ge_1_85 (count >= 2 barrier)": round(
                100.0 * float(np.mean(all_z0 >= 1.854)), 4
            ),
            "pct_cells_z0_ge_2_95 (count >= 3 barrier)": round(
                100.0 * float(np.mean(all_z0 >= 2.949)), 4
            ),
        }

    return {
        "images_evaluated": count,
        "feature_dim_C": int(X.shape[1]),
        "effective_rank": round(eff_rank, 2),
        "stable_rank": round(stable_rank, 2),
        "rank_utilization_pct": round(100.0 * eff_rank / X.shape[1], 2),
        "singular_values_top8": [round(float(v), 4) for v in s_norm[:8]],
        "top1_singular_variance_pct": round(top1_energy * 100.0, 2),
        "top2_singular_variance_pct": round(top2_energy * 100.0, 2),
        "top4_singular_variance_pct": round(top4_energy * 100.0, 2),
        "carrier_logit_z0_diagnostics": z0_stats,
    }


def audit_aspp_sampling_geometry(
    dilations: tuple[int, ...] = (1, 3, 6),
    kernel_size: int = 3,
    stride: int = 4,
) -> dict[str, Any]:
    """Audit ASPP-Lite dilated convolution grid coverage and gridding gaps."""
    results: dict[str, Any] = {}
    for d in dilations:
        rf_feature = (kernel_size - 1) * d + 1
        rf_pixels = (rf_feature - 1) * stride + 1
        # In a 3x3 dilated kernel with dilation d, sampling points are spaced by d*stride pixels.
        sampling_stride_px = d * stride
        # Area covered by convex hull of the 3x3 kernel: rf_pixels x rf_pixels
        total_hull_pixels = rf_pixels * rf_pixels
        # Sampled locations: exactly 9 points
        sampled_points = 9
        # Gridding sparsity ratio
        sparsity_pct = round(100.0 * (1.0 - sampled_points / max(total_hull_pixels, 1)), 4)
        results[f"dilation_{d}"] = {
            "kernel_size": kernel_size,
            "dilation": d,
            "rf_in_feature_grid": rf_feature,
            "rf_in_image_pixels": rf_pixels,
            "sampling_step_pixels": sampling_stride_px,
            "convex_hull_area_px": total_hull_pixels,
            "sampled_points": sampled_points,
            "sampling_sparsity_pct": sparsity_pct,
            "gridding_hole_risk": "HIGH" if d >= 6 else ("MEDIUM" if d >= 3 else "LOW"),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep Architectural Diagnostics & Error Analysis Toolkit")
    parser.add_argument("--manifest", type=str, default="data/sha_a_test.jsonl", help="Dataset manifest")
    parser.add_argument("--config", type=str, default=None, help="Model YAML config")
    parser.add_argument("--checkpoint", type=str, default=None, help="Trained model checkpoint (.pt)")
    parser.add_argument("--output", type=str, default="reports/architectural_diagnosis.json", help="Report path")
    parser.add_argument("--max_images", type=int, default=50, help="Max images for SVD evaluation")
    parser.add_argument("--device", type=str, default="cpu", help="Device for evaluation (cpu or cuda)")
    args = parser.parse_args()

    print(f"[DIAGNOSTICS] Starting Deep Architectural Diagnosis on manifest: {args.manifest}")
    full_report: dict[str, Any] = {}

    # Pillar 1 & 2: Dataset Collisions & Rayleigh Limits
    print("[1/3] Auditing Point Process Cell Collisions & Rayleigh Resolution Limits...")
    collision_report = audit_dataset_collisions(args.manifest)
    full_report["cell_collision_and_rayleigh_audit"] = collision_report
    print("      Done.")

    # Pillar 3: ASPP Sampling Geometry
    print("[2/3] Auditing ASPP-Lite Sampling Geometry & Gridding Holes...")
    aspp_report = audit_aspp_sampling_geometry(dilations=(1, 3, 6), kernel_size=3, stride=4)
    full_report["aspp_sampling_geometry"] = aspp_report
    print("      Done.")

    # Pillar 4: Model Feature SVD Rank & Logits (if config & checkpoint provided)
    if args.config is not None and args.checkpoint is not None:
        cfg_path = Path(args.config)
        ckpt_path = Path(args.checkpoint)
        if cfg_path.exists() and ckpt_path.exists():
            print(f"[3/3] Auditing SVD Feature Rank & Logit Range from checkpoint: {ckpt_path.name}...")
            from rmr_v3.model import RMRv3, RMRv3Config
            with open(cfg_path, "r", encoding="utf-8") as f:
                raw_cfg = yaml.safe_load(f)
            model_cfg = RMRv3Config.from_dict(raw_cfg["model"])
            model = RMRv3(model_cfg)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model", ckpt)
            model.load_state_dict(state_dict, strict=False)

            dev = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
            svd_report = audit_feature_svd(model, args.manifest, dev, max_images=args.max_images)
            full_report["model_svd_and_logit_diagnostics"] = svd_report
            print("      Done.")
        else:
            print("[3/3] Skipping Model SVD (Config or Checkpoint file not found).")
    else:
        print("[3/3] Skipping Model SVD (No config/checkpoint specified).")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"[DIAGNOSTICS] Comprehensive report written to: {out_path}")


if __name__ == "__main__":
    main()
