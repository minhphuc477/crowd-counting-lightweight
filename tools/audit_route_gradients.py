#!/usr/bin/env python3
"""Route-Specific Gradient Audit on Shared Pretrained Backbone.

Measures:
  g_C16: Gradient on shared backbone parameters induced by C16 route
  g_C32: Gradient on shared backbone parameters induced by C32 route

Computes:
  - Norm ratio: ||g_C32|| / ||g_C16||
  - Cosine alignment: cos(g_C16, g_C32)
  - Conflict rate: fraction of samples where cos(g_C16, g_C32) < 0
  - Stratification across Sparse (<300), Medium (300-1000), and Dense (>=1000) regimes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from hpc.data.factory import build_evaluation_dataset
from hpc.data.point_counts import build_exact_count_pyramid
from hpc.losses.factory import build_ntpc_criterion_from_config
from hpc.models.factory import assert_checkpoint_compatible, build_model_from_config


def make_natural_audit_crop(
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


def get_shared_backbone_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Identify parameters in the MobileNetV4 backbone that are shared upstream of C32 (i.e. stages up to C16)."""
    shared_params = []
    bb = model.backbone.backbone
    if hasattr(bb, "conv_stem") and bb.conv_stem is not None:
        shared_params.extend(p for p in bb.conv_stem.parameters() if p.requires_grad)
    if hasattr(bb, "bn1") and bb.bn1 is not None:
        shared_params.extend(p for p in bb.bn1.parameters() if p.requires_grad)
    if hasattr(bb, "blocks") and bb.blocks is not None:
        # stages 0, 1, 2 compute C4, C8, C16
        for block in bb.blocks[:3]:
            shared_params.extend(p for p in block.parameters() if p.requires_grad)
    return shared_params


def flatten_grads(parameters: List[nn.Parameter]) -> torch.Tensor:
    """Collect and flatten gradients from a parameter list into a single 1D tensor."""
    grads = []
    for p in parameters:
        if p.grad is not None:
            grads.append(p.grad.detach().view(-1))
        else:
            grads.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
    if not grads:
        return torch.tensor([], device=parameters[0].device if parameters else "cpu")
    return torch.cat(grads)


def audit_sample_route_gradients(
    model: nn.Module,
    criterion: nn.Module,
    image: torch.Tensor,
    targets: Dict[Any, torch.Tensor],
    full_gt_count: float,
    shared_params: List[nn.Parameter],
    device: torch.device,
) -> Dict[str, float]:
    """Compute g_C16, g_C32, norm ratio, and cosine gradient alignment on a single sample."""
    image = image.to(device)

    # 1. Forward backbone to obtain feature pyramid (c4, c8, c16, c32)
    features = model.backbone(image)
    if len(features) != 4:
        raise ValueError(f"Route audit requires 4 feature levels (C4, C8, C16, C32), got {len(features)}")
    c4, c8, c16, c32 = features

    # --- Route C16 Isolation ---
    # Detach c32 so no gradient flows from c32 into neck or backbone
    model.zero_grad(set_to_none=True)
    p4_c16 = model.neck(c4, c8, c16, c32.detach())
    mass_c16 = model.mass_from_p4(p4_c16)
    loss_c16, _ = criterion(mass_c16, targets)
    loss_c16.backward(retain_graph=True)
    g_c16_vec = flatten_grads(shared_params).float()

    # --- Route C32 Isolation ---
    # Detach c4, c8, c16 so gradient flows ONLY through c32 into neck and backbone
    model.zero_grad(set_to_none=True)
    p4_c32 = model.neck(c4.detach(), c8.detach(), c16.detach(), c32)
    mass_c32 = model.mass_from_p4(p4_c32)
    loss_c32, _ = criterion(mass_c32, targets)
    loss_c32.backward()
    g_c32_vec = flatten_grads(shared_params).float()

    # Metrics on shared backbone parameters
    norm_c16 = float(torch.norm(g_c16_vec, p=2).item())
    norm_c32 = float(torch.norm(g_c32_vec, p=2).item())
    dot_prod = float(torch.dot(g_c16_vec, g_c32_vec).item())

    denom = (norm_c16 * norm_c32) + 1e-12
    cos_sim = dot_prod / denom
    norm_ratio = norm_c32 / (norm_c16 + 1e-12)
    is_conflict = float(cos_sim < 0.0)

    crop_count = float(targets["N"].sum().item())

    return {
        "full_gt_count": full_gt_count,
        "crop_count": crop_count,
        "loss_c16": float(loss_c16.detach().item()),
        "loss_c32": float(loss_c32.detach().item()),
        "norm_c16": norm_c16,
        "norm_c32": norm_c32,
        "norm_ratio_c32_to_c16": norm_ratio,
        "cosine_similarity": cos_sim,
        "dot_product": dot_prod,
        "is_conflict": is_conflict,
    }


def run_route_gradient_audit(
    cfg: Dict[str, Any],
    checkpoint_path: str | None = None,
    split: str | None = None,
    max_samples: int | None = None,
    device_str: str | None = None,
) -> Dict[str, Any]:
    """Execute complete route gradient audit across the dataset."""
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Running Route Gradient Audit on device: {device}", flush=True)

    # Build model
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        assert_checkpoint_compatible(ckpt, cfg)
        model = build_model_from_config(cfg, load_pretrained=False).to(device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        print(f"Loaded checkpoint: {checkpoint_path} (epoch={ckpt.get('epoch', 'N/A')})", flush=True)
    else:
        model = build_model_from_config(cfg, load_pretrained=True).to(device)
        print("Initialized model with ImageNet pretrained backbone", flush=True)

    model.train()  # evaluate gradients with active graph
    # Freeze BatchNorm running statistics during probe to prevent drift
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()

    criterion_crop_stats = ckpt.get("resolved_crop_statistics") if checkpoint_path else None
    criterion = build_ntpc_criterion_from_config(cfg, crop_statistics=criterion_crop_stats).to(device)

    dataset, resolved_split = build_evaluation_dataset(cfg, split=split)
    n_samples = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    print(f"Evaluating {n_samples} images from split '{resolved_split}'", flush=True)

    shared_params = get_shared_backbone_parameters(model)
    print(f"Shared backbone parameters upstream of C32: {len(shared_params)} tensors ({sum(p.numel() for p in shared_params):,} weights)", flush=True)

    crop_size = int(cfg.get("dataset", {}).get("crop_size", 256))
    results: List[Dict[str, float]] = []

    start_t = time.time()
    for idx in range(n_samples):
        sample = dataset[idx]
        img_tensor = sample["image"].unsqueeze(0)  # (1, 3, H, W)
        pts = sample["gt_points"]
        full_gt_count = float(len(pts))

        crop_sample = make_natural_audit_crop(img_tensor, pts, max_crop=crop_size)
        if crop_sample is None:
            continue
        crop_image, crop_points = crop_sample
        _, _, ch, cw = crop_image.shape

        tree = build_exact_count_pyramid(
            [torch.from_numpy(crop_points).float()],
            height=ch,
            width=cw,
            block_sizes=(4, 8, 16, 32, 64),
            pad_multiple=64,
        )
        targets_b = {k: tree[k].to(device) for k in (4, 8, 16, 32, 64)}
        targets_b["N"] = tree["N"].to(device)

        res = audit_sample_route_gradients(
            model=model,
            criterion=criterion,
            image=crop_image,
            targets=targets_b,
            full_gt_count=full_gt_count,
            shared_params=shared_params,
            device=device,
        )
        results.append(res)
        if (idx + 1) % 25 == 0 or idx + 1 == n_samples:
            print(f"Audited [{idx + 1}/{n_samples}] samples ({time.time() - start_t:.1f}s)...", flush=True)

    # Aggregations & Stratification
    def aggregate_subset(subset: List[Dict[str, float]]) -> Dict[str, Any]:
        if not subset:
            return {"count": 0}
        cos_sims = np.array([r["cosine_similarity"] for r in subset], dtype=np.float64)
        norm_ratios = np.array([r["norm_ratio_c32_to_c16"] for r in subset], dtype=np.float64)
        norm_c16s = np.array([r["norm_c16"] for r in subset], dtype=np.float64)
        norm_c32s = np.array([r["norm_c32"] for r in subset], dtype=np.float64)
        conflicts = np.array([r["is_conflict"] for r in subset], dtype=np.float64)

        return {
            "count": len(subset),
            "norm_c16_mean": float(np.mean(norm_c16s)),
            "norm_c32_mean": float(np.mean(norm_c32s)),
            "norm_ratio_mean": float(np.mean(norm_ratios)),
            "norm_ratio_median": float(np.median(norm_ratios)),
            "norm_ratio_p90": float(np.percentile(norm_ratios, 90)),
            "cosine_similarity_mean": float(np.mean(cos_sims)),
            "cosine_similarity_median": float(np.median(cos_sims)),
            "cosine_similarity_p10": float(np.percentile(cos_sims, 10)),
            "conflict_fraction": float(np.mean(conflicts)),
            "conflict_percentage": f"{float(np.mean(conflicts)) * 100:.1f}%",
        }

    sparse_subset = [r for r in results if r["full_gt_count"] < 300]
    medium_subset = [r for r in results if 300 <= r["full_gt_count"] < 1000]
    dense_subset = [r for r in results if r["full_gt_count"] >= 1000]

    report = {
        "metadata": {
            "config": cfg.get("experiment", {}).get("name", "unnamed"),
            "checkpoint": checkpoint_path,
            "split": resolved_split,
            "total_samples": len(results),
            "crop_size": crop_size,
        },
        "overall": aggregate_subset(results),
        "stratification": {
            "sparse_lt300": aggregate_subset(sparse_subset),
            "medium_300_1000": aggregate_subset(medium_subset),
            "dense_ge1000": aggregate_subset(dense_subset),
        },
        "raw_samples": results,
    }

    # Print summary
    print("\n" + "=" * 70)
    print("        ROUTE-SPECIFIC GRADIENT AUDIT SUMMARY REPORT")
    print("=" * 70)
    ov = report["overall"]
    print(f"Overall Valid Samples: {ov['count']}")
    print(f"  ||g_C16|| Mean: {ov['norm_c16_mean']:.4f} | ||g_C32|| Mean: {ov['norm_c32_mean']:.4f}")
    print(f"  Norm Ratio (||g_C32|| / ||g_C16||): Mean={ov['norm_ratio_mean']:.2f}x, Median={ov['norm_ratio_median']:.2f}x, P90={ov['norm_ratio_p90']:.2f}x")
    print(f"  Cosine Alignment cos(g_C16, g_C32): Mean={ov['cosine_similarity_mean']:.4f}, Median={ov['cosine_similarity_median']:.4f}, P10={ov['cosine_similarity_p10']:.4f}")
    print(f"  Gradient Conflict Fraction (cos < 0): {ov['conflict_percentage']}")
    print("-" * 70)
    print("Density Stratification:")
    for name, sub in report["stratification"].items():
        if sub["count"] > 0:
            print(f"  [{name.upper()}] (n={sub['count']}): Ratio={sub['norm_ratio_mean']:.2f}x | Cosine={sub['cosine_similarity_mean']:.4f} | Conflict={sub['conflict_percentage']}")
    print("=" * 70 + "\n")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit route-specific gradients on shared backbone.")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt")
    parser.add_argument("--split", type=str, default="test_data", help="Dataset split to evaluate")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples to evaluate")
    parser.add_argument("--output", type=str, default=None, help="Path to save JSON report")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    report = run_route_gradient_audit(
        cfg=cfg,
        checkpoint_path=args.checkpoint,
        split=args.split,
        max_samples=args.max_samples,
        device_str=args.device,
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Saved audit report to: {out_path}")


if __name__ == "__main__":
    main()
