"""
Evaluation of Finite Horizon (FH) Boundary Mechanism in FH-CMICF B8.

This script tests the hypothesis:
Is cross-scale and spatial measure inconsistency driven by Finite Horizon block boundary
composition, or is it driven by visual crowd scale / backbone receptive field?

Specifically:
1. Cell-level within-block geometry (Boundary d=0 vs Interior d=1, Anchor, Terminal, Corners).
   Measures negative mass concentration, GT cell error, and continuous cross-scale mismatch.
2. Normalized FH units (0.5x, 1x, 2x, 4x FH spans in scaled space) vs Fixed Original Pixels (32, 64, 128, 256 px).
   Checks whether relative mismatch peaks at ~2 FH spans or at ~128 original pixels across scales.
3. Explicit boundary-crossing conditioning:
   - Size 0.5x FH (32 scaled px): Clean Inside (0 cuts) vs Boundary-Straddling (>=1 cuts).
   - Size 1.0x FH (64 scaled px): Grid-Aligned (1 intact FH block) vs Grid-Offset (+32px, 4-block junction).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hpc.models.micf_lite import MICFLite
from hpc.data.sha import ShanghaiTechDataset
from tools.eval_fh_cmicf_scale_consistency import (
    safe_torch_load,
    load_config,
    build_model_from_config,
    build_dataset,
    as_numpy_points,
    in_bounds_points,
    count_points_in_rect,
    conservative_region_mass,
    clipped_cell_edges,
    resize_image_tensor,
    discrete_mixed_difference,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit FH-CMICF boundary mechanism")
    p.add_argument("--checkpoint", type=str, default="runs/pilot_micf/b8_k4/best.pt")
    p.add_argument("--config", type=str, default="configs/pilot_micf/b8.yaml")
    p.add_argument("--dataset-root", type=str, default="./data/ShanghaiTech")
    p.add_argument("--part", type=str, default="part_A")
    p.add_argument("--split", type=str, default="test_data")
    p.add_argument("--scales", nargs="+", type=float, default=[0.5, 0.75, 1.0, 1.25, 1.5])
    p.add_argument("--reference-scale", type=float, default=1.0)
    p.add_argument("--fh-multipliers", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0])
    p.add_argument("--fixed-sizes", nargs="+", type=float, default=[32.0, 64.0, 128.0, 256.0])
    p.add_argument("--regions-per-size", type=int, default=16, help="Regions sampled per size.")
    p.add_argument("--max-samples", type=int, default=None, help="Debug limit on test samples.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=20260904, help="Random seed.")
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save audit outputs. Default: <checkpoint_dir>/eval_boundary_mechanism",
    )
    p.add_argument(
        "--allow-non-s16-k4",
        "--allow-non-b8",
        dest="allow_non_s16_k4",
        action="store_true",
        help="Disable the strict stride16/K4 model identity check.",
    )
    return p.parse_args()


# -----------------------------------------------------------------------------
# Fast Cumulative Evaluation
# -----------------------------------------------------------------------------

def build_cumulative_vertex_grid(y_mass: np.ndarray) -> np.ndarray:
    H, W = y_mass.shape
    C = np.zeros((H + 1, W + 1), dtype=np.float64)
    C[1:, 1:] = np.cumsum(np.cumsum(y_mass, axis=0), axis=1)
    return C


def resample_reference_grid_mass(
    ref_y_mass: np.ndarray,
    target_h: int,
    target_w: int,
    target_sh: int,
    target_sw: int,
    ref_sh: int,
    ref_sw: int,
    sx: float,
    sy: float,
    ref_sx: float,
    ref_sy: float,
    stride: int,
) -> np.ndarray:
    """
    Evaluates reference mass over each cell of the target grid using exact physical support:
    target cell [j*s, min((j+1)*s, sw)) x [i*s, min((i+1)*s, sh)) mapped to reference image
    coordinates and integrated against reference cell actual supports.
    """
    ref_h, ref_w = ref_y_mass.shape
    ref_x_edges = clipped_cell_edges(ref_sw, ref_w, stride)
    ref_y_edges = clipped_cell_edges(ref_sh, ref_h, stride)

    # Physical edges of target cells in reference pixel coordinates
    tgt_x_edges = clipped_cell_edges(target_sw, target_w, stride) * (ref_sx / sx)
    tgt_y_edges = clipped_cell_edges(target_sh, target_h, stride) * (ref_sy / sy)

    ref_dx = ref_x_edges[1:] - ref_x_edges[:-1]
    ref_dy = ref_y_edges[1:] - ref_y_edges[:-1]

    # Overlap matrices
    tgt_x0 = tgt_x_edges[:-1, None]
    tgt_x1 = tgt_x_edges[1:, None]
    ref_x0 = ref_x_edges[None, :-1]
    ref_x1 = ref_x_edges[None, 1:]
    overlap_x = np.maximum(0.0, np.minimum(tgt_x1, ref_x1) - np.maximum(tgt_x0, ref_x0))
    W_X = overlap_x / ref_dx[None, :]

    tgt_y0 = tgt_y_edges[:-1, None]
    tgt_y1 = tgt_y_edges[1:, None]
    ref_y0 = ref_y_edges[None, :-1]
    ref_y1 = ref_y_edges[None, 1:]
    overlap_y = np.maximum(0.0, np.minimum(tgt_y1, ref_y1) - np.maximum(tgt_y0, ref_y0))
    W_Y = overlap_y / ref_dy[None, :]

    return W_Y @ ref_y_mass @ W_X.T


# -----------------------------------------------------------------------------
# Module 1: Vectorized Cell Geometry Analysis
# -----------------------------------------------------------------------------

def analyze_cell_geometry_vectorized(
    pred_y: np.ndarray,
    ref_cell_masses: np.ndarray,
    points_scaled: np.ndarray,
    stride: int = 16,
    fh: int = 4,
) -> dict[str, dict[str, float]]:
    out_h, out_w = pred_y.shape

    # Rasterize GT points to cell grid using canonical rounding convention floor((p + 0.5) / stride)
    gt_grid = np.zeros((out_h, out_w), dtype=np.float64)
    for pt in points_scaled:
        px, py = pt[0], pt[1]
        cx = int(min(out_w - 1, max(0, math.floor((px + 0.5) / stride))))
        cy = int(min(out_h - 1, max(0, math.floor((py + 0.5) / stride))))
        gt_grid[cy, cx] += 1.0

    # Create coordinate grids
    cy_grid, cx_grid = np.meshgrid(np.arange(out_h), np.arange(out_w), indexing="ij")
    u = cx_grid % fh
    v = cy_grid % fh
    dx = np.minimum(u, fh - 1 - u)
    dy = np.minimum(v, fh - 1 - v)
    d_boundary = np.minimum(dx, dy)

    # Masks for categories
    masks = {
        "Boundary (d=0)": (d_boundary == 0),
        "Interior (d=1)": (d_boundary > 0),
        "Corner (4/16)": (dx == 0) & (dy == 0),
        "Anchor (0,0)": (u == 0) & (v == 0),
        "Terminal (3,3)": (u == fh - 1) & (v == fh - 1),
    }

    neg_mass = np.maximum(-pred_y, 0.0)
    pos_mass = np.maximum(pred_y, 0.0)
    abs_err = np.abs(pred_y - gt_grid)
    signed_err = pred_y - gt_grid
    mismatch = np.abs(pred_y - ref_cell_masses)

    results = {}
    for cat_name, mask in masks.items():
        n = int(np.sum(mask))
        if n == 0:
            continue
        results[cat_name] = {
            "count": n,
            "neg_mass": float(np.sum(neg_mass[mask])),
            "pos_mass": float(np.sum(pos_mass[mask])),
            "abs_err": float(np.sum(abs_err[mask])),
            "sq_err": float(np.sum(signed_err[mask] ** 2)),
            "signed_err": float(np.sum(signed_err[mask])),
            "mismatch": float(np.sum(mismatch[mask])),
            "ref_val": float(np.sum(ref_cell_masses[mask])),
        }

    phase_results = {}
    for vv in range(fh):
        for uu in range(fh):
            mask_phase = (u == uu) & (v == vv)
            n_p = int(np.sum(mask_phase))
            if n_p > 0:
                phase_results[(uu, vv)] = {
                    "count": n_p,
                    "neg_mass": float(np.sum(neg_mass[mask_phase])),
                    "pos_mass": float(np.sum(pos_mass[mask_phase])),
                    "abs_err": float(np.sum(abs_err[mask_phase])),
                    "sq_err": float(np.sum(signed_err[mask_phase] ** 2)),
                    "signed_err": float(np.sum(signed_err[mask_phase])),
                    "mismatch": float(np.sum(mismatch[mask_phase])),
                    "ref_val": float(np.sum(ref_cell_masses[mask_phase])),
                }

    return results, phase_results


# -----------------------------------------------------------------------------
# Module 2 & 3: Region Sampling
# -----------------------------------------------------------------------------

def sample_arbitrary_rects(
    image_h: int,
    image_w: int,
    sizes: list[float],
    aspect_ratios: list[float],
    regions_per_size: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    rectangles = []
    for nominal_size in sizes:
        feasible = []
        for aspect in aspect_ratios:
            width = float(nominal_size) * math.sqrt(float(aspect))
            height = float(nominal_size) / math.sqrt(float(aspect))
            if width <= float(image_w) and height <= float(image_h):
                feasible.append((float(aspect), width, height))
        if not feasible:
            continue
        for r_id in range(regions_per_size):
            aspect, width, height = feasible[rng.choice(len(feasible))]
            max_x0 = max(0.0, float(image_w) - width)
            max_y0 = max(0.0, float(image_h) - height)
            x0 = 0.0 if max_x0 == 0.0 else rng.random() * max_x0
            y0 = 0.0 if max_y0 == 0.0 else rng.random() * max_y0
            rectangles.append({
                "region_id": r_id,
                "nominal_size": float(nominal_size),
                "aspect": float(aspect),
                "x0": float(x0),
                "y0": float(y0),
                "x1": float(x0 + width),
                "y1": float(y0 + height),
                "width": float(width),
                "height": float(height),
            })
    return rectangles


def sample_conditioned_fh_rects(
    scaled_h: int,
    scaled_w: int,
    sx: float,
    sy: float,
    fh_span_px: int = 64,
    rng: np.random.Generator = None,
    num_samples: int = 16,
) -> list[dict[str, Any]]:
    rects = []
    n_blocks_x = scaled_w // fh_span_px
    n_blocks_y = scaled_h // fh_span_px

    if n_blocks_x > 0 and n_blocks_y > 0:
        # 1a. Size 32 Clean Interior (strictly inside a single 64x64 block)
        for i in range(num_samples):
            bx = rng.integers(0, n_blocks_x)
            by = rng.integers(0, n_blocks_y)
            ox = rng.uniform(8.0, 24.0)
            oy = rng.uniform(8.0, 24.0)
            sx0 = bx * fh_span_px + ox
            sy0 = by * fh_span_px + oy
            sx1 = sx0 + 32.0
            sy1 = sy0 + 32.0
            rects.append({
                "group": "size32_clean_interior",
                "scaled_size": 32.0,
                "x0_orig": sx0 / sx,
                "y0_orig": sy0 / sy,
                "x1_orig": sx1 / sx,
                "y1_orig": sy1 / sy,
                "cuts": 0,
            })

        # 1b. Size 32 Boundary Straddling (centered on boundary line)
        if n_blocks_x >= 2 or n_blocks_y >= 2:
            for i in range(num_samples):
                if n_blocks_x >= 2 and (n_blocks_y < 2 or rng.random() < 0.5):
                    bx = rng.integers(1, n_blocks_x)
                    by = rng.integers(0, n_blocks_y)
                    sx0 = bx * fh_span_px - 16.0
                    sy0 = by * fh_span_px + rng.uniform(8.0, 24.0)
                    cuts = 1
                else:
                    bx = rng.integers(0, n_blocks_x)
                    by = rng.integers(1, n_blocks_y)
                    sx0 = bx * fh_span_px + rng.uniform(8.0, 24.0)
                    sy0 = by * fh_span_px - 16.0
                    cuts = 1
                sx1 = sx0 + 32.0
                sy1 = sy0 + 32.0
                rects.append({
                    "group": "size32_boundary_straddling",
                    "scaled_size": 32.0,
                    "x0_orig": sx0 / sx,
                    "y0_orig": sy0 / sy,
                    "x1_orig": sx1 / sx,
                    "y1_orig": sy1 / sy,
                    "cuts": cuts,
                })

        # 2a. Size 64 Grid-Aligned (offset 0: exactly 1 intact FH block)
        for i in range(num_samples):
            bx = rng.integers(0, n_blocks_x)
            by = rng.integers(0, n_blocks_y)
            sx0 = bx * fh_span_px
            sy0 = by * fh_span_px
            sx1 = sx0 + 64.0
            sy1 = sy0 + 64.0
            rects.append({
                "group": "size64_grid_aligned",
                "scaled_size": 64.0,
                "x0_orig": sx0 / sx,
                "y0_orig": sy0 / sy,
                "x1_orig": sx1 / sx,
                "y1_orig": sy1 / sy,
                "cuts": 0,
            })

        # 2b. Size 64 Grid-Offset (+32, +32: junction of 4 FH blocks)
        if n_blocks_x >= 2 and n_blocks_y >= 2:
            for i in range(num_samples):
                bx = rng.integers(0, n_blocks_x - 1)
                by = rng.integers(0, n_blocks_y - 1)
                sx0 = bx * fh_span_px + 32.0
                sy0 = by * fh_span_px + 32.0
                sx1 = sx0 + 64.0
                sy1 = sy0 + 64.0
                rects.append({
                    "group": "size64_grid_offset_junction",
                    "scaled_size": 64.0,
                    "x0_orig": sx0 / sx,
                    "y0_orig": sy0 / sy,
                    "x1_orig": sx1 / sx,
                    "y1_orig": sy1 / sy,
                    "cuts": 2,
                })

    return rects


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(args.checkpoint).resolve().parent / "eval_boundary_mechanism"
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = safe_torch_load(args.checkpoint, map_location="cpu")
    cfg = load_config(checkpoint, args.config)
    m_id = cfg.get("experiment", {}).get("model_id", "MICF")
    model = build_model_from_config(cfg, checkpoint["state_dict"], device=device)

    stride = int(model.output_stride)
    fh = int(model.finite_horizon) if model.finite_horizon is not None else None

    if not args.allow_non_s16_k4:
        if model.head_type != "cumulative":
            raise RuntimeError(f"Expected cumulative head, got {model.head_type}")
        if stride != 16:
            raise RuntimeError(f"Expected output_stride=16, got {stride}")
        if fh is None or fh != 4:
            raise RuntimeError(f"Expected finite_horizon=4, got {fh}")

    dataset = build_dataset(cfg, root_override=args.dataset_root, part_override=args.part, split=args.split)
    n_total = len(dataset) if args.max_samples is None else min(len(dataset), int(args.max_samples))

    scales = [float(s) for s in args.scales]
    fh_mults = [float(m) for m in args.fh_multipliers]
    fixed_sizes = [float(sz) for sz in args.fixed_sizes]
    fh_span_scaled_px = stride * fh  # 16 * 4 = 64

    print("=" * 80)
    print("FINITE HORIZON BOUNDARY MECHANISM AUDIT")
    print(f"Model: {m_id} (output_stride={stride}, finite_horizon={fh}, block_span={fh_span_scaled_px}px)")
    print(f"Output Dir: {out_dir}")
    print(f"Images to evaluate: {n_total}")
    print(f"Scales: {scales}")
    print(f"FH Multipliers: {fh_mults} x ({fh_span_scaled_px}px / scale)")
    print(f"Fixed Sizes: {fixed_sizes} original pixels")
    print(f"Device: {device}")
    print("=" * 80)

    cell_accum = defaultdict(lambda: defaultdict(lambda: {
        "count": 0, "neg_mass": 0.0, "pos_mass": 0.0, "abs_err": 0.0,
        "sq_err": 0.0, "signed_err": 0.0, "mismatch": 0.0, "ref_val": 0.0
    }))
    phase_accum = defaultdict(lambda: defaultdict(lambda: {
        "count": 0, "neg_mass": 0.0, "pos_mass": 0.0, "abs_err": 0.0,
        "sq_err": 0.0, "signed_err": 0.0, "mismatch": 0.0, "ref_val": 0.0
    }))

    norm_fh_accum = defaultdict(list)
    fixed_sz_accum = defaultdict(list)
    cond_accum = defaultdict(list)

    rng = np.random.default_rng(args.seed)

    for idx in range(n_total):
        sample = dataset[idx]
        image = sample["image"].unsqueeze(0).to(device)
        _, _, orig_h, orig_w = image.shape
        points_raw = as_numpy_points(sample["gt_points"])
        points_in = in_bounds_points(points_raw, orig_h, orig_w)
        img_name = os.path.basename(str(sample["img_path"]))

        if (idx + 1) % 20 == 0 or idx == n_total - 1:
            print(f"Processing [{idx + 1:3d}/{n_total:3d}] {img_name}...")

        scale_maps: dict[float, dict[str, Any]] = {}
        with torch.inference_mode():
            for scale in scales:
                img_s, sh, sw, sx, sy = resize_image_tensor(image, scale)
                pred_count_t, pred_c = model.predict(img_s, pad_multiple=64)
                pred_c = pred_c.float()
                pred_y = discrete_mixed_difference(pred_c).squeeze(0).squeeze(0).cpu().numpy().astype(np.float64)
                scale_maps[scale] = {
                    "y_mass": pred_y,
                    "sh": sh, "sw": sw, "sx": sx, "sy": sy,
                    "count": float(pred_count_t.item())
                }

        ref_map = scale_maps[args.reference_scale]

        # 1. Module 1: Cell-Level Dissection (Vectorized)
        for scale in scales:
            s_map = scale_maps[scale]
            pts_s = points_in.copy()
            pts_s[:, 0] *= s_map["sx"]
            pts_s[:, 1] *= s_map["sy"]

            # Evaluate continuous reference cell masses
            if scale == args.reference_scale:
                ref_cell_masses = ref_map["y_mass"]
            else:
                ref_cell_masses = resample_reference_grid_mass(
                    ref_y_mass=ref_map["y_mass"],
                    target_h=s_map["y_mass"].shape[0],
                    target_w=s_map["y_mass"].shape[1],
                    target_sh=s_map["sh"],
                    target_sw=s_map["sw"],
                    ref_sh=ref_map["sh"],
                    ref_sw=ref_map["sw"],
                    sx=s_map["sx"],
                    sy=s_map["sy"],
                    ref_sx=ref_map["sx"],
                    ref_sy=ref_map["sy"],
                    stride=stride,
                )

            cell_res, phase_res = analyze_cell_geometry_vectorized(
                pred_y=s_map["y_mass"],
                ref_cell_masses=ref_cell_masses,
                points_scaled=pts_s,
                stride=stride,
                fh=fh,
            )

            for cat, metrics in cell_res.items():
                acc = cell_accum[scale][cat]
                for k, v in metrics.items():
                    acc[k] += v

            for phase_coord, metrics in phase_res.items():
                acc = phase_accum[scale][phase_coord]
                for k, v in metrics.items():
                    acc[k] += v

        # 2. Module 2: Normalized FH Units vs Fixed Original Pixels
        for scale in scales:
            s_map = scale_maps[scale]
            norm_sizes = [mult * (fh_span_scaled_px / scale) for mult in fh_mults]
            norm_rects = sample_arbitrary_rects(
                image_h=orig_h,
                image_w=orig_w,
                sizes=norm_sizes,
                aspect_ratios=[0.5, 1.0, 2.0],
                regions_per_size=args.regions_per_size,
                rng=rng,
            )

            for r in norm_rects:
                mult = min(fh_mults, key=lambda m: abs(m - (r["nominal_size"] / (fh_span_scaled_px / scale))))
                gt_r = count_points_in_rect(points_in, r["x0"], r["y0"], r["x1"], r["y1"])
                pred_r = conservative_region_mass(
                    s_map["y_mass"],
                    r["x0"] * s_map["sx"], r["y0"] * s_map["sy"],
                    r["x1"] * s_map["sx"], r["y1"] * s_map["sy"],
                    s_map["sh"], s_map["sw"], stride
                )
                ref_r = conservative_region_mass(
                    ref_map["y_mass"],
                    r["x0"] * ref_map["sx"], r["y0"] * ref_map["sy"],
                    r["x1"] * ref_map["sx"], r["y1"] * ref_map["sy"],
                    ref_map["sh"], ref_map["sw"], stride
                )

                norm_fh_accum[(scale, mult)].append({
                    "gt": gt_r,
                    "pred": pred_r,
                    "ref": ref_r,
                    "mismatch": abs(pred_r - ref_r),
                    "abs_err": abs(pred_r - gt_r),
                    "rel_mismatch": abs(pred_r - ref_r) / max(abs(ref_r), 1.0),
                    "rel_err": abs(pred_r - gt_r) / max(gt_r, 1.0),
                })

        # Fixed Original Pixels
        fixed_rects = sample_arbitrary_rects(
            image_h=orig_h,
            image_w=orig_w,
            sizes=fixed_sizes,
            aspect_ratios=[0.5, 1.0, 2.0],
            regions_per_size=args.regions_per_size,
            rng=rng,
        )
        for r in fixed_rects:
            sz = r["nominal_size"]
            gt_r = count_points_in_rect(points_in, r["x0"], r["y0"], r["x1"], r["y1"])
            ref_r = conservative_region_mass(
                ref_map["y_mass"],
                r["x0"] * ref_map["sx"], r["y0"] * ref_map["sy"],
                r["x1"] * ref_map["sx"], r["y1"] * ref_map["sy"],
                ref_map["sh"], ref_map["sw"], stride
            )
            for scale in scales:
                s_map = scale_maps[scale]
                pred_r = conservative_region_mass(
                    s_map["y_mass"],
                    r["x0"] * s_map["sx"], r["y0"] * s_map["sy"],
                    r["x1"] * s_map["sx"], r["y1"] * s_map["sy"],
                    s_map["sh"], s_map["sw"], stride
                )
                fixed_sz_accum[(scale, sz)].append({
                    "gt": gt_r,
                    "pred": pred_r,
                    "ref": ref_r,
                    "mismatch": abs(pred_r - ref_r),
                    "abs_err": abs(pred_r - gt_r),
                    "rel_mismatch": abs(pred_r - ref_r) / max(abs(ref_r), 1.0),
                    "rel_err": abs(pred_r - gt_r) / max(gt_r, 1.0),
                })

        # 3. Module 3: Boundary Conditioning
        for scale in scales:
            s_map = scale_maps[scale]
            cond_rects = sample_conditioned_fh_rects(
                scaled_h=s_map["sh"],
                scaled_w=s_map["sw"],
                sx=s_map["sx"],
                sy=s_map["sy"],
                fh_span_px=fh_span_scaled_px,
                rng=rng,
                num_samples=16,
            )
            for cr in cond_rects:
                gt_r = count_points_in_rect(points_in, cr["x0_orig"], cr["y0_orig"], cr["x1_orig"], cr["y1_orig"])
                pred_r = conservative_region_mass(
                    s_map["y_mass"],
                    cr["x0_orig"] * s_map["sx"], cr["y0_orig"] * s_map["sy"],
                    cr["x1_orig"] * s_map["sx"], cr["y1_orig"] * s_map["sy"],
                    s_map["sh"], s_map["sw"], stride
                )
                ref_r = conservative_region_mass(
                    ref_map["y_mass"],
                    cr["x0_orig"] * ref_map["sx"], cr["y0_orig"] * ref_map["sy"],
                    cr["x1_orig"] * ref_map["sx"], cr["y1_orig"] * ref_map["sy"],
                    ref_map["sh"], ref_map["sw"], stride
                )
                cond_accum[(scale, cr["group"])].append({
                    "gt": gt_r,
                    "pred": pred_r,
                    "ref": ref_r,
                    "mismatch": abs(pred_r - ref_r),
                    "abs_err": abs(pred_r - gt_r),
                    "rel_mismatch": abs(pred_r - ref_r) / max(abs(ref_r), 1.0),
                    "rel_err": abs(pred_r - gt_r) / max(gt_r, 1.0),
                })

    # =========================================================================
    # Reporting & Exporting
    # =========================================================================
    # =========================================================================
    # Reporting & Exporting
    # =========================================================================
    print("\n" + "=" * 100)
    print("MODULE 1: WITHIN-BLOCK CELL GEOMETRY AUDIT")
    print("Formula: Negative Variation Ratio (NVR) = sum(max(-Y, 0)) / max(sum(max(Y, 0)), 1e-12)")
    print("=" * 100)
    print(f"{'Scale':<8}{'Category':<22}{'N Cells':<12}{'NVR Share%':<12}{'NVR%':<12}{'Cell MAE':<12}{'RelMismatch':<14}")
    print("-" * 100)

    cell_summary_rows = []
    for scale in scales:
        for cat in ["Boundary (d=0)", "Interior (d=1)", "Corner (4/16)", "Anchor (0,0)", "Terminal (3,3)"]:
            acc = cell_accum[scale][cat]
            n = acc["count"]
            if n == 0:
                continue
            neg_mass_tot = acc["neg_mass"]
            pos_mass_tot = acc["pos_mass"]
            neg_mass_ratio = neg_mass_tot / max(pos_mass_tot, 1e-12)
            cell_mae = acc["abs_err"] / n
            cell_bias = acc["signed_err"] / n
            rel_mismatch = acc["mismatch"] / max(acc["ref_val"], 1e-12) if scale != 1.0 else 0.0

            tot_scale_neg = sum(cell_accum[scale][c]["neg_mass"] for c in ["Boundary (d=0)", "Interior (d=1)"])
            neg_mass_share = (neg_mass_tot / tot_scale_neg * 100.0) if tot_scale_neg > 0 else 0.0

            print(f"{scale:<8.2f}{cat:<22}{n:<12d}{neg_mass_share:>10.2f}%{neg_mass_ratio*100:>11.3f}%{cell_mae:>12.4f}{rel_mismatch*100:>13.2f}%")

            cell_summary_rows.append({
                "scale": scale,
                "category": cat,
                "n_cells": n,
                "neg_variation": neg_mass_tot,
                "pos_variation": pos_mass_tot,
                "nvr_pct": neg_mass_ratio * 100.0,
                "neg_mass_share_pct": neg_mass_share,
                "neg_mass_ratio_pct": neg_mass_ratio * 100.0,
                "cell_mae": cell_mae,
                "cell_bias": cell_bias,
                "rel_mismatch_pct": rel_mismatch * 100.0,
            })

    with open(out_dir / "cell_geometry_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(cell_summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(cell_summary_rows)

    # -------------------------------------------------------------------------
    # Module 1B: Full 16-Phase Local Matrix
    # -------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("MODULE 1B: FULL 16-PHASE LOCAL MATRIX (u in x-direction, v in y-direction)")
    print("Hypothesis Test: Does invalidity/error increase monotonically with distance from Anchor (0,0)?")
    print("=" * 100)

    phase_16_rows = []
    for scale in scales:
        print(f"\n--- Scale: {scale:.2f}x ---")
        print("Negative Variation Ratio (NVR) (%) Matrix [rows=v, cols=u]:")
        header = "v \\ u      " + "".join(f"u={u:<8d}" for u in range(fh))
        print(header)
        print("-" * len(header))
        for v in range(fh):
            row_str = f"v={v:<8d} "
            for u in range(fh):
                acc = phase_accum[scale][(u, v)]
                ratio = (acc["neg_mass"] / max(acc["pos_mass"], 1e-12)) * 100.0
                row_str += f"{ratio:>7.2f}%  "
            print(row_str)

        print("\nCell GT MAE Matrix [rows=v, cols=u]:")
        print(header)
        print("-" * len(header))
        for v in range(fh):
            row_str = f"v={v:<8d} "
            for u in range(fh):
                acc = phase_accum[scale][(u, v)]
                mae = acc["abs_err"] / max(acc["count"], 1)
                row_str += f"{mae:>8.4f}  "
            print(row_str)

        if scale != args.reference_scale:
            print(f"\nRelative Mismatch vs {args.reference_scale:.2f}x (%) Matrix [rows=v, cols=u]:")
            print(header)
            print("-" * len(header))
            for v in range(fh):
                row_str = f"v={v:<8d} "
                for u in range(fh):
                    acc = phase_accum[scale][(u, v)]
                    mism = (acc["mismatch"] / max(acc["ref_val"], 1e-12)) * 100.0
                    row_str += f"{mism:>7.2f}%  "
                print(row_str)

        dists, ratios, maes, misms = [], [], [], []
        for v in range(fh):
            for u in range(fh):
                acc = phase_accum[scale][(u, v)]
                n_cells = acc["count"]
                if n_cells == 0:
                    continue
                ratio = (acc["neg_mass"] / max(acc["pos_mass"], 1e-12)) * 100.0
                mae = acc["abs_err"] / n_cells
                signed_bias = acc["signed_err"] / n_cells
                mism = (acc["mismatch"] / max(acc["ref_val"], 1e-12)) * 100.0 if scale != args.reference_scale else 0.0
                d = u + v
                dists.append(d)
                ratios.append(ratio)
                maes.append(mae)
                misms.append(mism)
                phase_16_rows.append({
                    "scale": scale,
                    "u": u,
                    "v": v,
                    "manhattan_dist_from_origin": d,
                    "euclidean_dist_from_origin": math.sqrt(u*u + v*v),
                    "n_cells": n_cells,
                    "neg_variation": acc["neg_mass"],
                    "pos_variation": acc["pos_mass"],
                    "nvr_pct": ratio,
                    "neg_mass": acc["neg_mass"],
                    "pos_mass": acc["pos_mass"],
                    "neg_mass_ratio_pct": ratio,
                    "cell_mae": mae,
                    "cell_bias": signed_bias,
                    "rel_mismatch_pct": mism,
                })

        if len(dists) > 2 and np.std(ratios) > 1e-12:
            corr_ratio = float(np.corrcoef(dists, ratios)[0, 1])
            corr_mae = float(np.corrcoef(dists, maes)[0, 1])
            print(f"\nCorrelation with Distance from Anchor (u+v):")
            print(f"  Pearson r(dist, NVR)        = {corr_ratio:+.4f}")
            print(f"  Pearson r(dist, Cell MAE)   = {corr_mae:+.4f}")
            if scale != args.reference_scale and np.std(misms) > 1e-12:
                corr_mism = float(np.corrcoef(dists, misms)[0, 1])
                print(f"  Pearson r(dist, RelMismatch)= {corr_mism:+.4f}")

    with open(out_dir / "phase_16_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(phase_16_rows[0].keys()))
        writer.writeheader()
        writer.writerows(phase_16_rows)

    print("\n" + "=" * 100)
    print("MODULE 2: NORMALIZED FH UNITS vs FIXED ORIGINAL PIXELS")
    print("=" * 100)
    print("Part A: Normalized FH Units (0.5x, 1x, 2x, 4x FH spans in scaled space)")
    print(f"{'Scale':<8}{'FH Multiplier':<16}{'Orig Size(px)':<16}{'Rel Mismatch vs 1x':<22}{'Region GT MAE':<16}{'Rel GT MAE':<16}")
    print("-" * 100)

    norm_fh_summary = []
    for scale in scales:
        for mult in fh_mults:
            entries = norm_fh_accum[(scale, mult)]
            if not entries:
                continue
            mean_rel_mismatch = np.mean([e["rel_mismatch"] for e in entries])
            mean_abs_err = np.mean([e["abs_err"] for e in entries])
            mean_rel_err = np.mean([e["rel_err"] for e in entries])
            orig_px = mult * (fh_span_scaled_px / scale)
            print(f"{scale:<8.2f}{str(mult)+'x FH':<16}{orig_px:>14.1f}px{mean_rel_mismatch*100:>20.2f}%{mean_abs_err:>16.4f}{mean_rel_err*100:>15.2f}%")
            norm_fh_summary.append({
                "scale": scale,
                "fh_multiplier": mult,
                "orig_size_px": orig_px,
                "rel_mismatch_pct": mean_rel_mismatch * 100.0,
                "region_gt_mae": mean_abs_err,
                "rel_gt_mae_pct": mean_rel_err * 100.0,
            })

    with open(out_dir / "normalized_fh_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(norm_fh_summary[0].keys()))
        writer.writeheader()
        writer.writerows(norm_fh_summary)

    print("\nPart B: Fixed Original Pixels (32px, 64px, 128px, 256px)")
    print(f"{'Scale':<8}{'Fixed Size(px)':<16}{'FH Equivalent':<16}{'Rel Mismatch vs 1x':<22}{'Region GT MAE':<16}{'Rel GT MAE':<16}")
    print("-" * 100)

    fixed_sz_summary = []
    for scale in scales:
        for sz in fixed_sizes:
            entries = fixed_sz_accum[(scale, sz)]
            if not entries:
                continue
            mean_rel_mismatch = np.mean([e["rel_mismatch"] for e in entries])
            mean_abs_err = np.mean([e["abs_err"] for e in entries])
            mean_rel_err = np.mean([e["rel_err"] for e in entries])
            fh_equiv = sz / (fh_span_scaled_px / scale)
            print(f"{scale:<8.2f}{str(sz)+'px':<16}{fh_equiv:>14.2f}x{mean_rel_mismatch*100:>20.2f}%{mean_abs_err:>16.4f}{mean_rel_err*100:>15.2f}%")
            fixed_sz_summary.append({
                "scale": scale,
                "fixed_size_px": sz,
                "fh_equivalent": fh_equiv,
                "rel_mismatch_pct": mean_rel_mismatch * 100.0,
                "region_gt_mae": mean_abs_err,
                "rel_gt_mae_pct": mean_rel_err * 100.0,
            })

    with open(out_dir / "fixed_size_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fixed_sz_summary[0].keys()))
        writer.writeheader()
        writer.writerows(fixed_sz_summary)

    print("\n" + "=" * 100)
    print("MODULE 3: CONTROLLED BOUNDARY STRADDLING CONTRAST")
    print("=" * 100)
    print(f"{'Scale':<8}{'Group':<32}{'N Reg':<8}{'Rel Mismatch vs 1x':<22}{'Region GT MAE':<16}{'Rel GT MAE':<16}")
    print("-" * 100)

    cond_summary = []
    for scale in scales:
        for grp in ["size32_clean_interior", "size32_boundary_straddling", "size64_grid_aligned", "size64_grid_offset_junction"]:
            entries = cond_accum[(scale, grp)]
            if not entries:
                continue
            mean_rel_mismatch = np.mean([e["rel_mismatch"] for e in entries])
            mean_abs_err = np.mean([e["abs_err"] for e in entries])
            mean_rel_err = np.mean([e["rel_err"] for e in entries])
            print(f"{scale:<8.2f}{grp:<32}{len(entries):<8d}{mean_rel_mismatch*100:>20.2f}%{mean_abs_err:>16.4f}{mean_rel_err*100:>15.2f}%")
            cond_summary.append({
                "scale": scale,
                "group": grp,
                "n_regions": len(entries),
                "rel_mismatch_pct": mean_rel_mismatch * 100.0,
                "region_gt_mae": mean_abs_err,
                "rel_gt_mae_pct": mean_rel_err * 100.0,
            })

    with open(out_dir / "boundary_conditioning_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(cond_summary[0].keys()))
        writer.writeheader()
        writer.writerows(cond_summary)

    manifest = {
        "model_id": m_id,
        "checkpoint": str(args.checkpoint),
        "output_dir": str(out_dir),
        "stride": stride,
        "finite_horizon": fh,
        "fh_strict_local": getattr(model, "fh_strict_local", False),
        "fh_local_norm": getattr(model, "fh_local_norm", None),
        "scales": scales,
        "n_images": n_total,
    }
    with open(out_dir / "boundary_mechanism_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 100)
    print("Summary CSVs and manifest successfully written to:", out_dir)
    print("=" * 100)


if __name__ == "__main__":
    main()
