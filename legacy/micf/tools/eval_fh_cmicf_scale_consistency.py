from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hpc.data.sha import ShanghaiTechDataset
from hpc.losses.micf import discrete_mixed_difference
from hpc.models.micf_lite import MICFLite


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Cross-scale global-count and arbitrary-region consistency audit "
            "for an already-trained FH-CMICF B8 checkpoint. No retraining."
        )
    )

    p.add_argument(
        "--checkpoint",
        type=str,
        default="runs/pilot_micf/b8_k4/best.pt",
    )
    p.add_argument(
        "--config",
        type=str,
        default="configs/pilot_micf/b8.yaml",
        help="Fallback only if checkpoint does not contain its config.",
    )
    p.add_argument("--dataset-root", type=str, default=None)
    p.add_argument("--part", type=str, default=None)
    p.add_argument("--split", type=str, default="test_data")

    p.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[0.5, 0.75, 1.0, 1.25, 1.5],
    )
    p.add_argument(
        "--reference-scale",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--region-sizes",
        type=float,
        nargs="+",
        default=[32.0, 64.0, 128.0, 256.0, 512.0],
    )
    p.add_argument(
        "--aspect-ratios",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0],
    )
    p.add_argument(
        "--regions-per-size",
        type=int,
        default=32,
    )
    p.add_argument("--seed", type=int, default=20260904)

    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--amp",
        action="store_true",
        help="Optional CUDA autocast. Default is FP32 for audit stability.",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Debug only. Omit for all 182 SHA test images.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save audit outputs. Default: <checkpoint_dir>/eval_scale_consistency",
    )
    p.add_argument(
        "--conservation-tol",
        type=float,
        default=1e-4,
        help="Relative/absolute tolerance for full-image measure conservation checks.",
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
# Checkpoint / config / dataset loading
# -----------------------------------------------------------------------------


def safe_torch_load(path: str, map_location: str | torch.device = "cpu") -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_config(checkpoint: dict, config_path: str | None) -> dict:
    cfg = checkpoint.get("config")
    if cfg is not None:
        return cfg

    if config_path is None:
        raise ValueError("Checkpoint has no stored config. Provide --config.")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model_from_config(
    cfg: dict,
    state_dict: dict,
    device: torch.device,
) -> MICFLite:
    m_cfg = cfg.get("model", {})

    # pretrained=False is intentional at evaluation load time:
    # all learned weights come from state_dict, avoiding external downloads.
    model = MICFLite(
        backbone_name=m_cfg.get(
            "backbone",
            "mobilenetv4_conv_small_050.e3000_r224_in1k",
        ),
        pretrained=False,
        neck_width=int(m_cfg.get("neck_width", 32)),
        context_dilations=tuple(m_cfg.get("context_dilations", [1, 2, 3])),
        use_integral_context=bool(m_cfg.get("use_integral_context", False)),
        context_type=str(m_cfg.get("context_type", "directional")),
        head_type=m_cfg.get("head_type", "cumulative"),
        output_stride=int(m_cfg.get("output_stride", 16)),
        eps_d=float(m_cfg.get("eps_d", 1e-8)),
        extent_aware=bool(m_cfg.get("extent_aware", False)),
        finite_horizon=m_cfg.get("finite_horizon", None),
        fh_strict_local=bool(m_cfg.get("fh_strict_local", False)),
        fh_local_norm=str(m_cfg.get("fh_local_norm", "group")),
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/model mismatch.\n"
            f"Missing: {missing}\n"
            f"Unexpected: {unexpected}"
        )

    return model.to(device).eval()


def build_dataset(
    cfg: dict,
    root_override: str | None,
    part_override: str | None,
    split: str,
) -> ShanghaiTechDataset:
    ds_cfg = cfg.get("dataset", {})

    root = root_override if root_override is not None else ds_cfg.get(
        "root", "./data/ShanghaiTech"
    )
    part = part_override if part_override is not None else ds_cfg.get(
        "part", "part_A"
    )

    return ShanghaiTechDataset(
        root=root,
        part=part,
        split=split,
        crop_size=int(ds_cfg.get("crop_size", 256)),
        is_train=False,
        image_mean=ds_cfg.get("image_mean", [0.5, 0.5, 0.5]),
        image_std=ds_cfg.get("image_std", [0.5, 0.5, 0.5]),
        coordinate_base=int(ds_cfg.get("coordinate_base", 0)),
        annotation_bounds_policy="allow",
    )


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------


def as_numpy_points(points: Any) -> np.ndarray:
    if isinstance(points, torch.Tensor):
        arr = points.detach().cpu().numpy()
    else:
        arr = np.asarray(points)

    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return arr.reshape(-1, 2)


def in_bounds_points(points: np.ndarray, height: int, width: int) -> np.ndarray:
    if len(points) == 0:
        return points

    mask = (
        (points[:, 0] >= 0.0)
        & (points[:, 0] < float(width))
        & (points[:, 1] >= 0.0)
        & (points[:, 1] < float(height))
    )
    return points[mask]


def count_points_in_rect(
    points: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> float:
    if len(points) == 0:
        return 0.0

    mask = (
        (points[:, 0] >= x0)
        & (points[:, 0] < x1)
        & (points[:, 1] >= y0)
        & (points[:, 1] < y1)
    )
    return float(mask.sum())


def resize_image_tensor(
    image: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, int, int, float, float]:
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(f"Expected [1,3,H,W], got {tuple(image.shape)}")
    if scale <= 0:
        raise ValueError(f"Scale must be > 0, got {scale}")

    _, _, h, w = image.shape
    new_h = max(1, int(round(float(h) * float(scale))))
    new_w = max(1, int(round(float(w) * float(scale))))

    sy = float(new_h) / float(h)
    sx = float(new_w) / float(w)

    if new_h == h and new_w == w:
        return image, new_h, new_w, sx, sy

    try:
        resized = F.interpolate(
            image,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    except TypeError:
        resized = F.interpolate(
            image,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )

    return resized, new_h, new_w, sx, sy


# -----------------------------------------------------------------------------
# Conservative region query in resized-image coordinates
# -----------------------------------------------------------------------------


def clipped_cell_edges(length_px: int, n_cells: int, stride: int) -> np.ndarray:
    edges = np.arange(n_cells + 1, dtype=np.float64) * float(stride)
    edges = np.minimum(edges, float(length_px))
    edges[-1] = float(length_px)

    widths = np.diff(edges)
    if np.any(widths <= 0.0):
        raise RuntimeError(
            f"Invalid cell support: length={length_px}, n_cells={n_cells}, stride={stride}, "
            f"edges={edges.tolist()}"
        )
    return edges


def conservative_region_mass(
    y_mass: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    image_h: int,
    image_w: int,
    stride: int,
) -> float:
    if y_mass.ndim != 2:
        raise ValueError(f"Expected 2D y_mass, got {y_mass.shape}")

    out_h, out_w = y_mass.shape

    x0 = float(np.clip(x0, 0.0, float(image_w)))
    x1 = float(np.clip(x1, 0.0, float(image_w)))
    y0 = float(np.clip(y0, 0.0, float(image_h)))
    y1 = float(np.clip(y1, 0.0, float(image_h)))

    if x1 <= x0 or y1 <= y0:
        return 0.0

    x_edges = clipped_cell_edges(image_w, out_w, stride)
    y_edges = clipped_cell_edges(image_h, out_h, stride)

    ix0 = int(np.searchsorted(x_edges[1:], x0, side="right"))
    iy0 = int(np.searchsorted(y_edges[1:], y0, side="right"))

    ix1 = int(np.searchsorted(x_edges[:-1], x1, side="left"))
    iy1 = int(np.searchsorted(y_edges[:-1], y1, side="left"))

    ix0 = max(0, min(ix0, out_w))
    ix1 = max(ix0, min(ix1, out_w))
    iy0 = max(0, min(iy0, out_h))
    iy1 = max(iy0, min(iy1, out_h))

    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0

    cell_x0 = x_edges[ix0:ix1]
    cell_x1 = x_edges[ix0 + 1:ix1 + 1]
    cell_y0 = y_edges[iy0:iy1]
    cell_y1 = y_edges[iy0 + 1:iy1 + 1]

    overlap_x = np.maximum(
        0.0,
        np.minimum(cell_x1, x1) - np.maximum(cell_x0, x0),
    )
    overlap_y = np.maximum(
        0.0,
        np.minimum(cell_y1, y1) - np.maximum(cell_y0, y0),
    )

    cell_w = cell_x1 - cell_x0
    cell_h = cell_y1 - cell_y0

    wx = overlap_x / cell_w
    wy = overlap_y / cell_h
    weights = wy[:, None] * wx[None, :]

    block = y_mass[iy0:iy1, ix0:ix1]
    return float(np.sum(block * weights, dtype=np.float64))


# -----------------------------------------------------------------------------
# Arbitrary rectangle generation in ORIGINAL coordinates
# -----------------------------------------------------------------------------


def generate_rectangles(
    image_h: int,
    image_w: int,
    region_sizes: list[float],
    aspect_ratios: list[float],
    regions_per_size: int,
    seed: int,
    image_index: int,
) -> list[dict[str, float | int]]:
    rng = random.Random(int(seed) + 1000003 * int(image_index))
    rectangles: list[dict[str, float | int]] = []

    for nominal_size in region_sizes:
        feasible: list[tuple[float, float, float]] = []

        for aspect in aspect_ratios:
            if aspect <= 0:
                raise ValueError(f"Aspect ratio must be > 0, got {aspect}")

            width = float(nominal_size) * math.sqrt(float(aspect))
            height = float(nominal_size) / math.sqrt(float(aspect))

            if width <= float(image_w) and height <= float(image_h):
                feasible.append((float(aspect), width, height))

        if not feasible:
            continue

        for region_local_id in range(int(regions_per_size)):
            aspect, width, height = rng.choice(feasible)

            max_x0 = max(0.0, float(image_w) - width)
            max_y0 = max(0.0, float(image_h) - height)

            x0 = 0.0 if max_x0 == 0.0 else rng.random() * max_x0
            y0 = 0.0 if max_y0 == 0.0 else rng.random() * max_y0
            x1 = x0 + width
            y1 = y0 + height

            rectangles.append(
                {
                    "region_local_id": int(region_local_id),
                    "nominal_size": float(nominal_size),
                    "aspect_ratio": float(aspect),
                    "x0": float(x0),
                    "y0": float(y0),
                    "x1": float(x1),
                    "y1": float(y1),
                    "width": float(width),
                    "height": float(height),
                    "area": float(width * height),
                    "area_fraction": float((width * height) / (image_h * image_w)),
                }
            )

    return rectangles


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------


class RegionAccumulator:
    def __init__(self) -> None:
        self.n = 0
        self.sum_abs_gt = 0.0
        self.sum_sq_gt = 0.0
        self.sum_signed_gt = 0.0
        self.sum_abs_mismatch = 0.0
        self.sum_sq_mismatch = 0.0
        self.sum_rel_ref = 0.0
        self.sum_rel_gt = 0.0
        self.nonempty_gt = 0

    def update(
        self,
        pred: float,
        gt: float,
        ref_pred: float,
    ) -> None:
        signed_gt = float(pred - gt)
        abs_gt = abs(signed_gt)
        mismatch = abs(float(pred - ref_pred))

        self.n += 1
        self.sum_abs_gt += abs_gt
        self.sum_sq_gt += signed_gt * signed_gt
        self.sum_signed_gt += signed_gt
        self.sum_abs_mismatch += mismatch
        self.sum_sq_mismatch += mismatch * mismatch
        self.sum_rel_ref += mismatch / max(abs(float(ref_pred)), 1.0)
        self.sum_rel_gt += mismatch / max(float(gt), 1.0)
        if gt > 0.0:
            self.nonempty_gt += 1

    def summary(self) -> dict[str, float | int]:
        if self.n == 0:
            return {
                "n_regions": 0,
                "region_mae_gt": float("nan"),
                "region_rmse_gt": float("nan"),
                "region_bias_gt": float("nan"),
                "region_mismatch_mae_vs_1x": float("nan"),
                "region_mismatch_rmse_vs_1x": float("nan"),
                "region_relative_mismatch_vs_1x": float("nan"),
                "region_relative_mismatch_vs_gt": float("nan"),
                "nonempty_gt_fraction": float("nan"),
            }

        n = float(self.n)
        return {
            "n_regions": int(self.n),
            "region_mae_gt": float(self.sum_abs_gt / n),
            "region_rmse_gt": float(math.sqrt(self.sum_sq_gt / n)),
            "region_bias_gt": float(self.sum_signed_gt / n),
            "region_mismatch_mae_vs_1x": float(self.sum_abs_mismatch / n),
            "region_mismatch_rmse_vs_1x": float(math.sqrt(self.sum_sq_mismatch / n)),
            "region_relative_mismatch_vs_1x": float(self.sum_rel_ref / n),
            "region_relative_mismatch_vs_gt": float(self.sum_rel_gt / n),
            "nonempty_gt_fraction": float(self.nonempty_gt / n),
        }


def summarize_global_rows(rows: list[dict[str, Any]], scales: list[float]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    for scale in scales:
        subset = [r for r in rows if math.isclose(float(r["scale_requested"]), float(scale))]
        if not subset:
            continue

        err_raw = np.asarray(
            [float(r["pred_count"]) - float(r["gt_count_raw"]) for r in subset],
            dtype=np.float64,
        )
        err_in = np.asarray(
            [float(r["pred_count"]) - float(r["gt_count_inbounds"]) for r in subset],
            dtype=np.float64,
        )
        mismatch = np.asarray(
            [float(r["count_mismatch_vs_1x"]) for r in subset],
            dtype=np.float64,
        )
        rel_ref = np.asarray(
            [float(r["relative_count_mismatch_vs_1x"]) for r in subset],
            dtype=np.float64,
        )
        rel_gt = np.asarray(
            [float(r["relative_count_mismatch_vs_gt"]) for r in subset],
            dtype=np.float64,
        )

        output.append(
            {
                "scale_requested": float(scale),
                "n_images": int(len(subset)),
                "mae_gt_raw": float(np.mean(np.abs(err_raw))),
                "rmse_gt_raw": float(np.sqrt(np.mean(err_raw ** 2))),
                "bias_gt_raw": float(np.mean(err_raw)),
                "mae_gt_inbounds": float(np.mean(np.abs(err_in))),
                "rmse_gt_inbounds": float(np.sqrt(np.mean(err_in ** 2))),
                "bias_gt_inbounds": float(np.mean(err_in)),
                "count_mismatch_mae_vs_1x": float(np.mean(mismatch)),
                "count_mismatch_rmse_vs_1x": float(np.sqrt(np.mean(mismatch ** 2))),
                "relative_count_mismatch_vs_1x": float(np.mean(rel_ref)),
                "relative_count_mismatch_vs_gt": float(np.mean(rel_gt)),
                "mean_negative_mass": float(
                    np.mean([float(r["negative_mass"]) for r in subset])
                ),
                "mean_negative_mass_ratio": float(
                    np.mean([float(r["negative_mass_ratio"]) for r in subset])
                ),
            }
        )

    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    scales = [float(x) for x in args.scales]
    if not any(math.isclose(x, float(args.reference_scale)) for x in scales):
        raise ValueError(
            f"reference scale {args.reference_scale} must be present in --scales {scales}"
        )

    if len(set(scales)) != len(scales):
        raise ValueError(f"Duplicate scales are not allowed: {scales}")

    device = torch.device(args.device)
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(args.checkpoint).resolve().parent / "eval_scale_consistency"
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = safe_torch_load(args.checkpoint, map_location="cpu")
    cfg = load_config(checkpoint, args.config)
    m_id = cfg.get("experiment", {}).get("model_id", "MICF")

    if "state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint {args.checkpoint} has no 'state_dict'")

    model = build_model_from_config(
        cfg=cfg,
        state_dict=checkpoint["state_dict"],
        device=device,
    )

    if not args.allow_non_s16_k4:
        if model.head_type != "cumulative":
            raise RuntimeError(f"Expected cumulative head, got {model.head_type}")
        if int(model.output_stride) != 16:
            raise RuntimeError(f"Expected output_stride=16, got {model.output_stride}")
        if model.finite_horizon is None or int(model.finite_horizon) != 4:
            raise RuntimeError(
                f"Expected finite_horizon=4, got {model.finite_horizon}"
            )

    dataset = build_dataset(
        cfg=cfg,
        root_override=args.dataset_root,
        part_override=args.part,
        split=args.split,
    )

    n_total = len(dataset)
    if args.max_samples is not None:
        n_total = min(n_total, int(args.max_samples))

    if args.max_samples is None and len(dataset) != 182:
        print(
            f"WARNING: expected 182 ShanghaiTech Part A test images, found {len(dataset)}."
        )

    print("=" * 80)
    print(f"{m_id} (stride={model.output_stride}, K={model.finite_horizon}) SCALE / REGION CONSISTENCY AUDIT")
    print(f"Checkpoint       : {args.checkpoint}")
    print(f"Output Dir       : {out_dir}")
    print(f"Images           : {n_total}")
    print(f"Scales           : {scales}")
    print(f"Reference scale  : {args.reference_scale}")
    print(f"Region sizes     : {args.region_sizes}")
    print(f"Regions / size   : {args.regions_per_size}")
    print(f"Device           : {device}")
    print(f"AMP              : {bool(args.amp)}")
    print("=" * 80)

    per_image_rows: list[dict[str, Any]] = []
    region_acc: dict[tuple[float, float], RegionAccumulator] = defaultdict(RegionAccumulator)

    per_region_path = out_dir / "per_region.csv"
    per_region_file = per_region_path.open("w", newline="", encoding="utf-8")
    per_region_writer: csv.DictWriter | None = None

    stride = int(model.output_stride)

    if args.amp and device.type == "cuda":
        amp_context = lambda: torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        amp_context = nullcontext

    try:
        for image_index in range(n_total):
            sample = dataset[image_index]

            image = sample["image"].unsqueeze(0).to(device)
            _, _, orig_h, orig_w = image.shape

            points_raw = as_numpy_points(sample["gt_points"])
            points_in = in_bounds_points(points_raw, orig_h, orig_w)

            gt_count_raw = float(sample["gt_count"].item())
            gt_count_inbounds = float(len(points_in))
            image_name = os.path.basename(str(sample["img_path"]))

            scale_outputs: dict[float, dict[str, Any]] = {}

            with torch.inference_mode():
                for scale in scales:
                    image_s, scaled_h, scaled_w, sx, sy = resize_image_tensor(
                        image, scale
                    )

                    with amp_context():
                        pred_count_t, pred_c = model.predict(
                            image_s,
                            pad_multiple=64,
                        )

                    pred_c = pred_c.float()
                    pred_y = discrete_mixed_difference(pred_c).float()

                    pred_count = float(pred_count_t.detach().float().item())
                    pred_measure_sum = float(pred_y.sum().detach().cpu().item())

                    y_np = (
                        pred_y.squeeze(0)
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64, copy=False)
                    )

                    full_query = conservative_region_mass(
                        y_mass=y_np,
                        x0=0.0,
                        y0=0.0,
                        x1=float(scaled_w),
                        y1=float(scaled_h),
                        image_h=scaled_h,
                        image_w=scaled_w,
                        stride=stride,
                    )

                    tol = float(args.conservation_tol) * max(
                        1.0,
                        abs(pred_count),
                        abs(pred_measure_sum),
                    )

                    if abs(full_query - pred_measure_sum) > tol:
                        raise RuntimeError(
                            "Conservative query failed to conserve mass: "
                            f"image={image_name}, scale={scale}, "
                            f"full_query={full_query}, sumY={pred_measure_sum}, tol={tol}"
                        )

                    if abs(pred_count - pred_measure_sum) > tol:
                        raise RuntimeError(
                            "C corner count != sum(Delta C): "
                            f"image={image_name}, scale={scale}, "
                            f"pred_count={pred_count}, sumY={pred_measure_sum}, tol={tol}"
                        )

                    negative_mass = float(np.maximum(-y_np, 0.0).sum())
                    positive_mass = float(np.maximum(y_np, 0.0).sum())
                    negative_mass_ratio = negative_mass / max(positive_mass, 1e-12)

                    scale_outputs[float(scale)] = {
                        "pred_count": pred_count,
                        "pred_measure_sum": pred_measure_sum,
                        "y_mass": y_np,
                        "scaled_h": int(scaled_h),
                        "scaled_w": int(scaled_w),
                        "sx": float(sx),
                        "sy": float(sy),
                        "negative_mass": negative_mass,
                        "negative_mass_ratio": negative_mass_ratio,
                    }

            reference_key = next(
                k
                for k in scale_outputs.keys()
                if math.isclose(k, float(args.reference_scale))
            )
            ref = scale_outputs[reference_key]
            ref_count = float(ref["pred_count"])

            for scale in scales:
                result = scale_outputs[float(scale)]
                pred_count = float(result["pred_count"])
                mismatch = abs(pred_count - ref_count)

                row = {
                    "image_index": int(image_index),
                    "image_name": image_name,
                    "scale_requested": float(scale),
                    "realized_scale_x": float(result["sx"]),
                    "realized_scale_y": float(result["sy"]),
                    "original_h": int(orig_h),
                    "original_w": int(orig_w),
                    "scaled_h": int(result["scaled_h"]),
                    "scaled_w": int(result["scaled_w"]),
                    "gt_count_raw": gt_count_raw,
                    "gt_count_inbounds": gt_count_inbounds,
                    "pred_count": pred_count,
                    "pred_measure_sum": float(result["pred_measure_sum"]),
                    "abs_error_gt_raw": abs(pred_count - gt_count_raw),
                    "abs_error_gt_inbounds": abs(pred_count - gt_count_inbounds),
                    "signed_error_gt_raw": pred_count - gt_count_raw,
                    "signed_error_gt_inbounds": pred_count - gt_count_inbounds,
                    "pred_count_1x": ref_count,
                    "count_mismatch_vs_1x": mismatch,
                    "signed_count_delta_vs_1x": pred_count - ref_count,
                    "relative_count_mismatch_vs_1x": mismatch / max(abs(ref_count), 1.0),
                    "relative_count_mismatch_vs_gt": mismatch / max(gt_count_inbounds, 1.0),
                    "negative_mass": float(result["negative_mass"]),
                    "negative_mass_ratio": float(result["negative_mass_ratio"]),
                }
                per_image_rows.append(row)

            rectangles = generate_rectangles(
                image_h=orig_h,
                image_w=orig_w,
                region_sizes=[float(x) for x in args.region_sizes],
                aspect_ratios=[float(x) for x in args.aspect_ratios],
                regions_per_size=int(args.regions_per_size),
                seed=int(args.seed),
                image_index=int(image_index),
            )

            for region_index, rect in enumerate(rectangles):
                x0 = float(rect["x0"])
                y0 = float(rect["y0"])
                x1 = float(rect["x1"])
                y1 = float(rect["y1"])

                gt_region = count_points_in_rect(
                    points_in,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                )

                ref_x0 = x0 * float(ref["sx"])
                ref_x1 = x1 * float(ref["sx"])
                ref_y0 = y0 * float(ref["sy"])
                ref_y1 = y1 * float(ref["sy"])

                ref_region_pred = conservative_region_mass(
                    y_mass=ref["y_mass"],
                    x0=ref_x0,
                    y0=ref_y0,
                    x1=ref_x1,
                    y1=ref_y1,
                    image_h=int(ref["scaled_h"]),
                    image_w=int(ref["scaled_w"]),
                    stride=stride,
                )

                for scale in scales:
                    result = scale_outputs[float(scale)]

                    rx0 = x0 * float(result["sx"])
                    rx1 = x1 * float(result["sx"])
                    ry0 = y0 * float(result["sy"])
                    ry1 = y1 * float(result["sy"])

                    pred_region = conservative_region_mass(
                        y_mass=result["y_mass"],
                        x0=rx0,
                        y0=ry0,
                        x1=rx1,
                        y1=ry1,
                        image_h=int(result["scaled_h"]),
                        image_w=int(result["scaled_w"]),
                        stride=stride,
                    )

                    signed_gt = pred_region - gt_region
                    signed_delta = pred_region - ref_region_pred
                    abs_mismatch = abs(signed_delta)

                    region_row = {
                        "image_index": int(image_index),
                        "image_name": image_name,
                        "region_id": f"{image_index}:{region_index}",
                        "region_local_id": int(rect["region_local_id"]),
                        "nominal_size": float(rect["nominal_size"]),
                        "aspect_ratio": float(rect["aspect_ratio"]),
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1,
                        "width": float(rect["width"]),
                        "height": float(rect["height"]),
                        "area": float(rect["area"]),
                        "area_fraction": float(rect["area_fraction"]),
                        "scale_requested": float(scale),
                        "realized_scale_x": float(result["sx"]),
                        "realized_scale_y": float(result["sy"]),
                        "gt_region_count": float(gt_region),
                        "pred_region_count": float(pred_region),
                        "pred_region_count_1x": float(ref_region_pred),
                        "abs_error_gt": abs(signed_gt),
                        "signed_error_gt": signed_gt,
                        "abs_mismatch_vs_1x": abs_mismatch,
                        "signed_delta_vs_1x": signed_delta,
                        "relative_mismatch_vs_1x": (
                            abs_mismatch / max(abs(ref_region_pred), 1.0)
                        ),
                        "relative_mismatch_vs_gt": (
                            abs_mismatch / max(gt_region, 1.0)
                        ),
                    }

                    if per_region_writer is None:
                        per_region_writer = csv.DictWriter(
                            per_region_file,
                            fieldnames=list(region_row.keys()),
                        )
                        per_region_writer.writeheader()

                    per_region_writer.writerow(region_row)

                    region_acc[
                        (float(rect["nominal_size"]), float(scale))
                    ].update(
                        pred=float(pred_region),
                        gt=float(gt_region),
                        ref_pred=float(ref_region_pred),
                    )

            if (image_index + 1) % 10 == 0 or image_index + 1 == n_total:
                print(f"[{image_index + 1:3d}/{n_total}] {image_name}")

    finally:
        per_region_file.close()

    scale_summary = summarize_global_rows(per_image_rows, scales)

    region_summary: list[dict[str, Any]] = []
    for nominal_size in [float(x) for x in args.region_sizes]:
        for scale in scales:
            acc = region_acc.get((nominal_size, float(scale)))
            if acc is None:
                continue

            row = {
                "nominal_size": nominal_size,
                "scale_requested": float(scale),
            }
            row.update(acc.summary())
            region_summary.append(row)

    write_csv(out_dir / "per_image_scale.csv", per_image_rows)
    write_csv(out_dir / "scale_summary.csv", scale_summary)
    write_csv(out_dir / "region_summary.csv", region_summary)

    def pick_scale_summary(target: float) -> dict[str, Any] | None:
        for r in scale_summary:
            if math.isclose(float(r["scale_requested"]), float(target)):
                return r
        return None

    moderate_scales = [s for s in (0.75, 1.25) if s in scales]
    stress_scales = [s for s in (0.5, 1.5) if s in scales]

    def mean_metric(scale_list: list[float], key: str) -> float | None:
        vals = []
        for s in scale_list:
            r = pick_scale_summary(s)
            if r is not None:
                vals.append(float(r[key]))
        return float(np.mean(vals)) if vals else None

    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "model": {
            "model_id": m_id,
            "head_type": model.head_type,
            "output_stride": int(model.output_stride),
            "finite_horizon": (
                None if model.finite_horizon is None else int(model.finite_horizon)
            ),
            "fh_strict_local": getattr(model, "fh_strict_local", False),
            "fh_local_norm": getattr(model, "fh_local_norm", None),
            "fh_physical_span_at_1x_px": (
                None
                if model.finite_horizon is None
                else int(model.output_stride) * int(model.finite_horizon)
            ),
        },
        "protocol": {
            "dataset_split": args.split,
            "n_images": int(n_total),
            "scales": scales,
            "reference_scale": float(args.reference_scale),
            "region_sizes": [float(x) for x in args.region_sizes],
            "aspect_ratios": [float(x) for x in args.aspect_ratios],
            "regions_per_size": int(args.regions_per_size),
            "seed": int(args.seed),
            "inference": "direct full-image",
            "retraining": False,
            "region_coordinate_space": "original image pixel coordinates",
            "region_query": "conservative fractional-overlap integration of Y=Delta_xy(C)",
            "gt_region_convention": "in-bounds points, half-open rectangles [x0,x1)x[y0,y1)",
        },
        "derived": {
            "moderate_scale_count_mismatch_mae_vs_1x": mean_metric(
                moderate_scales,
                "count_mismatch_mae_vs_1x",
            ),
            "stress_scale_count_mismatch_mae_vs_1x": mean_metric(
                stress_scales,
                "count_mismatch_mae_vs_1x",
            ),
            "moderate_scale_relative_count_mismatch_vs_1x": mean_metric(
                moderate_scales,
                "relative_count_mismatch_vs_1x",
            ),
            "stress_scale_relative_count_mismatch_vs_1x": mean_metric(
                stress_scales,
                "relative_count_mismatch_vs_1x",
            ),
        },
        "scale_summary": scale_summary,
        "region_summary": region_summary,
    }

    with (out_dir / "scale_consistency_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2, allow_nan=True)

    print("\nSaved:")
    print(f"  {out_dir / 'scale_consistency_summary.json'}")
    print(f"  {out_dir / 'per_image_scale.csv'}")
    print(f"  {out_dir / 'scale_summary.csv'}")
    print(f"  {out_dir / 'per_region.csv'}")
    print(f"  {out_dir / 'region_summary.csv'}")

    print("\nGlobal scale summary:")
    for row in scale_summary:
        print(
            f"  scale={row['scale_requested']:.2f} | "
            f"MAE(rawGT)={row['mae_gt_raw']:.2f} | "
            f"Mismatch-vs-1x={row['count_mismatch_mae_vs_1x']:.2f} | "
            f"RelMismatch={100.0 * row['relative_count_mismatch_vs_1x']:.2f}%"
        )


if __name__ == "__main__":
    main()
