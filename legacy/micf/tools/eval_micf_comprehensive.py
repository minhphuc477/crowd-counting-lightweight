"""Comprehensive MICF checkpoint evaluator.

Evaluates one already-trained checkpoint without updating weights.

Metric families
---------------
1. Standard image-level:
   MAE, RMSE, NAE, SRE, signed bias, Median/P90/P95/Max AE.

2. Local/window:
   PMAE/PRMSE (both micro and macro), full-256 vs partial edge window breakdown,
   non-zero-window NAE, empty-window error, non-empty-window MAE, local cancellation.

3. Spatial:
   Canonical pixel-space GAME(0..3) and diagnostic stride-level GAME@stride16(0..3).

4. MICF validity:
   Macro & micro violation rate, macro & micro violation magnitude,
   macro & micro negative-mass ratio, total negative mass, count<->measure conservation error.

5. MICF representation:
   normalized cumulative-field MAE and normalized local-measure L1.

6. Three inference regimes:
   Window, Full-Tiled, Full-Direct.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
import yaml

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hpc.data.sha import ShanghaiTechDataset
from hpc.losses.micf import (
    cell_counts_to_cumulative_field,
    discrete_mixed_difference,
    points_to_count_map,
)
from hpc.metrics.counting import (
    benchmark_count_summary,
    count_metric_summary,
    diagnostic_count_summary,
)
from hpc.metrics.decomposition import (
    direct_tiled_discrepancy,
)
from hpc.metrics.game import (
    aggregate_game,
    game_errors_one_image,
)
from hpc.metrics.micf_validity import (
    ValidityRow,
    aggregate_validity,
    recovered_measure_validity,
)
from hpc.models.micf_lite import (
    MICFLite,
    compose_tiled_cumulative_field,
)


def in_bounds_points(points: np.ndarray, h: int, w: int) -> np.ndarray:
    """Filters 2D (x, y) points strictly within continuous domain [0, w) x [0, h)."""
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    x = points[:, 0]
    y = points[:, 1]
    mask = (x >= 0.0) & (x < float(w)) & (y >= 0.0) & (y < float(h))
    return points[mask]


def sanitize_for_json(obj: Any) -> Any:
    """Recursively replaces NaN and Inf with None for strict JSON standard conformance."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, tuple):
        return [sanitize_for_json(v) for v in obj]
    return obj


# ---------------------------------------------------------------------
# CLI / loading
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comprehensive MICF checkpoint evaluation."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint best.pt. Inferred from --config if omitted.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional YAML fallback. Normally the checkpoint already "
            "contains its config."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--part",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test_data",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--halo",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--controlled-halo",
        type=int,
        default=0,
        help="Halo size for controlled tiled evaluation without extent distortion (default: 0).",
    )
    parser.add_argument(
        "--direct-pad-multiple",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--game-levels",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
    )
    parser.add_argument(
        "--device",
        type=str,
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--legacy-debug",
        action="store_true",
        default=False,
        help="Compute and write legacy diagnostics, window metrics, and legacy GAME CSVs.",
    )
    parser.add_argument(
        "--verify-repo-tiled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Verify agreement with model.predict_tiled().",
    )
    parser.add_argument(
        "--strict-tiled-tolerance",
        type=float,
        default=1e-3,
    )
    return parser.parse_args()


def safe_torch_load(
    path: str,
    map_location: str | torch.device = "cpu",
) -> dict:
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=map_location,
        )


def load_config(
    checkpoint: dict,
    config_path: str | None,
) -> dict:
    cfg = checkpoint.get("config")
    if cfg is not None:
        return cfg

    if config_path is None:
        raise ValueError(
            "Checkpoint has no stored config. "
            "Provide --config."
        )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)


def build_model_from_config(
    cfg: dict,
    state_dict: dict,
    device: torch.device,
) -> MICFLite:
    m_cfg = cfg.get("model", {})

    model = MICFLite(
        backbone_name=m_cfg.get(
            "backbone",
            "mobilenetv4_conv_small_050.e3000_r224_in1k",
        ),
        pretrained=False,
        neck_width=int(
            m_cfg.get("neck_width", 32)
        ),
        context_dilations=tuple(
            m_cfg.get(
                "context_dilations",
                [1, 2, 3],
            )
        ),
        use_integral_context=bool(
            m_cfg.get(
                "use_integral_context",
                False,
            )
        ),
        context_type=str(
            m_cfg.get(
                "context_type",
                "directional",
            )
        ),
        head_type=m_cfg.get(
            "head_type",
            "cumulative",
        ),
        output_stride=int(
            m_cfg.get("output_stride", 16)
        ),
        eps_d=float(
            m_cfg.get("eps_d", 1e-8)
        ),
        extent_aware=bool(
            m_cfg.get(
                "extent_aware",
                False,
            )
        ),
        finite_horizon=m_cfg.get(
            "finite_horizon",
            None,
        ),
        fh_strict_local=bool(
            m_cfg.get(
                "fh_strict_local",
                False,
            )
        ),
        fh_local_norm=str(
            m_cfg.get(
                "fh_local_norm",
                "group",
            )
        ),
    )

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )
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

    root = (
        root_override
        if root_override is not None
        else ds_cfg.get(
            "root",
            "./data/ShanghaiTech",
        )
    )
    part = (
        part_override
        if part_override is not None
        else ds_cfg.get(
            "part",
            "part_A",
        )
    )

    return ShanghaiTechDataset(
        root=root,
        part=part,
        split=split,
        crop_size=int(
            ds_cfg.get("crop_size", 256)
        ),
        is_train=False,
        image_mean=ds_cfg.get(
            "image_mean",
            [0.5, 0.5, 0.5],
        ),
        image_std=ds_cfg.get(
            "image_std",
            [0.5, 0.5, 0.5],
        ),
        coordinate_base=int(
            ds_cfg.get(
                "coordinate_base",
                0,
            )
        ),
        annotation_bounds_policy="allow",
    )


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def as_numpy_points(
    points: Any,
) -> np.ndarray:
    if isinstance(points, torch.Tensor):
        arr = (
            points.detach()
            .cpu()
            .numpy()
        )
    else:
        arr = np.asarray(points)

    arr = np.asarray(
        arr,
        dtype=np.float32,
    )

    if arr.size == 0:
        return np.empty(
            (0, 2),
            dtype=np.float32,
        )

    return arr.reshape(-1, 2)


def in_bounds_points(
    points: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    if len(points) == 0:
        return points

    mask = (
        (points[:, 0] >= 0.0)
        & (points[:, 0] < float(width))
        & (points[:, 1] >= 0.0)
        & (points[:, 1] < float(height))
    )
    return points[mask]


def count_points_in_window(
    points: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> int:
    if len(points) == 0:
        return 0

    mask = (
        (points[:, 0] >= float(x0))
        & (points[:, 0] < float(x1))
        & (points[:, 1] >= float(y0))
        & (points[:, 1] < float(y1))
    )
    return int(mask.sum())


def compute_cancellation_ratio(
    signed_errors: Iterable[float],
    eps: float = 1e-12,
) -> float:
    errs = [float(e) for e in signed_errors]
    if not errs:
        return 0.0
    abs_sum = sum(abs(e) for e in errs)
    net_abs = abs(sum(errs))
    if abs_sum <= eps:
        return 0.0
    val = 1.0 - (net_abs / abs_sum)
    return float(min(1.0, max(0.0, val)))


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        path.write_text(
            "",
            encoding="utf-8",
        )
        return

    keys: list[str] = []
    seen: set[str] = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=keys,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(
    values: Iterable[float],
) -> float:
    arr = np.asarray(
        list(values),
        dtype=np.float64,
    )
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def finite_percentile(
    values: Iterable[float],
    q: float,
) -> float:
    arr = np.asarray(
        list(values),
        dtype=np.float64,
    )
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


# ---------------------------------------------------------------------
# Local window metrics
# ---------------------------------------------------------------------

def window_metric_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        return {}

    pred = np.asarray(
        [
            float(r["pred_count"])
            for r in rows
        ],
        dtype=np.float64,
    )
    gt = np.asarray(
        [
            float(r["gt_count"])
            for r in rows
        ],
        dtype=np.float64,
    )

    # Micro-level stats (all windows across dataset pooled equally)
    stats = count_metric_summary(
        pred,
        gt,
        eps=1.0,
    )

    # Macro-level stats (average over per-image window MAE)
    by_image: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        by_image.setdefault(int(r["image_index"]), []).append(r)

    per_image_wmae = []
    per_image_wrmse = []
    for img_idx, img_rows in by_image.items():
        img_pred = np.asarray([float(r["pred_count"]) for r in img_rows], dtype=np.float64)
        img_gt = np.asarray([float(r["gt_count"]) for r in img_rows], dtype=np.float64)
        err = img_pred - img_gt
        per_image_wmae.append(float(np.mean(np.abs(err))))
        per_image_wrmse.append(float(np.sqrt(np.mean(err * err))))

    wmae_macro = float(np.mean(per_image_wmae)) if per_image_wmae else float("nan")
    wrmse_macro = float(np.mean(per_image_wrmse)) if per_image_wrmse else float("nan")

    # Partition: Full 256x256 core windows vs Partial edge windows
    full_rows = [r for r in rows if bool(r.get("is_full_window", True))]
    edge_rows = [r for r in rows if not bool(r.get("is_full_window", True))]

    full_wmae = float("nan")
    full_wrmse = float("nan")
    if full_rows:
        f_pred = np.asarray([float(r["pred_count"]) for r in full_rows], dtype=np.float64)
        f_gt = np.asarray([float(r["gt_count"]) for r in full_rows], dtype=np.float64)
        f_err = f_pred - f_gt
        full_wmae = float(np.mean(np.abs(f_err)))
        full_wrmse = float(np.sqrt(np.mean(f_err * f_err)))

    edge_wmae = float("nan")
    edge_wrmse = float("nan")
    if edge_rows:
        e_pred = np.asarray([float(r["pred_count"]) for r in edge_rows], dtype=np.float64)
        e_gt = np.asarray([float(r["gt_count"]) for r in edge_rows], dtype=np.float64)
        e_err = e_pred - e_gt
        edge_wmae = float(np.mean(np.abs(e_err)))
        edge_wrmse = float(np.sqrt(np.mean(e_err * e_err)))

    positive = gt > 0
    empty = gt == 0

    if positive.any():
        nae_nonzero = float(
            np.mean(
                np.abs(
                    pred[positive]
                    - gt[positive]
                )
                / gt[positive]
            )
        )
        nonempty_mae = float(
            np.mean(
                np.abs(
                    pred[positive]
                    - gt[positive]
                )
            )
        )
    else:
        nae_nonzero = float("nan")
        nonempty_mae = float("nan")

    if empty.any():
        empty_mae = float(
            np.mean(
                np.abs(
                    pred[empty]
                )
            )
        )
        empty_mean_pred = float(
            np.mean(
                pred[empty]
            )
        )
    else:
        empty_mae = float("nan")
        empty_mean_pred = float("nan")

    stats.update(
        {
            # Primary naming: Window-MAE / Window-RMSE
            "window_mae": stats["mae"],
            "window_rmse": stats["rmse"],
            "window_mae_micro": stats["mae"],
            "window_rmse_micro": stats["rmse"],
            "window_mae_macro": wmae_macro,
            "window_rmse_macro": wrmse_macro,
            "full_window_mae": full_wmae,
            "full_window_rmse": full_wrmse,
            "full_window_count": len(full_rows),
            "edge_window_mae": edge_wmae,
            "edge_window_rmse": edge_wrmse,
            "edge_window_count": len(edge_rows),
            "nae_nonzero": nae_nonzero,
            "nonempty_window_mae": nonempty_mae,
            "empty_window_mae": empty_mae,
            "empty_window_mean_prediction": empty_mean_pred,
            "empty_window_fraction": float(empty.mean()),
        }
    )

    stats.pop("nae", None)
    stats.pop("sre", None)

    return stats


# ---------------------------------------------------------------------
# MICF / measure metrics
# ---------------------------------------------------------------------

def measure_validity_metrics(
    y_pred: torch.Tensor,
    tau: float = 1e-6,
    eps: float = 1e-6,
) -> dict[str, float]:
    y = y_pred.float()

    negative_raw = (-y).clamp_min(0.0)
    positive_raw = y.clamp_min(0.0)
    negative_tau = (-y - tau).clamp_min(0.0)

    neg_total = float(negative_raw.sum().item())
    pos_total = float(positive_raw.sum().item())

    neg_cells_raw = int((y < 0.0).sum().item())
    neg_cells_tau = int((y < -tau).sum().item())
    total_cells = int(y.numel())

    vr_raw = float(neg_cells_raw / max(total_cells, 1))
    vr_tau = float(neg_cells_tau / max(total_cells, 1))

    return {
        "violation_rate": vr_raw,
        "violation_rate_raw": vr_raw,
        "violation_rate_tau": vr_tau,
        "violation_magnitude": float(negative_raw.mean().item()),
        "violation_magnitude_tau": float(negative_tau.mean().item()),
        "negative_mass_ratio": float(neg_total / max(pos_total, 1e-12)),
        "negative_mass_total": neg_total,
        "positive_mass_total": pos_total,
        "neg_cell_count": neg_cells_raw,
        "neg_cell_count_raw": neg_cells_raw,
        "neg_cell_count_tau": neg_cells_tau,
        "total_cells": total_cells,
    }


def aggregate_validity_metrics(
    rows: list[dict[str, Any]],
    eps: float = 1e-6,
) -> dict[str, float]:
    if not rows:
        return {}

    macro_vr_raw = finite_mean(r.get("violation_rate_raw", r.get("violation_rate", 0.0)) for r in rows)
    macro_vr_tau = finite_mean(r.get("violation_rate_tau", r.get("violation_rate", 0.0)) for r in rows)
    macro_vm = finite_mean(r["violation_magnitude"] for r in rows)
    macro_nmr = finite_mean(r["negative_mass_ratio"] for r in rows)

    total_neg_cells_raw = sum(int(r.get("neg_cell_count_raw", r.get("neg_cell_count", 0))) for r in rows)
    total_neg_cells_tau = sum(int(r.get("neg_cell_count_tau", 0)) for r in rows)
    total_cells = sum(int(r["total_cells"]) for r in rows)
    total_neg_mass = sum(float(r["negative_mass_total"]) for r in rows)
    total_pos_mass = sum(float(r.get("positive_mass_total", 0.0)) for r in rows)

    micro_vr_raw = float(total_neg_cells_raw / max(total_cells, 1))
    micro_vr_tau = float(total_neg_cells_tau / max(total_cells, 1))
    micro_vm = float(total_neg_mass / max(total_cells, 1))
    micro_nmr = float(total_neg_mass / max(total_pos_mass, 1e-12))

    return {
        "macro_violation_rate": macro_vr_raw,
        "macro_violation_rate_raw": macro_vr_raw,
        "macro_violation_rate_tau": macro_vr_tau,
        "macro_violation_magnitude": macro_vm,
        "macro_negative_mass_ratio": macro_nmr,
        "micro_violation_rate": micro_vr_raw,
        "micro_violation_rate_raw": micro_vr_raw,
        "micro_violation_rate_tau": micro_vr_tau,
        "micro_violation_magnitude": micro_vm,
        "micro_negative_mass_ratio": micro_nmr,
        "negative_mass_total": float(total_neg_mass),
        "positive_mass_total": float(total_pos_mass),
        # Backward-compatibility aliases
        "violation_rate": macro_vr_raw,
        "violation_magnitude": macro_vm,
        "negative_mass_ratio": macro_nmr,
    }


def representation_metrics(
    pred_c: torch.Tensor,
    pred_y: torch.Tensor,
    gt_c: torch.Tensor,
    gt_y: torch.Tensor,
    gt_count: float,
) -> dict[str, float]:
    scale = max(
        float(gt_count),
        1.0,
    )

    c_nmae = float(
        (
            (
                pred_c.float()
                - gt_c.float()
            ).abs()
            / scale
        )
        .mean()
        .item()
    )

    y_nl1 = float(
        (
            pred_y.float()
            - gt_y.float()
        )
        .abs()
        .sum()
        .item()
        / scale
    )

    pred_count_from_c = float(
        pred_c[..., -1, -1]
        .reshape(-1)[0]
        .item()
    )
    pred_count_from_y = float(
        pred_y.sum().item()
    )

    conservation_error = abs(
        pred_count_from_c
        - pred_count_from_y
    )

    return {
        "cumulative_field_nmae": c_nmae,
        "measure_nl1": y_nl1,
        "conservation_error": conservation_error,
    }


# ---------------------------------------------------------------------
# GAME (Canonical Pixel-Space & Diagnostic Stride-16)
# ---------------------------------------------------------------------

def game_pixel_space_errors(
    pred_y: torch.Tensor,
    points_in_bounds: np.ndarray,
    img_h: int,
    img_w: int,
    stride: int,
    levels: Iterable[int],
) -> dict[int, float]:
    """Canonical crowd-counting benchmark pixel-space GAME.

    Partitions the continuous image area [0, W) x [0, H) into 2^L x 2^L non-overlapping regions.
    GT count is the exact number of annotated points falling within each sub-rectangle.
    Predicted count is the integrated count measure of pred_y over each sub-rectangle via
    interval overlap weighting, preserving total predicted count exactly across all levels.
    """
    _, _, out_h, out_w = pred_y.shape
    y_2d = pred_y.squeeze(0).squeeze(0).float()

    results: dict[int, float] = {}

    for level in levels:
        level = int(level)
        if level < 0:
            raise ValueError("GAME level must be >= 0")

        parts = 2 ** level
        total_err = 0.0

        y_edges = np.linspace(0.0, float(img_h), parts + 1)
        x_edges = np.linspace(0.0, float(img_w), parts + 1)

        for r in range(parts):
            py0, py1 = y_edges[r], y_edges[r + 1]
            gy0, gy1 = py0 / float(stride), py1 / float(stride)
            iy0, iy1 = int(math.floor(gy0)), min(out_h, int(math.ceil(gy1)))

            for c in range(parts):
                px0, px1 = x_edges[c], x_edges[c + 1]
                gx0, gx1 = px0 / float(stride), px1 / float(stride)
                ix0, ix1 = int(math.floor(gx0)), min(out_w, int(math.ceil(gx1)))

                # Exact GT point count in region
                if len(points_in_bounds) > 0:
                    pt_mask = (
                        (points_in_bounds[:, 0] >= px0)
                        & (points_in_bounds[:, 0] < px1)
                        & (points_in_bounds[:, 1] >= py0)
                        & (points_in_bounds[:, 1] < py1)
                    )
                    gt_count = float(pt_mask.sum())
                else:
                    gt_count = 0.0

                # Exact integrated measure prediction in region
                cell_block = y_2d[iy0:iy1, ix0:ix1]
                if cell_block.numel() > 0:
                    y_indices = torch.arange(iy0, iy1, device=pred_y.device, dtype=torch.float32)
                    x_indices = torch.arange(ix0, ix1, device=pred_y.device, dtype=torch.float32)

                    overlap_y = (
                        torch.min(y_indices + 1.0, torch.tensor(gy1, device=pred_y.device))
                        - torch.max(y_indices, torch.tensor(gy0, device=pred_y.device))
                    ).clamp_min(0.0)
                    overlap_x = (
                        torch.min(x_indices + 1.0, torch.tensor(gx1, device=pred_y.device))
                        - torch.max(x_indices, torch.tensor(gx0, device=pred_y.device))
                    ).clamp_min(0.0)

                    weights = overlap_y.unsqueeze(1) * overlap_x.unsqueeze(0)
                    pred_count = float((cell_block * weights).sum().item())
                else:
                    pred_count = 0.0

                total_err += abs(pred_count - gt_count)

        results[level] = float(total_err)

    return results


def game_stride16_errors(
    pred_y: torch.Tensor,
    gt_y: torch.Tensor,
    levels: Iterable[int],
) -> dict[int, float]:
    """Diagnostic GAME directly evaluated on the model's stride-16 count measure."""
    if pred_y.shape != gt_y.shape:
        raise ValueError(
            f"GAME shape mismatch: "
            f"{tuple(pred_y.shape)} vs {tuple(gt_y.shape)}"
        )

    _, _, H, W = pred_y.shape
    results: dict[int, float] = {}

    for level in levels:
        level = int(level)
        parts = 2 ** level

        y_edges = np.linspace(0, H, parts + 1, dtype=np.int64)
        x_edges = np.linspace(0, W, parts + 1, dtype=np.int64)

        total = 0.0
        for r in range(parts):
            y0, y1 = int(y_edges[r]), int(y_edges[r + 1])
            for c in range(parts):
                x0, x1 = int(x_edges[c]), int(x_edges[c + 1])

                pred_count = float(pred_y[..., y0:y1, x0:x1].sum().item())
                gt_count = float(gt_y[..., y0:y1, x0:x1].sum().item())
                total += abs(pred_count - gt_count)

        results[level] = float(total)

    return results


# ---------------------------------------------------------------------
# Tiled/window inference
# ---------------------------------------------------------------------

def validate_tiling_geometry(
    model: MICFLite,
    tile_size: int,
    halo: int,
) -> None:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if halo < 0:
        raise ValueError("halo must be >= 0")

    stride = int(model.output_stride)
    required = (
        stride
        if model.finite_horizon is None
        else stride * int(model.finite_horizon)
    )

    if tile_size % required != 0:
        raise ValueError(
            f"tile_size={tile_size} must be divisible by {required}"
        )
    if halo % required != 0:
        raise ValueError(
            f"halo={halo} must be divisible by {required}"
        )


@torch.no_grad()
def predict_windows_and_tiled(
    model: MICFLite,
    image: torch.Tensor,
    points_in_bounds: np.ndarray,
    tile_size: int,
    halo: int,
) -> tuple[
    float,
    list[dict[str, Any]],
    torch.Tensor,
    torch.Tensor,
]:
    """Return tiled count, per-window rows, tiled C, tiled Y."""
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(f"Expected [1,3,H,W], got {tuple(image.shape)}")

    validate_tiling_geometry(model, tile_size, halo)

    _, _, H, W = image.shape
    stride = int(model.output_stride)
    out_tile = tile_size // stride

    n_h = math.ceil(H / tile_size)
    n_w = math.ceil(W / tile_size)

    padded_h = n_h * tile_size
    padded_w = n_w * tile_size

    x_pad = F.pad(
        image,
        (0, padded_w - W, 0, padded_h - H),
        mode="constant",
        value=0.0,
    )

    is_cumulative = model.head_type in {"cumulative", "integrated_local"}

    c_local: list[list[torch.Tensor | None]] | None = None
    y_global_local = torch.zeros(
        (1, 1, n_h * out_tile, n_w * out_tile),
        device=image.device,
        dtype=torch.float32,
    )

    if is_cumulative:
        c_local = [[None] * n_w for _ in range(n_h)]

    rows: list[dict[str, Any]] = []

    for tile_r in range(n_h):
        for tile_c in range(n_w):
            y0 = tile_r * tile_size
            x0 = tile_c * tile_size

            y1_core = (tile_r + 1) * tile_size
            x1_core = (tile_c + 1) * tile_size

            valid_y1 = min(y1_core, H)
            valid_x1 = min(x1_core, W)

            valid_h = valid_y1 - y0
            valid_w = valid_x1 - x0

            hy0 = max(0, y0 - halo)
            hx0 = max(0, x0 - halo)
            hy1 = min(padded_h, y1_core + halo)
            hx1 = min(padded_w, x1_core + halo)

            crop = x_pad[..., hy0:hy1, hx0:hx1]
            field = model.forward_field(crop)

            ry0 = (y0 - hy0) // stride
            rx0 = (x0 - hx0) // stride

            if is_cumulative:
                if halo > 0:
                    y_full = discrete_mixed_difference(field)
                    y_core = y_full[..., ry0:(ry0 + out_tile), rx0:(rx0 + out_tile)]
                else:
                    c_core_raw = field[..., :out_tile, :out_tile]
                    y_core = discrete_mixed_difference(c_core_raw)

                c_tile = torch.cumsum(torch.cumsum(y_core, dim=-2), dim=-1)
                assert c_local is not None
                c_local[tile_r][tile_c] = c_tile.squeeze(0).squeeze(0)
            else:
                y_core = field[..., ry0:(ry0 + out_tile), rx0:(rx0 + out_tile)].float()

            oy0 = tile_r * out_tile
            ox0 = tile_c * out_tile
            y_global_local[..., oy0:(oy0 + out_tile), ox0:(ox0 + out_tile)] = y_core.float()

            valid_out_h = math.ceil(valid_h / stride)
            valid_out_w = math.ceil(valid_w / stride)

            pred_window = float(y_core[..., :valid_out_h, :valid_out_w].sum().item())
            gt_window = count_points_in_window(
                points_in_bounds,
                x0=x0,
                y0=y0,
                x1=valid_x1,
                y1=valid_y1,
            )

            signed = pred_window - float(gt_window)
            is_full = (valid_w == tile_size and valid_h == tile_size)

            rows.append(
                {
                    "tile_row": tile_r,
                    "tile_col": tile_c,
                    "x0": x0,
                    "y0": y0,
                    "x1": valid_x1,
                    "y1": valid_y1,
                    "valid_width": valid_w,
                    "valid_height": valid_h,
                    "is_full_window": is_full,
                    "gt_count": gt_window,
                    "pred_count": pred_window,
                    "signed_error": signed,
                    "abs_error": abs(signed),
                }
            )

    out_h_full = math.ceil(H / stride)
    out_w_full = math.ceil(W / stride)

    if is_cumulative:
        assert c_local is not None
        complete: list[list[torch.Tensor]] = []
        for row in c_local:
            if any(v is None for v in row):
                raise RuntimeError("Incomplete tile grid.")
            complete.append([v for v in row if v is not None])

        c_global_2d = compose_tiled_cumulative_field(complete)
        c_tiled = c_global_2d[:out_h_full, :out_w_full].unsqueeze(0).unsqueeze(0)
        y_tiled = discrete_mixed_difference(c_tiled)
    else:
        y_tiled = y_global_local[..., :out_h_full, :out_w_full]
        c_tiled = cell_counts_to_cumulative_field(y_tiled, orientation="TL")

    pred_tiled = float(c_tiled[..., -1, -1].reshape(-1)[0].item())
    sum_window_predictions = float(sum(float(r["pred_count"]) for r in rows))

    if not math.isclose(
        pred_tiled,
        sum_window_predictions,
        rel_tol=1e-5,
        abs_tol=1e-3,
    ):
        raise RuntimeError(
            "Tiled/window prediction mismatch: "
            f"C_corner={pred_tiled:.6f}, sum_windows={sum_window_predictions:.6f}"
        )

    return (
        pred_tiled,
        rows,
        c_tiled,
        y_tiled,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    if args.checkpoint is None:
        if args.config is None:
            raise ValueError("Must provide at least --checkpoint or --config.")
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        exp_cfg = cfg.get("experiment", {})
        save_dir = Path(exp_cfg.get("save_dir", f"./runs/pilot_micf/{exp_cfg.get('model_id', 'micf').lower()}"))
        checkpoint_path = save_dir / "best.pt"
    else:
        checkpoint_path = Path(args.checkpoint)

    checkpoint = safe_torch_load(str(checkpoint_path), map_location="cpu")
    cfg = load_config(checkpoint, args.config)

    state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        if all(isinstance(k, str) for k in checkpoint.keys()):
            state_dict = checkpoint
        else:
            raise ValueError("Checkpoint has no state_dict.")

    model = build_model_from_config(cfg, state_dict, device)
    dataset = build_dataset(cfg, args.dataset_root, args.part, args.split)

    if args.output_dir is None:
        output_dir = checkpoint_path.parent / "eval_comprehensive"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    n_eval = len(dataset)
    if args.max_samples is not None:
        n_eval = min(n_eval, int(args.max_samples))

    direct_predictions: list[float] = []
    tiled_predictions: list[float] = []
    controlled_tiled_predictions: list[float] = []
    full_gt_counts: list[float] = []
    spatial_gt_counts: list[float] = []

    direct_validity_rows: list[ValidityRow] = []
    controlled_validity_rows: list[ValidityRow] = []
    practical_validity_rows: list[ValidityRow] = []

    direct_game_rows: list[dict[int, float]] = []
    per_image_canonical_rows: list[dict[str, Any]] = []

    max_conservation_error: float = 0.0

    all_window_rows: list[dict[str, Any]] = []
    all_window_rows_controlled: list[dict[str, Any]] = []
    per_image_rows: list[dict[str, Any]] = []

    cancellation_ratios: list[float] = []

    direct_game_pixel_by_level: dict[int, list[float]] = {int(l): [] for l in args.game_levels}
    tiled_game_pixel_by_level: dict[int, list[float]] = {int(l): [] for l in args.game_levels}

    direct_game_stride16_by_level: dict[int, list[float]] = {int(l): [] for l in args.game_levels}
    tiled_game_stride16_by_level: dict[int, list[float]] = {int(l): [] for l in args.game_levels}

    direct_legacy_validity_rows: list[dict[str, Any]] = []
    tiled_legacy_validity_rows: list[dict[str, Any]] = []

    direct_repr_rows: list[dict[str, float]] = []
    tiled_repr_rows: list[dict[str, float]] = []

    is_cumulative = getattr(model, "head_type", "") == "cumulative"

    print("=" * 96)
    print("MICF COMPREHENSIVE CHECKPOINT EVALUATION (Paper-Ready Audit)")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Device     : {device}")
    print(f"Head       : {model.head_type} | stride={model.output_stride} | FH={model.finite_horizon}")
    print(f"Tile       : {args.tile_size} | halo={args.halo}")
    print(f"GAME       : {args.game_levels}")
    print(f"LegacyDebug: {args.legacy_debug}")
    print(f"Images     : {n_eval}")
    print("=" * 96)

    for image_index in range(n_eval):
        sample = dataset[image_index]
        image = sample["image"].unsqueeze(0).to(device)

        _, _, H, W = image.shape
        stride = int(model.output_stride)

        raw_points = as_numpy_points(sample["gt_points"])
        gt_count = float(sample["gt_count"].item())

        # Protocol: Do NOT artificially relocate OOB points by clipping to boundary.
        # Benchmark count metrics evaluate against official gt_count. Spatial metrics use valid in-bounds points.
        points_spatial = in_bounds_points(raw_points, H, W)
        gt_spatial_count = float(len(points_spatial))
        oob_count = float(len(raw_points) - len(points_spatial))

        points_inside = points_spatial
        gt_in_bounds = gt_spatial_count
        gt_out_of_bounds = oob_count
        spatial_gt_counts.append(gt_spatial_count)

        # ---------------------------------------------------------
        # Full-Direct
        # ---------------------------------------------------------
        pred_direct_count, direct_field = model.predict(
            image, pad_multiple=args.direct_pad_multiple
        )
        pred_direct_count = float(torch.as_tensor(pred_direct_count).item())

        if model.head_type in {"cumulative", "integrated_local"}:
            c_direct = direct_field.float()
            y_direct = discrete_mixed_difference(c_direct)

            count_from_c = float(c_direct[..., -1, -1].reshape(-1)[0].item())
            count_from_y = float(y_direct.sum().item())
            abs_cons_error = abs(count_from_c - count_from_y)
            if abs_cons_error > max_conservation_error:
                max_conservation_error = abs_cons_error
            cons_tol = 1e-4 * max(1.0, abs(count_from_c))
            if abs_cons_error > cons_tol:
                raise RuntimeError(
                    "Cumulative/measure conservation failed: "
                    f"C[-1,-1]={count_from_c:.8f}, sum(Y)={count_from_y:.8f}, "
                    f"error={abs_cons_error:.8f}, tol={cons_tol:.8f}"
                )
        else:
            y_direct = direct_field.float()
            c_direct = cell_counts_to_cumulative_field(y_direct, orientation="TL")

        # ---------------------------------------------------------
        # Window + Full-Tiled (Practical)
        # ---------------------------------------------------------
        (
            pred_tiled_count,
            window_rows,
            c_tiled,
            y_tiled,
        ) = predict_windows_and_tiled(
            model=model,
            image=image,
            points_in_bounds=points_inside,
            tile_size=args.tile_size,
            halo=args.halo,
        )

        # ---------------------------------------------------------
        # Full-Tiled (Controlled halo)
        # ---------------------------------------------------------
        if args.controlled_halo == args.halo:
            pred_ctrl_tiled_count = pred_tiled_count
            ctrl_window_rows = window_rows
            c_ctrl_tiled = c_tiled
            y_ctrl_tiled = y_tiled
        else:
            (
                pred_ctrl_tiled_count,
                ctrl_window_rows,
                c_ctrl_tiled,
                y_ctrl_tiled,
            ) = predict_windows_and_tiled(
                model=model,
                image=image,
                points_in_bounds=points_inside,
                tile_size=args.tile_size,
                halo=args.controlled_halo,
            )

        # ---------------------------------------------------------
        # Shared exact GT count measure on output grid
        # ---------------------------------------------------------
        out_h = c_direct.shape[-2]
        out_w = c_direct.shape[-1]

        if c_tiled.shape[-2:] != (out_h, out_w):
            raise RuntimeError(
                f"Direct/tiled output-grid mismatch: {c_direct.shape[-2:]} vs {c_tiled.shape[-2:]}"
            )

        gt_y_2d = points_to_count_map(
            points_inside,
            out_h=out_h,
            out_w=out_w,
            stride=stride,
            device=device,
            dtype=torch.float32,
        )
        gt_y = gt_y_2d.unsqueeze(0).unsqueeze(0)
        gt_c = cell_counts_to_cumulative_field(gt_y, orientation="TL")

        if not math.isclose(
            float(gt_y.sum().item()),
            gt_in_bounds,
            rel_tol=0.0,
            abs_tol=1e-5,
        ):
            raise RuntimeError("GT measure does not conserve spatial count.")

        # ---------------------------------------------------------
        # Optional repository tiled equivalence
        # ---------------------------------------------------------
        repo_tiled_count = float("nan")
        repo_tiled_diff = float("nan")

        if args.verify_repo_tiled:
            repo_tiled, _ = model.predict_tiled(
                image, tile_size=args.tile_size, halo=args.halo
            )
            repo_tiled_count = float(torch.as_tensor(repo_tiled).item())
            repo_tiled_diff = abs(repo_tiled_count - pred_tiled_count)
            if repo_tiled_diff > args.strict_tiled_tolerance:
                raise RuntimeError(
                    f"New tiled evaluator differs from model.predict_tiled(): "
                    f"ours={pred_tiled_count:.6f}, repo={repo_tiled_count:.6f}"
                )

        # ---------------------------------------------------------
        # Canonical validity & GAME per image
        # ---------------------------------------------------------
        if is_cumulative:
            valid_row_direct = recovered_measure_validity(y_direct, tau=1e-6)
            valid_row_ctrl = recovered_measure_validity(y_ctrl_tiled, tau=1e-6)
            valid_row_prac = recovered_measure_validity(y_tiled, tau=1e-6)

            direct_validity_rows.append(valid_row_direct)
            controlled_validity_rows.append(valid_row_ctrl)
            practical_validity_rows.append(valid_row_prac)

        game_row_direct = game_errors_one_image(
            y_direct,
            points_inside,
            image_h=H,
            image_w=W,
            stride=stride,
            levels=args.game_levels,
        )
        direct_game_rows.append(game_row_direct)

        canon_row: dict[str, Any] = {
            "image_index": image_index,
            "image_name": Path(sample["img_path"]).name,
            "height": H,
            "width": W,
            "gt_count": gt_count,
            "gt_spatial_count": gt_spatial_count,
            "direct_pred_count": pred_direct_count,
            "controlled_tiled_pred_count": pred_ctrl_tiled_count,
            "practical_tiled_pred_count": pred_tiled_count,
            "direct_abs_error": abs(pred_direct_count - gt_count),
            "controlled_tiled_abs_error": abs(pred_ctrl_tiled_count - gt_count),
            "practical_tiled_abs_error": abs(pred_tiled_count - gt_count),
            "direct_vs_controlled_abs": abs(pred_direct_count - pred_ctrl_tiled_count),
            "direct_vs_practical_abs": abs(pred_direct_count - pred_tiled_count),
            "direct_vr_tau": valid_row_direct.vr_tau if is_cumulative else None,
            "direct_positive_variation": valid_row_direct.positive_variation if is_cumulative else None,
            "direct_negative_variation": valid_row_direct.negative_variation if is_cumulative else None,
            "direct_nvr": valid_row_direct.nvr if is_cumulative else None,
            "direct_violating_cells": valid_row_direct.violating_cells if is_cumulative else None,
            "direct_total_cells": valid_row_direct.total_cells if is_cumulative else None,
        }
        for lvl, err in game_row_direct.items():
            canon_row[f"direct_game_{lvl}"] = err
        per_image_canonical_rows.append(canon_row)

        # ---------------------------------------------------------
        # Legacy validity, window bookkeeping & representation diagnostics
        # ---------------------------------------------------------
        cancel = float("nan")
        if args.legacy_debug:
            image_signed_errors: list[float] = []
            for local_idx, row in enumerate(window_rows):
                enriched = {
                    "image_index": image_index,
                    "window_index": local_idx,
                    "image_path": sample["img_path"],
                    "image_height": H,
                    "image_width": W,
                    "halo_type": "practical",
                    **row,
                }
                all_window_rows.append(enriched)
                image_signed_errors.append(float(row["signed_error"]))

            for local_idx, row in enumerate(ctrl_window_rows):
                enriched_ctrl = {
                    "image_index": image_index,
                    "window_index": local_idx,
                    "image_path": sample["img_path"],
                    "image_height": H,
                    "image_width": W,
                    "halo_type": "controlled",
                    **row,
                }
                all_window_rows_controlled.append(enriched_ctrl)

            cancel = compute_cancellation_ratio(image_signed_errors)
            cancellation_ratios.append(cancel)

            local_abs_sum = sum(abs(e) for e in image_signed_errors)
            net_abs = abs(sum(image_signed_errors))

            direct_game_pixel = game_pixel_space_errors(
                y_direct, points_inside, img_h=H, img_w=W, stride=stride, levels=args.game_levels
            )
            tiled_game_pixel = game_pixel_space_errors(
                y_tiled, points_inside, img_h=H, img_w=W, stride=stride, levels=args.game_levels
            )

            direct_game_stride16 = game_stride16_errors(y_direct, gt_y, args.game_levels)
            tiled_game_stride16 = game_stride16_errors(y_tiled, gt_y, args.game_levels)

            for level in args.game_levels:
                l_int = int(level)
                direct_game_pixel_by_level[l_int].append(direct_game_pixel[l_int])
                tiled_game_pixel_by_level[l_int].append(tiled_game_pixel[l_int])
                direct_game_stride16_by_level[l_int].append(direct_game_stride16[l_int])
                tiled_game_stride16_by_level[l_int].append(tiled_game_stride16[l_int])

            direct_valid = measure_validity_metrics(y_direct)
            tiled_valid = measure_validity_metrics(y_tiled)

            direct_repr = representation_metrics(
                c_direct, y_direct, gt_c, gt_y, gt_in_bounds
            )
            tiled_repr = representation_metrics(
                c_tiled, y_tiled, gt_c, gt_y, gt_in_bounds
            )

            direct_legacy_validity_rows.append(direct_valid)
            tiled_legacy_validity_rows.append(tiled_valid)
            direct_repr_rows.append(direct_repr)
            tiled_repr_rows.append(tiled_repr)

            row_dict: dict[str, Any] = {
                "image_index": image_index,
                "image_path": sample["img_path"],
                "height": H,
                "width": W,
                "num_windows": len(window_rows),
                "gt_count": gt_count,
                "gt_in_bounds_count": gt_in_bounds,
                "gt_out_of_bounds": gt_out_of_bounds,
                "pred_full_direct": pred_direct_count,
                "err_full_direct_signed": pred_direct_count - gt_count,
                "err_full_direct_abs": abs(pred_direct_count - gt_count),
                "pred_full_tiled": pred_tiled_count,
                "err_full_tiled_signed": pred_tiled_count - gt_count,
                "err_full_tiled_abs": abs(pred_tiled_count - gt_count),
                "pred_full_tiled_practical": pred_tiled_count,
                "err_full_tiled_practical_signed": pred_tiled_count - gt_count,
                "err_full_tiled_practical_abs": abs(pred_tiled_count - gt_count),
                "pred_full_tiled_controlled": pred_ctrl_tiled_count,
                "err_full_tiled_controlled_signed": pred_ctrl_tiled_count - gt_count,
                "err_full_tiled_controlled_abs": abs(pred_ctrl_tiled_count - gt_count),
                "halo_effect_signed": pred_tiled_count - pred_ctrl_tiled_count,
                "halo_effect_abs": abs(pred_tiled_count - pred_ctrl_tiled_count),
                "cancellation_ratio": cancel,
                "window_abs_error_sum": local_abs_sum,
                "window_net_abs_error": net_abs,
                "repo_tiled_count": repo_tiled_count,
                "repo_tiled_abs_diff": repo_tiled_diff,
            }

            for level in args.game_levels:
                row_dict[f"direct_game_pixel_L{level}"] = direct_game_pixel[level]
                row_dict[f"tiled_game_pixel_L{level}"] = tiled_game_pixel[level]
                row_dict[f"direct_game_stride16_L{level}"] = direct_game_stride16[level]
                row_dict[f"tiled_game_stride16_L{level}"] = tiled_game_stride16[level]

            for k, v in direct_valid.items():
                row_dict[f"direct_{k}"] = v
            for k, v in tiled_valid.items():
                row_dict[f"tiled_{k}"] = v
            for k, v in direct_repr.items():
                row_dict[f"direct_{k}"] = v
            for k, v in tiled_repr.items():
                row_dict[f"tiled_{k}"] = v

            per_image_rows.append(row_dict)

        direct_predictions.append(pred_direct_count)
        tiled_predictions.append(pred_tiled_count)
        controlled_tiled_predictions.append(pred_ctrl_tiled_count)
        full_gt_counts.append(gt_count)

        status_str = (
            f"[{image_index + 1:03d}/{n_eval:03d}] "
            f"GT={gt_count:.1f} | "
            f"Direct={pred_direct_count:.2f} (AE={abs(pred_direct_count-gt_count):.2f}) | "
            f"Tiled(h={args.halo})={pred_tiled_count:.2f} (AE={abs(pred_tiled_count-gt_count):.2f}) | "
            f"Tiled(ctrl={args.controlled_halo})={pred_ctrl_tiled_count:.2f} (AE={abs(pred_ctrl_tiled_count-gt_count):.2f})"
        )
        if args.legacy_debug and not math.isnan(cancel):
            status_str += f" | Cancel={100*cancel:.1f}%"
        print(status_str)

    # -----------------------------------------------------------------
    # Canonical Aggregation
    # -----------------------------------------------------------------
    gt_arr = np.asarray(full_gt_counts, dtype=np.float64)
    direct_arr = np.asarray(direct_predictions, dtype=np.float64)
    controlled_arr = np.asarray(controlled_tiled_predictions, dtype=np.float64)
    practical_arr = np.asarray(tiled_predictions, dtype=np.float64)

    benchmark_direct = benchmark_count_summary(direct_arr, gt_arr)
    benchmark_controlled = benchmark_count_summary(controlled_arr, gt_arr)
    benchmark_practical = benchmark_count_summary(practical_arr, gt_arr)

    if is_cumulative:
        validity_direct = aggregate_validity(direct_validity_rows)
        validity_controlled = aggregate_validity(controlled_validity_rows)
        validity_practical = aggregate_validity(practical_validity_rows)
    else:
        validity_direct = {"status": "nonnegative_by_construction"}
        validity_controlled = {"status": "nonnegative_by_construction"}
        validity_practical = {"status": "nonnegative_by_construction"}

    controlled_decomposition = direct_tiled_discrepancy(direct_arr, controlled_arr, gt_arr)
    practical_decomposition = direct_tiled_discrepancy(direct_arr, practical_arr, gt_arr)

    game_direct = aggregate_game(direct_game_rows)

    # Gate E check: GAME(0) invariant against spatial count
    if 0 in args.game_levels and len(direct_game_rows) > 0:
        game0 = float(game_direct.get("game_0", float("nan")))
        expected_game0 = float(np.mean(np.abs(direct_arr - np.asarray(spatial_gt_counts, dtype=np.float64))))
        tol = 1e-4 * max(1.0, abs(expected_game0))
        if abs(game0 - expected_game0) > tol:
            raise RuntimeError(
                "GAME(0) invariant failed against spatial GT count: "
                f"GAME0={game0:.8f}, expected={expected_game0:.8f}, tol={tol:.8f}"
            )

    canonical_summary: dict[str, Any] = {
        "benchmark": {
            "direct": benchmark_direct,
            "tiled_controlled": benchmark_controlled,
            "tiled_practical": benchmark_practical,
        },
        "method_validity": {
            "direct": validity_direct,
            "tiled_controlled": validity_controlled,
            "tiled_practical": validity_practical,
        },
        "decomposition": {
            "controlled": controlled_decomposition,
            "practical": practical_decomposition,
        },
        "spatial_optional": {
            "direct": game_direct,
        },
        "sanity": {
            "max_conservation_error": float(max_conservation_error),
        },
        "protocol": {
            "split": args.split,
            "tile_size": args.tile_size,
            "controlled_halo": args.controlled_halo,
            "practical_halo": args.halo,
            "direct_pad_multiple": args.direct_pad_multiple,
            "game_levels": [int(x) for x in args.game_levels],
            "spatial_points_protocol": "raw_in_bounds",
            "validity_tau": 1e-6,
            "nvr_eps": 1e-12,
            "benchmark_primary_inference": "full_image_direct",
            "legacy_debug": bool(args.legacy_debug),
        },
        "legacy_debug": None,
    }

    if args.legacy_debug:
        window_stats = window_metric_summary(all_window_rows)
        window_stats_controlled = window_metric_summary(all_window_rows_controlled)
        canonical_summary["legacy_debug"] = {
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_best_mae_stored": checkpoint.get("best_mae"),
            "model_head_type": model.head_type,
            "model_output_stride": model.output_stride,
            "model_finite_horizon": model.finite_horizon,
            "direct_diagnostics": diagnostic_count_summary(direct_arr, gt_arr),
            "controlled_diagnostics": diagnostic_count_summary(controlled_arr, gt_arr),
            "practical_diagnostics": diagnostic_count_summary(practical_arr, gt_arr),
            "window_controlled": window_stats_controlled,
            "window_practical": window_stats,
            "cancellation": {
                "mean": finite_mean(cancellation_ratios),
                "median": finite_percentile(cancellation_ratios, 50),
                "p90": finite_percentile(cancellation_ratios, 90),
            },
            "representation_direct": {
                key: finite_mean(row[key] for row in direct_repr_rows)
                for key in (direct_repr_rows[0].keys() if direct_repr_rows else [])
            },
            "representation_tiled": {
                key: finite_mean(row[key] for row in tiled_repr_rows)
                for key in (tiled_repr_rows[0].keys() if tiled_repr_rows else [])
            },
        }

    # -----------------------------------------------------------------
    # Save Outputs
    # -----------------------------------------------------------------
    summary_path = output_dir / "summary.json"
    canonical_image_csv = output_dir / "per_image_canonical.csv"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(canonical_summary), f, indent=2)

    write_csv(canonical_image_csv, per_image_canonical_rows)

    if args.legacy_debug:
        legacy_summary_path = output_dir / "comprehensive_summary.json"
        legacy_image_csv = output_dir / "comprehensive_per_image.csv"
        window_csv = output_dir / "comprehensive_per_window.csv"
        window_ctrl_csv = output_dir / "comprehensive_per_window_controlled.csv"
        with legacy_summary_path.open("w", encoding="utf-8") as f:
            json.dump(sanitize_for_json(canonical_summary), f, indent=2)
        write_csv(legacy_image_csv, per_image_rows)
        write_csv(window_csv, all_window_rows)
        write_csv(window_ctrl_csv, all_window_rows_controlled)

    # -----------------------------------------------------------------
    # Canonical Console Report (Section 25)
    # -----------------------------------------------------------------
    print()
    print("=" * 80)
    print("CANONICAL CROWD-COUNTING EVALUATION")
    print("=" * 80)
    print()
    print("Benchmark / Full-Image Direct")
    print(f"MAE   : {benchmark_direct['mae']:.4f}")
    print(f"RMSE  : {benchmark_direct['rmse']:.4f}")
    print(f"NAE   : {benchmark_direct['nae']:.4f}")
    print()
    if is_cumulative:
        print("PS-FH Cumulative Validity / Direct")
        print(f"VR_tau micro : {validity_direct['vr_tau_micro'] * 100:.4f}%")
        print(f"NVR micro    : {validity_direct['nvr_micro'] * 100:.4f}%")
        print()
    else:
        print("Model Representation : non-cumulative (nonnegative by construction)")
        print()
    print("Inference Decomposition")
    print(f"Direct vs Controlled | normalized discrepancy : {controlled_decomposition['mean_normalized_prediction_discrepancy']:.6f}")
    print(f"Direct vs Practical  | normalized discrepancy : {practical_decomposition['mean_normalized_prediction_discrepancy']:.6f}")
    print()
    print("Optional Spatial")
    for lvl in sorted(args.game_levels):
        val = game_direct.get(f"game_{lvl}", float("nan"))
        print(f"GAME({lvl}): {val:.4f}")
    print()
    print("Sanity")
    print(f"max |C[-1,-1] - sum(Delta_xy C)| : {max_conservation_error:.8e}")
    print("=" * 80)
    print()
    print(f"Summary    : {summary_path}")
    print(f"Per-image  : {canonical_image_csv}")
    print()


if __name__ == "__main__":
    main()
