"""Comprehensive MICF checkpoint evaluator.

Evaluates one already-trained checkpoint without updating weights.

Metric families
---------------
1. Standard image-level:
   MAE, RMSE, NAE, SRE, signed bias, Median/P90/P95/Max AE.

2. Local/window:
   PMAE/PRMSE, non-zero-window NAE, empty-window error,
   non-empty-window MAE, local cancellation.

3. Spatial:
   GAME(0..3) on the reconstructed stride-level count measure.

4. MICF validity:
   violation rate, violation magnitude, negative-mass ratio,
   total negative mass, count<->measure conservation error.

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
from hpc.models.micf_lite import (
    MICFLite,
    compose_tiled_cumulative_field,
)


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
        "--verify-repo-tiled",
        action="store_true",
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
# Standard count metrics
# ---------------------------------------------------------------------

def count_metric_summary(
    predictions: Iterable[float],
    targets: Iterable[float],
    eps: float = 1e-12,
) -> dict[str, float]:
    pred = np.asarray(
        list(predictions),
        dtype=np.float64,
    )
    gt = np.asarray(
        list(targets),
        dtype=np.float64,
    )

    if pred.shape != gt.shape:
        raise ValueError(
            f"Prediction/GT shape mismatch: "
            f"{pred.shape} vs {gt.shape}"
        )

    if pred.size == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "nae": float("nan"),
            "sre": float("nan"),
            "signed_bias": float("nan"),
            "median_ae": float("nan"),
            "p90_ae": float("nan"),
            "p95_ae": float("nan"),
            "max_ae": float("nan"),
        }

    err = pred - gt
    ae = np.abs(err)

    denom = np.maximum(
        gt,
        eps,
    )

    return {
        "mae": float(np.mean(ae)),
        "rmse": float(
            np.sqrt(
                np.mean(err * err)
            )
        ),
        "nae": float(
            np.mean(
                ae / denom
            )
        ),
        "sre": float(
            np.sqrt(
                np.mean(
                    (err * err)
                    / denom
                )
            )
        ),
        "signed_bias": float(
            np.mean(err)
        ),
        "median_ae": float(
            np.median(ae)
        ),
        "p90_ae": float(
            np.percentile(
                ae,
                90,
            )
        ),
        "p95_ae": float(
            np.percentile(
                ae,
                95,
            )
        ),
        "max_ae": float(
            np.max(ae)
        ),
    }


def window_metric_summary(
    rows: list[dict[str, Any]],
) -> dict[str, float]:
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

    stats = count_metric_summary(
        pred,
        gt,
        eps=1.0,
    )

    # Ordinary NAE is not meaningful for GT=0 windows.
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
            "pmae": stats["mae"],
            "prmse": stats["rmse"],
            "nae_nonzero": nae_nonzero,
            "nonempty_window_mae": (
                nonempty_mae
            ),
            "empty_window_mae": (
                empty_mae
            ),
            "empty_window_mean_prediction": (
                empty_mean_pred
            ),
            "empty_window_fraction": float(
                empty.mean()
            ),
        }
    )

    # Remove the arbitrary eps-protected window NAE/SRE aliases.
    # Keep only the explicitly meaningful non-zero-window NAE.
    stats.pop("nae", None)
    stats.pop("sre", None)

    return stats


# ---------------------------------------------------------------------
# MICF / measure metrics
# ---------------------------------------------------------------------

def measure_validity_metrics(
    y_pred: torch.Tensor,
    eps: float = 1e-6,
) -> dict[str, float]:
    y = y_pred.float()

    negative = (
        -y
    ).clamp_min(0.0)

    neg_total = float(
        negative.sum().item()
    )
    abs_total = float(
        y.abs().sum().item()
    )

    return {
        "violation_rate": float(
            (y < 0).float()
            .mean()
            .item()
        ),
        "violation_magnitude": float(
            negative.mean().item()
        ),
        "negative_mass_ratio": float(
            neg_total
            / (abs_total + eps)
        ),
        "negative_mass_total": (
            neg_total
        ),
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
        "cumulative_field_nmae": (
            c_nmae
        ),
        "measure_nl1": y_nl1,
        "conservation_error": (
            conservation_error
        ),
    }


# ---------------------------------------------------------------------
# GAME
# ---------------------------------------------------------------------

def game_errors(
    pred_y: torch.Tensor,
    gt_y: torch.Tensor,
    levels: Iterable[int],
) -> dict[int, float]:
    """GAME on the model's stride-level count measure.

    Both tensors must be [1,1,H,W] and spatially aligned.
    """
    if pred_y.shape != gt_y.shape:
        raise ValueError(
            f"GAME shape mismatch: "
            f"{tuple(pred_y.shape)} "
            f"vs {tuple(gt_y.shape)}"
        )

    if pred_y.ndim != 4:
        raise ValueError(
            "GAME expects [1,1,H,W]."
        )

    _, _, H, W = pred_y.shape

    results: dict[int, float] = {}

    for level in levels:
        level = int(level)
        if level < 0:
            raise ValueError(
                "GAME level must be >= 0"
            )

        parts = 2 ** level

        # Integer boundaries cover every cell exactly once.
        y_edges = np.linspace(
            0,
            H,
            parts + 1,
            dtype=np.int64,
        )
        x_edges = np.linspace(
            0,
            W,
            parts + 1,
            dtype=np.int64,
        )

        total = 0.0

        for r in range(parts):
            y0 = int(y_edges[r])
            y1 = int(y_edges[r + 1])

            for c in range(parts):
                x0 = int(x_edges[c])
                x1 = int(x_edges[c + 1])

                pred_count = float(
                    pred_y[
                        ...,
                        y0:y1,
                        x0:x1,
                    ]
                    .sum()
                    .item()
                )
                gt_count = float(
                    gt_y[
                        ...,
                        y0:y1,
                        x0:x1,
                    ]
                    .sum()
                    .item()
                )

                total += abs(
                    pred_count
                    - gt_count
                )

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
        raise ValueError(
            "tile_size must be positive"
        )
    if halo < 0:
        raise ValueError(
            "halo must be >= 0"
        )

    stride = int(model.output_stride)

    required = (
        stride
        if model.finite_horizon is None
        else stride
        * int(model.finite_horizon)
    )

    if tile_size % required != 0:
        raise ValueError(
            f"tile_size={tile_size} "
            f"must be divisible by {required}"
        )

    if halo % required != 0:
        raise ValueError(
            f"halo={halo} "
            f"must be divisible by {required}"
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
    """Return tiled count, per-window rows, tiled C, tiled Y.

    Cumulative path mirrors MICFLite.predict_tiled():
        halo forward
        -> mixed difference
        -> core Y
        -> local core C
        -> exact global composition
    """
    if (
        image.ndim != 4
        or image.shape[0] != 1
    ):
        raise ValueError(
            f"Expected [1,3,H,W], "
            f"got {tuple(image.shape)}"
        )

    validate_tiling_geometry(
        model,
        tile_size,
        halo,
    )

    _, _, H, W = image.shape
    stride = int(model.output_stride)
    out_tile = tile_size // stride

    n_h = math.ceil(
        H / tile_size
    )
    n_w = math.ceil(
        W / tile_size
    )

    padded_h = n_h * tile_size
    padded_w = n_w * tile_size

    x_pad = F.pad(
        image,
        (
            0,
            padded_w - W,
            0,
            padded_h - H,
        ),
        mode="constant",
        value=0.0,
    )

    is_cumulative = (
        model.head_type
        in {
            "cumulative",
            "integrated_local",
        }
    )

    c_local: (
        list[list[torch.Tensor | None]]
        | None
    ) = None

    y_global_local = torch.zeros(
        (
            1,
            1,
            n_h * out_tile,
            n_w * out_tile,
        ),
        device=image.device,
        dtype=torch.float32,
    )

    if is_cumulative:
        c_local = [
            [None] * n_w
            for _ in range(n_h)
        ]

    rows: list[dict[str, Any]] = []

    for tile_r in range(n_h):
        for tile_c in range(n_w):
            y0 = tile_r * tile_size
            x0 = tile_c * tile_size

            y1_core = (
                tile_r + 1
            ) * tile_size
            x1_core = (
                tile_c + 1
            ) * tile_size

            valid_y1 = min(
                y1_core,
                H,
            )
            valid_x1 = min(
                x1_core,
                W,
            )

            valid_h = (
                valid_y1 - y0
            )
            valid_w = (
                valid_x1 - x0
            )

            hy0 = max(
                0,
                y0 - halo,
            )
            hx0 = max(
                0,
                x0 - halo,
            )
            hy1 = min(
                padded_h,
                y1_core + halo,
            )
            hx1 = min(
                padded_w,
                x1_core + halo,
            )

            crop = x_pad[
                ...,
                hy0:hy1,
                hx0:hx1,
            ]

            field = model.forward_field(
                crop
            )

            ry0 = (
                y0 - hy0
            ) // stride
            rx0 = (
                x0 - hx0
            ) // stride

            if is_cumulative:
                if halo > 0:
                    y_full = (
                        discrete_mixed_difference(
                            field
                        )
                    )
                    y_core = y_full[
                        ...,
                        ry0:(
                            ry0
                            + out_tile
                        ),
                        rx0:(
                            rx0
                            + out_tile
                        ),
                    ]
                else:
                    c_core_raw = field[
                        ...,
                        :out_tile,
                        :out_tile,
                    ]
                    y_core = (
                        discrete_mixed_difference(
                            c_core_raw
                        )
                    )

                c_tile = torch.cumsum(
                    torch.cumsum(
                        y_core,
                        dim=-2,
                    ),
                    dim=-1,
                )

                assert c_local is not None
                c_local[
                    tile_r
                ][tile_c] = (
                    c_tile.squeeze(0)
                    .squeeze(0)
                )
            else:
                y_core = field[
                    ...,
                    ry0:(
                        ry0
                        + out_tile
                    ),
                    rx0:(
                        rx0
                        + out_tile
                    ),
                ].float()

            oy0 = (
                tile_r
                * out_tile
            )
            ox0 = (
                tile_c
                * out_tile
            )

            y_global_local[
                ...,
                oy0:(
                    oy0 + out_tile
                ),
                ox0:(
                    ox0 + out_tile
                ),
            ] = y_core.float()

            valid_out_h = math.ceil(
                valid_h / stride
            )
            valid_out_w = math.ceil(
                valid_w / stride
            )

            pred_window = float(
                y_core[
                    ...,
                    :valid_out_h,
                    :valid_out_w,
                ]
                .sum()
                .item()
            )

            gt_window = (
                count_points_in_window(
                    points_in_bounds,
                    x0=x0,
                    y0=y0,
                    x1=valid_x1,
                    y1=valid_y1,
                )
            )

            signed = (
                pred_window
                - float(gt_window)
            )

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
                    "gt_count": gt_window,
                    "pred_count": pred_window,
                    "signed_error": signed,
                    "abs_error": abs(
                        signed
                    ),
                }
            )

    out_h_full = math.ceil(
        H / stride
    )
    out_w_full = math.ceil(
        W / stride
    )

    if is_cumulative:
        assert c_local is not None

        complete: list[
            list[torch.Tensor]
        ] = []

        for row in c_local:
            if any(
                v is None
                for v in row
            ):
                raise RuntimeError(
                    "Incomplete tile grid."
                )

            complete.append(
                [
                    v
                    for v in row
                    if v is not None
                ]
            )

        c_global_2d = (
            compose_tiled_cumulative_field(
                complete
            )
        )

        c_tiled = c_global_2d[
            :out_h_full,
            :out_w_full,
        ].unsqueeze(0).unsqueeze(0)

        y_tiled = (
            discrete_mixed_difference(
                c_tiled
            )
        )
    else:
        y_tiled = y_global_local[
            ...,
            :out_h_full,
            :out_w_full,
        ]
        c_tiled = (
            cell_counts_to_cumulative_field(
                y_tiled,
                orientation="TL",
            )
        )

    pred_tiled = float(
        c_tiled[
            ...,
            -1,
            -1,
        ]
        .reshape(-1)[0]
        .item()
    )

    sum_window_predictions = float(
        sum(
            float(r["pred_count"])
            for r in rows
        )
    )

    if not math.isclose(
        pred_tiled,
        sum_window_predictions,
        rel_tol=1e-5,
        abs_tol=1e-3,
    ):
        raise RuntimeError(
            "Tiled/window prediction mismatch: "
            f"C_corner={pred_tiled:.6f}, "
            f"sum_windows="
            f"{sum_window_predictions:.6f}"
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
    device = torch.device(
        args.device
    )

    if args.checkpoint is None:
        if args.config is None:
            raise ValueError(
                "Must provide at least --checkpoint or --config."
            )
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        exp_cfg = cfg.get("experiment", {})
        save_dir = Path(exp_cfg.get("save_dir", f"./runs/pilot_micf/{exp_cfg.get('model_id', 'micf').lower()}"))
        checkpoint_path = save_dir / "best.pt"
    else:
        checkpoint_path = Path(args.checkpoint)

    checkpoint = safe_torch_load(
        str(checkpoint_path),
        map_location="cpu",
    )

    cfg = load_config(
        checkpoint,
        args.config,
    )

    state_dict = checkpoint.get(
        "state_dict"
    )

    if state_dict is None:
        if all(
            isinstance(k, str)
            for k in checkpoint.keys()
        ):
            state_dict = checkpoint
        else:
            raise ValueError(
                "Checkpoint has no state_dict."
            )

    model = build_model_from_config(
        cfg,
        state_dict,
        device,
    )

    dataset = build_dataset(
        cfg,
        args.dataset_root,
        args.part,
        args.split,
    )

    if args.output_dir is None:
        output_dir = (
            checkpoint_path.parent
            / "eval_comprehensive"
        )
    else:
        output_dir = Path(
            args.output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    n_eval = len(dataset)
    if args.max_samples is not None:
        n_eval = min(
            n_eval,
            int(args.max_samples),
        )

    direct_predictions: list[float] = []
    tiled_predictions: list[float] = []
    full_gt_counts: list[float] = []

    all_window_rows: list[
        dict[str, Any]
    ] = []
    per_image_rows: list[
        dict[str, Any]
    ] = []

    cancellation_ratios: list[
        float
    ] = []

    direct_game_by_level: dict[
        int,
        list[float],
    ] = {
        int(level): []
        for level in args.game_levels
    }

    tiled_game_by_level: dict[
        int,
        list[float],
    ] = {
        int(level): []
        for level in args.game_levels
    }

    direct_validity_rows: list[
        dict[str, float]
    ] = []
    tiled_validity_rows: list[
        dict[str, float]
    ] = []

    direct_repr_rows: list[
        dict[str, float]
    ] = []
    tiled_repr_rows: list[
        dict[str, float]
    ] = []

    print("=" * 96)
    print(
        "MICF COMPREHENSIVE CHECKPOINT EVALUATION"
    )
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Device     : {device}")
    print(
        f"Head       : {model.head_type}"
        f" | stride={model.output_stride}"
        f" | FH={model.finite_horizon}"
    )
    print(
        f"Tile       : {args.tile_size}"
        f" | halo={args.halo}"
    )
    print(
        f"GAME       : {args.game_levels}"
    )
    print(
        f"Images     : {n_eval}"
    )
    print("=" * 96)

    for image_index in range(n_eval):
        sample = dataset[image_index]

        image = (
            sample["image"]
            .unsqueeze(0)
            .to(device)
        )

        _, _, H, W = image.shape
        stride = int(
            model.output_stride
        )

        raw_points = as_numpy_points(
            sample["gt_points"]
        )
        points_inside = in_bounds_points(
            raw_points,
            height=H,
            width=W,
        )

        gt_count = float(
            sample["gt_count"].item()
        )
        gt_in_bounds = float(
            len(points_inside)
        )
        gt_out_of_bounds = (
            gt_count
            - gt_in_bounds
        )

        # ---------------------------------------------------------
        # Full-Direct
        # ---------------------------------------------------------
        pred_direct_count, direct_field = (
            model.predict(
                image,
                pad_multiple=(
                    args.direct_pad_multiple
                ),
            )
        )

        pred_direct_count = float(
            torch.as_tensor(
                pred_direct_count
            ).item()
        )

        if (
            model.head_type
            in {
                "cumulative",
                "integrated_local",
            }
        ):
            c_direct = (
                direct_field.float()
            )
            y_direct = (
                discrete_mixed_difference(
                    c_direct
                )
            )
        else:
            y_direct = (
                direct_field.float()
            )
            c_direct = (
                cell_counts_to_cumulative_field(
                    y_direct,
                    orientation="TL",
                )
            )

        # ---------------------------------------------------------
        # Window + Full-Tiled
        # ---------------------------------------------------------
        (
            pred_tiled_count,
            window_rows,
            c_tiled,
            y_tiled,
        ) = predict_windows_and_tiled(
            model=model,
            image=image,
            points_in_bounds=(
                points_inside
            ),
            tile_size=args.tile_size,
            halo=args.halo,
        )

        # ---------------------------------------------------------
        # Shared exact GT count measure on output grid
        # ---------------------------------------------------------
        out_h = c_direct.shape[-2]
        out_w = c_direct.shape[-1]

        if (
            c_tiled.shape[-2:]
            != (out_h, out_w)
        ):
            raise RuntimeError(
                "Direct/tiled output-grid mismatch: "
                f"{c_direct.shape[-2:]} "
                f"vs {c_tiled.shape[-2:]}"
            )

        gt_y_2d = points_to_count_map(
            points_inside,
            out_h=out_h,
            out_w=out_w,
            stride=stride,
            device=device,
            dtype=torch.float32,
        )

        gt_y = (
            gt_y_2d
            .unsqueeze(0)
            .unsqueeze(0)
        )

        gt_c = (
            cell_counts_to_cumulative_field(
                gt_y,
                orientation="TL",
            )
        )

        if not math.isclose(
            float(gt_y.sum().item()),
            gt_in_bounds,
            rel_tol=0.0,
            abs_tol=1e-5,
        ):
            raise RuntimeError(
                "GT measure does not conserve "
                "in-bounds count."
            )

        # ---------------------------------------------------------
        # Optional repository tiled equivalence
        # ---------------------------------------------------------
        repo_tiled_count = float("nan")
        repo_tiled_diff = float("nan")

        if args.verify_repo_tiled:
            repo_tiled, _ = (
                model.predict_tiled(
                    image,
                    tile_size=(
                        args.tile_size
                    ),
                    halo=args.halo,
                )
            )

            repo_tiled_count = float(
                torch.as_tensor(
                    repo_tiled
                ).item()
            )
            repo_tiled_diff = abs(
                repo_tiled_count
                - pred_tiled_count
            )

            if (
                repo_tiled_diff
                > args.strict_tiled_tolerance
            ):
                raise RuntimeError(
                    "New tiled evaluator differs "
                    "from model.predict_tiled(): "
                    f"ours={pred_tiled_count:.6f}, "
                    f"repo={repo_tiled_count:.6f}"
                )

        # ---------------------------------------------------------
        # Window bookkeeping + cancellation
        # ---------------------------------------------------------
        image_signed_errors: list[
            float
        ] = []

        for local_idx, row in enumerate(
            window_rows
        ):
            enriched = {
                "image_index": image_index,
                "window_index": local_idx,
                "image_path": sample[
                    "img_path"
                ],
                "image_height": H,
                "image_width": W,
                **row,
            }

            all_window_rows.append(
                enriched
            )
            image_signed_errors.append(
                float(
                    row["signed_error"]
                )
            )

        local_abs_sum = float(
            sum(
                abs(e)
                for e in image_signed_errors
            )
        )
        net_abs = abs(
            float(
                sum(
                    image_signed_errors
                )
            )
        )

        if local_abs_sum > 1e-12:
            cancel = (
                1.0
                - net_abs
                / local_abs_sum
            )
            cancel = float(
                min(
                    1.0,
                    max(
                        0.0,
                        cancel,
                    ),
                )
            )
        else:
            cancel = 0.0

        cancellation_ratios.append(
            cancel
        )

        sum_window_gt = float(
            sum(
                float(r["gt_count"])
                for r in window_rows
            )
        )
        sum_window_pred = float(
            sum(
                float(r["pred_count"])
                for r in window_rows
            )
        )

        if not math.isclose(
            sum_window_gt,
            gt_in_bounds,
            rel_tol=0.0,
            abs_tol=1e-5,
        ):
            raise RuntimeError(
                "Window GT partition mismatch."
            )

        if not math.isclose(
            sum_window_pred,
            pred_tiled_count,
            rel_tol=1e-5,
            abs_tol=1e-3,
        ):
            raise RuntimeError(
                "Window prediction partition "
                "mismatch."
            )

        # ---------------------------------------------------------
        # GAME
        # ---------------------------------------------------------
        direct_game = game_errors(
            y_direct,
            gt_y,
            args.game_levels,
        )
        tiled_game = game_errors(
            y_tiled,
            gt_y,
            args.game_levels,
        )

        for level in args.game_levels:
            level = int(level)
            direct_game_by_level[
                level
            ].append(
                direct_game[level]
            )
            tiled_game_by_level[
                level
            ].append(
                tiled_game[level]
            )

        # ---------------------------------------------------------
        # MICF validity / representation
        # ---------------------------------------------------------
        direct_valid = (
            measure_validity_metrics(
                y_direct
            )
        )
        tiled_valid = (
            measure_validity_metrics(
                y_tiled
            )
        )

        direct_repr = (
            representation_metrics(
                c_direct,
                y_direct,
                gt_c,
                gt_y,
                gt_in_bounds,
            )
        )
        tiled_repr = (
            representation_metrics(
                c_tiled,
                y_tiled,
                gt_c,
                gt_y,
                gt_in_bounds,
            )
        )

        direct_validity_rows.append(
            direct_valid
        )
        tiled_validity_rows.append(
            tiled_valid
        )
        direct_repr_rows.append(
            direct_repr
        )
        tiled_repr_rows.append(
            tiled_repr
        )

        # ---------------------------------------------------------
        # Per-image row
        # ---------------------------------------------------------
        row: dict[str, Any] = {
            "image_index": image_index,
            "image_path": sample[
                "img_path"
            ],
            "height": H,
            "width": W,
            "num_windows": len(
                window_rows
            ),
            "gt_count": gt_count,
            "gt_in_bounds_count": (
                gt_in_bounds
            ),
            "gt_out_of_bounds": (
                gt_out_of_bounds
            ),
            "pred_full_direct": (
                pred_direct_count
            ),
            "err_full_direct_signed": (
                pred_direct_count
                - gt_count
            ),
            "err_full_direct_abs": abs(
                pred_direct_count
                - gt_count
            ),
            "pred_full_tiled": (
                pred_tiled_count
            ),
            "err_full_tiled_signed": (
                pred_tiled_count
                - gt_count
            ),
            "err_full_tiled_abs": abs(
                pred_tiled_count
                - gt_count
            ),
            "cancellation_ratio": (
                cancel
            ),
            "window_abs_error_sum": (
                local_abs_sum
            ),
            "window_net_abs_error": (
                net_abs
            ),
            "sum_window_gt": (
                sum_window_gt
            ),
            "sum_window_predictions": (
                sum_window_pred
            ),
            "repo_tiled_count": (
                repo_tiled_count
            ),
            "repo_tiled_abs_diff": (
                repo_tiled_diff
            ),
        }

        for level, value in (
            direct_game.items()
        ):
            row[
                f"direct_game_L{level}"
            ] = value

        for level, value in (
            tiled_game.items()
        ):
            row[
                f"tiled_game_L{level}"
            ] = value

        for key, value in (
            direct_valid.items()
        ):
            row[
                f"direct_{key}"
            ] = value

        for key, value in (
            tiled_valid.items()
        ):
            row[
                f"tiled_{key}"
            ] = value

        for key, value in (
            direct_repr.items()
        ):
            row[
                f"direct_{key}"
            ] = value

        for key, value in (
            tiled_repr.items()
        ):
            row[
                f"tiled_{key}"
            ] = value

        per_image_rows.append(
            row
        )

        direct_predictions.append(
            pred_direct_count
        )
        tiled_predictions.append(
            pred_tiled_count
        )
        full_gt_counts.append(
            gt_count
        )

        print(
            f"[{image_index + 1:03d}/"
            f"{n_eval:03d}] "
            f"GT={gt_count:.1f} | "
            f"Direct={pred_direct_count:.2f} "
            f"(AE={abs(pred_direct_count-gt_count):.2f}) | "
            f"Tiled={pred_tiled_count:.2f} "
            f"(AE={abs(pred_tiled_count-gt_count):.2f}) | "
            f"Cancel={100*cancel:.1f}%"
        )

    # -----------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------
    direct_stats = count_metric_summary(
        direct_predictions,
        full_gt_counts,
        eps=1.0,
    )
    tiled_stats = count_metric_summary(
        tiled_predictions,
        full_gt_counts,
        eps=1.0,
    )
    window_stats = window_metric_summary(
        all_window_rows
    )

    summary: dict[str, Any] = {
        "checkpoint": str(
            checkpoint_path.resolve()
        ),
        "checkpoint_epoch": (
            checkpoint.get("epoch")
        ),
        "checkpoint_best_mae_stored": (
            checkpoint.get(
                "best_mae"
            )
        ),
        "model_head_type": (
            model.head_type
        ),
        "model_output_stride": (
            model.output_stride
        ),
        "model_finite_horizon": (
            model.finite_horizon
        ),
        "dataset_split": args.split,
        "num_images": n_eval,
        "num_windows": len(
            all_window_rows
        ),
        "tile_size": args.tile_size,
        "halo": args.halo,
        "game_levels": [
            int(x)
            for x in args.game_levels
        ],
        "window": window_stats,
        "full_tiled": tiled_stats,
        "full_direct": direct_stats,
        "direct_minus_tiled": {
            "mae_gap": (
                direct_stats["mae"]
                - tiled_stats["mae"]
            ),
            "rmse_gap": (
                direct_stats["rmse"]
                - tiled_stats["rmse"]
            ),
            "nae_gap": (
                direct_stats["nae"]
                - tiled_stats["nae"]
            ),
            "sre_gap": (
                direct_stats["sre"]
                - tiled_stats["sre"]
            ),
        },
        "cancellation": {
            "mean": finite_mean(
                cancellation_ratios
            ),
            "median": (
                finite_percentile(
                    cancellation_ratios,
                    50,
                )
            ),
            "p90": (
                finite_percentile(
                    cancellation_ratios,
                    90,
                )
            ),
        },
        "game_direct": {
            f"L{level}": finite_mean(
                values
            )
            for level, values
            in direct_game_by_level.items()
        },
        "game_tiled": {
            f"L{level}": finite_mean(
                values
            )
            for level, values
            in tiled_game_by_level.items()
        },
        "micf_validity_direct": {
            key: finite_mean(
                row[key]
                for row
                in direct_validity_rows
            )
            for key in (
                direct_validity_rows[0].keys()
                if direct_validity_rows
                else []
            )
        },
        "micf_validity_tiled": {
            key: finite_mean(
                row[key]
                for row
                in tiled_validity_rows
            )
            for key in (
                tiled_validity_rows[0].keys()
                if tiled_validity_rows
                else []
            )
        },
        "representation_direct": {
            key: finite_mean(
                row[key]
                for row
                in direct_repr_rows
            )
            for key in (
                direct_repr_rows[0].keys()
                if direct_repr_rows
                else []
            )
        },
        "representation_tiled": {
            key: finite_mean(
                row[key]
                for row
                in tiled_repr_rows
            )
            for key in (
                tiled_repr_rows[0].keys()
                if tiled_repr_rows
                else []
            )
        },
    }

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------
    summary_path = (
        output_dir
        / "comprehensive_summary.json"
    )
    image_csv = (
        output_dir
        / "comprehensive_per_image.csv"
    )
    window_csv = (
        output_dir
        / "comprehensive_per_window.csv"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            allow_nan=True,
        )

    write_csv(
        image_csv,
        per_image_rows,
    )
    write_csv(
        window_csv,
        all_window_rows,
    )

    # -----------------------------------------------------------------
    # Console report
    # -----------------------------------------------------------------
    print()
    print("=" * 96)
    print(
        "STANDARD COUNT METRICS"
    )
    print("=" * 96)
    print(
        f"{'Metric':<18}"
        f"{'Window':>14}"
        f"{'Tiled':>14}"
        f"{'Direct':>14}"
    )
    print("-" * 60)

    print(
        f"{'MAE/PMAE':<18}"
        f"{window_stats.get('mae', float('nan')):>14.4f}"
        f"{tiled_stats['mae']:>14.4f}"
        f"{direct_stats['mae']:>14.4f}"
    )
    print(
        f"{'RMSE/PRMSE':<18}"
        f"{window_stats.get('rmse', float('nan')):>14.4f}"
        f"{tiled_stats['rmse']:>14.4f}"
        f"{direct_stats['rmse']:>14.4f}"
    )
    print(
        f"{'NAE':<18}"
        f"{window_stats.get('nae_nonzero', float('nan')):>14.4f}"
        f"{tiled_stats['nae']:>14.4f}"
        f"{direct_stats['nae']:>14.4f}"
    )
    print(
        f"{'SRE':<18}"
        f"{'n/a':>14}"
        f"{tiled_stats['sre']:>14.4f}"
        f"{direct_stats['sre']:>14.4f}"
    )
    print(
        f"{'Signed Bias':<18}"
        f"{window_stats.get('signed_bias', float('nan')):>14.4f}"
        f"{tiled_stats['signed_bias']:>14.4f}"
        f"{direct_stats['signed_bias']:>14.4f}"
    )

    print()
    print("=" * 96)
    print(
        "TAIL / FAILURE METRICS"
    )
    print("=" * 96)

    for key, label in (
        ("median_ae", "Median AE"),
        ("p90_ae", "P90 AE"),
        ("p95_ae", "P95 AE"),
        ("max_ae", "Max AE"),
    ):
        print(
            f"{label:<18}"
            f"{window_stats.get(key, float('nan')):>14.4f}"
            f"{tiled_stats[key]:>14.4f}"
            f"{direct_stats[key]:>14.4f}"
        )

    print()
    print("=" * 96)
    print(
        "LOCAL / PATCH DIAGNOSTICS"
    )
    print("=" * 96)
    print(
        f"Window PMAE                 : "
        f"{window_stats.get('pmae', float('nan')):.4f}"
    )
    print(
        f"Window PRMSE                : "
        f"{window_stats.get('prmse', float('nan')):.4f}"
    )
    print(
        f"Window NAE (GT>0 only)      : "
        f"{window_stats.get('nae_nonzero', float('nan')):.4f}"
    )
    print(
        f"Non-empty Window MAE        : "
        f"{window_stats.get('nonempty_window_mae', float('nan')):.4f}"
    )
    print(
        f"Empty Window MAE            : "
        f"{window_stats.get('empty_window_mae', float('nan')):.4f}"
    )
    print(
        f"Empty Window Mean Pred      : "
        f"{window_stats.get('empty_window_mean_prediction', float('nan')):.4f}"
    )
    print(
        f"Empty Window Fraction       : "
        f"{100*window_stats.get('empty_window_fraction', float('nan')):.2f}%"
    )
    print(
        f"Mean Cancellation Ratio     : "
        f"{100*summary['cancellation']['mean']:.2f}%"
    )

    print()
    print("=" * 96)
    print(
        "GAME SPATIAL ERROR"
    )
    print("=" * 96)
    for level in args.game_levels:
        level = int(level)
        print(
            f"GAME({level}) "
            f"Tiled={summary['game_tiled'][f'L{level}']:.4f} "
            f"| Direct={summary['game_direct'][f'L{level}']:.4f}"
        )

    print()
    print("=" * 96)
    print(
        "MICF VALIDITY"
    )
    print("=" * 96)

    vd = summary[
        "micf_validity_direct"
    ]
    vt = summary[
        "micf_validity_tiled"
    ]

    for key in (
        "violation_rate",
        "violation_magnitude",
        "negative_mass_ratio",
        "negative_mass_total",
    ):
        print(
            f"{key:<28}"
            f"Tiled={vt.get(key, float('nan')):.6f} "
            f"| Direct={vd.get(key, float('nan')):.6f}"
        )

    print()
    print("=" * 96)
    print(
        "REPRESENTATION DIAGNOSTICS"
    )
    print("=" * 96)

    rd = summary[
        "representation_direct"
    ]
    rt = summary[
        "representation_tiled"
    ]

    for key in (
        "cumulative_field_nmae",
        "measure_nl1",
        "conservation_error",
    ):
        print(
            f"{key:<28}"
            f"Tiled={rt.get(key, float('nan')):.6f} "
            f"| Direct={rd.get(key, float('nan')):.6f}"
        )

    print()
    print("=" * 96)
    print(
        "DIRECT - TILED GAPS"
    )
    print("=" * 96)
    for key, value in (
        summary[
            "direct_minus_tiled"
        ].items()
    ):
        print(
            f"{key:<28}: "
            f"{value:.6f}"
        )

    print()
    print(f"Summary    : {summary_path}")
    print(f"Per-image  : {image_csv}")
    print(f"Per-window : {window_csv}")


if __name__ == "__main__":
    main()
