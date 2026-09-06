from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


def _points_array(
    points: np.ndarray,
) -> np.ndarray:
    arr = np.asarray(
        points,
        dtype=np.float64,
    )

    if arr.size == 0:
        return np.empty(
            (0, 2),
            dtype=np.float64,
        )

    return arr.reshape(-1, 2)


def _cell_actual_bounds(
    row: int,
    col: int,
    *,
    stride: int,
    image_h: int,
    image_w: int,
) -> tuple[float, float, float, float]:
    y0 = float(
        row * stride
    )
    x0 = float(
        col * stride
    )

    y1 = float(
        min(
            (row + 1) * stride,
            image_h,
        )
    )

    x1 = float(
        min(
            (col + 1) * stride,
            image_w,
        )
    )

    return (
        x0,
        y0,
        x1,
        y1,
    )


def _intersection_area(
    a_x0: float,
    a_y0: float,
    a_x1: float,
    a_y1: float,
    b_x0: float,
    b_y0: float,
    b_x1: float,
    b_y1: float,
) -> float:
    w = max(
        0.0,
        min(a_x1, b_x1)
        - max(a_x0, b_x0),
    )

    h = max(
        0.0,
        min(a_y1, b_y1)
        - max(a_y0, b_y0),
    )

    return float(
        w * h
    )


def _clipped_cell_edges_1d(length_px: int, n_cells: int, stride: int) -> np.ndarray:
    edges = np.arange(n_cells + 1, dtype=np.float64) * float(stride)
    edges = np.minimum(edges, float(length_px))
    edges[-1] = float(length_px)
    return edges


def region_count_from_cell_measure(
    pred_y: torch.Tensor | np.ndarray,
    *,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    image_h: int,
    image_w: int,
    stride: int,
) -> float:
    """
    Integrate a cell-count measure over a pixel-space rectangle.

    Each output cell's total mass is uniformly distributed
    over its actual in-image support.

    This is an evaluation convention used only for GAME.
    """
    if isinstance(pred_y, torch.Tensor):
        if pred_y.ndim == 4:
            if pred_y.shape[0] != 1 or pred_y.shape[1] != 1:
                raise ValueError(f"Expected [1,1,H,W], got {tuple(pred_y.shape)}")
            y = pred_y[0, 0].detach().to(dtype=torch.float64).cpu().numpy()
        elif pred_y.ndim == 2:
            y = pred_y.detach().to(dtype=torch.float64).cpu().numpy()
        else:
            raise ValueError(f"Expected 2D or [1,1,H,W], got {tuple(pred_y.shape)}")
    else:
        y = np.asarray(pred_y, dtype=np.float64)

    out_h, out_w = y.shape
    x_edges = _clipped_cell_edges_1d(image_w, out_w, stride)
    y_edges = _clipped_cell_edges_1d(image_h, out_h, stride)

    cell_w = x_edges[1:] - x_edges[:-1]
    cell_h = y_edges[1:] - y_edges[:-1]

    overlap_x = np.maximum(0.0, np.minimum(float(x1), x_edges[1:]) - np.maximum(float(x0), x_edges[:-1]))
    wx = overlap_x / cell_w

    overlap_y = np.maximum(0.0, np.minimum(float(y1), y_edges[1:]) - np.maximum(float(y0), y_edges[:-1]))
    wy = overlap_y / cell_h

    return float(wy @ y @ wx)


def _gt_region_count(
    points: np.ndarray,
    *,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> float:
    if points.size == 0:
        return 0.0

    mask = (
        (points[:, 0] >= x0)
        & (points[:, 0] < x1)
        & (points[:, 1] >= y0)
        & (points[:, 1] < y1)
    )

    return float(mask.sum())


def game_errors_one_image(
    pred_y: torch.Tensor | np.ndarray,
    points: np.ndarray,
    *,
    image_h: int,
    image_w: int,
    stride: int,
    levels: Iterable[int] = (0, 1, 2, 3),
) -> dict[int, float]:
    if isinstance(pred_y, torch.Tensor):
        if pred_y.ndim == 4:
            y = pred_y[0, 0].detach().to(dtype=torch.float64).cpu().numpy()
        elif pred_y.ndim == 2:
            y = pred_y.detach().to(dtype=torch.float64).cpu().numpy()
        else:
            raise ValueError(f"Expected 2D or [1,1,H,W], got {tuple(pred_y.shape)}")
    else:
        y = np.asarray(pred_y, dtype=np.float64)

    out_h, out_w = y.shape
    pts = _points_array(points)

    x_edges = _clipped_cell_edges_1d(image_w, out_w, stride)
    y_edges = _clipped_cell_edges_1d(image_h, out_h, stride)
    cell_w = x_edges[1:] - x_edges[:-1]
    cell_h = y_edges[1:] - y_edges[:-1]

    out: dict[int, float] = {}

    for level_raw in levels:
        level = int(level_raw)
        if level < 0:
            raise ValueError("GAME level must be >= 0")

        parts = 2 ** level
        part_x_edges = np.linspace(0.0, float(image_w), parts + 1)
        part_y_edges = np.linspace(0.0, float(image_h), parts + 1)

        # Vectorized 1D overlap weights:
        # Shape [parts, out_w]
        p_x0 = part_x_edges[:-1, None]
        p_x1 = part_x_edges[1:, None]
        c_x0 = x_edges[None, :-1]
        c_x1 = x_edges[None, 1:]
        overlap_x = np.maximum(0.0, np.minimum(p_x1, c_x1) - np.maximum(p_x0, c_x0))
        W_X = overlap_x / cell_w[None, :]

        # Shape [parts, out_h]
        p_y0 = part_y_edges[:-1, None]
        p_y1 = part_y_edges[1:, None]
        c_y0 = y_edges[None, :-1]
        c_y1 = y_edges[None, 1:]
        overlap_y = np.maximum(0.0, np.minimum(p_y1, c_y1) - np.maximum(p_y0, c_y0))
        W_Y = overlap_y / cell_h[None, :]

        # Predicted count per GAME partition rectangle: shape [parts, parts]
        pred_counts = W_Y @ y @ W_X.T

        # GT count per GAME partition rectangle: shape [parts, parts]
        gt_counts = np.zeros((parts, parts), dtype=np.float64)
        if pts.size > 0:
            for pt in pts:
                px, py = pt[0], pt[1]
                if 0.0 <= px < float(image_w) and 0.0 <= py < float(image_h):
                    bx = int(np.clip(np.searchsorted(part_x_edges[1:], px, side="right"), 0, parts - 1))
                    by = int(np.clip(np.searchsorted(part_y_edges[1:], py, side="right"), 0, parts - 1))
                    gt_counts[by, bx] += 1.0

        total_abs_error = float(np.sum(np.abs(pred_counts - gt_counts)))
        out[level] = total_abs_error

    return out


def aggregate_game(
    per_image_game: list[dict[int, float]],
) -> dict[str, float]:
    if not per_image_game:
        return {}

    levels = sorted(
        set().union(
            *[
                set(x.keys())
                for x in per_image_game
            ]
        )
    )

    out: dict[str, float] = {}

    for level in levels:
        vals = [
            float(x[level])
            for x in per_image_game
            if level in x
        ]

        out[
            f"game_{level}"
        ] = float(
            np.mean(vals)
        )

    return out
