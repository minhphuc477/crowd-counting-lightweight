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


def region_count_from_cell_measure(
    pred_y: torch.Tensor,
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
    if pred_y.ndim == 4:
        if (
            pred_y.shape[0] != 1
            or pred_y.shape[1] != 1
        ):
            raise ValueError(
                f"Expected [1,1,H,W], got "
                f"{tuple(pred_y.shape)}"
            )
        y = (
            pred_y[0, 0]
            .detach()
            .to(dtype=torch.float64)
            .cpu()
            .numpy()
        )
    elif pred_y.ndim == 2:
        y = (
            pred_y.detach()
            .to(dtype=torch.float64)
            .cpu()
            .numpy()
        )
    else:
        raise ValueError(
            f"Expected 2D or [1,1,H,W], got "
            f"{tuple(pred_y.shape)}"
        )

    out_h, out_w = y.shape

    total = 0.0

    for r in range(out_h):
        for c in range(out_w):
            (
                cx0,
                cy0,
                cx1,
                cy1,
            ) = _cell_actual_bounds(
                r,
                c,
                stride=stride,
                image_h=image_h,
                image_w=image_w,
            )

            support_area = (
                (cx1 - cx0)
                * (cy1 - cy0)
            )

            if support_area <= 0.0:
                continue

            overlap = _intersection_area(
                cx0,
                cy0,
                cx1,
                cy1,
                x0,
                y0,
                x1,
                y1,
            )

            if overlap <= 0.0:
                continue

            total += float(
                y[r, c]
                * (
                    overlap
                    / support_area
                )
            )

    return float(total)


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

    return float(
        mask.sum()
    )


def game_errors_one_image(
    pred_y: torch.Tensor,
    points: np.ndarray,
    *,
    image_h: int,
    image_w: int,
    stride: int,
    levels: Iterable[int] = (0, 1, 2, 3),
) -> dict[int, float]:
    pts = _points_array(
        points
    )

    out: dict[int, float] = {}

    for level_raw in levels:
        level = int(
            level_raw
        )

        if level < 0:
            raise ValueError(
                "GAME level must be >= 0"
            )

        parts = 2 ** level

        x_edges = np.linspace(
            0.0,
            float(image_w),
            parts + 1,
        )

        y_edges = np.linspace(
            0.0,
            float(image_h),
            parts + 1,
        )

        total_abs_error = 0.0

        for r in range(parts):
            for c in range(parts):
                x0 = float(
                    x_edges[c]
                )
                x1 = float(
                    x_edges[c + 1]
                )
                y0 = float(
                    y_edges[r]
                )
                y1 = float(
                    y_edges[r + 1]
                )

                pred_count = (
                    region_count_from_cell_measure(
                        pred_y,
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        image_h=image_h,
                        image_w=image_w,
                        stride=stride,
                    )
                )

                gt_count = (
                    _gt_region_count(
                        pts,
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                    )
                )

                total_abs_error += abs(
                    pred_count
                    - gt_count
                )

        out[level] = float(
            total_abs_error
        )

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
