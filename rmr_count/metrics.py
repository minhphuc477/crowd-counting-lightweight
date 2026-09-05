from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch

from .operators import RegionSet, regional_sum


def count_from_map(y: torch.Tensor) -> torch.Tensor:
    return y.sum(dim=(-2, -1))


def compute_nae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Canonical crowd-counting NAE (NWPU / UCF-QNRF standard).

    Samples with gt == 0 are excluded to avoid division by zero or arbitrary clamping.
    NAE = (1 / |{i: gt_i > 0}|) * sum_{gt_i > 0} (|pred_i - gt_i| / gt_i)
    """
    preds = np.asarray(predictions, dtype=np.float64).reshape(-1)
    gts = np.asarray(targets, dtype=np.float64).reshape(-1)
    if preds.size == 0:
        return float("nan")
    mask = gts > 0.0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(preds[mask] - gts[mask]) / gts[mask]))


def _clipped_cell_edges_1d(length_px: int, n_cells: int, stride: int) -> np.ndarray:
    edges = np.arange(n_cells + 1, dtype=np.float64) * float(stride)
    edges = np.minimum(edges, float(length_px))
    edges[-1] = float(length_px)
    return edges


def game_physical_image(
    pred_y: torch.Tensor | np.ndarray,
    points_xy: torch.Tensor | np.ndarray,
    image_h: int,
    image_w: int,
    stride: int = 4,
    levels: tuple[int, ...] = (0, 1, 2, 3),
) -> dict[int, float]:
    """Canonical physical-support GAME(L) on exact image dimensions and point coordinates.

    Matches hpc.metrics.game:
    - partitions the physical image into 2^L x 2^L regions
    - integrates cell-count measure with partial boundary overlap weights
    - exact GT counts directly from raw point coordinates in physical image space
    - rigorously satisfies GAME(0) == |pred - gt| (overall absolute error).
    """
    if isinstance(pred_y, torch.Tensor):
        if pred_y.ndim == 4:
            y = pred_y[0, 0].detach().to(dtype=torch.float64).cpu().numpy()
        elif pred_y.ndim == 3:
            y = pred_y[0].detach().to(dtype=torch.float64).cpu().numpy()
        elif pred_y.ndim == 2:
            y = pred_y.detach().to(dtype=torch.float64).cpu().numpy()
        else:
            raise ValueError(f"Expected 2D/3D/4D, got shape {tuple(pred_y.shape)}")
    else:
        y = np.asarray(pred_y, dtype=np.float64)

    if isinstance(points_xy, torch.Tensor):
        pts = points_xy.detach().cpu().numpy().reshape(-1, 2)
    else:
        pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)

    out_h, out_w = y.shape
    x_edges = _clipped_cell_edges_1d(image_w, out_w, stride)
    y_edges = _clipped_cell_edges_1d(image_h, out_h, stride)
    cell_w = np.maximum(x_edges[1:] - x_edges[:-1], 1e-8)
    cell_h = np.maximum(y_edges[1:] - y_edges[:-1], 1e-8)

    out: dict[int, float] = {}
    for level in levels:
        parts = 2 ** level
        part_x_edges = np.linspace(0.0, float(image_w), parts + 1)
        part_y_edges = np.linspace(0.0, float(image_h), parts + 1)

        p_x0 = part_x_edges[:-1, None]
        p_x1 = part_x_edges[1:, None]
        c_x0 = x_edges[None, :-1]
        c_x1 = x_edges[None, 1:]
        overlap_x = np.maximum(0.0, np.minimum(p_x1, c_x1) - np.maximum(p_x0, c_x0))
        W_X = overlap_x / cell_w[None, :]

        p_y0 = part_y_edges[:-1, None]
        p_y1 = part_y_edges[1:, None]
        c_y0 = y_edges[None, :-1]
        c_y1 = y_edges[None, 1:]
        overlap_y = np.maximum(0.0, np.minimum(p_y1, c_y1) - np.maximum(p_y0, c_y0))
        W_Y = overlap_y / cell_h[None, :]

        pred_counts = W_Y @ y @ W_X.T

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


# Legacy wrapper kept for backward compatibility in train validation loop
def game_single(pred: torch.Tensor, target: torch.Tensor, level: int) -> float:
    """Fast stride-grid GAME for lightweight validation monitoring."""
    if pred.ndim == 3:
        pred = pred[0]
    if target.ndim == 3:
        target = target[0]
    h, w = pred.shape
    n = 2 ** level
    ys = [round(i * h / n) for i in range(n + 1)]
    xs = [round(i * w / n) for i in range(n + 1)]
    err = 0.0
    for iy in range(n):
        for ix in range(n):
            p = pred[ys[iy]:ys[iy + 1], xs[ix]:xs[ix + 1]].sum()
            t = target[ys[iy]:ys[iy + 1], xs[ix]:xs[ix + 1]].sum()
            err += float((p - t).abs().item())
    return err


def summarize_predictions(rows: list[dict]) -> dict[str, float]:
    gt = np.asarray([r["gt"] for r in rows], dtype=np.float64)
    pred = np.asarray([r["pred"] for r in rows], dtype=np.float64)
    ae = np.abs(pred - gt)
    out = {
        "MAE": float(ae.mean()),
        "RMSE": float(np.sqrt(np.mean((pred - gt) ** 2))),
        "NAE": compute_nae(pred, gt),
        "Bias": float(np.mean(pred - gt)),
        "MedianAE": float(np.median(ae)),
        "P90AE": float(np.quantile(ae, 0.90)),
        "P95AE": float(np.quantile(ae, 0.95)),
        "MaxAE": float(ae.max(initial=0.0)),
    }
    for level in range(4):
        key = f"GAME{level}"
        vals = [r[key] for r in rows if key in r]
        if vals:
            out[key] = float(np.mean(vals))
    return out


def bootstrap_ci(
    values: np.ndarray,
    statistic=np.mean,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 123,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    n = len(values)
    stats = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[i] = statistic(values[idx])
    lo = np.quantile(stats, alpha / 2)
    hi = np.quantile(stats, 1 - alpha / 2)
    return float(lo), float(hi)


def density_stratified_mae(rows: list[dict]) -> dict[str, float]:
    bins = {
        "sparse_le100": lambda n: n <= 100,
        "mid_101_500": lambda n: 100 < n <= 500,
        "dense_gt500": lambda n: n > 500,
    }
    out = {}
    for name, fn in bins.items():
        vals = [abs(r["pred"] - r["gt"]) for r in rows if fn(r["gt"])]
        if vals:
            out[name] = float(np.mean(vals))
    return out
