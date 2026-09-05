from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch

from .operators import RegionSet, regional_sum


def count_from_map(y: torch.Tensor) -> torch.Tensor:
    return y.sum(dim=(-2, -1))


def game_single(pred: torch.Tensor, target: torch.Tensor, level: int) -> float:
    """Mass-preserving GAME(L) on one [1,H,W] count map."""
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
        "NAE": float(np.mean(ae / np.maximum(gt, 1.0))),
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
