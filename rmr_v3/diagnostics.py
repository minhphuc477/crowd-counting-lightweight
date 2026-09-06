from __future__ import annotations

import math
from typing import Sequence
import numpy as np
import torch

from rmr_count.operators import regional_sum


@torch.no_grad()
def regional_reliability_rows(
    outputs: dict,
    target_y: torch.Tensor,
) -> list[dict]:
    regions = outputs["regions"]

    mu = outputs["b_region"].float()
    disp = outputs["region_dispersion"].float()
    weight = outputs["region_weight"].float()
    rate_var = outputs["region_rate_variance"].float()

    gt = regional_sum(
        target_y.float(),
        regions.boxes,
        out_dtype=torch.float32,
    )

    area = regions.area.float().view(1, 1, -1)

    abs_count_error = (mu - gt).abs()
    abs_rate_error = (
        abs_count_error
        / area.clamp_min(1.0)
    )

    rows = []

    bsz = mu.shape[0]

    for bi in range(bsz):
        for ri in range(mu.shape[-1]):
            rows.append(
                {
                    "batch_index": bi,
                    "region_index": ri,
                    "scale_id": int(
                        regions.scale_id[ri].item()
                    ),
                    "area": float(
                        regions.area[ri].item()
                    ),
                    "gt_count": float(
                        gt[bi, 0, ri].item()
                    ),
                    "pred_count": float(
                        mu[bi, 0, ri].item()
                    ),
                    "dispersion": float(
                        disp[bi, 0, ri].item()
                    ),
                    "rate_variance": float(
                        rate_var[bi, 0, ri].item()
                    ),
                    "weight": float(
                        weight[bi, 0, ri].item()
                    ),
                    "abs_count_error": float(
                        abs_count_error[
                            bi, 0, ri
                        ].item()
                    ),
                    "abs_rate_error": float(
                        abs_rate_error[
                            bi, 0, ri
                        ].item()
                    ),
                }
            )

    return rows


def compute_reliability_correlations(rows: list[dict], eps: float = 1e-8) -> dict[str, float]:
    """Compute Pearson and Spearman correlations between uncertainty and error."""
    if not rows:
        return {
            "pearson_rate_var_error": 0.0,
            "spearman_rate_var_error": 0.0,
            "spearman_weight_error": 0.0,
        }

    rate_vars = np.array([r["rate_variance"] for r in rows], dtype=np.float64)
    weights = np.array([r["weight"] for r in rows], dtype=np.float64)
    errors = np.array([r["abs_rate_error"] for r in rows], dtype=np.float64)

    def _pearson(x: np.ndarray, y: np.ndarray) -> float:
        if len(x) < 2 or np.std(x) < eps or np.std(y) < eps:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    def _spearman(x: np.ndarray, y: np.ndarray) -> float:
        if len(x) < 2:
            return 0.0
        rx = np.argsort(np.argsort(x)).astype(np.float64)
        ry = np.argsort(np.argsort(y)).astype(np.float64)
        return _pearson(rx, ry)

    return {
        "pearson_rate_var_error": _pearson(rate_vars, errors),
        "spearman_rate_var_error": _spearman(rate_vars, errors),
        "spearman_weight_error": _spearman(weights, errors),
    }


@torch.no_grad()
def summarize_diagnostics(
    outputs: dict,
    target_y: torch.Tensor,
    density_bins: tuple[float, float] = (100.0, 500.0),
) -> dict:
    """Collect all diagnostic summaries for validation."""
    rows = regional_reliability_rows(outputs, target_y)
    corrs = compute_reliability_correlations(rows)

    # Weight distribution
    w = outputs["region_weight"].float()
    w_min = float(w.min().item())
    w_max = float(w.max().item())
    w_mean = float(w.mean().item())
    w_std = float(w.std().item())

    # Dispersion distribution
    d = outputs["region_dispersion"].float().cpu().numpy().flatten()
    d_mean = float(np.mean(d))
    d_p10 = float(np.percentile(d, 10))
    d_p50 = float(np.percentile(d, 50))
    d_p90 = float(np.percentile(d, 90))

    # Energy trace
    energy_trace = outputs.get("energy_trace", [])
    energy_reduction = 0.0
    if energy_trace:
        e_init = float(energy_trace[0]["before"].mean().item())
        e_final = float(energy_trace[-1]["after"].mean().item())
        if e_init > 1e-8:
            energy_reduction = (e_init - e_final) / e_init

    # Density bin MAE
    pred_y = outputs["y"]
    pred_cnt = pred_y.sum(dim=(-1, -2, -3)).float().cpu()
    gt_cnt = target_y.sum(dim=(-1, -2, -3)).float().cpu()
    ae = (pred_cnt - gt_cnt).abs()

    lo, hi = density_bins
    sparse = ae[gt_cnt <= lo]
    moderate = ae[(gt_cnt > lo) & (gt_cnt <= hi)]
    dense = ae[gt_cnt > hi]

    def _mean_or_nan(t: torch.Tensor) -> float:
        return float(t.mean().item()) if t.numel() > 0 else float("nan")

    return {
        "correlations": corrs,
        "weight_stats": {
            "mean": w_mean,
            "std": w_std,
            "min": w_min,
            "max": w_max,
        },
        "dispersion_stats": {
            "mean": d_mean,
            "p10": d_p10,
            "p50": d_p50,
            "p90": d_p90,
        },
        "energy_reduction": energy_reduction,
        "density_bin_mae": {
            "sparse": _mean_or_nan(sparse),
            "moderate": _mean_or_nan(moderate),
            "dense": _mean_or_nan(dense),
        },
    }
