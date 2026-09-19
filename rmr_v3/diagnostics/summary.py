from __future__ import annotations

import numpy as np
import torch

from .calibration import (
    compute_dispersion_saturation,
    compute_nb_interval_coverage,
    compute_uncertainty_calibration_bins,
)
from .rows import (
    compute_reliability_correlations,
    regional_reliability_rows,
)
from .trajectory import compute_solver_trajectory_diagnostics


@torch.no_grad()
def summarize_diagnostics(
    outputs: dict,
    target_y: torch.Tensor,
    density_bins: tuple[float, float] = (100.0, 500.0),
    disp_min: float = 0.5,
    disp_max: float = 500.0,
) -> dict:
    """Collect all diagnostic summaries for validation."""
    rows = regional_reliability_rows(outputs, target_y)
    corrs = compute_reliability_correlations(rows)
    calib = compute_uncertainty_calibration_bins(rows)
    nb_cov = compute_nb_interval_coverage(rows)
    sat = compute_dispersion_saturation(rows, disp_min=disp_min, disp_max=disp_max)
    traj = compute_solver_trajectory_diagnostics(outputs, target_y)

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
        "calibration": calib,
        "nb_interval_coverage": nb_cov,
        "dispersion_saturation": sat,
        "solver_trajectory": traj,
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
