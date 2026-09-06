from __future__ import annotations

import math
from typing import Any, Sequence
import numpy as np
import scipy.stats
import torch

from rmr_core.operators import regional_sum


@torch.no_grad()
def regional_reliability_rows(
    outputs: dict,
    target_y: torch.Tensor,
) -> list[dict]:
    """Extract fine-grained regional reliability and prediction diagnostics."""
    regions = outputs["regions"]

    mu = outputs["b_region"].float()
    disp = outputs["region_dispersion"].float()
    weight = outputs["region_weight"].float()
    solver_weight = outputs.get("solver_region_weight", weight).float()
    rate_var = outputs["region_rate_variance"].float()
    count_var = outputs.get("region_count_variance")
    if count_var is None:
        count_var = mu + mu.square() / disp.clamp_min(1e-6)
    count_var = count_var.float()

    gt = regional_sum(
        target_y.float(),
        regions.boxes,
        out_dtype=torch.float32,
    )

    area = regions.area.float().view(1, 1, -1).clamp_min(1.0)

    abs_count_error = (mu - gt).abs()
    abs_rate_error = abs_count_error / area
    std_residual = abs_count_error / count_var.clamp_min(1e-6).sqrt()

    rows = []
    bsz = mu.shape[0]

    for bi in range(bsz):
        for ri in range(mu.shape[-1]):
            rows.append(
                {
                    "batch_index": bi,
                    "region_index": ri,
                    "scale_id": int(regions.scale_id[ri].item()),
                    "area": float(regions.area[ri].item()),
                    "gt_count": float(gt[bi, 0, ri].item()),
                    "pred_count": float(mu[bi, 0, ri].item()),
                    "dispersion": float(disp[bi, 0, ri].item()),
                    "count_variance": float(count_var[bi, 0, ri].item()),
                    "rate_variance": float(rate_var[bi, 0, ri].item()),
                    "weight": float(weight[bi, 0, ri].item()),
                    "solver_weight": float(solver_weight[bi, 0, ri].item()),
                    "abs_count_error": float(abs_count_error[bi, 0, ri].item()),
                    "abs_rate_error": float(abs_rate_error[bi, 0, ri].item()),
                    "std_residual": float(std_residual[bi, 0, ri].item()),
                }
            )

    return rows


def _safe_pearson(x: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> float:
    if len(x) < 2 or float(np.std(x)) < eps or float(np.std(y)) < eps:
        return 0.0
    val = float(np.corrcoef(x, y)[0, 1])
    return val if np.isfinite(val) else 0.0


def _safe_spearman(x: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> float:
    if len(x) < 2 or float(np.std(x)) < eps or float(np.std(y)) < eps:
        return 0.0
    res = scipy.stats.spearmanr(x, y)
    stat = getattr(res, "statistic", getattr(res, "correlation", 0.0))
    return float(stat) if np.isfinite(stat) else 0.0


def compute_reliability_correlations(
    rows: list[dict],
    eps: float = 1e-8,
) -> dict[str, float]:
    """Compute Pearson and Spearman correlations between uncertainty and error.

    Includes global correlations and per-scale correlations (32px, 64px, 128px),
    as well as disentangled solver weight vs predicted weight correlations.
    """
    if not rows:
        out = {
            "pearson_rate_var_error": 0.0,
            "spearman_rate_var_error": 0.0,
            "spearman_weight_error": 0.0,
            "spearman_pred_weight_error": 0.0,
        }
        for s in (32, 64, 128):
            out[f"pearson_rate_var_error_{s}"] = 0.0
            out[f"spearman_rate_var_error_{s}"] = 0.0
            out[f"spearman_weight_error_{s}"] = 0.0
            out[f"spearman_pred_weight_error_{s}"] = 0.0
        return out

    rate_vars = np.array([r["rate_variance"] for r in rows], dtype=np.float64)
    solver_weights = np.array([r.get("solver_weight", r["weight"]) for r in rows], dtype=np.float64)
    pred_weights = np.array([r["weight"] for r in rows], dtype=np.float64)
    errors = np.array([r["abs_rate_error"] for r in rows], dtype=np.float64)
    scale_ids = np.array([r["scale_id"] for r in rows], dtype=np.int64)

    results = {
        "pearson_rate_var_error": _safe_pearson(rate_vars, errors, eps),
        "spearman_rate_var_error": _safe_spearman(rate_vars, errors, eps),
        "spearman_weight_error": _safe_spearman(solver_weights, errors, eps),
        "spearman_pred_weight_error": _safe_spearman(pred_weights, errors, eps),
    }

    scale_map = {0: 32, 1: 64, 2: 128}
    for sid, s_px in scale_map.items():
        mask = scale_ids == sid
        if np.any(mask):
            rv_s = rate_vars[mask]
            err_s = errors[mask]
            sw_s = solver_weights[mask]
            pw_s = pred_weights[mask]
            results[f"pearson_rate_var_error_{s_px}"] = _safe_pearson(rv_s, err_s, eps)
            results[f"spearman_rate_var_error_{s_px}"] = _safe_spearman(rv_s, err_s, eps)
            results[f"spearman_weight_error_{s_px}"] = _safe_spearman(sw_s, err_s, eps)
            results[f"spearman_pred_weight_error_{s_px}"] = _safe_spearman(pw_s, err_s, eps)
        else:
            results[f"pearson_rate_var_error_{s_px}"] = 0.0
            results[f"spearman_rate_var_error_{s_px}"] = 0.0
            results[f"spearman_weight_error_{s_px}"] = 0.0
            results[f"spearman_pred_weight_error_{s_px}"] = 0.0

    return results


def compute_uncertainty_calibration_bins(
    rows: list[dict],
    num_bins: int = 4,
) -> dict[str, Any]:
    """Bin predicted rate variance into quantiles (e.g. quartiles) and report error and standardized residual."""
    if not rows:
        return {
            "mean_std_residual": 0.0,
            "p50_std_residual": 0.0,
            "p90_std_residual": 0.0,
            "bins": [],
        }

    rate_vars = np.array([r["rate_variance"] for r in rows], dtype=np.float64)
    abs_cnt_errs = np.array([r["abs_count_error"] for r in rows], dtype=np.float64)
    abs_rate_errs = np.array([r["abs_rate_error"] for r in rows], dtype=np.float64)
    std_resids = np.array([r.get("std_residual", 0.0) for r in rows], dtype=np.float64)

    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(rate_vars, quantiles)

    bin_summaries = []
    for k in range(num_bins):
        low_edge = bin_edges[k]
        high_edge = bin_edges[k + 1]
        if k == num_bins - 1:
            mask = (rate_vars >= low_edge) & (rate_vars <= high_edge)
        else:
            mask = (rate_vars >= low_edge) & (rate_vars < high_edge)

        if np.any(mask):
            bin_summaries.append({
                "bin_index": k,
                "var_min": float(np.min(rate_vars[mask])),
                "var_max": float(np.max(rate_vars[mask])),
                "count": int(np.sum(mask)),
                "mean_abs_count_error": float(np.mean(abs_cnt_errs[mask])),
                "mean_abs_rate_error": float(np.mean(abs_rate_errs[mask])),
                "mean_std_residual": float(np.mean(std_resids[mask])),
            })
        else:
            bin_summaries.append({
                "bin_index": k,
                "var_min": float(low_edge),
                "var_max": float(high_edge),
                "count": 0,
                "mean_abs_count_error": 0.0,
                "mean_abs_rate_error": 0.0,
                "mean_std_residual": 0.0,
            })

    return {
        "mean_std_residual": float(np.mean(std_resids)),
        "p50_std_residual": float(np.percentile(std_resids, 50)),
        "p90_std_residual": float(np.percentile(std_resids, 90)),
        "bins": bin_summaries,
    }


def compute_dispersion_saturation(
    rows: list[dict],
    disp_min: float = 0.5,
    disp_max: float = 500.0,
    eps: float = 1e-4,
) -> dict[str, float]:
    """Track fraction of regions saturated at dispersion boundaries."""
    if not rows:
        return {"dispersion_sat_low_fraction": 0.0, "dispersion_sat_high_fraction": 0.0}

    disps = np.array([r["dispersion"] for r in rows], dtype=np.float64)
    low_frac = float(np.mean(disps <= disp_min + eps))
    high_frac = float(np.mean(disps >= disp_max - eps))
    return {
        "dispersion_sat_low_fraction": low_frac,
        "dispersion_sat_high_fraction": high_frac,
    }


@torch.no_grad()
def compute_solver_trajectory_diagnostics(
    outputs: dict,
    target_y: torch.Tensor,
) -> dict[str, Any]:
    """Compute trajectory MAE over iterates Y_0 -> Y_1 -> Y_2, regional disagreement,

    energy monotonicity, and harmful-correction rate.
    """
    iterates = outputs.get("iterates", [])
    if not iterates:
        return {}

    regions = outputs["regions"]
    boxes = regions.boxes
    b_region = outputs["b_region"].float()
    target_float = target_y.float()

    gt_reg = regional_sum(target_float, boxes, out_dtype=torch.float32)

    scale_ids = regions.scale_id
    scale_map = {0: 32, 1: 64, 2: 128}

    results: dict[str, Any] = {}

    for t_idx, y_t in enumerate(iterates):
        reg_t = regional_sum(y_t.float(), boxes, out_dtype=torch.float32)
        abs_err = (reg_t - gt_reg).abs()
        results[f"mae_reg_y{t_idx}"] = float(abs_err.mean().item())

        for sid, s_px in scale_map.items():
            mask = scale_ids == sid
            if mask.any():
                results[f"mae_reg_{s_px}_y{t_idx}"] = float(abs_err[..., mask].mean().item())
            else:
                results[f"mae_reg_{s_px}_y{t_idx}"] = 0.0

        # Disagreement with predicted regional evidence mu
        disagree = (reg_t - b_region).abs()
        results[f"reg_disagreement_y{t_idx}"] = float(disagree.mean().item())

    # Energy monotonicity
    energy_trace = outputs.get("energy_trace", [])
    if energy_trace:
        mono_count = 0
        total_transitions = len(energy_trace)
        for step in energy_trace:
            e_before = float(step["before"].mean().item())
            e_after = float(step["after"].mean().item())
            if e_after <= e_before + 1e-8:
                mono_count += 1
        results["energy_monotonic_fraction"] = float(mono_count / max(1, total_transitions))
    else:
        results["energy_monotonic_fraction"] = 1.0

    # Image-level harmful correction rate: e_0 = |sum Y_0 - N*|, e_T = |sum Y_T - N*|
    y0 = iterates[0]
    y_final = iterates[-1]

    cnt0 = y0.sum(dim=(-1, -2, -3)).float().cpu()
    cnt_final = y_final.sum(dim=(-1, -2, -3)).float().cpu()
    gt_cnt = target_float.sum(dim=(-1, -2, -3)).float().cpu()

    e0 = (cnt0 - gt_cnt).abs()
    e_final = (cnt_final - gt_cnt).abs()
    delta_e = e_final - e0  # < 0 means solver helped, > 0 means solver hurt

    help_mask = delta_e < -1e-4
    harm_mask = delta_e > 1e-4
    neutral_mask = ~help_mask & ~harm_mask

    results["solver_help_fraction"] = float(help_mask.float().mean().item())
    results["solver_harm_fraction"] = float(harm_mask.float().mean().item())
    results["solver_neutral_fraction"] = float(neutral_mask.float().mean().item())
    results["solver_delta_e_mean"] = float(delta_e.mean().item())

    return results


@torch.no_grad()
def summarize_diagnostics(
    outputs: dict,
    target_y: torch.Tensor,
    density_bins: tuple[float, float] = (100.0, 500.0),
) -> dict:
    """Collect all diagnostic summaries for validation."""
    rows = regional_reliability_rows(outputs, target_y)
    corrs = compute_reliability_correlations(rows)
    calib = compute_uncertainty_calibration_bins(rows)
    sat = compute_dispersion_saturation(rows)
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
