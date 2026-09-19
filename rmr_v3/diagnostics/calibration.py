from __future__ import annotations

from typing import Any, Sequence
import numpy as np
import scipy.stats

from .rows import resolve_scale_map


def _bin_subset(
    rate_vars: np.ndarray,
    abs_cnt_errs: np.ndarray,
    abs_rate_errs: np.ndarray,
    std_resids: np.ndarray,
    num_bins: int,
) -> tuple[dict[str, Any], list[dict]]:
    if len(rate_vars) == 0:
        return {
            "mean_std_residual": 0.0,
            "p50_std_residual": 0.0,
            "p90_std_residual": 0.0,
            "bins": [],
        }, []

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

    summary = {
        "mean_std_residual": float(np.mean(std_resids)),
        "p50_std_residual": float(np.percentile(std_resids, 50)),
        "p90_std_residual": float(np.percentile(std_resids, 90)),
        "bins": bin_summaries,
    }
    return summary, bin_summaries


def compute_uncertainty_calibration_bins(
    rows: list[dict],
    num_bins: int = 4,
    scale_map: dict[int, int | str] | None = None,
) -> dict[str, Any]:
    """Bin predicted rate variance into quantiles and report error and standardized residual."""
    if not rows:
        empty: dict[str, Any] = {
            "mean_std_residual": 0.0,
            "p50_std_residual": 0.0,
            "p90_std_residual": 0.0,
            "bins": [],
            "calibration_pooled": [],
            "calibration_16": [],
            "calibration_32": [],
            "calibration_64": [],
            "calibration_128": [],
            "mean_std_residual_16": 0.0,
            "mean_std_residual_32": 0.0,
            "mean_std_residual_64": 0.0,
            "mean_std_residual_128": 0.0,
        }
        return empty

    rate_vars = np.array([r["rate_variance"] for r in rows], dtype=np.float64)
    abs_cnt_errs = np.array([r["abs_count_error"] for r in rows], dtype=np.float64)
    abs_rate_errs = np.array([r["abs_rate_error"] for r in rows], dtype=np.float64)
    std_resids = np.array([r.get("std_residual", 0.0) for r in rows], dtype=np.float64)
    scale_ids = np.array([r["scale_id"] for r in rows], dtype=np.int64)

    pooled_summary, pooled_bins = _bin_subset(rate_vars, abs_cnt_errs, abs_rate_errs, std_resids, num_bins)

    out: dict[str, Any] = {
        "mean_std_residual": pooled_summary["mean_std_residual"],
        "p50_std_residual": pooled_summary["p50_std_residual"],
        "p90_std_residual": pooled_summary["p90_std_residual"],
        "bins": pooled_bins,
        "calibration_pooled": pooled_bins,
    }

    resolved_scale_map = resolve_scale_map(scale_ids, scale_map)
    for sid, s_px in resolved_scale_map.items():
        mask = scale_ids == sid
        if np.any(mask):
            s_sum, s_bins = _bin_subset(
                rate_vars[mask], abs_cnt_errs[mask], abs_rate_errs[mask], std_resids[mask], num_bins
            )
            out[f"calibration_{s_px}"] = s_bins
            out[f"mean_std_residual_{s_px}"] = s_sum["mean_std_residual"]
            out[f"p50_std_residual_{s_px}"] = s_sum["p50_std_residual"]
            out[f"p90_std_residual_{s_px}"] = s_sum["p90_std_residual"]
        else:
            out[f"calibration_{s_px}"] = []
            out[f"mean_std_residual_{s_px}"] = 0.0
            out[f"p50_std_residual_{s_px}"] = 0.0
            out[f"p90_std_residual_{s_px}"] = 0.0

    return out


def compute_nb_interval_coverage(
    rows: list[dict],
    nominal_levels: Sequence[float] = (0.50, 0.80, 0.95),
    eps: float = 1e-6,
    scale_map: dict[int, int | str] | None = None,
) -> dict[str, float]:
    """Compute empirical coverage and calibration gaps for theoretical Negative Binomial predictive intervals."""
    levels_pct = [int(round(a * 100)) for a in nominal_levels]
    empty_out: dict[str, float] = {}
    for pct in levels_pct:
        empty_out[f"coverage_{pct}"] = 0.0
        empty_out[f"calib_gap_{pct}"] = 0.0
        for s in (16, 32, 64, 128):
            empty_out[f"coverage_{pct}_{s}"] = 0.0
            empty_out[f"calib_gap_{pct}_{s}"] = 0.0
    if not rows:
        return empty_out

    mu = np.array([r["pred_count"] for r in rows], dtype=np.float64)
    disp = np.array([r["dispersion"] for r in rows], dtype=np.float64)
    gt = np.array([r["gt_count"] for r in rows], dtype=np.float64)
    scale_ids = np.array([r["scale_id"] for r in rows], dtype=np.int64)

    mu_pos = np.maximum(0.0, mu)
    disp_pos = np.maximum(eps, disp)
    p = np.clip(disp_pos / (disp_pos + mu_pos), 1e-8, 1.0)

    out: dict[str, float] = {}
    resolved_scale_map = resolve_scale_map(scale_ids, scale_map)

    for pct, alpha in zip(levels_pct, nominal_levels):
        alpha_val = float(alpha)
        q_low = scipy.stats.nbinom.ppf((1.0 - alpha_val) / 2.0, disp_pos, p)
        q_high = scipy.stats.nbinom.ppf(1.0 - (1.0 - alpha_val) / 2.0, disp_pos, p)
        hits = (gt >= q_low) & (gt <= q_high)

        cov_all = float(np.mean(hits))
        out[f"coverage_{pct}"] = cov_all
        out[f"calib_gap_{pct}"] = float(cov_all - alpha_val)

        for sid, s_px in resolved_scale_map.items():
            mask = scale_ids == sid
            if np.any(mask):
                cov_s = float(np.mean(hits[mask]))
                out[f"coverage_{pct}_{s_px}"] = cov_s
                out[f"calib_gap_{pct}_{s_px}"] = float(cov_s - alpha_val)
            else:
                out[f"coverage_{pct}_{s_px}"] = 0.0
                out[f"calib_gap_{pct}_{s_px}"] = 0.0

    return out


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
