from __future__ import annotations

from typing import Any
import numpy as np
import scipy.stats
import torch

from rmr_core.operators import regional_sum


@torch.no_grad()
def regional_reliability_rows(
    outputs: dict,
    target_y: torch.Tensor,
    max_regions: int | None = None,
) -> list[dict]:
    """Extract fine-grained regional reliability and prediction diagnostics.

    Parameters
    ----------
    outputs : dict
        Model forward output dictionary containing regions, b_region, etc.
    target_y : torch.Tensor
        Ground truth density map (B, 1, H, W).
    max_regions : int | None, default=None
        If specified and total regions per sample exceeds this threshold,
        uniformly steps across regions to bound Python heap memory allocation
        while preserving statistical distribution properties (e.g. Pearson/Spearman).
    """
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

    pred_field = outputs.get("y", outputs.get("y0"))
    target_float = target_y.float()
    if pred_field is not None and target_float.shape[-2:] != pred_field.shape[-2:]:
        target_float = target_float[..., :pred_field.shape[-2], :pred_field.shape[-1]]

    gt = regional_sum(
        target_float,
        regions.boxes,
        out_dtype=torch.float32,
    )

    area = regions.area.float().view(1, 1, -1).clamp_min(1.0)

    abs_count_error = (mu - gt).abs()
    abs_rate_error = abs_count_error / area
    std_residual = abs_count_error / count_var.clamp_min(1e-6).sqrt()

    rows = []
    bsz = mu.shape[0]
    num_total_regions = mu.shape[-1]
    if max_regions is not None and num_total_regions > max_regions:
        region_indices = np.linspace(0, num_total_regions - 1, max_regions, dtype=int).tolist()
    else:
        region_indices = list(range(num_total_regions))

    for bi in range(bsz):
        for ri in region_indices:
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


def resolve_scale_map(
    scale_ids: np.ndarray | torch.Tensor | None = None,
    scale_map: dict[int, int | str] | None = None,
) -> dict[int, int | str]:
    """Resolve mapping from integer scale_id to physical pixel scale label (e.g. 16, 32, 64, 128)."""
    if scale_map is not None:
        return scale_map
    if scale_ids is None:
        return {0: 32, 1: 64, 2: 128}
    if isinstance(scale_ids, torch.Tensor):
        sids = scale_ids.detach().cpu().numpy()
    else:
        sids = np.asarray(scale_ids)
    unique_sids = sorted([int(s) for s in np.unique(sids) if s >= 0])
    if unique_sids == [0, 1, 2, 3, 4]:
        return {0: 16, 1: 32, 2: 64, 3: "64_32", 4: 128}
    if unique_sids == [0, 1, 2, 3]:
        return {0: 16, 1: 32, 2: 64, 3: 128}
    if unique_sids == [0, 1, 2]:
        return {0: 32, 1: 64, 2: 128}
    if not unique_sids:
        return {0: 32, 1: 64, 2: 128}
    return {sid: sid for sid in unique_sids}


def compute_reliability_correlations(
    rows: list[dict],
    eps: float = 1e-8,
    scale_map: dict[int, int | str] | None = None,
) -> dict[str, float]:
    """Compute Pearson and Spearman correlations between uncertainty and error."""
    if not rows:
        out = {
            "pearson_rate_var_error": 0.0,
            "spearman_rate_var_error": 0.0,
            "spearman_weight_error": 0.0,
            "spearman_pred_weight_error": 0.0,
        }
        for s in (16, 32, 64, 128):
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

    resolved_scale_map = resolve_scale_map(scale_ids, scale_map)
    for sid, s_px in resolved_scale_map.items():
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
