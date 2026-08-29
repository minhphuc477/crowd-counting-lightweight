from typing import Dict, Optional, Union

import numpy as np
import torch

from .counting import compute_mae, compute_rmse, compute_nae


def _flat(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x).reshape(-1)


def evaluate_subgroup_diagnostics(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    luminances: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Dict[str, float]:
    """Custom robustness diagnostics; these are NOT official NWPU scene/illumination groups."""
    preds, gts = _flat(predictions), _flat(targets)
    if preds.shape != gts.shape:
        raise ValueError("predictions and targets must have the same shape")
    if len(preds) == 0:
        return {}

    results: Dict[str, float] = {}
    empty = gts == 0
    if np.any(empty):
        e = preds[empty]
        results.update(
            empty_count=int(empty.sum()),
            empty_mae=float(np.mean(np.abs(e))),
            empty_pred_mean=float(np.mean(e)),
            empty_pred_p95=float(np.percentile(e, 95)),
        )
    else:
        results.update(empty_count=0, empty_mae=0.0, empty_pred_mean=0.0, empty_pred_p95=0.0)

    for name, mask in [
        ("bin_0", gts == 0),
        ("bin_1_10", (gts >= 1) & (gts <= 10)),
        ("bin_11_100", (gts >= 11) & (gts <= 100)),
        ("bin_101_1000", (gts >= 101) & (gts <= 1000)),
        ("bin_gt1000", gts > 1000),
    ]:
        if np.any(mask):
            results[f"{name}_mae"] = compute_mae(preds[mask], gts[mask])
            results[f"{name}_rmse"] = compute_rmse(preds[mask], gts[mask])
            results[f"{name}_count"] = int(mask.sum())

    # NTPC paper diagnostics.  Boundaries are disjoint by construction:
    # sparse < 300, medium 300--999, dense >= 1000.
    for name, mask in [
        ("bin_sparse", gts < 300),
        ("bin_medium", (gts >= 300) & (gts < 1000)),
        ("bin_dense", gts >= 1000),
    ]:
        if np.any(mask):
            error = preds[mask] - gts[mask]
            results[f"{name}_mae"] = compute_mae(preds[mask], gts[mask])
            results[f"{name}_rmse"] = compute_rmse(preds[mask], gts[mask])
            results[f"{name}_bias"] = float(np.mean(error))
            results[f"{name}_count"] = int(mask.sum())

    threshold = np.percentile(gts, 90)
    top = gts >= threshold
    if np.any(top):
        results["top10_dense_mae"] = compute_mae(preds[top], gts[top])
        results["top10_dense_rmse"] = compute_rmse(preds[top], gts[top])

    if luminances is not None:
        lums = _flat(luminances)
        if len(lums) != len(gts):
            raise ValueError("luminances length must match predictions")
        q25, q50, q75 = np.percentile(lums, [25, 50, 75])
        masks = {
            "lum_q1_dark": lums <= q25,
            "lum_q2": (lums > q25) & (lums <= q50),
            "lum_q3": (lums > q50) & (lums <= q75),
            "lum_q4_bright": lums > q75,
        }
        for name, mask in masks.items():
            if np.any(mask):
                results[f"{name}_mae"] = compute_mae(preds[mask], gts[mask])
    return results


def evaluate_nwpu_official_groups(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    levels: Union[np.ndarray, torch.Tensor],
    illuminations: Union[np.ndarray, torch.Tensor],
) -> Dict[str, object]:
    """Reproduce NWPU official aggregation when official level/illum labels are available."""
    preds, gts = _flat(predictions), _flat(targets)
    levels, illuminations = _flat(levels).astype(int), _flat(illuminations).astype(int)
    if not (len(preds) == len(gts) == len(levels) == len(illuminations)):
        raise ValueError("All NWPU arrays must have equal length")

    def meter(mask):
        if not np.any(mask):
            return {"mae": float("nan"), "rmse": float("nan"), "nae": float("nan")}
        return {
            "mae": compute_mae(preds[mask], gts[mask]),
            "rmse": compute_rmse(preds[mask], gts[mask]),
            "nae": compute_nae(preds[mask], gts[mask], ignore_zero=True),
        }

    overall = meter(np.ones(len(gts), dtype=bool))
    level_stats = [meter(levels == i) for i in range(5)]
    illum_stats = [meter(illuminations == i) for i in range(4)]
    return {
        "overall": overall,
        "levels": level_stats,
        "illums": illum_stats,
        "mmae": {
            "mmae_level": float(np.nanmean([x["mae"] for x in level_stats])),
            "mmae_illum": float(np.nanmean([x["mae"] for x in illum_stats])),
        },
    }
