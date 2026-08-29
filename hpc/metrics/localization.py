"""Crowd Localization Metrics via Hungarian Bipartite Matching (P2PNet / STEERER / CLTR standard).

Evaluates:
  - Precision, Recall, and F1-score at distance thresholds sigma in {4, 8} pixels.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple, Union

import numpy as np
import scipy.ndimage as ndi
from scipy.optimize import linear_sum_assignment
import torch


def extract_points_from_mass_map(
    mass_map: Union[np.ndarray, torch.Tensor],
    stride: int = 4,
    threshold_rel: float = 0.05,
    threshold_abs: float = 0.01,
    min_distance_px: int = 4,
) -> np.ndarray:
    """Extract (x, y) continuous coordinate head locations from mass density map D via local maxima."""
    if isinstance(mass_map, torch.Tensor):
        mass_map = mass_map.detach().cpu().float().squeeze().numpy()
    if mass_map.ndim != 2:
        raise ValueError(f"Expected 2D mass map, got shape {mass_map.shape}")
    max_val = float(np.max(mass_map))
    if max_val < threshold_abs:
        return np.empty((0, 2), dtype=np.float32)
    thresh = max(threshold_abs, threshold_rel * max_val)
    radius_cells = max(1, int(np.ceil(float(min_distance_px) / float(stride))))
    window_size = 2 * radius_cells + 1
    local_max = (ndi.maximum_filter(mass_map, size=window_size) == mass_map)
    peak_mask = local_max & (mass_map >= thresh)
    peak_y, peak_x = np.nonzero(peak_mask)
    if len(peak_x) == 0:
        return np.empty((0, 2), dtype=np.float32)
    offset = (float(stride) - 1.0) / 2.0
    orig_x = peak_x.astype(np.float32) * float(stride) + offset
    orig_y = peak_y.astype(np.float32) * float(stride) + offset
    return np.stack([orig_x, orig_y], axis=1)


def match_points(
    pred_xy: Union[np.ndarray, torch.Tensor],
    gt_xy: Union[np.ndarray, torch.Tensor],
    threshold: float,
) -> Tuple[int, int, int]:
    """Hungarian minimum distance one-to-one matching with distance gating."""
    if isinstance(pred_xy, torch.Tensor):
        pred_xy = pred_xy.detach().cpu().float().numpy()
    if isinstance(gt_xy, torch.Tensor):
        gt_xy = gt_xy.detach().cpu().float().numpy()

    pred_xy = np.asarray(pred_xy, dtype=np.float32).reshape(-1, 2)
    gt_xy = np.asarray(gt_xy, dtype=np.float32).reshape(-1, 2)

    np_pts = len(pred_xy)
    ng_pts = len(gt_xy)

    if np_pts == 0:
        return 0, 0, ng_pts
    if ng_pts == 0:
        return 0, np_pts, 0

    diff = pred_xy[:, None, :] - gt_xy[None, :, :]
    distance = np.sqrt(np.sum(diff ** 2, axis=-1))

    # Distance-gated Hungarian matching
    penalty = 1e6
    gated_distance = np.where(distance <= threshold, distance, penalty)
    pred_idx, gt_idx = linear_sum_assignment(gated_distance)
    matched_distance = distance[pred_idx, gt_idx]

    tp = int(np.sum(matched_distance <= threshold))
    fp = int(np_pts - tp)
    fn = int(ng_pts - tp)

    return tp, fp, fn


def evaluate_localization_single_image(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    distance_thresholds: Tuple[float, ...] = (4.0, 8.0, 16.0),
) -> Dict[float, Dict[str, float]]:
    res = {}
    for sigma in distance_thresholds:
        tp, fp, fn = match_points(pred_points, gt_points, threshold=sigma)
        m = localization_metrics(tp, fp, fn)
        res[sigma] = m
    return res


def localization_metrics(total_tp: int, total_fp: int, total_fn: int) -> Dict[str, float]:
    """Calculate dataset-level Precision, Recall, and F1-score."""
    precision = float(total_tp / max(total_tp + total_fp, 1)) if (total_tp + total_fp) > 0 else 0.0
    recall = float(total_tp / max(total_tp + total_fn, 1)) if (total_tp + total_fn) > 0 else 0.0
    f1 = float((2.0 * precision * recall) / max(precision + recall, 1e-12)) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


def evaluate_dataset_localization(
    predictions_list: Sequence[Union[np.ndarray, torch.Tensor]],
    ground_truths_list: Sequence[Union[np.ndarray, torch.Tensor]],
    distance_thresholds: Tuple[float, ...] = (4.0, 8.0, 16.0),
) -> Dict[str, float]:
    """Aggregate localization performance across the entire test set."""
    if len(predictions_list) != len(ground_truths_list):
        raise ValueError("predictions and ground_truths lists must have the same length")

    accum = {sigma: {"tp": 0, "fp": 0, "fn": 0} for sigma in distance_thresholds}

    for pred_pts, gt_pts in zip(predictions_list, ground_truths_list):
        for sigma in distance_thresholds:
            tp, fp, fn = match_points(pred_pts, gt_pts, threshold=sigma)
            accum[sigma]["tp"] += tp
            accum[sigma]["fp"] += fp
            accum[sigma]["fn"] += fn

    summary: Dict[str, float] = {}
    for sigma in distance_thresholds:
        m = localization_metrics(accum[sigma]["tp"], accum[sigma]["fp"], accum[sigma]["fn"])
        sig_str = f"sigma_{int(sigma)}" if float(sigma).is_integer() else f"sigma_{float(sigma):g}"
        summary[f"{sig_str}_precision"] = m["precision"]
        summary[f"{sig_str}_recall"] = m["recall"]
        summary[f"{sig_str}_f1"] = m["f1"]
        summary[f"{sig_str}_tp"] = m["tp"]
        summary[f"{sig_str}_fp"] = m["fp"]
        summary[f"{sig_str}_fn"] = m["fn"]

    return summary
