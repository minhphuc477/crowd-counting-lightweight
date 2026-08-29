"""Crowd Localization Metrics via Hungarian Bipartite Matching (P2PNet / STEERER / CLTR standard).

Evaluates:
  - Precision, Recall, and F1-score at distance thresholds sigma in {4, 8} pixels.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import torch


def match_points(
    pred_xy: Union[np.ndarray, torch.Tensor],
    gt_xy: Union[np.ndarray, torch.Tensor],
    threshold: float,
) -> Tuple[int, int, int]:
    """Hungarian minimum distance one-to-one matching between predicted and ground-truth points.
    
    Args:
        pred_xy: [Np, 2] array/tensor of predicted point coordinates in pixel space.
        gt_xy: [Ng, 2] array/tensor of ground-truth head coordinates in pixel space.
        threshold: maximum euclidean distance (in pixels) for a valid positive match.
        
    Returns:
        TP, FP, FN counts.
    """
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

    # Pairwise Euclidean distance matrix [Np, Ng]
    diff = pred_xy[:, None, :] - gt_xy[None, :, :]
    distance = np.sqrt(np.sum(diff ** 2, axis=-1))

    # Hungarian minimum distance assignment
    pred_idx, gt_idx = linear_sum_assignment(distance)
    matched_distance = distance[pred_idx, gt_idx]

    tp = int(np.sum(matched_distance <= threshold))
    fp = int(np_pts - tp)
    fn = int(ng_pts - tp)

    return tp, fp, fn


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
        sig_str = f"sigma_{int(sigma)}"
        summary[f"{sig_str}_precision"] = m["precision"]
        summary[f"{sig_str}_recall"] = m["recall"]
        summary[f"{sig_str}_f1"] = m["f1"]
        summary[f"{sig_str}_tp"] = m["tp"]
        summary[f"{sig_str}_fp"] = m["fp"]
        summary[f"{sig_str}_fn"] = m["fn"]

    return summary
