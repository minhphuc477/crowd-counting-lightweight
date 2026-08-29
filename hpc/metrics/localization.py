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
    image_hw: Tuple[int, int] | None = None,
) -> np.ndarray:
    """Extract (x, y) continuous coordinate head locations from mass map D via local maxima."""
    if isinstance(mass_map, torch.Tensor):
        mass_map = mass_map.detach().cpu().float().squeeze().numpy()
    if mass_map.ndim != 2:
        raise ValueError(f"Expected 2D mass map, got shape {mass_map.shape}")
    if not np.isfinite(mass_map).all():
        raise ValueError("Mass map contains non-finite values (NaN or Inf)")
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
    if image_hw is not None:
        image_h, image_w = float(image_hw[0]), float(image_hw[1])
        x0 = peak_x.astype(np.float32) * float(stride)
        y0 = peak_y.astype(np.float32) * float(stride)
        x1 = np.minimum(x0 + float(stride) - 1.0, image_w - 1.0)
        y1 = np.minimum(y0 + float(stride) - 1.0, image_h - 1.0)
        orig_x = 0.5 * (x0 + x1)
        orig_y = 0.5 * (y0 + y1)
    else:
        offset = (float(stride) - 1.0) / 2.0
        orig_x = peak_x.astype(np.float32) * float(stride) + offset
        orig_y = peak_y.astype(np.float32) * float(stride) + offset
    return np.stack([orig_x, orig_y], axis=1)


from scipy.spatial import cKDTree


def match_points(
    pred_xy: Union[np.ndarray, torch.Tensor],
    gt_xy: Union[np.ndarray, torch.Tensor],
    threshold: float,
) -> Tuple[int, int, int]:
    """Hungarian minimum distance one-to-one matching with distance gating.
    
    Supports dense direct evaluation on small sets and scalable O(N log N)
    sparse spatial graph component matching on large sets (e.g. 10k-20k points).
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

    if np_pts * ng_pts <= 2500:
        diff = pred_xy[:, None, :] - gt_xy[None, :, :]
        distance = np.sqrt(np.sum(diff ** 2, axis=-1))
        penalty = 1e6
        gated_distance = np.where(distance <= threshold, distance, penalty)
        pred_idx, gt_idx = linear_sum_assignment(gated_distance)
        matched_distance = distance[pred_idx, gt_idx]
        tp = int(np.sum(matched_distance <= threshold))
        return tp, int(np_pts - tp), int(ng_pts - tp)

    # Scalable sparse candidate bipartite matching
    pred_tree = cKDTree(pred_xy)
    gt_tree = cKDTree(gt_xy)
    neighbors = pred_tree.query_ball_tree(gt_tree, r=threshold)

    parent: Dict[int, int] = {}

    def find(i: int) -> int:
        path = []
        while i in parent and parent[i] != i:
            path.append(i)
            i = parent[i]
        for p in path:
            parent[p] = i
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    active_preds = []
    has_edges = False
    for p_i, gt_indices in enumerate(neighbors):
        if len(gt_indices) > 0:
            has_edges = True
            active_preds.append(p_i)
            for g_j in gt_indices:
                union(p_i, g_j + np_pts)

    if not has_edges:
        return 0, np_pts, ng_pts

    comp_preds: Dict[int, list[int]] = {}
    comp_gts: Dict[int, list[int]] = {}
    for p_i in active_preds:
        root = find(p_i)
        if root not in comp_preds:
            comp_preds[root] = []
            comp_gts[root] = []
        comp_preds[root].append(p_i)
        for g_j in neighbors[p_i]:
            comp_gts[root].append(g_j)

    for root in comp_gts:
        comp_gts[root] = list(set(comp_gts[root]))

    tp = 0
    penalty = 1e6
    for root, p_list in comp_preds.items():
        g_list = comp_gts[root]
        p_sub = pred_xy[p_list]
        g_sub = gt_xy[g_list]
        diff = p_sub[:, None, :] - g_sub[None, :, :]
        dist_sub = np.sqrt(np.sum(diff ** 2, axis=-1))
        gated_sub = np.where(dist_sub <= threshold, dist_sub, penalty)
        pi, gi = linear_sum_assignment(gated_sub)
        tp += int(np.sum(dist_sub[pi, gi] <= threshold))

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
