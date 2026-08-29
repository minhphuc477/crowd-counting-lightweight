"""Crowd-localization metrics via exact distance-gated bipartite matching.

Evaluates:
  - Precision, Recall, and F1-score at distance thresholds sigma in {4, 8} pixels.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple, Union

import numpy as np
import scipy.ndimage as ndi
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree
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
    # A flat maximum is one peak, not one point per equal-valued pixel. Select the
    # plateau pixel nearest its centroid (row-major tie break) deterministically.
    labels, component_count = ndi.label(peak_mask, structure=np.ones((3, 3), dtype=np.uint8))
    selected_yx: list[np.ndarray] = []
    for label_index in range(1, component_count + 1):
        coords = np.argwhere(labels == label_index)
        center = coords.mean(axis=0)
        selected_yx.append(coords[np.argmin(np.square(coords - center).sum(axis=1))])
    if selected_yx:
        peak_y, peak_x = np.asarray(selected_yx, dtype=np.int64).T
    else:
        peak_y = peak_x = np.empty(0, dtype=np.int64)
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


def match_points(
    pred_xy: Union[np.ndarray, torch.Tensor],
    gt_xy: Union[np.ndarray, torch.Tensor],
    threshold: float,
) -> Tuple[int, int, int]:
    """Return TP/FP/FN using exact maximum-cardinality gated matching.

    A KD-tree constructs only point pairs within ``threshold``; SciPy then finds
    an exact maximum-cardinality matching on that sparse bipartite graph. Metric
    correctness depends on cardinality, not on minimizing the sum of distances.
    """
    if isinstance(pred_xy, torch.Tensor):
        pred_xy = pred_xy.detach().cpu().float().numpy()
    if isinstance(gt_xy, torch.Tensor):
        gt_xy = gt_xy.detach().cpu().float().numpy()

    pred_xy = np.asarray(pred_xy, dtype=np.float32).reshape(-1, 2)
    gt_xy = np.asarray(gt_xy, dtype=np.float32).reshape(-1, 2)

    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError(f"threshold must be finite and non-negative, got {threshold}")
    if not np.isfinite(pred_xy).all() or not np.isfinite(gt_xy).all():
        raise ValueError("Localization points contain NaN or Inf")

    np_pts = len(pred_xy)
    ng_pts = len(gt_xy)

    if np_pts == 0:
        return 0, 0, ng_pts
    if ng_pts == 0:
        return 0, np_pts, 0

    pred_tree = cKDTree(pred_xy)
    gt_tree = cKDTree(gt_xy)
    neighbors = pred_tree.query_ball_tree(gt_tree, r=threshold)
    edge_count = sum(len(indices) for indices in neighbors)
    if edge_count == 0:
        return 0, np_pts, ng_pts

    indptr = np.empty(np_pts + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(np.fromiter((len(x) for x in neighbors), dtype=np.int64), out=indptr[1:])
    indices = np.fromiter(
        (gt_index for row in neighbors for gt_index in row),
        dtype=np.int64,
        count=edge_count,
    )
    graph = csr_matrix(
        (np.ones(edge_count, dtype=np.int8), indices, indptr),
        shape=(np_pts, ng_pts),
    )
    matched_gt_for_pred = maximum_bipartite_matching(graph, perm_type="column")
    tp = int(np.count_nonzero(matched_gt_for_pred >= 0))

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
