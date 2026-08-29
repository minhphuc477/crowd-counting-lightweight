"""Crowd Localization Metrics for Point-Supervised Crowd Counting & Localization.

Implements standard benchmark localization evaluation (NWPU-Crowd / SHA convention):
  1. Extract predicted head points from continuous mass map D(x, y) via local maxima (NMS).
  2. Bipartite distance matching with ground-truth points under threshold sigma:
       d(p_pred, p_gt) <= sigma
  3. Calculate True Positives (TP), False Positives (FP), False Negatives (FN).
  4. Compute Precision, Recall, and F1-measure:
       Precision = TP / (TP + FP)
       Recall = TP / (TP + FN)
       F1 = 2 * Precision * Recall / (Precision + Recall)
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Union

import numpy as np
import scipy.ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import torch


def extract_points_from_mass_map(
    mass_map: Union[np.ndarray, torch.Tensor],
    stride: int = 4,
    threshold_rel: float = 0.05,
    threshold_abs: float = 0.01,
    min_distance_px: int = 4,
) -> np.ndarray:
    """Extract (x, y) continuous coordinate head locations from mass density map D.
    
    Args:
        mass_map: (H/stride, W/stride) 2D array of predicted count mass.
        stride: spatial downscaling factor of mass map relative to original image (default 4).
        threshold_rel: relative threshold relative to max mass (default 0.05).
        threshold_abs: absolute minimum mass per cell to be a candidate peak.
        min_distance_px: minimum pixel distance between distinct head peaks in original image space.
        
    Returns:
        (M, 2) numpy array of predicted (x, y) coordinates in original image pixel space.
    """
    if isinstance(mass_map, torch.Tensor):
        mass_map = mass_map.detach().cpu().float().squeeze().numpy()
        
    if mass_map.ndim != 2:
        raise ValueError(f"Expected 2D mass map, got shape {mass_map.shape}")
        
    max_val = float(np.max(mass_map))
    if max_val < threshold_abs:
        return np.empty((0, 2), dtype=np.float32)
        
    thresh = max(threshold_abs, threshold_rel * max_val)
    
    # Neighborhood window in mass grid coordinates
    window_size = max(3, int(round(min_distance_px / stride)))
    if window_size % 2 == 0:
        window_size += 1
        
    local_max = (ndi.maximum_filter(mass_map, size=window_size) == mass_map)
    peak_mask = local_max & (mass_map >= thresh)
    
    # Indices in grid (row = y, col = x)
    peak_y, peak_x = np.nonzero(peak_mask)
    
    if len(peak_x) == 0:
        return np.empty((0, 2), dtype=np.float32)
        
    # Convert from mass grid to image pixel coordinates (center of stride-4 cell)
    orig_x = (peak_x.astype(np.float32) + 0.5) * stride
    orig_y = (peak_y.astype(np.float32) + 0.5) * stride
    
    pred_pts = np.stack([orig_x, orig_y], axis=1)
    return pred_pts


def evaluate_localization_single_image(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    distance_thresholds: Tuple[float, ...] = (4.0, 8.0, 16.0, 32.0),
) -> Dict[float, Dict[str, float]]:
    """Match predicted points to ground-truth points for one image across multiple distance thresholds sigma.
    
    Uses distance-gated bipartite matching to guarantee optimal assignments under cutoff threshold.
    """
    pred_pts = np.asarray(pred_points, dtype=np.float32).reshape(-1, 2)
    gt_pts = np.asarray(gt_points, dtype=np.float32).reshape(-1, 2)
    
    n_pred = len(pred_pts)
    n_gt = len(gt_pts)
    
    res = {}
    
    if n_pred == 0 and n_gt == 0:
        for sigma in distance_thresholds:
            res[sigma] = {"tp": 0, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
        return res
        
    if n_pred == 0:
        for sigma in distance_thresholds:
            res[sigma] = {"tp": 0, "fp": 0, "fn": n_gt, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        return res
        
    if n_gt == 0:
        for sigma in distance_thresholds:
            res[sigma] = {"tp": 0, "fp": n_pred, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        return res
        
    # Pairwise Euclidean distance matrix (n_pred, n_gt)
    cost_matrix = cdist(pred_pts, gt_pts, metric="euclidean")
    
    for sigma in distance_thresholds:
        # Distance-gated cost matrix: costs > sigma penalize heavily (1e6)
        penalty = 1e6
        gated_cost = np.where(cost_matrix <= sigma, cost_matrix, penalty)
        pred_ind, gt_ind = linear_sum_assignment(gated_cost)
        
        valid_matches = np.sum(cost_matrix[pred_ind, gt_ind] <= sigma)
        tp = int(valid_matches)
        fp = int(n_pred - tp)
        fn = int(n_gt - tp)
        
        prec = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0
        rec = tp / max(tp + fn, 1) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        res[sigma] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        }
        
    return res


def evaluate_dataset_localization(
    predictions_list: List[np.ndarray],
    ground_truths_list: List[np.ndarray],
    distance_thresholds: Tuple[float, ...] = (4.0, 8.0, 16.0, 32.0),
) -> Dict[str, float]:
    """Compute overall dataset-level Precision, Recall, and F1 at all distance thresholds."""
    if len(predictions_list) != len(ground_truths_list):
        raise ValueError("predictions and ground_truths lists must have the same length")
        
    accum = {sigma: {"tp": 0, "fp": 0, "fn": 0} for sigma in distance_thresholds}
    
    for pred_pts, gt_pts in zip(predictions_list, ground_truths_list):
        img_res = evaluate_localization_single_image(pred_pts, gt_pts, distance_thresholds=distance_thresholds)
        for sigma in distance_thresholds:
            accum[sigma]["tp"] += img_res[sigma]["tp"]
            accum[sigma]["fp"] += img_res[sigma]["fp"]
            accum[sigma]["fn"] += img_res[sigma]["fn"]
            
    summary: Dict[str, float] = {}
    for sigma in distance_thresholds:
        tp = accum[sigma]["tp"]
        fp = accum[sigma]["fp"]
        fn = accum[sigma]["fn"]
        
        prec = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0
        rec = tp / max(tp + fn, 1) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        sig_str = f"sigma_{int(sigma)}"
        summary[f"{sig_str}_precision"] = float(prec)
        summary[f"{sig_str}_recall"] = float(rec)
        summary[f"{sig_str}_f1"] = float(f1)
        summary[f"{sig_str}_tp"] = tp
        summary[f"{sig_str}_fp"] = fp
        summary[f"{sig_str}_fn"] = fn
        
    return summary
