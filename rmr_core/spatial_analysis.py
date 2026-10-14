from __future__ import annotations

from typing import Any
import numpy as np
import torch
import torch.nn.functional as F


def extract_subpixel_peaks(
    density_map: torch.Tensor | np.ndarray,
    threshold: float = 0.05,
    nms_kernel: int = 3,
    stride: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect discrete crowd head peaks and refine locations to subpixel physical coordinates.

    Steps:
      1. Local 2D maximum suppression via MaxPool2D(nms_kernel, stride=1, padding=nms_kernel//2).
      2. Threshold filtering: keep peaks with intensity > threshold.
      3. Subpixel parabolic centroid refinement on 3x3 local neighborhood:
           dx = (f(y0, x0+1) - f(y0, x0-1)) / (2 * (2*f(y0, x0) - f(y0, x0+1) - f(y0, x0-1) + eps))
           dy = (f(y0+1, x0) - f(y0-1, x0)) / (2 * (2*f(y0, x0) - f(y0+1, x0) - f(y0-1, x0) + eps))
         bounded to [-0.5, 0.5] to prevent divergent extrapolation.
      4. Physical image coordinates:
           x_phys = (x0 + dx + 0.5) * stride
           y_phys = (y0 + dy + 0.5) * stride

    Returns:
      points_xy: np.ndarray of shape (N, 2) in physical pixel coordinates.
      scores: np.ndarray of shape (N,) peak intensity values.
    """
    if isinstance(density_map, np.ndarray):
        density_t = torch.from_numpy(density_map).float()
    else:
        density_t = density_map.detach().float().cpu()

    while density_t.ndim < 4:
        density_t = density_t.unsqueeze(0)

    # 1. 2D Local Maximum Suppression
    pad = nms_kernel // 2
    max_pool = F.max_pool2d(density_t, kernel_size=nms_kernel, stride=1, padding=pad)
    peak_mask = (density_t == max_pool) & (density_t > float(threshold))
    peak_mask = peak_mask[0, 0]
    dens_2d = density_t[0, 0]
    h, w = dens_2d.shape

    y_indices, x_indices = torch.nonzero(peak_mask, as_tuple=True)
    if y_indices.numel() == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.float32)

    # 2. Subpixel parabolic interpolation
    eps = 1e-6
    points = []
    scores = []
    for y0, x0 in zip(y_indices.tolist(), x_indices.tolist(), strict=False):
        c = float(dens_2d[y0, x0])
        # x-axis refinement
        x_left = float(dens_2d[y0, x0 - 1]) if x0 > 0 else c
        x_right = float(dens_2d[y0, x0 + 1]) if x0 < w - 1 else c
        denom_x = 2.0 * (2.0 * c - x_left - x_right)
        dx = (x_right - x_left) / (denom_x + eps) if abs(denom_x) > eps else 0.0
        dx = max(-0.5, min(0.5, dx))

        # y-axis refinement
        y_top = float(dens_2d[y0 - 1, x0]) if y0 > 0 else c
        y_bottom = float(dens_2d[y0 + 1, x0]) if y0 < h - 1 else c
        denom_y = 2.0 * (2.0 * c - y_top - y_bottom)
        dy = (y_bottom - y_top) / (denom_y + eps) if abs(denom_y) > eps else 0.0
        dy = max(-0.5, min(0.5, dy))

        phys_x = (float(x0) + dx + 0.5) * float(stride)
        phys_y = (float(y0) + dy + 0.5) * float(stride)
        points.append((phys_x, phys_y))
        scores.append(c)

    return np.asarray(points, dtype=np.float32), np.asarray(scores, dtype=np.float32)


def evaluate_point_localization(
    pred_points: np.ndarray | torch.Tensor,
    gt_points: np.ndarray | torch.Tensor,
    distance_thresholds: tuple[float, ...] = (4.0, 8.0, 16.0),
) -> dict[str, float]:
    """Evaluate point-level crowd detection and localization performance.

    Standard benchmark matching (P2PNet, STEERER, FIDTM):
      At each tolerance distance sigma in distance_thresholds:
        - Match each prediction to the nearest unmatched GT point within distance <= sigma.
        - Precision = TP / (TP + FP + eps)
        - Recall = TP / (TP + FN + eps)
        - F1 = 2 * P * R / (P + R + eps)
    """
    if isinstance(pred_points, torch.Tensor):
        preds = pred_points.detach().cpu().numpy().reshape(-1, 2)
    else:
        preds = np.asarray(pred_points, dtype=np.float64).reshape(-1, 2)

    if isinstance(gt_points, torch.Tensor):
        gts = gt_points.detach().cpu().numpy().reshape(-1, 2)
    else:
        gts = np.asarray(gt_points, dtype=np.float64).reshape(-1, 2)

    n_pred = len(preds)
    n_gt = len(gts)
    results: dict[str, float] = {
        "pred_points": float(n_pred),
        "gt_points": float(n_gt),
    }

    if n_pred == 0 and n_gt == 0:
        for d in distance_thresholds:
            results[f"Precision@{int(d)}"] = 1.0
            results[f"Recall@{int(d)}"] = 1.0
            results[f"F1@{int(d)}"] = 1.0
        return results

    if n_pred == 0 or n_gt == 0:
        for d in distance_thresholds:
            results[f"Precision@{int(d)}"] = 0.0
            results[f"Recall@{int(d)}"] = 0.0
            results[f"F1@{int(d)}"] = 0.0
        return results

    # Compute pairwise Euclidean distance matrix (n_pred, n_gt)
    diff = preds[:, None, :] - gts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))

    for d in distance_thresholds:
        thresh = float(d)
        matched_gt = set()
        tp = 0
        pairs = []
        for i in range(n_pred):
            for j in range(n_gt):
                if dist[i, j] <= thresh:
                    pairs.append((dist[i, j], i, j))
        pairs.sort(key=lambda x: x[0])

        matched_preds = set()
        for _, p_idx, g_idx in pairs:
            if p_idx not in matched_preds and g_idx not in matched_gt:
                matched_preds.add(p_idx)
                matched_gt.add(g_idx)
                tp += 1

        fp = n_pred - tp
        fn = n_gt - tp
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float((2.0 * prec * rec) / (prec + rec)) if (prec + rec) > 0 else 0.0
        results[f"Precision@{int(d)}"] = prec
        results[f"Recall@{int(d)}"] = rec
        results[f"F1@{int(d)}"] = f1

    return results


def compute_count_confidence_interval(
    pred_count: float,
    b_variance: torch.Tensor | np.ndarray | None = None,
    confidence_level: float = 0.95,
) -> tuple[float, float, float]:
    """Compute analytical asymptotic confidence interval on predicted crowd count.

    Uses the Negative-Binomial heteroscedastic regional variance:
      sigma_total = sqrt(sum_R Var(N_R) / K_overlap) or Poisson-limit sqrt(N).
      CI = [N - z * sigma, N + z * sigma].

    Returns:
      (ci_lower, ci_upper, std_err)
    """
    count = max(float(pred_count), 0.0)
    if b_variance is not None:
        if isinstance(b_variance, torch.Tensor):
            var_arr = b_variance.detach().cpu().numpy().reshape(-1)
        else:
            var_arr = np.asarray(b_variance, dtype=np.float64).reshape(-1)
        if var_arr.size > 0 and np.isfinite(var_arr).all() and (var_arr > 0).any():
            # Account for 50% spatial box overlap (redundancy ~ 4x)
            total_var = float(np.sum(var_arr) / 4.0)
            std_err = float(np.sqrt(max(total_var, 1e-4)))
        else:
            std_err = float(np.sqrt(count))
    else:
        std_err = float(np.sqrt(count))

    # Standard normal quantile: 95% -> 1.96, 90% -> 1.645, 99% -> 2.576
    if abs(confidence_level - 0.95) < 0.01:
        z = 1.95996
    elif abs(confidence_level - 0.90) < 0.01:
        z = 1.64485
    elif abs(confidence_level - 0.99) < 0.01:
        z = 2.57583
    else:
        z = 1.95996

    ci_lower = max(0.0, count - z * std_err)
    ci_upper = count + z * std_err
    return float(ci_lower), float(ci_upper), float(std_err)


def compute_crowd_risk_index(
    density_map: np.ndarray | torch.Tensor,
    scale_map: np.ndarray | None = None,
    area_sqm: float = 100.0,
) -> dict[str, Any]:
    """Classify crowd congestion risk level based on local density and peak concentration.

    Risk Categories (based on Still et al. Crowd Safety Standards):
      - NORMAL: < 1.5 persons/m^2 (Free flowing)
      - MODERATE: 1.5 - 3.0 persons/m^2 (Restricted movement)
      - CONGESTED: 3.0 - 5.0 persons/m^2 (High density, crowd control required)
      - CRITICAL_RISK: > 5.0 persons/m^2 (Immediate stampede danger)
    """
    if isinstance(density_map, torch.Tensor):
        dens = density_map.detach().cpu().numpy().squeeze()
    else:
        dens = np.asarray(density_map, dtype=np.float64).squeeze()

    total_count = float(dens.sum())
    mean_density = total_count / max(float(area_sqm), 1.0)
    peak_cell = float(dens.max(initial=0.0))

    total_cells = max(dens.size, 1)
    cell_area = float(area_sqm) / total_cells
    peak_density_m2 = peak_cell / max(cell_area, 1e-4)

    if peak_density_m2 > 5.0 or mean_density > 4.0:
        risk_level = "CRITICAL_RISK"
    elif peak_density_m2 > 3.0 or mean_density > 2.5:
        risk_level = "CONGESTED"
    elif peak_density_m2 > 1.5 or mean_density > 1.0:
        risk_level = "MODERATE"
    else:
        risk_level = "NORMAL"

    high_dens_cells = int(np.sum(dens / max(cell_area, 1e-4) > 1.5))
    high_dens_pct = float(high_dens_cells / total_cells * 100.0)

    return {
        "risk_level": risk_level,
        "mean_density_m2": float(mean_density),
        "peak_density_m2": float(peak_density_m2),
        "high_density_area_pct": high_dens_pct,
        "total_count": total_count,
    }


def extract_perspective_field(
    scale_router_weights: torch.Tensor | np.ndarray,
    scale_sizes_px: tuple[int, ...] = (32, 64, 128),
) -> np.ndarray:
    """Extract continuous expected head scale field S(x, y) = sum_k d_k * pi_k(x, y).

    Provides an unsupervised self-supervised perspective/depth map (in pixels)
    without requiring camera calibration or 3D LiDAR annotations.
    """
    if isinstance(scale_router_weights, torch.Tensor):
        pi = scale_router_weights.detach().cpu().numpy()
    else:
        pi = np.asarray(scale_router_weights, dtype=np.float64)

    while pi.ndim > 3:
        pi = pi[0]

    k_scales = len(scale_sizes_px)
    if pi.shape[0] != k_scales:
        raise ValueError(f"Expected {k_scales} scale channels, got {pi.shape[0]}")

    scales = np.asarray(scale_sizes_px, dtype=np.float64).reshape(-1, 1, 1)
    scale_field = np.sum(pi * scales, axis=0)
    return np.asarray(scale_field, dtype=np.float32)
