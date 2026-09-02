"""Foreground / Background Error Decomposition (Modolo et al. WACV 2021).

Partitions spatial prediction into foreground cells (y > 0) and background cells (y == 0):
- FG Deficit: sum_{y > 0} max(0, y - m_pred) [heads missed in crowd regions]
- FG Surplus: sum_{y > 0} max(0, m_pred - y) [overpredicted mass in crowd regions]
- BG Excess:  sum_{y == 0} m_pred            [false-positive mass in background regions]
- BG Compensation: min(FG Deficit, BG Excess) [misses masked by false positives]
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from hpc.losses.ntpc import sum_pool_mass_pyramid


def decompose_fg_bg_errors(
    pred_mass_stride4: torch.Tensor,
    gt_targets_dict: torch.Tensor | Dict[int, torch.Tensor],
    stride: int = 16,
) -> Dict[str, float]:
    """Decompose counting error into foreground misses and background false-positive compensation.

    Args:
        pred_mass_stride4: Predicted mass map at stride 4, or pooled mass at `stride`.
        gt_targets_dict: Exact ground truth tensor at `stride` or target pyramid dict.
        stride: Spatial resolution to evaluate decomposition at (default 16).
    """
    if isinstance(gt_targets_dict, dict):
        if stride not in gt_targets_dict:
            raise KeyError(f"Target pyramid does not contain requested stride {stride}")
        y = gt_targets_dict[stride].detach().float().squeeze()
    else:
        y = gt_targets_dict.detach().float().squeeze()

    if pred_mass_stride4.ndim == 4:
        pred_mass_s4 = pred_mass_stride4
    elif pred_mass_stride4.ndim == 3:
        pred_mass_s4 = pred_mass_stride4.unsqueeze(1)
    else:
        pred_mass_s4 = pred_mass_stride4.unsqueeze(0).unsqueeze(0)

    if pred_mass_s4.shape[-2:] == y.shape[-2:]:
        m = pred_mass_s4.squeeze().detach().float()
    else:
        # Pool predicted mass to target stride
        k = stride // 4
        if k == 1:
            m = pred_mass_s4.squeeze().detach().float()
        else:
            m = (
                torch.nn.functional.avg_pool2d(pred_mass_s4, kernel_size=k, stride=k) * (k * k)
            ).squeeze().detach().float()

    y_flat = y.flatten().cpu().numpy()
    m_flat = m.flatten().cpu().numpy()

    gt_total = float(y_flat.sum())
    pred_total = float(m_flat.sum())

    fg_mask = (y_flat > 0)
    bg_mask = (y_flat == 0)

    fg_gt = float(y_flat[fg_mask].sum())
    fg_pred = float(m_flat[fg_mask].sum())
    bg_pred = float(m_flat[bg_mask].sum())

    fg_cell_errors = y_flat[fg_mask] - m_flat[fg_mask]
    fg_deficit = float(np.sum(np.maximum(0.0, fg_cell_errors)))
    fg_surplus = float(np.sum(np.maximum(0.0, -fg_cell_errors)))

    bg_compensation = float(min(fg_deficit, bg_pred))
    compensation_ratio = float(bg_compensation / (fg_deficit + 1e-8)) if fg_deficit > 0 else 0.0

    signed_error = pred_total - gt_total
    abs_error = abs(signed_error)

    return {
        "stride": float(stride),
        "gt_total": gt_total,
        "pred_total": pred_total,
        "signed_error": signed_error,
        "abs_error": abs_error,
        "n_fg_cells": float(fg_mask.sum()),
        "n_bg_cells": float(bg_mask.sum()),
        "fg_gt": fg_gt,
        "fg_pred": fg_pred,
        "bg_pred": bg_pred,
        "bg_mass_fraction": float(bg_pred / (pred_total + 1e-8)) if pred_total > 0 else 0.0,
        "fg_deficit": fg_deficit,
        "fg_surplus": fg_surplus,
        "bg_compensation": bg_compensation,
        "compensation_ratio": compensation_ratio,
    }
