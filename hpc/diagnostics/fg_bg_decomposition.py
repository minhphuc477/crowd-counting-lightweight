"""Cell Occupancy Error Decomposition (Occupied vs Empty Grid Cells).

Partitions spatial prediction into occupied cells (y > 0) and empty cells (y == 0)
at a specified grid stride (default 16):
- Occupied Deficit: sum_{y > 0} max(0, y - m_pred) [heads missed in occupied cells]
- Occupied Surplus: sum_{y > 0} max(0, m_pred - y) [excess mass in occupied cells]
- Empty Cell Mass:  sum_{y == 0} m_pred            [mass placed on empty cells]
- Empty Cell Compensation: min(Occupied Deficit, Empty Cell Mass) [deficit masked by empty mass]

NOTE: This measures spatial cell sparsity at grid stride, NOT semantic foreground/background
segmentation (e.g. Modolo et al. WACV 2021). Empty cells between heads in a crowd region
are counted as empty grid cells, not semantic background.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch


def decompose_cell_occupancy_errors(
    pred_mass_stride4: torch.Tensor,
    gt_targets_dict: torch.Tensor | Dict[int, torch.Tensor],
    stride: int = 16,
) -> Dict[str, float]:
    """Decompose counting error into occupied cell deficit and empty cell compensation.

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

    occupied_mask = (y_flat > 0)
    empty_mask = (y_flat == 0)

    occupied_gt = float(y_flat[occupied_mask].sum())
    occupied_pred = float(m_flat[occupied_mask].sum())
    empty_pred = float(m_flat[empty_mask].sum())

    cell_errors = y_flat[occupied_mask] - m_flat[occupied_mask]
    occupied_deficit = float(np.sum(np.maximum(0.0, cell_errors)))
    occupied_surplus = float(np.sum(np.maximum(0.0, -cell_errors)))

    empty_compensation = float(min(occupied_deficit, empty_pred))
    compensation_ratio = (
        float(empty_compensation / (occupied_deficit + 1e-8))
        if occupied_deficit > 0
        else 0.0
    )

    signed_error = pred_total - gt_total
    abs_error = abs(signed_error)

    return {
        "stride": float(stride),
        "gt_total": gt_total,
        "pred_total": pred_total,
        "signed_error": signed_error,
        "abs_error": abs_error,
        "n_occupied_cells": float(occupied_mask.sum()),
        "n_empty_cells": float(empty_mask.sum()),
        "occupied_gt": occupied_gt,
        "occupied_pred": occupied_pred,
        "empty_cell_mass": empty_pred,
        "empty_cell_mass_fraction": (
            float(empty_pred / (pred_total + 1e-8)) if pred_total > 0 else 0.0
        ),
        "occupied_deficit": occupied_deficit,
        "occupied_surplus": occupied_surplus,
        "empty_cell_compensation": empty_compensation,
        "compensation_ratio": compensation_ratio,
        # Backward compatibility aliases
        "n_fg_cells": float(occupied_mask.sum()),
        "n_bg_cells": float(empty_mask.sum()),
        "fg_gt": occupied_gt,
        "fg_pred": occupied_pred,
        "bg_pred": empty_pred,
        "bg_mass_fraction": (
            float(empty_pred / (pred_total + 1e-8)) if pred_total > 0 else 0.0
        ),
        "fg_deficit": occupied_deficit,
        "fg_surplus": occupied_surplus,
        "bg_compensation": empty_compensation,
    }


# Backward-compatible alias
decompose_fg_bg_errors = decompose_cell_occupancy_errors
