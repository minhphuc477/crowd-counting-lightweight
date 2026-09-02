"""Local Multiplicity Calibration: E[m_pred | y_gt = k] at stride 4, 8, 16.

Directly tests whether a crowd counter saturates in high local multiplicity regimes:
- Evaluates E[m_hat | y = k] for k in {0, 1, 2, 3, 4, 5, 6, 7, 8, >=9}.
- Multiplicity Ratio: E[m_hat | y = k] / k.
  - Linear / Calibrated: ratio ~ 1.0 across all k.
  - Saturated: ratio drops sharply as k increases (e.g. 0.95 -> 0.70 -> 0.50).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from hpc.losses.ntpc import sum_pool_mass_pyramid


class MultiplicityAccumulator:
    """Accumulator for local multiplicity pairs (y_gt, m_pred) across a dataset."""

    def __init__(self, strides: Sequence[int] = (4, 8, 16), max_k: int = 8):
        self.strides = tuple(strides)
        self.max_k = max_k
        # Storage for each stride: list of y values, list of m values
        self._pairs: Dict[int, Tuple[List[np.ndarray], List[np.ndarray]]] = {
            s: ([], []) for s in self.strides
        }

    def add_image(
        self,
        pred_mass_stride4: torch.Tensor,
        gt_targets_dict: Dict[int, torch.Tensor],
    ) -> None:
        """Add one image's mass map and ground truth targets to the accumulator."""
        if pred_mass_stride4.ndim == 4:
            m_s4 = pred_mass_stride4
        else:
            m_s4 = pred_mass_stride4.unsqueeze(0).unsqueeze(0)

        for s in self.strides:
            if s not in gt_targets_dict:
                continue
            y_t = gt_targets_dict[s].detach().float().squeeze()
            y = y_t.flatten().cpu().numpy()

            if m_s4.shape[-2:] == y_t.shape[-2:]:
                m = m_s4.squeeze().detach().float().flatten().cpu().numpy()
            else:
                k_factor = s // 4
                if k_factor == 1:
                    m = m_s4.squeeze().detach().float().flatten().cpu().numpy()
                else:
                    m = (
                        torch.nn.functional.avg_pool2d(m_s4, kernel_size=k_factor, stride=k_factor) * (k_factor * k_factor)
                    ).squeeze().detach().float().flatten().cpu().numpy()

            self._pairs[s][0].append(y)
            self._pairs[s][1].append(m)

    def summarize(self) -> Dict[int, Dict[str, Any]]:
        """Compute conditional expectation and variance E[m_pred | y = k] for each stride."""
        summary: Dict[int, Dict[str, Any]] = {}

        for s in self.strides:
            if not self._pairs[s][0]:
                continue
            all_y = np.concatenate(self._pairs[s][0])
            all_m = np.concatenate(self._pairs[s][1])

            bin_results: Dict[str, Dict[str, float]] = {}

            # Bins for k = 0, 1, ..., max_k
            for k in range(self.max_k + 1):
                mask = (all_y == float(k))
                n_cells = int(mask.sum())
                if n_cells > 0:
                    vals = all_m[mask]
                    mean_m = float(np.mean(vals))
                    std_m = float(np.std(vals))
                    ratio = float(mean_m / k) if k > 0 else float("nan")
                    abs_cal_err = float(abs(mean_m - k))
                else:
                    mean_m = float("nan")
                    std_m = float("nan")
                    ratio = float("nan")
                    abs_cal_err = float("nan")

                bin_results[f"k_{k}"] = {
                    "k": float(k),
                    "n_cells": float(n_cells),
                    "mean_pred": mean_m,
                    "std_pred": std_m,
                    "ratio_pred_gt": ratio,
                    "abs_cal_error": abs_cal_err,
                }

            # Overflow bin: k > max_k
            mask_over = (all_y > float(self.max_k))
            n_over = int(mask_over.sum())
            if n_over > 0:
                y_over = all_y[mask_over]
                m_over = all_m[mask_over]
                mean_y = float(np.mean(y_over))
                mean_m = float(np.mean(m_over))
                std_m = float(np.std(m_over))
                ratio = float(mean_m / mean_y) if mean_y > 0 else float("nan")
            else:
                mean_y = float("nan")
                mean_m = float("nan")
                std_m = float("nan")
                ratio = float("nan")

            bin_results[f"k_gt_{self.max_k}"] = {
                "k_label": f">{self.max_k}",
                "n_cells": float(n_over),
                "mean_gt": mean_y,
                "mean_pred": mean_m,
                "std_pred": std_m,
                "ratio_pred_gt": ratio,
            }

            summary[s] = bin_results

        return summary
