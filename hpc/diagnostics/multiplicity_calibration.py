"""Local Multiplicity Calibration: E[m_pred | y_gt = k] at stride 4, 8, 16.

Directly tests whether a crowd counter saturates in high local multiplicity regimes:
- Evaluates E[m_hat | y = k] for k in {0, 1, 2, 3, 4, 5, 6, 7, 8, >=9}.
- Multiplicity Ratio: E[m_hat | y = k] / k.
- Image-Cluster Bootstrap: Resamples whole images B times with replacement to produce
  cluster-robust standard errors, 95% Confidence Intervals, and paired difference tests.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch


class MultiplicityAccumulator:
    """Accumulator for local multiplicity pairs (y_gt, m_pred) with image-cluster bootstrap support."""

    def __init__(self, strides: Sequence[int] = (4, 8, 16), max_k: int = 8):
        self.strides = tuple(strides)
        self.max_k = max_k
        # Storage for each stride: list of y values, list of m values
        self._pairs: Dict[int, Tuple[List[np.ndarray], List[np.ndarray]]] = {
            s: ([], []) for s in self.strides
        }
        # Image-level records: list of dicts, one per image
        # image_record[s][k] = (n_cells, sum_m)
        self._image_records: List[Dict[int, Dict[int, Tuple[int, float]]]] = []

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

        img_stats: Dict[int, Dict[int, Tuple[int, float]]] = {}

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

            # Store per-image bin aggregates
            bin_aggregates: Dict[int, Tuple[int, float]] = {}
            for k in range(self.max_k + 1):
                mask = (y == float(k))
                n_c = int(mask.sum())
                sum_v = float(m[mask].sum()) if n_c > 0 else 0.0
                bin_aggregates[k] = (n_c, sum_v)

            # Overflow bin (> max_k) keyed by max_k + 1
            mask_over = (y > float(self.max_k))
            n_o = int(mask_over.sum())
            sum_o = float(m[mask_over].sum()) if n_o > 0 else 0.0
            sum_y_o = float(y[mask_over].sum()) if n_o > 0 else 0.0
            bin_aggregates[self.max_k + 1] = (n_o, sum_o, sum_y_o)  # type: ignore

            img_stats[s] = bin_aggregates

        self._image_records.append(img_stats)

    def summarize(self) -> Dict[int, Dict[str, Any]]:
        """Compute pooled conditional expectation and variance E[m_pred | y = k] for each stride."""
        summary: Dict[int, Dict[str, Any]] = {}

        for s in self.strides:
            if not self._pairs[s][0]:
                continue
            all_y = np.concatenate(self._pairs[s][0])
            all_m = np.concatenate(self._pairs[s][1])

            bin_results: Dict[str, Dict[str, float]] = {}

            # Count contributing images
            for k in range(self.max_k + 1):
                mask = (all_y == float(k))
                n_cells = int(mask.sum())
                n_images = sum(
                    1 for rec in self._image_records
                    if s in rec and k in rec[s] and rec[s][k][0] > 0
                )
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
                    "n_contributing_images": float(n_images),
                    "mean_pred": mean_m,
                    "std_pred": std_m,
                    "ratio_pred_gt": ratio,
                    "abs_cal_error": abs_cal_err,
                }

            # Overflow bin: k > max_k
            mask_over = (all_y > float(self.max_k))
            n_over = int(mask_over.sum())
            n_images_over = sum(
                1 for rec in self._image_records
                if s in rec and (self.max_k + 1) in rec[s] and rec[s][self.max_k + 1][0] > 0
            )
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
                "n_contributing_images": float(n_images_over),
                "mean_gt": mean_y,
                "mean_pred": mean_m,
                "std_pred": std_m,
                "ratio_pred_gt": ratio,
            }

            summary[s] = bin_results

        return summary

    def cluster_bootstrap(
        self,
        stride: int = 16,
        n_boot: int = 1000,
        seed: int = 42,
        ci_level: float = 0.95,
    ) -> Dict[str, Dict[str, float]]:
        """Compute image-cluster bootstrap confidence intervals for E[m_pred | y = k]."""
        n_images = len(self._image_records)
        if n_images < 2:
            return {}

        rng = np.random.RandomState(seed)
        alpha = (1.0 - ci_level) / 2.0

        # Extract per-image arrays for the requested stride
        # k_counts[k, i], k_sums[k, i]
        all_k = list(range(self.max_k + 2))  # 0..max_k plus overflow
        k_counts = np.zeros((len(all_k), n_images), dtype=np.float64)
        k_sums = np.zeros((len(all_k), n_images), dtype=np.float64)

        for i, rec in enumerate(self._image_records):
            if stride not in rec:
                continue
            for idx_k, k in enumerate(all_k):
                if k in rec[stride]:
                    entry = rec[stride][k]
                    k_counts[idx_k, i] = entry[0]
                    k_sums[idx_k, i] = entry[1]

        # Resample image indices with replacement B times
        boot_indices = rng.randint(0, n_images, size=(n_boot, n_images))

        results: Dict[str, Dict[str, float]] = {}
        for idx_k, k in enumerate(all_k):
            k_name = f"k_{k}" if k <= self.max_k else f"k_gt_{self.max_k}"
            counts_boot = k_counts[idx_k, boot_indices]  # [B, n_images]
            sums_boot = k_sums[idx_k, boot_indices]      # [B, n_images]

            total_counts = np.sum(counts_boot, axis=1)   # [B]
            total_sums = np.sum(sums_boot, axis=1)       # [B]

            valid_boot = total_counts > 0
            if np.sum(valid_boot) < 10:
                results[k_name] = {
                    "mean": float("nan"),
                    "se": float("nan"),
                    "ci_lower": float("nan"),
                    "ci_upper": float("nan"),
                    "n_contributing_images": float(np.sum(k_counts[idx_k] > 0)),
                }
                continue

            boot_estimates = total_sums[valid_boot] / total_counts[valid_boot]
            results[k_name] = {
                "mean": float(np.mean(boot_estimates)),
                "se": float(np.std(boot_estimates)),
                "ci_lower": float(np.percentile(boot_estimates, alpha * 100.0)),
                "ci_upper": float(np.percentile(boot_estimates, (1.0 - alpha) * 100.0)),
                "n_contributing_images": float(np.sum(k_counts[idx_k] > 0)),
            }

        return results

    @staticmethod
    def cluster_bootstrap_paired_diff(
        acc_a: MultiplicityAccumulator,
        acc_b: MultiplicityAccumulator,
        stride: int = 16,
        n_boot: int = 1000,
        seed: int = 42,
        ci_level: float = 0.95,
    ) -> Dict[str, Dict[str, Any]]:
        """Compute paired image-cluster bootstrap CI for Delta_k = E_A[m|y=k] - E_B[m|y=k]."""
        n_images = min(len(acc_a._image_records), len(acc_b._image_records))
        if n_images < 2:
            return {}

        rng = np.random.RandomState(seed)
        alpha = (1.0 - ci_level) / 2.0
        all_k = list(range(acc_a.max_k + 2))

        # Build arrays for A and B
        counts_a = np.zeros((len(all_k), n_images), dtype=np.float64)
        sums_a = np.zeros((len(all_k), n_images), dtype=np.float64)
        counts_b = np.zeros((len(all_k), n_images), dtype=np.float64)
        sums_b = np.zeros((len(all_k), n_images), dtype=np.float64)

        for i in range(n_images):
            rec_a = acc_a._image_records[i]
            rec_b = acc_b._image_records[i]
            if stride in rec_a:
                for idx_k, k in enumerate(all_k):
                    if k in rec_a[stride]:
                        counts_a[idx_k, i] = rec_a[stride][k][0]
                        sums_a[idx_k, i] = rec_a[stride][k][1]
            if stride in rec_b:
                for idx_k, k in enumerate(all_k):
                    if k in rec_b[stride]:
                        counts_b[idx_k, i] = rec_b[stride][k][0]
                        sums_b[idx_k, i] = rec_b[stride][k][1]

        boot_indices = rng.randint(0, n_images, size=(n_boot, n_images))
        results: Dict[str, Dict[str, Any]] = {}

        for idx_k, k in enumerate(all_k):
            k_name = f"k_{k}" if k <= acc_a.max_k else f"k_gt_{acc_a.max_k}"
            ca_boot = np.sum(counts_a[idx_k, boot_indices], axis=1)
            sa_boot = np.sum(sums_a[idx_k, boot_indices], axis=1)
            cb_boot = np.sum(counts_b[idx_k, boot_indices], axis=1)
            sb_boot = np.sum(sums_b[idx_k, boot_indices], axis=1)

            valid = (ca_boot > 0) & (cb_boot > 0)
            if np.sum(valid) < 10:
                results[k_name] = {
                    "diff_mean": float("nan"),
                    "diff_se": float("nan"),
                    "diff_ci_lower": float("nan"),
                    "diff_ci_upper": float("nan"),
                    "significant": False,
                }
                continue

            diff_boot = (sa_boot[valid] / ca_boot[valid]) - (sb_boot[valid] / cb_boot[valid])
            ci_low = float(np.percentile(diff_boot, alpha * 100.0))
            ci_high = float(np.percentile(diff_boot, (1.0 - alpha) * 100.0))
            sig = bool((ci_low > 0.0) or (ci_high < 0.0))

            results[k_name] = {
                "diff_mean": float(np.mean(diff_boot)),
                "diff_se": float(np.std(diff_boot)),
                "diff_ci_lower": ci_low,
                "diff_ci_upper": ci_high,
                "significant": sig,
            }

        return results
