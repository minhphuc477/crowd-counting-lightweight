"""Tail support and training distribution profiling for crowd counting datasets.

Computes point-set and grid-level multiplicity metrics:
- Total count N
- Spatial density (heads per 10,000 px^2)
- Nearest-neighbor (NN) distance statistics via KDTree (median, p10, min, fraction <4px, <8px, <16px)
- Maximum cell count at stride 4, 8, 16 (local multiplicity peaks)
- Empirical percentile ranks of test samples relative to training distribution
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import scipy.spatial
import torch

from hpc.data.point_counts import build_exact_count_pyramid


def compute_image_spatial_statistics(
    image_hw: Tuple[int, int],
    points: np.ndarray,
) -> Dict[str, float]:
    """Compute exact spatial density and nearest-neighbor statistics for a single point set."""
    h, w = image_hw
    n_pts = len(points)
    area_10k = (h * w) / 10000.0 if h * w > 0 else 1.0
    density_10k = float(n_pts / area_10k)

    if n_pts < 2:
        return {
            "gt_count": float(n_pts),
            "density_10k": density_10k,
            "nn_median": float("nan"),
            "nn_p10": float("nan"),
            "nn_min": float("nan"),
            "nn_frac_lt_4": 0.0,
            "nn_frac_lt_8": 0.0,
            "nn_frac_lt_16": 0.0,
            "max_y_4": float(n_pts),
            "max_y_8": float(n_pts),
            "max_y_16": float(n_pts),
        }

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
    pts[:, 0] = np.clip(pts[:, 0], 0.0, max(0.0, float(w - 1)))
    pts[:, 1] = np.clip(pts[:, 1], 0.0, max(0.0, float(h - 1)))
    tree = scipy.spatial.cKDTree(pts)
    dists, _ = tree.query(pts, k=2)
    nn_dists = dists[:, 1]

    # Grid multiplicity at stride 4, 8, 16
    pyramid = build_exact_count_pyramid(
        [torch.from_numpy(pts).float()],
        height=h,
        width=w,
        block_sizes=(4, 8, 16),
        pad_multiple=64,
    )
    max_y_4 = float(pyramid[4].max().item())
    max_y_8 = float(pyramid[8].max().item())
    max_y_16 = float(pyramid[16].max().item())

    return {
        "gt_count": float(n_pts),
        "density_10k": density_10k,
        "nn_median": float(np.median(nn_dists)),
        "nn_p10": float(np.percentile(nn_dists, 10)),
        "nn_min": float(np.min(nn_dists)),
        "nn_frac_lt_4": float(np.mean(nn_dists < 4.0)),
        "nn_frac_lt_8": float(np.mean(nn_dists < 8.0)),
        "nn_frac_lt_16": float(np.mean(nn_dists < 16.0)),
        "max_y_4": max_y_4,
        "max_y_8": max_y_8,
        "max_y_16": max_y_16,
    }


def compute_dataset_support_profile(
    dataset: Sequence[Any],
    max_samples: int | None = None,
) -> List[Dict[str, Any]]:
    """Compute spatial statistics for all samples in a dataset."""
    n_samples = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    records = []
    for idx in range(n_samples):
        sample = dataset[idx]
        img = sample["image"]
        h, w = int(img.shape[-2]), int(img.shape[-1])
        pts = np.asarray(sample["gt_points"], dtype=np.float32).reshape(-1, 2)
        stats = compute_image_spatial_statistics((h, w), pts)
        stats["index"] = idx
        stats["img_path"] = sample.get("img_path", f"sample_{idx}")
        stats["height"] = h
        stats["width"] = w
        records.append(stats)
    return records


def compute_relative_percentiles(
    query_records: List[Dict[str, Any]],
    reference_records: List[Dict[str, Any]],
    keys: Sequence[str] = (
        "gt_count",
        "density_10k",
        "nn_median",
        "nn_p10",
        "nn_frac_lt_4",
        "max_y_4",
        "max_y_8",
        "max_y_16",
    ),
) -> List[Dict[str, float]]:
    """Compute empirical percentile ranks of query records relative to reference records."""
    ref_arrays: Dict[str, np.ndarray] = {}
    for k in keys:
        vals = [r[k] for r in reference_records if not math.isnan(r[k])]
        ref_arrays[k] = np.sort(np.asarray(vals, dtype=np.float64))

    pctl_records = []
    for q in query_records:
        row: Dict[str, float] = {}
        for k in keys:
            q_val = q[k]
            if math.isnan(q_val) or len(ref_arrays[k]) == 0:
                row[f"{k}_pctl"] = float("nan")
            else:
                rank = np.searchsorted(ref_arrays[k], q_val, side="right")
                row[f"{k}_pctl"] = float(rank / len(ref_arrays[k]) * 100.0)
        pctl_records.append(row)
    return pctl_records
