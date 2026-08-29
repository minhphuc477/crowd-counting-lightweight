import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import WeightedRandomSampler


def compute_image_density_and_luminance(image_path: str, points: np.ndarray) -> Tuple[float, float, int]:
    count = int(len(points))
    with Image.open(image_path) as img:
        w, h = img.size
        gray = img.convert("L").resize((64, 64), Image.Resampling.BILINEAR)
        lum = float(np.mean(np.asarray(gray)))
    area = max(float(w * h), 1.0)
    density_scalar = math.log1p(1e6 * count / area)
    return density_scalar, lum, count


def build_density_luminance_sampler(
    image_paths: List[str],
    points_list: List[np.ndarray],
    num_density_bins: int = 5,
    num_luminance_bins: int = 4,
    power: float = 0.5,
    num_samples: Optional[int] = None,
) -> Tuple[WeightedRandomSampler, Dict[str, Any]]:
    """2D density/luminance sampler with a dedicated empty-scene density bin."""
    if len(image_paths) != len(points_list):
        raise ValueError("image_paths and points_list length mismatch")
    if not image_paths:
        raise ValueError("Cannot build sampler for an empty dataset")
    if num_density_bins < 2:
        raise ValueError("num_density_bins must be >=2 so bin 0 can be reserved for empty scenes")
    if num_luminance_bins < 1:
        raise ValueError("num_luminance_bins must be >=1")
    if power < 0:
        raise ValueError("power must be non-negative")

    stats_rows = [compute_image_density_and_luminance(p, pts) for p, pts in zip(image_paths, points_list)]
    densities = np.asarray([r[0] for r in stats_rows], dtype=np.float64)
    luminances = np.asarray([r[1] for r in stats_rows], dtype=np.float64)
    counts = np.asarray([r[2] for r in stats_rows], dtype=np.int64)

    # Reserve density bin 0 for true empty images. Positive images use bins 1..K-1.
    d_bins = np.zeros(len(counts), dtype=np.int64)
    pos = counts > 0
    n_pos_bins = num_density_bins - 1
    if np.any(pos):
        pos_density = densities[pos]
        if n_pos_bins > 1:
            d_quantiles = np.quantile(pos_density, np.linspace(0, 1, n_pos_bins + 1)[1:-1])
            d_bins[pos] = 1 + np.digitize(pos_density, d_quantiles)
        else:
            d_quantiles = np.array([], dtype=np.float64)
            d_bins[pos] = 1
    else:
        d_quantiles = np.array([], dtype=np.float64)

    l_quantiles = (
        np.quantile(luminances, np.linspace(0, 1, num_luminance_bins + 1)[1:-1])
        if num_luminance_bins > 1
        else np.array([], dtype=np.float64)
    )
    l_bins = np.digitize(luminances, l_quantiles)

    group_counts: Dict[Tuple[int, int], int] = {}
    for db, lb in zip(d_bins, l_bins):
        key = (int(db), int(lb))
        group_counts[key] = group_counts.get(key, 0) + 1

    weights = [1.0 / (group_counts[(int(db), int(lb))] ** power) for db, lb in zip(d_bins, l_bins)]
    weights_tensor = torch.tensor(weights, dtype=torch.double)
    total_samples = int(num_samples) if num_samples is not None else len(image_paths)
    if total_samples <= 0:
        raise ValueError("num_samples must be positive")

    sampler = WeightedRandomSampler(weights_tensor, num_samples=total_samples, replacement=True)
    return sampler, {
        "positive_density_quantiles": d_quantiles.tolist(),
        "luminance_quantiles": l_quantiles.tolist(),
        "group_counts": {f"{k[0]}_{k[1]}": v for k, v in group_counts.items()},
        "empty_count": int(np.sum(counts == 0)),
        "mean_count": float(np.mean(counts)),
        "max_count": int(np.max(counts)),
    }


def build_density_quantile_sampler(
    points_list: List[np.ndarray],
    num_bins: int = 4,
    num_samples: Optional[int] = None,
) -> Tuple[WeightedRandomSampler, Dict[str, Any]]:
    """Builds a pure 4-quantile density-balanced sampler.

    Partitions dataset into exact quartiles (Q0: sparse, Q1: low-med, Q2: high-med, Q3: dense)
    and samples each quantile with equal probability (25% each).
    """
    counts = np.asarray([int(len(pts)) for pts in points_list], dtype=np.float64)
    N = len(counts)
    if N == 0:
        raise ValueError("Cannot build sampler for empty dataset")

    # Compute quartile boundaries: 25%, 50%, 75%
    quantiles = np.quantile(counts, np.linspace(0, 1, num_bins + 1)[1:-1])
    bin_ids = np.digitize(counts, quantiles)  # 0, 1, 2, 3

    bin_counts = np.bincount(bin_ids, minlength=num_bins)

    # Each bin receives equal probability 1/num_bins; each sample in bin b receives 1 / (num_bins * bin_counts[b])
    weights = np.zeros(N, dtype=np.float64)
    for b in range(num_bins):
        c = bin_counts[b]
        if c > 0:
            weights[bin_ids == b] = 1.0 / (float(num_bins) * float(c))

    weights_tensor = torch.tensor(weights, dtype=torch.double)
    total_samples = int(num_samples) if num_samples is not None else N
    sampler = WeightedRandomSampler(weights_tensor, num_samples=total_samples, replacement=True)

    return sampler, {
        "quartile_boundaries": quantiles.tolist(),
        "bin_counts": bin_counts.tolist(),
        "min_count": float(np.min(counts)),
        "max_count": float(np.max(counts)),
    }
