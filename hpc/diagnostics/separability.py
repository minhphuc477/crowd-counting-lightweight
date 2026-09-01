"""D-K / G-K: Inter-person separability collapse diagnostics across encoder depth.

Measures whether compact encoders merge neighboring-person representations
earlier in the depth hierarchy (C4 -> C8 -> C16 -> C32 -> P4) across raw
inter-person Euclidean spacing bins (d_min in pixels).

Uses continuous aligned pixel-center coordinate mapping and same-scene far-control pairs.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree
import torch
import torch.nn as nn
import torch.nn.functional as F


# Raw Euclidean spacing bins (in image pixels)
RAW_SPACING_BINS = {
    "le8": (0.0, 8.0),       # <= 8px (high risk of receptive field overlap at stride 8/16)
    "8_16": (8.0, 16.0),     # 8px - 16px (stride 16 cell boundary)
    "16_32": (16.0, 32.0),   # 16px - 32px
    "gt32": (32.0, 1e6),     # > 32px (isolated / well-separated nearest neighbours)
}


def sample_feature_at_image_coord(
    feat: torch.Tensor,
    xy: torch.Tensor,
    reduction: int,
    origin_xy: Tuple[float, float] = (0.0, 0.0),
) -> torch.Tensor:
    """Sample convolutional feature maps at zero-based image pixel-center coordinates.

    For the current timm MobileNetV4 with symmetric odd-kernel padding:
        image_x = origin_x + feature_x * reduction
        image_y = origin_y + feature_y * reduction

    grid_sample uses align_corners=False.
    """
    if feat.ndim != 4 or feat.shape[0] != 1:
        raise ValueError(f"Expected feature shape (1,C,H,W), got {tuple(feat.shape)}")
    if reduction <= 0:
        raise ValueError(f"reduction must be positive, got {reduction}")

    _, _, feat_h, feat_w = feat.shape
    xy = xy.to(device=feat.device, dtype=torch.float32)
    origin_x, origin_y = origin_xy

    feat_x = (xy[:, 0] - float(origin_x)) / float(reduction)
    feat_y = (xy[:, 1] - float(origin_y)) / float(reduction)

    # Feature-index coordinate -> grid_sample coordinate for align_corners=False
    norm_x = 2.0 * (feat_x + 0.5) / float(feat_w) - 1.0
    norm_y = 2.0 * (feat_y + 0.5) / float(feat_h) - 1.0

    grid = torch.stack((norm_x, norm_y), dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        feat,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return sampled[0, :, :, 0].transpose(0, 1).contiguous()


def deterministic_far_control_pairs(
    points: np.ndarray,
    min_distance_px: float = 32.0,
    max_pairs: int = 100,
    oversample_factor: int = 10,
) -> List[Tuple[int, int, float]]:
    """Deterministically sample far pairs from the SAME image.

    This is independent of nearest-neighbour membership and therefore acts as the
    control pool for near-pair separability.
    """
    pts = np.asarray(points, dtype=np.float32)
    n = len(pts)
    if n < 2 or max_pairs <= 0:
        return []

    # Stable image-specific RNG seed; annotation ordering does not matter
    # because evaluate_separability_single_image sorts points beforehand.
    digest = hashlib.blake2b(pts.tobytes(), digest_size=8).digest()
    seed = int.from_bytes(digest, byteorder="little", signed=False)
    rng = np.random.default_rng(seed)

    target_candidates = max(max_pairs, max_pairs * oversample_factor)
    found: Dict[Tuple[int, int], float] = {}
    max_trials = max(2000, target_candidates * 100)

    for _ in range(max_trials):
        if len(found) >= target_candidates:
            break
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n - 1))
        if j >= i:
            j += 1
        a, b = sorted((i, j))
        key = (a, b)
        if key in found:
            continue
        d = float(np.linalg.norm(pts[a] - pts[b]))
        if d > min_distance_px:
            found[key] = d

    pairs = [(i, j, d) for (i, j), d in found.items()]
    pairs.sort(key=lambda x: (x[2], x[0], x[1]))
    if len(pairs) > max_pairs:
        idx = np.linspace(0, len(pairs) - 1, max_pairs, dtype=int)
        pairs = [pairs[k] for k in idx]
    return pairs


def evaluate_separability_single_image(
    model: nn.Module,
    image: torch.Tensor,  # (1, 3, H, W)
    points: np.ndarray,    # (N, 2)
    device: torch.device = torch.device("cpu"),
    max_pairs_per_bin: int = 100,
) -> Dict[str, Any]:
    """Evaluate inter-person separability across stages (C4, C8, C16, C32, P4) on one image."""
    model.eval()
    if len(points) < 2:
        return {"num_points": len(points), "bins": {}}

    if image.ndim == 3:
        image = image.unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        feats = model.backbone(image)
        p4 = model.neck(*feats)
        mass = model.mass_from_p4(p4)

    stages: Dict[str, Tuple[torch.Tensor, int]] = {
        "C4": (feats[0], 4),
        "C8": (feats[1], 8),
        "C16": (feats[2], 16),
        "P4_mass": (mass, 4),
    }
    if len(feats) >= 4:
        stages["C32"] = (feats[3], 32)

    points_np = np.asarray(points, dtype=np.float32)
    # Remove annotation-order dependence entirely
    order = np.lexsort((points_np[:, 1], points_np[:, 0]))
    points_np = points_np[order]

    tree = cKDTree(points_np)
    # k=2 because first neighbor is the query point itself
    nn_distances, nn_indices = tree.query(points_np, k=2)
    nearest_dist = nn_distances[:, 1].astype(np.float32)
    nearest_idx = nn_indices[:, 1].astype(np.int64)

    # Deduplicate neighbor pairs and bin by raw Euclidean distance d_ij
    pairs_by_bin: Dict[str, List[Tuple[int, int, float]]] = {k: [] for k in RAW_SPACING_BINS}
    seen_pairs = set()
    for i, (j, d_raw) in enumerate(zip(nearest_idx.tolist(), nearest_dist.tolist())):
        j = int(j)
        pair_key = tuple(sorted((i, j)))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        for bname, (low, high) in RAW_SPACING_BINS.items():
            if low < d_raw <= high:
                pairs_by_bin[bname].append((i, j, float(d_raw)))
                break

    for bname, pairs in pairs_by_bin.items():
        pairs = sorted(pairs, key=lambda x: (x[2], x[0], x[1]))
        if len(pairs) > max_pairs_per_bin:
            idx = np.linspace(0, len(pairs) - 1, max_pairs_per_bin, dtype=int)
            pairs = [pairs[k] for k in idx]
        pairs_by_bin[bname] = pairs

    # Add independent same-scene far control pool
    pairs_by_bin["far_control_gt32"] = deterministic_far_control_pairs(
        points_np,
        min_distance_px=32.0,
        max_pairs=max_pairs_per_bin,
    )

    bin_results: Dict[str, Dict[str, Any]] = {}
    for bname, pair_list in pairs_by_bin.items():
        if not pair_list:
            continue
        p1_coords = torch.from_numpy(np.asarray([points_np[i] for i, j, d in pair_list], dtype=np.float32)).to(device)
        p2_coords = torch.from_numpy(np.asarray([points_np[j] for i, j, d in pair_list], dtype=np.float32)).to(device)
        mid_coords = 0.5 * (p1_coords + p2_coords)

        stage_metrics: Dict[str, Any] = {
            "num_pairs": len(pair_list),
            "mean_raw_dist_px": float(np.mean([d for i, j, d in pair_list])),
        }
        for sname, (sfeat, reduction) in stages.items():
            f1 = sample_feature_at_image_coord(sfeat, p1_coords, reduction=reduction)
            f2 = sample_feature_at_image_coord(sfeat, p2_coords, reduction=reduction)
            f_mid = sample_feature_at_image_coord(sfeat, mid_coords, reduction=reduction)

            if sname == "P4_mass":
                # For scalar mass map: peak-to-trough ratio = min(m1, m2) / m_mid
                m1 = f1.squeeze(-1).clamp_min(1e-8)
                m2 = f2.squeeze(-1).clamp_min(1e-8)
                mmid = f_mid.squeeze(-1).clamp_min(1e-8)
                ptr = (torch.minimum(m1, m2) / mmid).cpu().numpy()
                stage_metrics[f"{sname}_peak_to_trough_ratio"] = float(np.median(ptr))
                stage_metrics[f"{sname}_merged_fraction"] = float(np.mean(ptr <= 1.0))
                stage_metrics["P4_mass_merged_count"] = int(np.sum(ptr <= 1.0))
                stage_metrics["P4_mass_peak_to_trough_values"] = ptr.astype(float).tolist()
            else:
                f_avg = 0.5 * (f1 + f2)
                cos_sim_mid = F.cosine_similarity(f_mid, f_avg, dim=-1)
                cos_dissim_indiv = 1.0 - F.cosine_similarity(f1, f2, dim=-1)
                stage_metrics[f"{sname}_cos_sim_to_midpoint"] = float(cos_sim_mid.mean().item())
                stage_metrics[f"{sname}_inter_person_dissimilarity"] = float(cos_dissim_indiv.mean().item())
                stage_metrics[f"{sname}_inter_person_dissimilarity_sum"] = float(cos_dissim_indiv.sum().item())

        bin_results[bname] = stage_metrics

    return {
        "num_points": len(points),
        "mean_knn_spacing_px": float(nearest_dist.mean()),
        "bins": bin_results,
    }
