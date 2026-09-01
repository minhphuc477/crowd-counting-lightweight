"""D-K / G-K: Inter-person separability collapse diagnostics across encoder depth.

Measures whether compact encoders merge neighboring-person representations
earlier in the depth hierarchy (C8 / C16) than early layers (C4), causing local
counting and localization failure in dense crowd regions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SPACING_BINS = {
    "le8": (0.0, 8.0),
    "8_16": (8.0, 16.0),
    "16_32": (16.0, 32.0),
    "gt32": (32.0, 1e6),
}


def compute_knn_spacing(points: np.ndarray, k: int = 1) -> np.ndarray:
    """Compute nearest-neighbor Euclidean distance for each point."""
    if len(points) <= 1:
        return np.full(len(points), 1000.0, dtype=np.float32)
    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    knn_dist = np.min(dist, axis=-1)
    return knn_dist.astype(np.float32)


def sample_feature_at_coord(feat: torch.Tensor, xy: torch.Tensor, stride: float) -> torch.Tensor:
    """Bilinear sample feature tensor (1, C, H, W) at (x, y) coordinates.
    
    xy: (N, 2) in image pixel coordinates.
    stride: spatial reduction factor of the feature map (e.g. 4.0, 8.0, 16.0).
    Returns: (N, C) feature vectors.
    """
    b, c, h, w = feat.shape
    # Normalize image coordinates to [-1, 1] relative to feature grid
    # Pixel center convention: x in [0, W_orig], grid cell center is x / stride
    norm_x = (xy[:, 0] / (w * stride)) * 2.0 - 1.0
    norm_y = (xy[:, 1] / (h * stride)) * 2.0 - 1.0
    grid = torch.stack((norm_x, norm_y), dim=-1).unsqueeze(0).unsqueeze(2)  # (1, N, 1, 2)
    
    sampled = F.grid_sample(feat, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return sampled.squeeze(0).squeeze(-1).permute(1, 0)  # (N, C)


def evaluate_separability_single_image(
    model: nn.Module,
    image: torch.Tensor,  # (1, 3, H, W)
    points: np.ndarray,    # (N, 2)
    device: torch.device = torch.device("cpu"),
    max_pairs_per_bin: int = 100,
) -> Dict[str, Any]:
    """Evaluate inter-person separability across stages (C4, C8, C16, P4) on one image."""
    model.eval()
    if len(points) < 2:
        return {"num_points": len(points), "bins": {}}
    
    if image.ndim == 3:
        image = image.unsqueeze(0)
    image = image.to(device)
    
    with torch.no_grad():
        feats = model.backbone(image)
        p4 = model.neck(*feats)
        mass = model.head_out(model.head_act(model.head_norm(model.head_dw(p4))) if not model.use_repblock else model.head_refine(p4))
        mass = F.softplus(mass.float()) + model.eps_d
        
    stages: Dict[str, Tuple[torch.Tensor, float]] = {
        "C4": (feats[0], 4.0),
        "C8": (feats[1], 8.0),
        "C16": (feats[2], 16.0),
        "P4_mass": (mass, 4.0),
    }
    if len(feats) >= 4:
        stages["C32"] = (feats[3], 32.0)
        
    # Find candidate neighboring pairs (i, j)
    diff = points[:, None, :] - points[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Assign each pair to a spacing bin
    pairs_by_bin: Dict[str, List[Tuple[int, int, float]]] = {k: [] for k in SPACING_BINS}
    n_pts = len(points)
    for i in range(n_pts):
        j = int(np.argmin(dists[i]))
        d = float(dists[i, j])
        if i < j:  # unique pair
            for bname, (low, high) in SPACING_BINS.items():
                if low < d <= high:
                    if len(pairs_by_bin[bname]) < max_pairs_per_bin:
                        pairs_by_bin[bname].append((i, j, d))
                    break
                    
    bin_results: Dict[str, Dict[str, float]] = {}
    for bname, pair_list in pairs_by_bin.items():
        if not pair_list:
            continue
        p1_coords = torch.from_numpy(np.array([points[i] for i, j, d in pair_list], dtype=np.float32)).to(device)
        p2_coords = torch.from_numpy(np.array([points[j] for i, j, d in pair_list], dtype=np.float32)).to(device)
        mid_coords = 0.5 * (p1_coords + p2_coords)
        
        stage_metrics: Dict[str, float] = {"num_pairs": len(pair_list)}
        for sname, (sfeat, s_stride) in stages.items():
            f1 = sample_feature_at_coord(sfeat, p1_coords, s_stride)
            f2 = sample_feature_at_coord(sfeat, p2_coords, s_stride)
            f_mid = sample_feature_at_coord(sfeat, mid_coords, s_stride)
            
            if sname == "P4_mass":
                # For mass map (1 channel scalar): peak-to-trough ratio = min(m1, m2) / m_mid
                m1 = f1.squeeze(-1).clamp_min(1e-8)
                m2 = f2.squeeze(-1).clamp_min(1e-8)
                mmid = f_mid.squeeze(-1).clamp_min(1e-8)
                ptr = (torch.minimum(m1, m2) / mmid).cpu().numpy()
                stage_metrics[f"{sname}_peak_to_trough_ratio"] = float(np.median(ptr))
                stage_metrics[f"{sname}_merged_fraction"] = float(np.mean(ptr <= 1.0))
            else:
                # Feature cosine dissimilarity: 1 - cos(f_mid, (f1 + f2)/2)
                f_avg = 0.5 * (f1 + f2)
                cos_sim_mid = F.cosine_similarity(f_mid, f_avg, dim=-1)
                cos_dissim_indiv = 1.0 - F.cosine_similarity(f1, f2, dim=-1)
                stage_metrics[f"{sname}_cos_sim_to_midpoint"] = float(cos_sim_mid.mean().item())
                stage_metrics[f"{sname}_inter_person_dissimilarity"] = float(cos_dissim_indiv.mean().item())
                
        bin_results[bname] = stage_metrics

    return {
        "num_points": len(points),
        "knn_spacing_mean": float(np.mean(compute_knn_spacing(points))),
        "knn_spacing_p10": float(np.percentile(compute_knn_spacing(points), 10)),
        "bins": bin_results,
    }
