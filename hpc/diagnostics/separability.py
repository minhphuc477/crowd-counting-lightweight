"""D-K / G-K: Inter-person separability collapse diagnostics across encoder depth.

Measures whether compact encoders merge neighboring-person representations
earlier in the depth hierarchy (C8 / C16 / C32) after normalizing for local head scale,
causing local counting and localization failure.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Normalized spacing bins: r = d_ij / min(s_head_i, s_head_j)
NORMALIZED_SPACING_BINS = {
    "le_0p5": (0.0, 0.5),     # Severe physical crowding / overlapping heads
    "0p5_1p0": (0.5, 1.0),    # Touching / immediate neighbors
    "1p0_2p0": (1.0, 2.0),    # Near neighbors
    "gt_2p0": (2.0, 1e6),     # Isolated / well-separated heads
}


def compute_head_scale_proxies(points: np.ndarray, k: int = 3) -> np.ndarray:
    """Compute local head-scale proxy s_head for each point as mean distance to k-nearest neighbors.
    
    Standard proxy in crowd counting when explicit bounding boxes are unavailable.
    """
    n = len(points)
    if n <= 1:
        return np.full(n, 50.0, dtype=np.float32)
    k_actual = min(k, n - 1)
    diff = points[:, None, :] - points[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    # Sort distances for each point
    sorted_dists = np.sort(dists, axis=-1)
    knn_mean = np.mean(sorted_dists[:, :k_actual], axis=-1)
    return knn_mean.astype(np.float32)


def sample_feature_at_image_coord(
    feat: torch.Tensor,
    xy: torch.Tensor,
    img_h: int,
    img_w: int,
) -> torch.Tensor:
    """Sample feature tensor (1, C, H_feat, W_feat) at continuous image-space coordinates (N, 2).
    
    Under align_corners=False:
      u = -1.0 + (2.0 * x) / img_w
      v = -1.0 + (2.0 * y) / img_h
    Returns: (N, C) feature vectors.
    """
    norm_x = (xy[:, 0] / float(img_w)) * 2.0 - 1.0
    norm_y = (xy[:, 1] / float(img_h)) * 2.0 - 1.0
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
    """Evaluate inter-person separability across stages (C4, C8, C16, C32, P4) on one image."""
    model.eval()
    if len(points) < 2:
        return {"num_points": len(points), "bins": {}}
    
    if image.ndim == 3:
        image = image.unsqueeze(0)
    image = image.to(device)
    _, _, img_h, img_w = image.shape
    
    with torch.no_grad():
        feats = model.backbone(image)
        p4 = model.neck(*feats)
        mass = model.head_out(model.head_act(model.head_norm(model.head_dw(p4))) if not model.use_repblock else model.head_refine(p4))
        mass = F.softplus(mass.float()) + model.eps_d
        
    stages: Dict[str, torch.Tensor] = {
        "C4": feats[0],
        "C8": feats[1],
        "C16": feats[2],
        "P4_mass": mass,
    }
    if len(feats) >= 4:
        stages["C32"] = feats[3]
        
    # Compute local head-scale proxies
    s_heads = compute_head_scale_proxies(points, k=3)
    
    diff = points[:, None, :] - points[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Deduplicate neighbor pairs and bin by normalized spacing r = d_ij / min(s_i, s_j)
    pairs_by_bin: Dict[str, List[Tuple[int, int, float, float]]] = {k: [] for k in NORMALIZED_SPACING_BINS}
    seen_pairs = set()
    n_pts = len(points)
    for i in range(n_pts):
        j = int(np.argmin(dists[i]))
        pair_key = tuple(sorted((i, j)))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        
        d_raw = float(dists[i, j])
        s_min = max(1e-4, float(min(s_heads[i], s_heads[j])))
        r_norm = d_raw / s_min
        
        for bname, (low, high) in NORMALIZED_SPACING_BINS.items():
            if low < r_norm <= high:
                if len(pairs_by_bin[bname]) < max_pairs_per_bin:
                    pairs_by_bin[bname].append((i, j, d_raw, r_norm))
                break
                
    bin_results: Dict[str, Dict[str, float]] = {}
    for bname, pair_list in pairs_by_bin.items():
        if not pair_list:
            continue
        p1_coords = torch.from_numpy(np.array([points[i] for i, j, d, r in pair_list], dtype=np.float32)).to(device)
        p2_coords = torch.from_numpy(np.array([points[j] for i, j, d, r in pair_list], dtype=np.float32)).to(device)
        mid_coords = 0.5 * (p1_coords + p2_coords)
        
        stage_metrics: Dict[str, float] = {
            "num_pairs": len(pair_list),
            "mean_raw_dist_px": float(np.mean([d for i, j, d, r in pair_list])),
            "mean_norm_ratio": float(np.mean([r for i, j, d, r in pair_list])),
        }
        for sname, sfeat in stages.items():
            f1 = sample_feature_at_image_coord(sfeat, p1_coords, img_h, img_w)
            f2 = sample_feature_at_image_coord(sfeat, p2_coords, img_h, img_w)
            f_mid = sample_feature_at_image_coord(sfeat, mid_coords, img_h, img_w)
            
            if sname == "P4_mass":
                # For scalar mass map: peak-to-trough ratio = min(m1, m2) / m_mid
                m1 = f1.squeeze(-1).clamp_min(1e-8)
                m2 = f2.squeeze(-1).clamp_min(1e-8)
                mmid = f_mid.squeeze(-1).clamp_min(1e-8)
                ptr = (torch.minimum(m1, m2) / mmid).cpu().numpy()
                stage_metrics[f"{sname}_peak_to_trough_ratio"] = float(np.median(ptr))
                stage_metrics[f"{sname}_merged_fraction"] = float(np.mean(ptr <= 1.0))
            else:
                f_avg = 0.5 * (f1 + f2)
                cos_sim_mid = F.cosine_similarity(f_mid, f_avg, dim=-1)
                cos_dissim_indiv = 1.0 - F.cosine_similarity(f1, f2, dim=-1)
                stage_metrics[f"{sname}_cos_sim_to_midpoint"] = float(cos_sim_mid.mean().item())
                stage_metrics[f"{sname}_inter_person_dissimilarity"] = float(cos_dissim_indiv.mean().item())
                
        bin_results[bname] = stage_metrics

    return {
        "num_points": len(points),
        "head_scale_proxy_mean": float(np.mean(s_heads)),
        "bins": bin_results,
    }
