"""D-K / G-K: Inter-person separability collapse diagnostics across encoder depth.

Measures whether compact encoders merge neighboring-person representations
earlier in the depth hierarchy (C4 -> C8 -> C16 -> C32 -> P4) across raw
inter-person Euclidean spacing bins (d_min in pixels).

Note: On point-only annotations (without head bounding boxes or perspective maps),
spacing is reported as raw Euclidean pixel distance d_min, and scale is explicitly
uncontrolled.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Raw Euclidean spacing bins (in image pixels)
RAW_SPACING_BINS = {
    "le8": (0.0, 8.0),       # <= 8px (high risk of receptive field overlap at stride 8/16)
    "8_16": (8.0, 16.0),     # 8px - 16px (stride 16 cell boundary)
    "16_32": (16.0, 32.0),   # 16px - 32px
    "gt32": (32.0, 1e6),     # > 32px (isolated / well-separated)
}


def sample_feature_at_image_coord(
    feat: torch.Tensor,
    xy: torch.Tensor,
    img_h: int,
    img_w: int,
) -> torch.Tensor:
    """Sample feature tensor (1, C, H_feat, W_feat) at continuous image-space coordinates (N, 2).
    
    Under align_corners=False pixel-center mapping:
      Pixel center of continuous coordinate x in [0, img_w - 1] is x + 0.5.
      Normalized coordinate: u = -1.0 + 2.0 * (x + 0.5) / img_w.
      Similarly for y: v = -1.0 + 2.0 * (y + 0.5) / img_h.
    Returns: (N, C) feature vectors.
    """
    norm_x = ((xy[:, 0] + 0.5) / float(img_w)) * 2.0 - 1.0
    norm_y = ((xy[:, 1] + 0.5) / float(img_h)) * 2.0 - 1.0
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
        
    diff = points[:, None, :] - points[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Deduplicate neighbor pairs and bin by raw Euclidean distance d_ij
    pairs_by_bin: Dict[str, List[Tuple[int, int, float]]] = {k: [] for k in RAW_SPACING_BINS}
    seen_pairs = set()
    n_pts = len(points)
    for i in range(n_pts):
        j = int(np.argmin(dists[i]))
        pair_key = tuple(sorted((i, j)))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        
        d_raw = float(dists[i, j])
        for bname, (low, high) in RAW_SPACING_BINS.items():
            if low < d_raw <= high:
                pairs_by_bin[bname].append((i, j, d_raw))
                break

    for bname, pairs in pairs_by_bin.items():
        if len(pairs) > max_pairs_per_bin:
            idx = np.linspace(0, len(pairs) - 1, max_pairs_per_bin, dtype=int)
            pairs_by_bin[bname] = [pairs[k] for k in idx]

    bin_results: Dict[str, Dict[str, Any]] = {}
    for bname, pair_list in pairs_by_bin.items():
        if not pair_list:
            continue
        p1_coords = torch.from_numpy(np.array([points[i] for i, j, d in pair_list], dtype=np.float32)).to(device)
        p2_coords = torch.from_numpy(np.array([points[j] for i, j, d in pair_list], dtype=np.float32)).to(device)
        mid_coords = 0.5 * (p1_coords + p2_coords)
        
        stage_metrics: Dict[str, Any] = {
            "num_pairs": len(pair_list),
            "mean_raw_dist_px": float(np.mean([d for i, j, d in pair_list])),
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
        "mean_knn_spacing_px": float(np.mean(np.min(dists, axis=-1))),
        "bins": bin_results,
    }
