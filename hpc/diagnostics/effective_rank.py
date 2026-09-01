"""D-L / G-L: Normalized effective representation rank collapse diagnostics.

Evaluates singular-value spectrum, covariance-energy participation ratio,
and spectral entropy of intermediate activations sampled strictly at matched
crowd point locations across depth (C4 -> C8 -> C16 -> C32).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .separability import sample_feature_at_image_coord


def compute_spectral_rank_metrics(x: torch.Tensor, eps: float = 1e-10) -> Dict[str, float]:
    """Compute SVD spectrum, Covariance-Energy Participation Ratio, and Spectral Entropy.
    
    x: Matrix of shape (M, C) where M is sample count, C is nominal channels.
    Normalized by min(M - 1, C) to eliminate spatial resolution bias across depth.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected 2D matrix (M, C), got shape {tuple(x.shape)}")
    m, c = x.shape
    max_rank = max(1, min(m - 1, c))
    
    if m < 2 or c < 2:
        return {
            "nominal_channels": float(c),
            "sample_count": float(m),
            "max_observable_rank": float(max_rank),
            "participation_ratio": 1.0,
            "normalized_participation_ratio": 1.0 / float(max_rank),
            "spectral_entropy_rank": 1.0 / float(max_rank),
            "top1_energy_ratio": 1.0,
        }
        
    # Center features along sample dimension
    x_centered = x.float() - x.float().mean(dim=0, keepdim=True)
    
    try:
        s = torch.linalg.svdvals(x_centered)
    except Exception:
        s = torch.from_numpy(np.linalg.svd(x_centered.cpu().numpy(), compute_uv=False))
        
    s = s.clamp_min(0.0)
    s2 = s.square()
    total_energy = s2.sum()

    tiny = torch.finfo(total_energy.dtype).tiny
    if not bool(torch.isfinite(total_energy)) or float(total_energy) <= tiny:
        return {
            "nominal_channels": float(c),
            "sample_count": float(m),
            "max_observable_rank": float(max_rank),
            "participation_ratio": 0.0,
            "normalized_participation_ratio": 0.0,
            "spectral_entropy_rank": 0.0,
            "top1_energy_ratio": 0.0,
        }

    # Scale-invariant spectral probabilities
    p = s2 / total_energy

    pr = float((1.0 / p.square().sum().clamp_min(eps)).item())
    norm_pr = pr / float(max_rank)

    entropy = -float((p * torch.log(p.clamp_min(eps))).sum().item())
    entropy_rank = float(np.exp(entropy) / float(max_rank))

    top1 = float(p[0].item())

    return {
        "nominal_channels": float(c),
        "sample_count": float(m),
        "max_observable_rank": float(max_rank),
        "participation_ratio": pr,
        "normalized_participation_ratio": norm_pr,
        "spectral_entropy_rank": entropy_rank,
        "top1_energy_ratio": top1,
    }


def evaluate_effective_rank_single_image(
    model: nn.Module,
    image: torch.Tensor,          # (1, 3, H, W)
    points: Optional[np.ndarray], # (N, 2)
    device: torch.device = torch.device("cpu"),
    max_crowd_samples: int = 128,
) -> Dict[str, Any]:
    """Evaluate effective rank at C4, C8, C16, C32 strictly sampled at matched crowd locations."""
    if points is None or len(points) < 4:
        return {
            "valid": False,
            "reason": "fewer_than_4_crowd_points",
            "matched_sample_count": int(0 if points is None else len(points)),
            "stages": {},
        }
        
    model.eval()
    if image.ndim == 3:
        image = image.unsqueeze(0)
    image = image.to(device)
    _, _, img_h, img_w = image.shape
    
    with torch.no_grad():
        feats = model.backbone(image)
        
    stages: Dict[str, Tuple[torch.Tensor, int]] = {
        "C4": (feats[0], 4),
        "C8": (feats[1], 8),
        "C16": (feats[2], 16),
    }
    if len(feats) >= 4:
        stages["C32"] = (feats[3], 32)
        
    points_np = np.asarray(points, dtype=np.float32)
    # Make subset independent of annotation ordering
    order = np.lexsort((points_np[:, 1], points_np[:, 0]))
    points_sorted = points_np[order]
    n_pts = len(points_sorted)

    if n_pts > max_crowd_samples:
        rng = np.random.default_rng(20260901)
        indices = np.sort(rng.choice(n_pts, size=max_crowd_samples, replace=False))
        sample_coords = points_sorted[indices]
    else:
        sample_coords = points_sorted
    coords_t = torch.from_numpy(np.asarray(sample_coords, dtype=np.float32)).to(device)
        
    stage_metrics: Dict[str, Dict[str, float]] = {}
    for sname, (sfeat, reduction) in stages.items():
        sampled_feat = sample_feature_at_image_coord(sfeat, coords_t, reduction=reduction)
        stage_metrics[sname] = compute_spectral_rank_metrics(sampled_feat)
        
    # Depthwise decay ratios
    c4_pr = stage_metrics["C4"]["normalized_participation_ratio"]
    c16_pr = stage_metrics["C16"]["normalized_participation_ratio"]
    decay_16_4 = float(c16_pr / max(1e-6, c4_pr))
    
    res = {
        "valid": True,
        "sampling_mode": "crowd_points",
        "matched_sample_count": len(coords_t),
        "depth_decay_c16_to_c4": decay_16_4,
        "stages": stage_metrics,
    }
    if "C32" in stage_metrics:
        c32_pr = stage_metrics["C32"]["normalized_participation_ratio"]
        res["depth_decay_c32_to_c16"] = float(c32_pr / max(1e-6, c16_pr))
        res["depth_decay_c32_to_c4"] = float(c32_pr / max(1e-6, c4_pr))
        
    return res
