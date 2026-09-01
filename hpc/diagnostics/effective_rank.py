"""D-L / G-L: Normalized effective representation rank collapse diagnostics.

Evaluates singular-value spectrum, participation ratio, and spectral entropy
of intermediate activations across depth (C4 -> C8 -> C16 -> C32) and across
density/foreground regions.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_spectral_rank_metrics(x: torch.Tensor, eps: float = 1e-10) -> Dict[str, float]:
    """Compute SVD spectrum, Participation Ratio, and Spectral Entropy on matrix X (N, C).
    
    Returns:
      nominal_channels: C
      participation_ratio: (sum sigma)^2 / sum(sigma^2)
      normalized_participation_ratio: PR / C  (in range [1/C, 1.0])
      spectral_entropy_rank: exp(entropy) / C (in range [1/C, 1.0])
      top1_energy_ratio: sigma_1 / sum(sigma)
      top5_energy_ratio: sum(sigma[:5]) / sum(sigma)
    """
    if x.ndim != 2:
        raise ValueError(f"Expected 2D matrix (N, C), got shape {tuple(x.shape)}")
    n, c = x.shape
    if n < 2 or c < 2:
        return {
            "nominal_channels": float(c),
            "participation_ratio": 1.0,
            "normalized_participation_ratio": 1.0 / max(1, c),
            "spectral_entropy_rank": 1.0 / max(1, c),
            "top1_energy_ratio": 1.0,
            "top5_energy_ratio": 1.0,
        }
    # Center features along sample dimension
    x_centered = x.float() - x.float().mean(dim=0, keepdim=True)
    # SVD: singular values of X
    # Using torch.linalg.svdvals for fast spectrum computation
    try:
        s = torch.linalg.svdvals(x_centered)
    except Exception:
        # Fallback to numpy if torch SVD fails on edge case
        s = torch.from_numpy(np.linalg.svd(x_centered.cpu().numpy(), compute_uv=False))
        
    s = s.clamp_min(eps)
    sum_s = s.sum()
    sum_s2 = (s ** 2).sum()
    
    # Participation Ratio
    pr = float(((sum_s ** 2) / sum_s2).item())
    norm_pr = pr / float(c)
    
    # Spectral Entropy
    p = s / sum_s
    entropy = -float((p * torch.log(p.clamp_min(eps))).sum().item())
    entropy_rank = float(np.exp(entropy) / c)
    
    top1 = float((s[0] / sum_s).item())
    top5 = float((s[:min(5, len(s))].sum() / sum_s).item())
    
    return {
        "nominal_channels": float(c),
        "participation_ratio": pr,
        "normalized_participation_ratio": norm_pr,
        "spectral_entropy_rank": entropy_rank,
        "top1_energy_ratio": top1,
        "top5_energy_ratio": top5,
    }


def evaluate_effective_rank_single_image(
    model: nn.Module,
    image: torch.Tensor,          # (1, 3, H, W)
    points: Optional[np.ndarray], # (N, 2)
    device: torch.device = torch.device("cpu"),
    fg_radius: float = 16.0,
) -> Dict[str, Any]:
    """Evaluate effective rank at C4, C8, C16, C32 in foreground vs whole image."""
    model.eval()
    if image.ndim == 3:
        image = image.unsqueeze(0)
    image = image.to(device)
    _, _, h, w = image.shape
    
    with torch.no_grad():
        feats = model.backbone(image)
        
    stages: Dict[str, Tuple[torch.Tensor, float]] = {
        "C4": (feats[0], 4.0),
        "C8": (feats[1], 8.0),
        "C16": (feats[2], 16.0),
    }
    if len(feats) >= 4:
        stages["C32"] = (feats[3], 32.0)
        
    stage_metrics: Dict[str, Dict[str, float]] = {}
    
    for sname, (sfeat, s_stride) in stages.items():
        _, c, sh, sw = sfeat.shape
        # Flatten feature map to (H*W, C)
        feat_flat = sfeat.squeeze(0).permute(1, 2, 0).reshape(-1, c)
        
        # Whole-image spectral metrics
        whole_metrics = compute_spectral_rank_metrics(feat_flat)
        
        # Foreground spectral metrics (if points exist)
        fg_metrics: Dict[str, float] = {}
        if points is not None and len(points) > 0:
            # Create boolean mask on feature grid
            # Coordinates in feature space: (pts / stride)
            pts_feat = points / s_stride
            grid_y, grid_x = torch.meshgrid(
                torch.arange(sh, device=device, dtype=torch.float32),
                torch.arange(sw, device=device, dtype=torch.float32),
                indexing="ij",
            )
            # Compute distance to nearest point
            pts_t = torch.tensor(pts_feat, dtype=torch.float32, device=device)
            # Reshape for broadcast: grid is (sh, sw, 1, 2), pts is (1, 1, N, 2)
            grid_pos = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(2)
            pts_pos = pts_t.unsqueeze(0).unsqueeze(0)
            dists = torch.sqrt(((grid_pos - pts_pos) ** 2).sum(dim=-1))
            min_dist, _ = dists.min(dim=-1)
            fg_mask = (min_dist <= (fg_radius / s_stride)).flatten()
            
            if fg_mask.sum() >= 4:
                fg_flat = feat_flat[fg_mask]
                fg_metrics = {f"fg_{k}": v for k, v in compute_spectral_rank_metrics(fg_flat).items()}
                
        stage_metrics[sname] = {**whole_metrics, **fg_metrics}
        
    # Depthwise rank decay: C16 rank / C4 rank
    c4_pr = stage_metrics["C4"]["normalized_participation_ratio"]
    c16_pr = stage_metrics["C16"]["normalized_participation_ratio"]
    depth_decay = c16_pr / max(1e-6, c4_pr)
    
    return {
        "depth_decay_c16_to_c4": depth_decay,
        "stages": stage_metrics,
    }
