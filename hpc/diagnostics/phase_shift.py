"""D-R / G-R: Sampling-phase and translation instability diagnostics.

Evaluates whether small +/-1, +/-2 pixel translations induce count and mass map
inconsistencies on the central valid support using natural source crop shifts
to avoid all artificial padding and GroupNorm wrap-around boundary distortions.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_SHIFTS: Tuple[Tuple[int, int], ...] = (
    (0, 0),
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (-1, -1),
    (2, 0),
    (-2, 0),
    (0, 2),
    (0, -2),
    (2, 2),
    (-2, -2),
)


def natural_shift_view(
    image: torch.Tensor,
    dx: int,
    dy: int,
    max_shift: int,
) -> torch.Tensor:
    """Crop a shifted sub-window from original image canvas without boundary wrap-around."""
    _, _, h, w = image.shape
    out_h = h - 2 * max_shift
    out_w = w - 2 * max_shift
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"Image of size ({h}, {w}) too small for max_shift={max_shift}")
    y0 = max_shift - dy
    x0 = max_shift - dx
    return image[..., y0:y0 + out_h, x0:x0 + out_w]


def make_inverse_align_grid(
    h_feat: int,
    w_feat: int,
    dx_feat: float,
    dy_feat: float,
    device: torch.device,
) -> torch.Tensor:
    """Build grid_sample grid (1, H, W, 2) for inverse-aligning under align_corners=False."""
    u = -1.0 + (2.0 * torch.arange(w_feat, device=device, dtype=torch.float32) + 1.0) / float(w_feat)
    v = -1.0 + (2.0 * torch.arange(h_feat, device=device, dtype=torch.float32) + 1.0) / float(h_feat)
    grid_y, grid_x = torch.meshgrid(v, u, indexing="ij")
    
    delta_u = (2.0 * float(dx_feat)) / float(w_feat)
    delta_v = (2.0 * float(dy_feat)) / float(h_feat)
    
    grid = torch.stack((grid_x + delta_u, grid_y + delta_v), dim=-1).unsqueeze(0)
    return grid


def inverse_align_feature(
    feat: torch.Tensor,
    dx_img: int,
    dy_img: int,
    stride: float,
    device: torch.device,
) -> torch.Tensor:
    """Inverse-align feature tensor (1, C, H, W) by image-space shift (dx_img, dy_img)."""
    if dx_img == 0 and dy_img == 0:
        return feat
    _, _, h, w = feat.shape
    dx_feat = float(dx_img) / float(stride)
    dy_feat = float(dy_img) / float(stride)
    grid = make_inverse_align_grid(h, w, dx_feat, dy_feat, device)
    return F.grid_sample(feat, grid, mode="bilinear", padding_mode="border", align_corners=False)


def crop_valid_center(
    feat: torch.Tensor,
    margin_px: int,
    stride: int,
) -> torch.Tensor:
    """Crop the interior region of feature map, safely removed from boundary wrap-around."""
    m = math.ceil(margin_px / stride)
    if feat.shape[-2] <= 2 * m or feat.shape[-1] <= 2 * m:
        raise ValueError(
            f"margin_px={margin_px} (m={m}) too large for feature map of shape {feat.shape[-2:]}"
        )
    return feat[..., m:-m, m:-m]


def evaluate_phase_shift_single_image(
    model: nn.Module,
    image: torch.Tensor,  # (1, 3, H, W)
    shifts: Tuple[Tuple[int, int], ...] = DEFAULT_SHIFTS,
    device: torch.device = torch.device("cpu"),
    border_margin_px: int = 96,
) -> Dict[str, float]:
    """Compute phase shift variance and inverse-aligned consistency using natural shifted sub-windows."""
    model.eval()
    if image.ndim == 3:
        image = image.unsqueeze(0)
    image = image.to(device)
    _, _, original_h, original_w = image.shape
    
    max_shift = max(max(abs(dx), abs(dy)) for dx, dy in shifts)
    view_h = original_h - 2 * max_shift
    view_w = original_w - 2 * max_shift
    
    margin = border_margin_px
    while margin > 16 and (view_h <= 2 * margin or view_w <= 2 * margin):
        margin -= 16
        
    inv_mass_maps: List[torch.Tensor] = []
    inv_c4_feats: List[torch.Tensor] = []
    inv_c8_feats: List[torch.Tensor] = []
    inv_c16_feats: List[torch.Tensor] = []
    inv_c32_feats: List[torch.Tensor] = []
    has_c32 = False
    
    with torch.no_grad():
        for dx, dy in shifts:
            img_shifted = natural_shift_view(image, dx=dx, dy=dy, max_shift=max_shift)
            feats = model.backbone(img_shifted)
            p4 = model.neck(*feats)
            mass = model.mass_from_p4(p4)
            
            # Inverse-align and crop valid center
            inv_mass = inverse_align_feature(mass, dx, dy, stride=4.0, device=device)
            inv_mass = crop_valid_center(inv_mass, margin_px=margin, stride=4)
            inv_mass_maps.append(inv_mass)
            
            inv_c4 = inverse_align_feature(feats[0], dx, dy, stride=4.0, device=device)
            inv_c4 = crop_valid_center(inv_c4, margin_px=margin, stride=4)
            inv_c4_feats.append(inv_c4)
            
            inv_c8 = inverse_align_feature(feats[1], dx, dy, stride=8.0, device=device)
            inv_c8 = crop_valid_center(inv_c8, margin_px=margin, stride=8)
            inv_c8_feats.append(inv_c8)
            
            inv_c16 = inverse_align_feature(feats[2], dx, dy, stride=16.0, device=device)
            inv_c16 = crop_valid_center(inv_c16, margin_px=margin, stride=16)
            inv_c16_feats.append(inv_c16)
            
            if len(feats) >= 4:
                has_c32 = True
                inv_c32 = inverse_align_feature(feats[3], dx, dy, stride=32.0, device=device)
                inv_c32 = crop_valid_center(inv_c32, margin_px=margin, stride=32)
                inv_c32_feats.append(inv_c32)
                
    counts = [float(m.sum().item()) for m in inv_mass_maps]
    base_count = counts[0]
    counts_arr = np.asarray(counts, dtype=np.float64)
    count_mean = float(counts_arr.mean())
    count_std = float(counts_arr.std())
    count_relative_std = float(count_std / max(1.0, count_mean))
    count_max_rel_delta = float(np.max(np.abs(counts_arr - base_count)) / max(1.0, base_count))
    
    base_mass = inv_mass_maps[0]
    mass_diffs = [float((m - base_mass).abs().mean().item()) for m in inv_mass_maps[1:]]
    
    def compute_aligned_cos_sim(feat_list: List[torch.Tensor]) -> float:
        if len(feat_list) <= 1:
            return 1.0
        base = feat_list[0]
        vals = []
        for feat in feat_list[1:]:
            sim = F.cosine_similarity(base, feat, dim=1).mean()
            vals.append(float(sim.item()))
        return float(np.mean(vals)) if vals else 1.0

    c4_cos = compute_aligned_cos_sim(inv_c4_feats)
    c8_cos = compute_aligned_cos_sim(inv_c8_feats)
    c16_cos = compute_aligned_cos_sim(inv_c16_feats)
    c32_cos = compute_aligned_cos_sim(inv_c32_feats) if has_c32 else float("nan")

    res = {
        "base_count": base_count,
        "count_mean": count_mean,
        "count_std": count_std,
        "count_relative_std": count_relative_std,
        "count_max_rel_delta": count_max_rel_delta,
        "mass_mae_mean": float(np.mean(mass_diffs)) if mass_diffs else 0.0,
        "feature_c4_cos_sim_aligned": c4_cos,
        "feature_c8_cos_sim_aligned": c8_cos,
        "feature_c16_cos_sim_aligned": c16_cos,
        "effective_margin_px": float(margin),
    }
    if has_c32:
        res["feature_c32_cos_sim_aligned"] = c32_cos
    return res
