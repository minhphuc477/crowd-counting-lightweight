"""D-R / G-R: Sampling-phase and translation instability diagnostics.

Evaluates whether small +/-1, +/-2 pixel translations induce count and mass map
inconsistencies on the common valid interior support, eliminating border truncation
artifacts.
"""

from __future__ import annotations

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


def shift_tensor(x: torch.Tensor, dx: int, dy: int, mode: str = "replicate") -> torch.Tensor:
    """Shift 2D spatial tensor (B, C, H, W) by integer (dx, dy) with padding."""
    if dx == 0 and dy == 0:
        return x
    b, c, h, w = x.shape
    pad_l = max(0, dx)
    pad_r = max(0, -dx)
    pad_t = max(0, dy)
    pad_b = max(0, -dy)
    padded = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode=mode)
    start_y = 0 if dy >= 0 else -dy
    start_x = 0 if dx >= 0 else -dx
    return padded[:, :, start_y:start_y + h, start_x:start_x + w]


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


def evaluate_phase_shift_single_image(
    model: nn.Module,
    image: torch.Tensor,  # (1, 3, H, W)
    shifts: Tuple[Tuple[int, int], ...] = DEFAULT_SHIFTS,
    device: torch.device = torch.device("cpu"),
    border_margin_px: int = 8,
) -> Dict[str, float]:
    """Compute phase shift variance and inverse-aligned consistency on common valid interior support."""
    model.eval()
    if image.ndim == 3:
        image = image.unsqueeze(0)
    image = image.to(device)
    _, _, h, w = image.shape
    
    inv_mass_maps: List[torch.Tensor] = []
    inv_c4_feats: List[torch.Tensor] = []
    inv_c8_feats: List[torch.Tensor] = []
    inv_c16_feats: List[torch.Tensor] = []
    inv_c32_feats: List[torch.Tensor] = []
    has_c32 = False
    
    with torch.no_grad():
        for dx, dy in shifts:
            img_shifted = shift_tensor(image, dx, dy, mode="replicate")
            feats = model.backbone(img_shifted)
            p4 = model.neck(*feats)
            mass = model.head_out(model.head_act(model.head_norm(model.head_dw(p4))) if not model.use_repblock else model.head_refine(p4))
            mass = F.softplus(mass.float()) + model.eps_d
            
            # Inverse-align mass map at stride 4
            inv_mass = inverse_align_feature(mass, dx, dy, stride=4.0, device=device)
            inv_mass_maps.append(inv_mass)
            
            # Inverse-align backbone features at respective strides
            inv_c4 = inverse_align_feature(feats[0], dx, dy, stride=4.0, device=device)
            inv_c8 = inverse_align_feature(feats[1], dx, dy, stride=8.0, device=device)
            inv_c16 = inverse_align_feature(feats[2], dx, dy, stride=16.0, device=device)
            inv_c4_feats.append(inv_c4)
            inv_c8_feats.append(inv_c8)
            inv_c16_feats.append(inv_c16)
            
            if len(feats) >= 4:
                has_c32 = True
                inv_c32 = inverse_align_feature(feats[3], dx, dy, stride=32.0, device=device)
                inv_c32_feats.append(inv_c32)
                
    # Define common valid interior mask on mass map (stride 4)
    base_mass = inv_mass_maps[0]
    _, _, mh, mw = base_mass.shape
    b_margin_m = max(1, border_margin_px // 4)
    interior_mask = torch.zeros((1, 1, mh, mw), dtype=torch.bool, device=device)
    if mh > 2 * b_margin_m and mw > 2 * b_margin_m:
        interior_mask[:, :, b_margin_m:-b_margin_m, b_margin_m:-b_margin_m] = True
    else:
        interior_mask[:] = True
        
    # Measure interior count on common valid support for all shifts
    interior_counts: List[float] = []
    mass_diffs: List[float] = []
    interior_mass_diffs: List[float] = []
    
    for m in inv_mass_maps:
        int_cnt = float(m[interior_mask].sum().item())
        interior_counts.append(int_cnt)
        if len(interior_counts) > 1:
            diff = torch.abs(m - base_mass)
            mass_diffs.append(float(diff.mean().item()))
            interior_mass_diffs.append(float(diff[interior_mask].mean().item()))
            
    base_count = interior_counts[0]
    counts_arr = np.array(interior_counts, dtype=np.float64)
    count_std = float(np.std(counts_arr))
    count_mean = float(np.mean(counts_arr))
    count_relative_std = float(count_std / max(1.0, count_mean))
    count_max_delta = float(np.max(np.abs(counts_arr - base_count)))
    count_max_rel_delta = float(count_max_delta / max(1.0, base_count))
    
    # Helper to compute aligned feature cosine similarity on interior
    def compute_aligned_cos_sim(feat_list: List[torch.Tensor], stride: float) -> float:
        if len(feat_list) <= 1:
            return 1.0
        base_f = feat_list[0]
        _, _, fh, fw = base_f.shape
        f_margin = max(1, border_margin_px // int(stride))
        if fh > 2 * f_margin and fw > 2 * f_margin:
            f_mask = (slice(None), slice(None), slice(f_margin, -f_margin), slice(f_margin, -f_margin))
        else:
            f_mask = (slice(None), slice(None), slice(None), slice(None))
            
        sims: List[float] = []
        base_crop = base_f[f_mask]
        for f in feat_list[1:]:
            f_crop = f[f_mask]
            cos = F.cosine_similarity(base_crop, f_crop, dim=1).mean().item()
            sims.append(float(cos))
        return float(np.mean(sims)) if sims else 1.0

    c4_cos = compute_aligned_cos_sim(inv_c4_feats, stride=4.0)
    c8_cos = compute_aligned_cos_sim(inv_c8_feats, stride=8.0)
    c16_cos = compute_aligned_cos_sim(inv_c16_feats, stride=16.0)
    c32_cos = compute_aligned_cos_sim(inv_c32_feats, stride=32.0) if has_c32 else float("nan")

    res = {
        "interior_base_count": base_count,
        "interior_count_mean": count_mean,
        "interior_count_std": count_std,
        "interior_count_relative_std": count_relative_std,
        "interior_count_max_rel_delta": count_max_rel_delta,
        "interior_mass_mae_mean": float(np.mean(interior_mass_diffs)) if interior_mass_diffs else 0.0,
        "feature_c4_cos_sim_aligned": c4_cos,
        "feature_c8_cos_sim_aligned": c8_cos,
        "feature_c16_cos_sim_aligned": c16_cos,
    }
    if has_c32:
        res["feature_c32_cos_sim_aligned"] = c32_cos
    return res
