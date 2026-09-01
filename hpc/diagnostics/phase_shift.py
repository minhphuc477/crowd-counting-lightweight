"""D-R / G-R: Sampling-phase and translation instability diagnostics.

Evaluates whether small +/-1, +/-2 pixel translations induce count and mass map
inconsistencies in compact models, and whether instability is concentrated in
dense/small-head regions versus boundary padding.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

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
    # If dx > 0 (shift right): pad left by dx, slice [:w]
    # If dx < 0 (shift left): pad right by -dx, slice [abs(dx):abs(dx)+w]
    pad_l = max(0, dx)
    pad_r = max(0, -dx)
    pad_t = max(0, dy)
    pad_b = max(0, -dy)
    padded = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode=mode)
    start_y = 0 if dy >= 0 else -dy
    start_x = 0 if dx >= 0 else -dx
    return padded[:, :, start_y:start_y + h, start_x:start_x + w]


def evaluate_phase_shift_single_image(
    model: nn.Module,
    image: torch.Tensor,  # (1, 3, H, W)
    shifts: Tuple[Tuple[int, int], ...] = DEFAULT_SHIFTS,
    device: torch.device = torch.device("cpu"),
    border_margin: int = 16,
) -> Dict[str, float]:
    """Compute phase shift variance and inverse-aligned mass consistency on one image."""
    model.eval()
    if image.ndim == 3:
        image = image.unsqueeze(0)
    image = image.to(device)
    _, _, h, w = image.shape
    
    counts: List[float] = []
    mass_maps: List[torch.Tensor] = []
    c4_feats: List[torch.Tensor] = []
    c8_feats: List[torch.Tensor] = []
    c16_feats: List[torch.Tensor] = []
    
    with torch.no_grad():
        for dx, dy in shifts:
            img_shifted = shift_tensor(image, dx, dy, mode="replicate")
            # Extract features and mass map
            feats = model.backbone(img_shifted)
            p4 = model.neck(*feats)
            mass = model.head_out(model.head_act(model.head_norm(model.head_dw(p4))) if not model.use_repblock else model.head_refine(p4))
            mass = F.softplus(mass.float()) + model.eps_d
            
            c4_feats.append(feats[0])
            c8_feats.append(feats[1])
            c16_feats.append(feats[2])
            
            cnt = float(mass.sum().item())
            counts.append(cnt)
            
            # Inverse-align mass map at stride 4 (scale translation dx/4, dy/4)
            # Since shift is in image pixels, round or shift mass by dx/4 if divisible, or resample
            # Exact pixel-space inverse shift using grid_sample:
            inv_dx_norm = -float(dx) * 2.0 / max(1, w)
            inv_dy_norm = -float(dy) * 2.0 / max(1, h)
            
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(-1, 1, mass.shape[2], device=device),
                torch.linspace(-1, 1, mass.shape[3], device=device),
                indexing="ij",
            )
            grid = torch.stack((grid_x + inv_dx_norm, grid_y + inv_dy_norm), dim=-1).unsqueeze(0)
            inv_mass = F.grid_sample(mass, grid, mode="bilinear", padding_mode="border", align_corners=False)
            mass_maps.append(inv_mass)
            
    base_count = counts[0]
    counts_arr = np.array(counts, dtype=np.float64)
    count_std = float(np.std(counts_arr))
    count_mean = float(np.mean(counts_arr))
    count_relative_std = float(count_std / max(1.0, count_mean))
    count_max_delta = float(np.max(np.abs(counts_arr - base_count)))
    count_max_rel_delta = float(count_max_delta / max(1.0, base_count))
    
    # Mass map inverse-aligned MAE (excluding border to isolate core from padding noise)
    base_mass = mass_maps[0]
    _, _, mh, mw = base_mass.shape
    b_margin_m = max(1, border_margin // 4)
    interior_mask = torch.zeros((1, 1, mh, mw), dtype=torch.bool, device=device)
    if mh > 2 * b_margin_m and mw > 2 * b_margin_m:
        interior_mask[:, :, b_margin_m:-b_margin_m, b_margin_m:-b_margin_m] = True
    else:
        interior_mask[:] = True
        
    mass_diffs: List[float] = []
    interior_mass_diffs: List[float] = []
    for m in mass_maps[1:]:
        diff = torch.abs(m - base_mass)
        mass_diffs.append(float(diff.mean().item()))
        interior_mass_diffs.append(float(diff[interior_mask].mean().item()))
        
    # Feature cosine similarity stability across 1-px shifts
    # Comparing (1, 0) and (0, 1) shifts with base (0, 0)
    c4_cos = float(F.cosine_similarity(c4_feats[0], c4_feats[1], dim=1).mean().item()) if len(c4_feats) > 1 else 1.0
    c8_cos = float(F.cosine_similarity(c8_feats[0], c8_feats[1], dim=1).mean().item()) if len(c8_feats) > 1 else 1.0
    c16_cos = float(F.cosine_similarity(c16_feats[0], c16_feats[1], dim=1).mean().item()) if len(c16_feats) > 1 else 1.0

    return {
        "base_count": base_count,
        "count_mean": count_mean,
        "count_std": count_std,
        "count_relative_std": count_relative_std,
        "count_max_delta": count_max_delta,
        "count_max_rel_delta": count_max_rel_delta,
        "mass_mae_mean": float(np.mean(mass_diffs)) if mass_diffs else 0.0,
        "interior_mass_mae_mean": float(np.mean(interior_mass_diffs)) if interior_mass_diffs else 0.0,
        "feature_c4_cos_sim": c4_cos,
        "feature_c8_cos_sim": c8_cos,
        "feature_c16_cos_sim": c16_cos,
    }
