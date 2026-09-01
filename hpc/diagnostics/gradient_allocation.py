"""D-M / G-M: Foreground crowd vs background gradient allocation diagnostics.

Evaluates whether backward gradient energy is concentrated on crowd heads
or diluted by background distractors in a frozen model state.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def evaluate_gradient_allocation_single_batch(
    model: nn.Module,
    criterion: nn.Module,
    images: torch.Tensor,               # (B, 3, H, W)
    targets: Dict[int | str, torch.Tensor],
    points_list: Optional[List[np.ndarray]] = None,
    valid_hw: Optional[Tuple[int, int]] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Compute area-normalized gradient energy allocation in a strictly frozen model state."""
    # Keep model in eval mode to prevent BatchNorm buffer mutations
    model.eval()
    model.zero_grad(set_to_none=True)
    
    images = images.to(device)
    targets = {k: v.to(device) for k, v in targets.items()}
    
    captured_grads: Dict[str, torch.Tensor] = {}
    
    def make_hook(name: str):
        def hook(grad):
            captured_grads[name] = grad.detach()
        return hook
        
    with torch.set_grad_enabled(True):
        feats = model.backbone(images)
        for idx, name in enumerate(["C4", "C8", "C16", "C32"]):
            if idx < len(feats):
                feats[idx].register_hook(make_hook(name))
                
        p4 = model.neck(*feats)
        mass = model.head_out(model.head_act(model.head_norm(model.head_dw(p4))) if not model.use_repblock else model.head_refine(p4))
        mass = F.softplus(mass.float()) + model.eps_d
        
        loss, logs = criterion(mass, targets)
        loss.backward()
        
    results: Dict[str, float] = {"loss": float(loss.item())}
    
    # Analyze gradient allocation on captured feature stages
    for sname, g in captured_grads.items():
        # g: (B, C, H, W) -> spatial gradient energy density: (B, H, W)
        g_sq = (g ** 2).sum(dim=1)  # (B, H, W)
        
        stride = 4 if sname == "C4" else (8 if sname == "C8" else (16 if sname == "C16" else 32))
        if stride in targets:
            tgt_block = targets[stride].float()
            fg_mask = (tgt_block > 0).squeeze(1) if tgt_block.ndim == 4 else (tgt_block > 0)
        else:
            tgt_16 = targets.get(16, targets.get("N", None))
            if tgt_16 is not None and tgt_16.ndim >= 2:
                fg_mask = F.interpolate((tgt_16 > 0).float().unsqueeze(1), size=g_sq.shape[-2:], mode="nearest").squeeze(1).bool()
            else:
                fg_mask = torch.ones_like(g_sq, dtype=torch.bool)
                
        if valid_hw is not None:
            valid_h = math.ceil(valid_hw[0] / stride)
            valid_w = math.ceil(valid_hw[1] / stride)
            valid_mask = torch.zeros_like(g_sq, dtype=torch.bool)
            valid_mask[..., :valid_h, :valid_w] = True
        else:
            valid_mask = torch.ones_like(g_sq, dtype=torch.bool)

        fg_mask = fg_mask & valid_mask
        bg_mask = (~fg_mask) & valid_mask

        n_total = float(valid_mask.sum().item())
        n_fg = float(fg_mask.sum().item())
        n_bg = float(bg_mask.sum().item())
        
        fg_area_frac = n_fg / max(1.0, n_total)
        
        fg_energy = float(g_sq[fg_mask].sum().item()) if n_fg > 0 else 0.0
        bg_energy = float(g_sq[bg_mask].sum().item()) if n_bg > 0 else 0.0
        total_energy = fg_energy + bg_energy + 1e-12
        
        fg_energy_frac = fg_energy / total_energy
        
        # Area-normalized gradient enrichment: (E_fg / E_total) / (A_fg / A_total)
        # Value > 1.0 means foreground gradients are denser than background
        enrichment = fg_energy_frac / max(1e-6, fg_area_frac)
        
        # Mean gradient density ratio
        mean_fg_density = (fg_energy / max(1.0, n_fg))
        mean_bg_density = (bg_energy / max(1.0, n_bg))
        density_ratio = mean_fg_density / max(1e-12, mean_bg_density)
        
        results[f"{sname}_fg_area_fraction"] = fg_area_frac
        results[f"{sname}_fg_energy_fraction"] = fg_energy_frac
        results[f"{sname}_gradient_enrichment"] = enrichment
        results[f"{sname}_gradient_density_ratio"] = density_ratio
        
    model.zero_grad(set_to_none=True)
    return results
