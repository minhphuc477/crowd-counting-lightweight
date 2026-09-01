"""D-M / G-M: Foreground crowd vs background gradient allocation diagnostics.

Evaluates whether backward gradient energy is concentrated on crowd heads
or diluted by background distractors in compact models.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

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
    device: torch.device = torch.device("cpu"),
    fg_radius: float = 16.0,
) -> Dict[str, float]:
    """Compute gradient energy allocation between foreground crowd and background."""
    model.train()
    images = images.to(device)
    targets = {k: v.to(device) for k, v in targets.items()}
    
    # Enable gradient tracking on intermediate backbone features via hooks
    captured_grads: Dict[str, torch.Tensor] = {}
    
    def make_hook(name: str):
        def hook(grad):
            captured_grads[name] = grad.detach()
        return hook
        
    feats = model.backbone(images)
    for idx, name in enumerate(["C4", "C8", "C16"]):
        if idx < len(feats):
            feats[idx].register_hook(make_hook(name))
            
    p4 = model.neck(*feats)
    mass = model.head_out(model.head_act(model.head_norm(model.head_dw(p4))) if not model.use_repblock else model.head_refine(p4))
    mass = F.softplus(mass.float()) + model.eps_d
    
    loss, logs = criterion(mass, targets)
    loss.backward()
    
    results: Dict[str, float] = {"loss": float(loss.item())}
    
    # Analyze gradient allocation on C4 / C8 / C16
    for sname in ["C4", "C8", "C16"]:
        if sname not in captured_grads:
            continue
        g = captured_grads[sname]  # (B, C, H, W)
        # Compute spatial gradient energy: (B, H, W)
        g_spatial = torch.norm(g, p=2, dim=1)  # (B, H, W)
        
        # Build foreground mask from target blocks or points
        # If target pyramid has block level 4 or 16, use target_pyramid > 0 as crowd mask
        stride = 4 if sname == "C4" else (8 if sname == "C8" else 16)
        if stride in targets:
            tgt_block = targets[stride].float()
            fg_mask = (tgt_block > 0).squeeze(1) if tgt_block.ndim == 4 else (tgt_block > 0)
        else:
            # Fallback: non-zero target mass at reduction 16 upsampled
            tgt_16 = targets.get(16, targets.get("N", None))
            if tgt_16 is not None and tgt_16.ndim >= 2:
                fg_mask = F.interpolate((tgt_16 > 0).float().unsqueeze(1), size=g_spatial.shape[-2:], mode="nearest").squeeze(1).bool()
            else:
                fg_mask = torch.ones_like(g_spatial, dtype=torch.bool)
                
        bg_mask = ~fg_mask
        
        fg_energy = float(((g_spatial[fg_mask] ** 2).sum()).item()) if fg_mask.any() else 0.0
        bg_energy = float(((g_spatial[bg_mask] ** 2).sum()).item()) if bg_mask.any() else 0.0
        total_energy = fg_energy + bg_energy + 1e-10
        
        fg_density = float(g_spatial[fg_mask].mean().item()) if fg_mask.any() else 0.0
        bg_density = float(g_spatial[bg_mask].mean().item()) if bg_mask.any() else 0.0
        
        results[f"{sname}_fg_energy_fraction"] = fg_energy / total_energy
        results[f"{sname}_fg_to_bg_density_ratio"] = fg_density / max(1e-8, bg_density)
        
    return results
