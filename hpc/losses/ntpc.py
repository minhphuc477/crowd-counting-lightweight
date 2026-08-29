"""Neural Tree-Pólya Crowd Counting (NTPC) Loss Module.

This module implements the unified probabilistic and deterministic hierarchical
formulations for the 5 decisive research ablation experiments:

  - R0: Multi-Scale Exact Regional Regression (Baseline)
  - R1: S-DCNet-style Deterministic Allocation (Prior-Art Match)
  - R2: Flat Dirichlet-Multinomial at Leaf 16 (No Hierarchy)
  - R3: Neural DTM Tree: 64 -> 32 -> 16 (Proposed Core Contribution)
  - R4: Full NTPC: R3 + Density-Adaptive Fine-Level 16 -> 8 (Proposed Full Method)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dirichlet_multinomial import dirichlet_multinomial_nll, normalize_positive_mass
from .negative_binomial import negative_binomial_nll_mean_dispersion


def sum_pool_mass_pyramid(
    mass: torch.Tensor,
    block_sizes: Tuple[int, ...] = (8, 16, 32, 64),
    stride: int = 4,
) -> Dict[int, torch.Tensor]:
    """Extract spatial count pyramid via linear sum-pooling from single mass map.
    
    Args:
        mass: (B, 1, H/4, W/4) positive mass density map.
        block_sizes: pixel block sizes.
        stride: stride of mass map relative to original image (default 4).
        
    Returns:
        dict: {block_size: (B, H_b, W_b)} pooled counts.
    """
    pyramid: Dict[int, torch.Tensor] = {}
    for bs in block_sizes:
        scale_factor = bs // stride
        if scale_factor < 1:
            raise ValueError(f"Block size {bs} smaller than mass stride {stride}")
        if scale_factor == 1:
            pooled = mass.squeeze(1)
        else:
            pooled = F.avg_pool2d(
                mass,
                kernel_size=scale_factor,
                stride=scale_factor,
            ).squeeze(1) * (scale_factor ** 2)
        pyramid[bs] = pooled
    return pyramid


def group_four_children(
    child_grid: torch.Tensor,
) -> torch.Tensor:
    """Group a 2D child grid (B, 2H, 2W) into (B, H, W, 4) under 2x2 parent blocks.
    
    Child ordering in 4-vector: [top-left, top-right, bottom-left, bottom-right].
    """
    b, h2, w2 = child_grid.shape
    h, w = h2 // 2, w2 // 2
    x = child_grid.view(b, h, 2, w, 2)
    x = x.permute(0, 1, 3, 2, 4).contiguous()
    return x.view(b, h, w, 4)


@dataclass
class NTPCConfig:
    """Configuration for Neural Tree-Pólya Crowd Counting loss."""
    mode: str = "r4_full_ntpc"  # "r0_exact" | "r1_deterministic" | "r2_flat_dm" | "r3_tree_dtm" | "r4_full_ntpc"
    
    # Root Negative-Binomial dispersion parameter
    root_dispersion: float = 50.0
    
    # Concentration parameters for Dirichlet-Multinomial allocations
    kappa_root64: float = 20.0
    kappa_64_32: float = 20.0
    kappa_32_16: float = 20.0
    kappa_16_8: float = 20.0
    kappa_flat16: float = 20.0
    
    # Density threshold for adaptive 16->8 fine-level supervision
    dense_threshold_16: float = 2.0
    
    # Component loss weights
    w_root_nb: float = 1.0
    w_root64: float = 1.0
    w_64_32: float = 1.0
    w_32_16: float = 1.0
    w_16_8: float = 1.0
    w_flat_16: float = 1.0
    w_exact_regression: float = 1.0
    w_deterministic_alloc: float = 1.0
    
    eps: float = 1e-8


class NTPCLoss(nn.Module):
    """Neural Tree-Pólya Crowd Counting loss criterion."""

    def __init__(self, cfg: NTPCConfig | None = None):
        super().__init__()
        self.cfg = cfg or NTPCConfig()

    def forward(
        self,
        mass: torch.Tensor,
        target_pyramid: Dict[int | str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute loss and detailed diagnostics.
        
        Args:
            mass: (B, 1, H/4, W/4) predicted positive mass density map.
            target_pyramid: dictionary containing ground-truth integer counts
                            for blocks {8, 16, 32, 64} and total count 'N'.
                            
        Returns:
            total_loss: scalar tensor for backpropagation.
            logs: dictionary of individual loss terms for monitoring.
        """
        # Ensure mass is positive and compute spatial pyramid
        mass = mass.float()
        pred_pyramid = sum_pool_mass_pyramid(mass, block_sizes=(8, 16, 32, 64), stride=4)
        
        pred_n = mass.sum(dim=(1, 2, 3))  # (B,)
        target_n = target_pyramid["N"].to(device=mass.device, dtype=torch.float32)  # (B,)
        b = mass.shape[0]

        logs: Dict[str, torch.Tensor] = {
            "root_nb": torch.tensor(0.0, device=mass.device),
            "root_to_64": torch.tensor(0.0, device=mass.device),
            "64_to_32": torch.tensor(0.0, device=mass.device),
            "32_to_16": torch.tensor(0.0, device=mass.device),
            "16_to_8_dense": torch.tensor(0.0, device=mass.device),
            "flat_16": torch.tensor(0.0, device=mass.device),
            "deterministic_alloc": torch.tensor(0.0, device=mass.device),
            "exact_regression": torch.tensor(0.0, device=mass.device),
            "total": torch.tensor(0.0, device=mass.device),
        }

        # -------------------------------------------------------------
        # MODE R0: Exact Regional Multi-Scale Regression Baseline
        # -------------------------------------------------------------
        if self.cfg.mode == "r0_exact":
            l_n = F.l1_loss(pred_n, target_n)
            l_64 = F.l1_loss(pred_pyramid[64], target_pyramid[64].float())
            l_32 = F.l1_loss(pred_pyramid[32], target_pyramid[32].float())
            l_16 = F.l1_loss(pred_pyramid[16], target_pyramid[16].float())
            
            total = l_n + l_64 + l_32 + l_16
            logs["exact_regression"] = total.detach()
            logs["total"] = total.detach()
            return total, logs

        # -------------------------------------------------------------
        # ALL PROBABILISTIC / S-DC MODES (R1, R2, R3, R4) INCLUDE ROOT NB
        # -------------------------------------------------------------
        l_root_nb = negative_binomial_nll_mean_dispersion(
            target=target_n,
            mean=pred_n,
            dispersion=self.cfg.root_dispersion,
            reduction="mean",
        )
        logs["root_nb"] = l_root_nb.detach()
        total = self.cfg.w_root_nb * l_root_nb

        # -------------------------------------------------------------
        # MODE R1: S-DCNet-style Deterministic Hierarchical Allocation
        # L_det = sum_p || Y_child(p) - Y_p * pi_p ||_1
        # -------------------------------------------------------------
        if self.cfg.mode == "r1_deterministic":
            # 64 -> 32 deterministic allocation
            y64 = target_pyramid[64].float()
            y32 = target_pyramid[32].float()
            m32 = pred_pyramid[32].float()
            
            y32_grouped = group_four_children(y32)  # (B, H64, W64, 4)
            m32_grouped = group_four_children(m32)  # (B, H64, W64, 4)
            pi32 = normalize_positive_mass(m32_grouped, dim=-1, eps=self.cfg.eps)
            
            # Expected deterministic allocation: Y_p * pi_p
            expected_y32 = y64.unsqueeze(-1) * pi32
            mask_64 = y64 > 0
            l_det_64_32 = F.l1_loss(expected_y32[mask_64], y32_grouped[mask_64]) if mask_64.any() else torch.tensor(0.0, device=mass.device)

            # 32 -> 16 deterministic allocation
            y16 = target_pyramid[16].float()
            m16 = pred_pyramid[16].float()
            y16_grouped = group_four_children(y16)  # (B, H32, W32, 4)
            m16_grouped = group_four_children(m16)  # (B, H32, W32, 4)
            pi16 = normalize_positive_mass(m16_grouped, dim=-1, eps=self.cfg.eps)
            
            expected_y16 = y32.unsqueeze(-1) * pi16
            mask_32 = y32 > 0
            l_det_32_16 = F.l1_loss(expected_y16[mask_32], y16_grouped[mask_32]) if mask_32.any() else torch.tensor(0.0, device=mass.device)

            l_det = l_det_64_32 + l_det_32_16
            total = total + self.cfg.w_deterministic_alloc * l_det
            logs["deterministic_alloc"] = l_det.detach()
            logs["total"] = total.detach()
            return total, logs

        # -------------------------------------------------------------
        # MODE R2: Flat Dirichlet-Multinomial at Leaf 16 (No Hierarchy)
        # N -> 28x28 (784 leaf blocks)
        # -------------------------------------------------------------
        if self.cfg.mode == "r2_flat_dm":
            y16_flat = target_pyramid[16].reshape(b, -1).float()  # (B, 784)
            m16_flat = pred_pyramid[16].reshape(b, -1).float()    # (B, 784)
            pi16_flat = normalize_positive_mass(m16_flat, dim=-1, eps=self.cfg.eps)
            
            l_flat16 = dirichlet_multinomial_nll(
                target_counts=y16_flat,
                probs=pi16_flat,
                concentration=self.cfg.kappa_flat16,
                valid_mask=target_n > 0,
                reduction="mean",
            )
            total = total + self.cfg.w_flat_16 * l_flat16
            logs["flat_16"] = l_flat16.detach()
            logs["total"] = total.detach()
            return total, logs

        # -------------------------------------------------------------
        # MODE R3 & R4: Neural DTM Tree Allocation (64 -> 32 -> 16)
        # -------------------------------------------------------------
        # Level 1: Root N -> 64 blocks
        y64_flat = target_pyramid[64].reshape(b, -1).float()  # (B, 49)
        m64_flat = pred_pyramid[64].reshape(b, -1).float()    # (B, 49)
        pi64_flat = normalize_positive_mass(m64_flat, dim=-1, eps=self.cfg.eps)
        
        l_root64 = dirichlet_multinomial_nll(
            target_counts=y64_flat,
            probs=pi64_flat,
            concentration=self.cfg.kappa_root64,
            valid_mask=target_n > 0,
            reduction="mean",
        )
        logs["root_to_64"] = l_root64.detach()
        total = total + self.cfg.w_root64 * l_root64

        # Level 2: 64 -> 32 blocks (conditional on 64 parent count)
        y64 = target_pyramid[64].float()
        y32 = target_pyramid[32].float()
        m32 = pred_pyramid[32].float()
        
        y32_grouped = group_four_children(y32)  # (B, H64, W64, 4)
        m32_grouped = group_four_children(m32)  # (B, H64, W64, 4)
        pi32 = normalize_positive_mass(m32_grouped, dim=-1, eps=self.cfg.eps)
        
        l_64_32 = dirichlet_multinomial_nll(
            target_counts=y32_grouped,
            probs=pi32,
            concentration=self.cfg.kappa_64_32,
            valid_mask=y64 > 0,
            reduction="mean",
        )
        logs["64_to_32"] = l_64_32.detach()
        total = total + self.cfg.w_64_32 * l_64_32

        # Level 3: 32 -> 16 blocks (conditional on 32 parent count)
        y16 = target_pyramid[16].float()
        m16 = pred_pyramid[16].float()
        
        y16_grouped = group_four_children(y16)  # (B, H32, W32, 4)
        m16_grouped = group_four_children(m16)  # (B, H32, W32, 4)
        pi16 = normalize_positive_mass(m16_grouped, dim=-1, eps=self.cfg.eps)
        
        l_32_16 = dirichlet_multinomial_nll(
            target_counts=y16_grouped,
            probs=pi16,
            concentration=self.cfg.kappa_32_16,
            valid_mask=y32 > 0,
            reduction="mean",
        )
        logs["32_to_16"] = l_32_16.detach()
        total = total + self.cfg.w_32_16 * l_32_16

        # -------------------------------------------------------------
        # MODE R4: Dense-Adaptive Fine-Level Allocation (16 -> 8)
        # Evaluated ONLY on congested parent blocks: Y_p^(16) >= tau_D
        # -------------------------------------------------------------
        if self.cfg.mode == "r4_full_ntpc" and 8 in target_pyramid:
            y8 = target_pyramid[8].float()
            m8 = pred_pyramid[8].float()
            
            y8_grouped = group_four_children(y8)  # (B, H16, W16, 4)
            m8_grouped = group_four_children(m8)  # (B, H16, W16, 4)
            pi8 = normalize_positive_mass(m8_grouped, dim=-1, eps=self.cfg.eps)
            
            dense_mask = y16 >= self.cfg.dense_threshold_16
            l_16_8 = dirichlet_multinomial_nll(
                target_counts=y8_grouped,
                probs=pi8,
                concentration=self.cfg.kappa_16_8,
                valid_mask=dense_mask,
                reduction="mean",
            )
            logs["16_to_8_dense"] = l_16_8.detach()
            total = total + self.cfg.w_16_8 * l_16_8

        logs["total"] = total.detach()
        return total, logs
