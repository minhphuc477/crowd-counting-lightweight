"""Neural Tree-Pólya Crowd Counting (NTPC) Loss Module.

Implements the exact joint probabilistic Tree-Pólya NLL and controlled ablation formulations:
  - R0: Multi-Scale Exact Regional L1 Regression (Baseline)
  - R1: Deterministic Conserved Allocation (Prior-Art Match / S-DC style)
  - R2: Flat Dirichlet-Multinomial at Leaf 16 (No Hierarchy)
  - R3: Hierarchical Multinomial Tree: 64 -> 32 -> 16 (Hierarchy without Overdispersion)
  - R4: Neural DTM Tree: 64 -> 32 -> 16 (Base Joint Dirichlet-Tree Likelihood)
  - R5: Full NTPC: R4 + Density-Adaptive Fine-Level 16 -> 8 Auxiliary
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .negative_binomial import negative_binomial_nll_mean_dispersion


def block_sum(x: torch.Tensor, k: int) -> torch.Tensor:
    """Non-overlapping exact block sum via reshape."""
    has_channel = (x.ndim == 4)
    if not has_channel:
        x = x.unsqueeze(1)
        
    B, C, H, W = x.shape
    if H % k != 0 or W % k != 0:
        raise ValueError(f"Shape ({H}, {W}) not divisible by factor {k}")
        
    out = x.reshape(B, C, H // k, k, W // k, k).sum(dim=(3, 5))
    return out if has_channel else out.squeeze(1)


def sum_pool_mass_pyramid(
    mass: torch.Tensor,
    block_sizes: Tuple[int, ...] = (8, 16, 32, 64),
    stride: int = 4,
) -> Dict[int, torch.Tensor]:
    """Extract spatial count pyramid via non-overlapping block sums from single mass map.
    
    Args:
        mass: (B, 1, H/4, W/4) or (B, H/4, W/4) positive mass density map.
        block_sizes: image pixel block sizes.
        stride: stride of mass map relative to original image (default 4).
        
    Returns:
        dict: {block_size: (B, H_b, W_b)} pooled counts.
    """
    mass = mass.float()
    if mass.ndim == 3:
        mass = mass.unsqueeze(1)
        
    pyramid: Dict[int, torch.Tensor] = {}
    for bs in block_sizes:
        scale_factor = bs // stride
        if scale_factor < 1:
            raise ValueError(f"Block size {bs} smaller than mass stride {stride}")
        if scale_factor == 1:
            pooled = mass.squeeze(1)
        else:
            pooled = block_sum(mass, scale_factor).squeeze(1)
        pyramid[bs] = pooled
    return pyramid


def group_2x2_flat(x: torch.Tensor) -> torch.Tensor:
    """Group a 2D child grid (B, 2H, 2W) into (B, P, 4) where P = H * W.
    
    Child ordering in 4-vector: [top-left, top-right, bottom-left, bottom-right].
    """
    if x.ndim == 4 and x.shape[1] == 1:
        x = x.squeeze(1)
    B, H2, W2 = x.shape
    if H2 % 2 != 0 or W2 % 2 != 0:
        raise ValueError("Grid height and width must be even")
        
    tl = x[:, 0::2, 0::2]
    tr = x[:, 0::2, 1::2]
    bl = x[:, 1::2, 0::2]
    br = x[:, 1::2, 1::2]
    
    children = torch.stack([tl, tr, bl, br], dim=-1)  # (B, H, W, 4)
    return children.reshape(B, -1, 4)                 # (B, P, 4)


group_four_children = group_2x2_flat


def probs_from_positive_mass(mass: torch.Tensor, tiny: float = 1e-12) -> torch.Tensor:
    """Normalize positive mass into probability simplex without adding independent eps."""
    m = mass.float().clamp_min(tiny)
    return m / m.sum(dim=-1, keepdim=True)


def alpha_from_mass(mass: torch.Tensor, kappa: float, tiny: float = 1e-12) -> torch.Tensor:
    """Compute Dirichlet concentration vector alpha = kappa * pi."""
    pi = probs_from_positive_mass(mass, tiny=tiny)
    return float(kappa) * pi


def dm_nll_none(
    y: torch.Tensor,
    alpha: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-element Dirichlet-Multinomial NLL.
    
    Args:
        y: [..., K] non-negative exact integer counts.
        alpha: [..., K] strictly positive concentration parameters.
        
    Returns:
        [...] unreduced NLL tensor (float32). For n=0, NLL is mathematically exactly 0.
    """
    y = y.float()
    alpha = alpha.float().clamp_min(eps)

    n = y.sum(dim=-1)
    alpha0 = alpha.sum(dim=-1)

    log_prob = (
        torch.lgamma(n + 1.0)
        - torch.lgamma(y + 1.0).sum(dim=-1)
        + torch.lgamma(alpha0)
        - torch.lgamma(n + alpha0)
        + (torch.lgamma(y + alpha) - torch.lgamma(alpha)).sum(dim=-1)
    )

    nll = -log_prob
    # Zero counts contribute exactly zero loss
    zero_mask = (n == 0)
    if zero_mask.any():
        nll = torch.where(zero_mask, torch.zeros_like(nll), nll)
    return nll


def multinomial_nll_none(
    y: torch.Tensor,
    pi: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Per-element Multinomial NLL (DM limit as kappa -> infinity)."""
    y = y.float()
    pi = pi.float().clamp_min(eps)
    n = y.sum(dim=-1)

    log_prob = (
        torch.lgamma(n + 1.0)
        - torch.lgamma(y + 1.0).sum(dim=-1)
        + (y * torch.log(pi)).sum(dim=-1)
    )
    nll = -log_prob
    zero_mask = (n == 0)
    if zero_mask.any():
        nll = torch.where(zero_mask, torch.zeros_like(nll), nll)
    return nll


def tree_level_nll_per_image(
    y_child_map: torch.Tensor,
    mu_child_map: torch.Tensor,
    kappa: float,
) -> torch.Tensor:
    """Compute per-image sum of node DM NLLs for a tree level. Returns (B,)."""
    y = group_2x2_flat(y_child_map)
    m = group_2x2_flat(mu_child_map)
    alpha = alpha_from_mass(m, kappa)
    node_nll = dm_nll_none(y, alpha)  # (B, P)
    return node_nll.sum(dim=1)        # (B,)


def root_grid_nll_per_image(
    y64: torch.Tensor,
    mu64: torch.Tensor,
    kappa64: float,
) -> torch.Tensor:
    """Compute root-to-64 DM NLL per image. Returns (B,)."""
    y = y64.float().flatten(1)
    m = mu64.float().flatten(1)
    alpha = alpha_from_mass(m, kappa64)
    return dm_nll_none(y, alpha)      # (B,)


@dataclass
class NTPCConfig:
    """Configuration for Neural Tree-Pólya Crowd Counting loss."""
    mode: str = "r5_full_ntpc"  # "r0_exact" | "r1_deterministic" | "r2_flat_dm" | "r3_multinomial_tree" | "r4_dtm_tree" | "r5_full_ntpc"
    
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
        """
        mass = mass.float()
        pred_pyramid = sum_pool_mass_pyramid(mass, block_sizes=(8, 16, 32, 64), stride=4)
        
        pred_n = mass.flatten(1).sum(dim=1).reshape(-1)  # (B,)
        target_n = target_pyramid["N"].to(device=mass.device, dtype=torch.float32).reshape(-1)  # (B,)
        b = mass.shape[0]

        if pred_n.shape != target_n.shape:
            raise RuntimeError(f"Shape mismatch: pred_n {pred_n.shape} vs target_n {target_n.shape}")

        logs: Dict[str, torch.Tensor] = {
            "root_nb": torch.tensor(0.0, device=mass.device),
            "root_to_64": torch.tensor(0.0, device=mass.device),
            "64_to_32": torch.tensor(0.0, device=mass.device),
            "32_to_16": torch.tensor(0.0, device=mass.device),
            "16_to_8_dense": torch.tensor(0.0, device=mass.device),
            "flat_16": torch.tensor(0.0, device=mass.device),
            "multinomial_tree": torch.tensor(0.0, device=mass.device),
            "deterministic_alloc": torch.tensor(0.0, device=mass.device),
            "exact_regression": torch.tensor(0.0, device=mass.device),
            "total": torch.tensor(0.0, device=mass.device),
        }

        # -------------------------------------------------------------
        # MODE R0: Multi-Scale Exact Regional L1 Regression Baseline
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
        # ALL PROBABILISTIC / S-DC MODES (R1-R5) INCLUDE ROOT NB
        # -------------------------------------------------------------
        l_root_nb_per_image = negative_binomial_nll_mean_dispersion(
            target=target_n,
            mean=pred_n,
            dispersion=self.cfg.root_dispersion,
            reduction="none",
        )  # (B,)
        logs["root_nb"] = l_root_nb_per_image.mean().detach()

        # -------------------------------------------------------------
        # MODE R1: Deterministic Conserved Allocation (S-DCNet style)
        # L_det = (1/|V|) * sum_{p in V} (sum_c |Y_{p,c} - Y_p * pi_{p,c}|) / (Y_p + eps)
        # -------------------------------------------------------------
        if self.cfg.mode == "r1_deterministic":
            y64 = target_pyramid[64].float()
            y32_grouped = group_2x2_flat(target_pyramid[32].float())
            m32_grouped = group_2x2_flat(pred_pyramid[32].float())
            pi32 = probs_from_positive_mass(m32_grouped)
            
            y64_flat = y64.reshape(b, -1)
            expected_y32 = y64_flat.unsqueeze(-1) * pi32
            mask_64 = y64_flat > 0
            if mask_64.any():
                diff_64 = (expected_y32[mask_64] - y32_grouped[mask_64]).abs().sum(dim=-1)
                l_det_64_32 = (diff_64 / y64_flat[mask_64].clamp_min(1.0)).mean()
            else:
                l_det_64_32 = torch.tensor(0.0, device=mass.device)

            y32_flat = target_pyramid[32].float().reshape(b, -1)
            y16_grouped = group_2x2_flat(target_pyramid[16].float())
            m16_grouped = group_2x2_flat(pred_pyramid[16].float())
            pi16 = probs_from_positive_mass(m16_grouped)
            
            expected_y16 = y32_flat.unsqueeze(-1) * pi16
            mask_32 = y32_flat > 0
            if mask_32.any():
                diff_32 = (expected_y16[mask_32] - y16_grouped[mask_32]).abs().sum(dim=-1)
                l_det_32_16 = (diff_32 / y32_flat[mask_32].clamp_min(1.0)).mean()
            else:
                l_det_32_16 = torch.tensor(0.0, device=mass.device)

            l_det = l_det_64_32 + l_det_32_16
            total = self.cfg.w_root_nb * l_root_nb_per_image.mean() + self.cfg.w_deterministic_alloc * l_det
            logs["deterministic_alloc"] = l_det.detach()
            logs["total"] = total.detach()
            return total, logs

        # -------------------------------------------------------------
        # MODE R2: Flat Dirichlet-Multinomial at Leaf 16 (No Hierarchy)
        # N -> all leaf 16x16 blocks
        # -------------------------------------------------------------
        if self.cfg.mode == "r2_flat_dm":
            y16_flat = target_pyramid[16].float().flatten(1)
            m16_flat = pred_pyramid[16].float().flatten(1)
            alpha_flat = alpha_from_mass(m16_flat, self.cfg.kappa_flat16)
            l_flat16_per_image = dm_nll_none(y16_flat, alpha_flat)  # (B,)
            
            total_per_image = self.cfg.w_root_nb * l_root_nb_per_image + self.cfg.w_flat_16 * l_flat16_per_image
            total = total_per_image.mean()
            logs["flat_16"] = l_flat16_per_image.mean().detach()
            logs["total"] = total.detach()
            return total, logs

        # -------------------------------------------------------------
        # MODE R3: Hierarchical Multinomial Tree (64 -> 32 -> 16)
        # -------------------------------------------------------------
        if self.cfg.mode == "r3_multinomial_tree":
            # Root -> 64
            y64_flat = target_pyramid[64].float().flatten(1)
            m64_flat = pred_pyramid[64].float().flatten(1)
            pi64_flat = probs_from_positive_mass(m64_flat)
            l_multi_r64_per_image = multinomial_nll_none(y64_flat, pi64_flat)  # (B,)

            # 64 -> 32
            y32_grouped = group_2x2_flat(target_pyramid[32].float())
            m32_grouped = group_2x2_flat(pred_pyramid[32].float())
            pi32 = probs_from_positive_mass(m32_grouped)
            l_multi_64_32_per_image = multinomial_nll_none(y32_grouped, pi32).sum(dim=1)  # (B,)

            # 32 -> 16
            y16_grouped = group_2x2_flat(target_pyramid[16].float())
            m16_grouped = group_2x2_flat(pred_pyramid[16].float())
            pi16 = probs_from_positive_mass(m16_grouped)
            l_multi_32_16_per_image = multinomial_nll_none(y16_grouped, pi16).sum(dim=1)  # (B,)

            l_tree_multi_per_image = l_multi_r64_per_image + l_multi_64_32_per_image + l_multi_32_16_per_image
            total_per_image = self.cfg.w_root_nb * l_root_nb_per_image + l_tree_multi_per_image
            total = total_per_image.mean()
            logs["multinomial_tree"] = l_tree_multi_per_image.mean().detach()
            logs["total"] = total.detach()
            return total, logs

        # -------------------------------------------------------------
        # MODE R4 & R5: Base Joint Neural DTM Tree Likelihood (Per-Image Joint NLL)
        # L_i = L_NB,i + L_root64,i + sum_p L_{64->32,i} + sum_p L_{32->16,i}
        # -------------------------------------------------------------
        l_root64_per_image = root_grid_nll_per_image(
            target_pyramid[64],
            pred_pyramid[64],
            self.cfg.kappa_root64,
        )  # (B,)

        l_64_32_per_image = tree_level_nll_per_image(
            target_pyramid[32],
            pred_pyramid[32],
            self.cfg.kappa_64_32,
        )  # (B,)

        l_32_16_per_image = tree_level_nll_per_image(
            target_pyramid[16],
            pred_pyramid[16],
            self.cfg.kappa_32_16,
        )  # (B,)

        l_base_joint_per_image = (
            self.cfg.w_root_nb * l_root_nb_per_image
            + self.cfg.w_root64 * l_root64_per_image
            + self.cfg.w_64_32 * l_64_32_per_image
            + self.cfg.w_32_16 * l_32_16_per_image
        )  # (B,)

        logs["root_to_64"] = l_root64_per_image.mean().detach()
        logs["64_to_32"] = l_64_32_per_image.mean().detach()
        logs["32_to_16"] = l_32_16_per_image.mean().detach()

        total = l_base_joint_per_image.mean()

        # -------------------------------------------------------------
        # MODE R5: Dense-Adaptive Fine-Level Auxiliary (16 -> 8)
        # Composite training objective: L_total = L_base_joint + lambda_8 * L_dense8
        # -------------------------------------------------------------
        if self.cfg.mode in ("r5_full_ntpc", "r4_full_ntpc") and 8 in target_pyramid:
            y8_grouped = group_2x2_flat(target_pyramid[8].float())
            m8_grouped = group_2x2_flat(pred_pyramid[8].float())
            alpha8 = alpha_from_mass(m8_grouped, self.cfg.kappa_16_8)

            node_nll_8 = dm_nll_none(y8_grouped, alpha8)  # (B, P)
            dense_mask = (target_pyramid[16].float().reshape(b, -1) >= float(self.cfg.dense_threshold_16))

            if dense_mask.any():
                l_dense8 = node_nll_8[dense_mask].mean()
            else:
                l_dense8 = torch.tensor(0.0, device=mass.device)

            logs["16_to_8_dense"] = l_dense8.detach()
            total = total + self.cfg.w_16_8 * l_dense8

        logs["total"] = total.detach()
        return total, logs
