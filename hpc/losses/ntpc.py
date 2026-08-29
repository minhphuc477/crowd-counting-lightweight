"""Neural Tree-Pólya Crowd Counting (NTPC) Loss Module.

Supports full tree hierarchy down to stride-4:
  N -> 64 -> 32 -> 16 -> 8 -> 4

Modes supported:
  - r0_exact: Multi-Scale Regional L1 Regression Baseline
  - r1_deterministic: S-DCNet Deterministic Allocation Baseline
  - r2_flat_dm: Flat Dirichlet-Multinomial at 16 (No hierarchy)
  - r3_multinomial_tree: Pure Multinomial Tree: 64 -> 32 -> 16
  - r4_dtm_tree16: DTM Tree to 16: Root -> 64 -> 32 -> 16
  - r4_dtm_tree8:  DTM Tree to 8:  Root -> 64 -> 32 -> 16 -> 8
  - r4_dtm_tree4:  Full DTM Tree to 4: Root -> 64 -> 32 -> 16 -> 8 -> 4 (Full Stride-4 Hierarchy)
  - r5_full_ntpc: Alias for full stride-4 hierarchy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .negative_binomial import negative_binomial_nll_mean_dispersion


def block_sum(x: torch.Tensor, k: int) -> torch.Tensor:
    """Non-overlapping exact sum pooling via reshape."""
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
    block_sizes: Tuple[int, ...] = (4, 8, 16, 32, 64),
    stride: int = 4,
) -> Dict[int, torch.Tensor]:
    """Extract spatial count pyramid via non-overlapping block sums from single mass map.
    
    mass: (B, 1, H/4, W/4) or (B, H/4, W/4) positive mass density map.
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


mass_pyramid = sum_pool_mass_pyramid


def group_2x2_flat(x: torch.Tensor) -> torch.Tensor:
    """Group a 2D child grid (B, 2H, 2W) into (B, P, 4) where P = H * W.
    
    Child ordering: [top-left, top-right, bottom-left, bottom-right].
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

    children = torch.stack([tl, tr, bl, br], dim=-1)
    return children.reshape(B, -1, 4)


group_four_children = group_2x2_flat


def probs_from_positive_mass(mass: torch.Tensor, tiny: float = 1e-12) -> torch.Tensor:
    """Normalize positive mass into probability simplex."""
    m = mass.float().clamp_min(tiny)
    return m / m.sum(dim=-1, keepdim=True)


mass_to_prob = probs_from_positive_mass


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
    
    For n=0 (empty parent), NLL is mathematically exactly 0 and gradient is 0.
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
    zero_mask = (n == 0)
    if zero_mask.any():
        nll = torch.where(zero_mask, torch.zeros_like(nll), nll)
    return nll


def dm_from_mass(
    y: torch.Tensor,
    child_mass: torch.Tensor,
    kappa: float,
) -> torch.Tensor:
    alpha = alpha_from_mass(child_mass, kappa=kappa)
    return dm_nll_none(y, alpha)


def multinomial_nll_none(
    y: torch.Tensor,
    pi: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Per-element Multinomial NLL."""
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


def tree_level_dm_nll(
    child_gt_map: torch.Tensor,
    child_pred_map: torch.Tensor,
    kappa: float,
) -> torch.Tensor:
    """Sum of node DM NLLs for a 2x2 split level per image. Returns (B,)."""
    gt_children = group_2x2_flat(child_gt_map)
    pred_children = group_2x2_flat(child_pred_map)
    node_nll = dm_from_mass(gt_children, pred_children, kappa)
    return node_nll.sum(dim=1)


tree_level_nll_per_image = tree_level_dm_nll


def root_to_64_nll(
    y64: torch.Tensor,
    mu64: torch.Tensor,
    kappa: float,
) -> torch.Tensor:
    """Root-to-64 DM NLL per image. Returns (B,)."""
    y = y64.float().flatten(1)
    m = mu64.float().flatten(1)
    return dm_from_mass(y, m, kappa)


root_grid_nll_per_image = root_to_64_nll


@dataclass
class NTPCConfig:
    mode: str = "r4_dtm_tree4"  # "r0_exact" | "r1_deterministic" | "r2_flat_dm" | "r3_multinomial_tree" | "r4_dtm_tree16" | "r4_dtm_tree8" | "r4_dtm_tree4" | "r5_full_ntpc"
    root_dispersion: float = 50.0
    kappa_root64: float = 20.0
    kappa_64_32: float = 20.0
    kappa_32_16: float = 20.0
    kappa_16_8: float = 20.0
    kappa_8_4: float = 20.0
    kappa_flat16: float = 20.0
    dense_threshold_16: float = 2.0
    w_root_nb: float = 1.0
    w_root64: float = 1.0
    w_64_32: float = 1.0
    w_32_16: float = 1.0
    w_16_8: float = 1.0
    w_8_4: float = 1.0
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
        mass = mass.float()
        pred_pyramid = sum_pool_mass_pyramid(mass, block_sizes=(4, 8, 16, 32, 64), stride=4)

        pred_n = mass.flatten(1).sum(dim=1).reshape(-1)
        target_n = target_pyramid["N"].to(device=mass.device, dtype=torch.float32).reshape(-1)
        b = mass.shape[0]

        logs: Dict[str, torch.Tensor] = {
            "root_nb": torch.tensor(0.0, device=mass.device),
            "root_to_64": torch.tensor(0.0, device=mass.device),
            "64_to_32": torch.tensor(0.0, device=mass.device),
            "32_to_16": torch.tensor(0.0, device=mass.device),
            "16_to_8": torch.tensor(0.0, device=mass.device),
            "8_to_4": torch.tensor(0.0, device=mass.device),
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
        )
        logs["root_nb"] = l_root_nb_per_image.mean().detach()

        # -------------------------------------------------------------
        # MODE R1: Deterministic Conserved Allocation (S-DCNet style)
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
        # -------------------------------------------------------------
        if self.cfg.mode == "r2_flat_dm":
            y16_flat = target_pyramid[16].float().flatten(1)
            m16_flat = pred_pyramid[16].float().flatten(1)
            alpha_flat = alpha_from_mass(m16_flat, self.cfg.kappa_flat16)
            l_flat16_per_image = dm_nll_none(y16_flat, alpha_flat)

            total_per_image = self.cfg.w_root_nb * l_root_nb_per_image + self.cfg.w_flat_16 * l_flat16_per_image
            total = total_per_image.mean()
            logs["flat_16"] = l_flat16_per_image.mean().detach()
            logs["total"] = total.detach()
            return total, logs

        # -------------------------------------------------------------
        # MODE R3: Hierarchical Multinomial Tree (64 -> 32 -> 16)
        # -------------------------------------------------------------
        if self.cfg.mode == "r3_multinomial_tree":
            y64_flat = target_pyramid[64].float().flatten(1)
            m64_flat = pred_pyramid[64].float().flatten(1)
            pi64_flat = probs_from_positive_mass(m64_flat)
            l_multi_r64_per_image = multinomial_nll_none(y64_flat, pi64_flat)

            y32_grouped = group_2x2_flat(target_pyramid[32].float())
            m32_grouped = group_2x2_flat(pred_pyramid[32].float())
            pi32 = probs_from_positive_mass(m32_grouped)
            l_multi_64_32_per_image = multinomial_nll_none(y32_grouped, pi32).sum(dim=1)

            y16_grouped = group_2x2_flat(target_pyramid[16].float())
            m16_grouped = group_2x2_flat(pred_pyramid[16].float())
            pi16 = probs_from_positive_mass(m16_grouped)
            l_multi_32_16_per_image = multinomial_nll_none(y16_grouped, pi16).sum(dim=1)

            l_tree_multi_per_image = l_multi_r64_per_image + l_multi_64_32_per_image + l_multi_32_16_per_image
            total_per_image = self.cfg.w_root_nb * l_root_nb_per_image + l_tree_multi_per_image
            total = total_per_image.mean()
            logs["multinomial_tree"] = l_tree_multi_per_image.mean().detach()
            logs["total"] = total.detach()
            return total, logs

        # -------------------------------------------------------------
        # DTM TREE MODES (R4 / R5 / T0 / T1 / T2)
        # -------------------------------------------------------------
        l_root64_per_image = root_to_64_nll(target_pyramid[64], pred_pyramid[64], self.cfg.kappa_root64)
        l_64_32_per_image = tree_level_dm_nll(target_pyramid[32], pred_pyramid[32], self.cfg.kappa_64_32)
        l_32_16_per_image = tree_level_dm_nll(target_pyramid[16], pred_pyramid[16], self.cfg.kappa_32_16)

        logs["root_to_64"] = l_root64_per_image.mean().detach()
        logs["64_to_32"] = l_64_32_per_image.mean().detach()
        logs["32_to_16"] = l_32_16_per_image.mean().detach()

        per_image_total = (
            self.cfg.w_root_nb * l_root_nb_per_image
            + self.cfg.w_root64 * l_root64_per_image
            + self.cfg.w_64_32 * l_64_32_per_image
            + self.cfg.w_32_16 * l_32_16_per_image
        )

        # Depth extension: 16 -> 8
        if self.cfg.mode in ("r4_dtm_tree8", "r4_dtm_tree4", "r4_dtm_tree", "r5_full_ntpc", "r4_full_ntpc") and 8 in target_pyramid:
            l_16_8_per_image = tree_level_dm_nll(target_pyramid[8], pred_pyramid[8], self.cfg.kappa_16_8)
            logs["16_to_8"] = l_16_8_per_image.mean().detach()
            per_image_total = per_image_total + self.cfg.w_16_8 * l_16_8_per_image

        # Depth extension: 8 -> 4 (Full Stride-4 Hierarchy)
        if self.cfg.mode in ("r4_dtm_tree4", "r5_full_ntpc", "r4_full_ntpc") and 4 in target_pyramid:
            l_8_4_per_image = tree_level_dm_nll(target_pyramid[4], pred_pyramid[4], self.cfg.kappa_8_4)
            logs["8_to_4"] = l_8_4_per_image.mean().detach()
            per_image_total = per_image_total + self.cfg.w_8_4 * l_8_4_per_image

        total = per_image_total.mean()
        logs["total"] = total.detach()
        return total, logs


FullNTPCLoss = NTPCLoss
