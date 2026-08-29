"""Probabilistic Hierarchy Loss Module for HPC-Lite: Adaptive Hierarchical Probabilistic Count Tree.

Implements the unified formulation:
  L = L_{TreeProb} + lambda_Z * L_{HardZero} + lambda_C * L_{LocalContrast} + lambda_M * L_{mass}

Where:
1. L_{TreeProb} = L_{Root-NB} + sum_{p in T} L_{Branch}(p)
   - Root: N ~ NB(mu_N, r)
   - Branching: (Y_{c1},...,Y_{c4}) | Y_p ~ Dirichlet-Multinomial(Y_p, kappa_B * pi) or Multinomial(Y_p, pi)
   - Adaptive Hierarchy: 64 -> 32 -> 16 -> (dense only) 8
2. L_{LocalContrast}: Supervised contrastive representation learning across discrete density classes on P4
3. L_{HardZero}: Top-rho false positive mining on empty background 16x16 leaves
4. L_{mass}: Exact multi-scale L1 calibration
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def group_quadtree_children(child_grid: torch.Tensor) -> torch.Tensor:
    """Group child spatial grid (B, 1, 2H, 2W) or (B, 2H, 2W) into parent quadtree groups (B, H, W, 4).
    
    The 4 children of parent (h, w) are located at indices:
      (2h, 2w), (2h, 2w+1), (2h+1, 2w), (2h+1, 2w+1).
    """
    if child_grid.ndim == 3:
        child_grid = child_grid.unsqueeze(1)  # (B, 1, 2H, 2W)
    B, C, H2, W2 = child_grid.shape
    H, W = H2 // 2, W2 // 2
    # (B, 1, H, 2, W, 2) -> (B, H, W, 4)
    grouped = child_grid.view(B, C, H, 2, W, 2).permute(0, 2, 4, 1, 3, 5).reshape(B, H, W, 4)
    return grouped


class RootNegativeBinomialLoss(nn.Module):
    """Negative-Binomial negative log-likelihood for total image crowd count N."""

    def __init__(self, dispersion: float = 10.0, eps: float = 1e-6):
        super().__init__()
        self.dispersion = float(dispersion)
        self.eps = float(eps)

    def forward(self, pred_mass: torch.Tensor, gt_count: torch.Tensor) -> torch.Tensor:
        """Compute root-level Negative Binomial NLL."""
        mu = pred_mass.view(-1).clamp(min=self.eps)
        y = gt_count.view(-1)
        r = self.dispersion

        log_prob = (
            torch.lgamma(y + r)
            - torch.lgamma(torch.tensor(r, device=mu.device, dtype=mu.dtype))
            - torch.lgamma(y + 1.0)
            + r * torch.log(r / (mu + r))
            + y * torch.log(mu / (mu + r))
        )
        return -log_prob.mean()


class AdaptiveHierarchicalCountTreeLoss(nn.Module):
    """Density-Adaptive Probabilistic Count Tree Loss across quadtree branching levels.
    
    Levels:
      - 64 -> 32 (Macro spatial partition)
      - 32 -> 16 (Meso spatial partition)
      - 16 -> 8  (Micro spatial partition, activated selectively on dense parents Y_p >= dense_threshold)
      
    Modes:
      - 'dirichlet_multinomial': (Y_1,...,Y_4) | Y_p ~ DM(Y_p, kappa_B * pi)
      - 'multinomial':           (Y_1,...,Y_4) | Y_p ~ Multinomial(Y_p, pi)
    """

    def __init__(
        self,
        mode: str = "dirichlet_multinomial",
        concentration_alpha: float = 10.0,
        dense_threshold_16: float = 10.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.mode = mode.lower()
        self.concentration_alpha = float(concentration_alpha)
        self.dense_threshold_16 = float(dense_threshold_16)
        self.eps = float(eps)

    def _compute_branch_nll(
        self,
        mu_children: torch.Tensor,
        y_children: torch.Tensor,
        parent_min_count: float = 1.0,
    ) -> torch.Tensor:
        """Compute branch negative log-likelihood for quadtree children."""
        mu_grouped = group_quadtree_children(mu_children)  # (B, H, W, 4)
        y_grouped = group_quadtree_children(y_children)    # (B, H, W, 4)

        mu_flat = mu_grouped.reshape(-1, 4)
        y_flat = y_grouped.reshape(-1, 4)

        y_p = y_flat.sum(dim=-1, keepdim=True)  # (N_parents, 1)
        active_mask = (y_p.squeeze(-1) >= parent_min_count)

        if not active_mask.any():
            return torch.tensor(0.0, device=mu_children.device, requires_grad=True)

        y_active = y_flat[active_mask]       # (M, 4)
        mu_active = mu_flat[active_mask]     # (M, 4)
        yp_active = y_p[active_mask]         # (M, 1)

        # Allocation probabilities: pi_i = mu_i / (sum(mu_j) + eps)
        pi = (mu_active / (mu_active.sum(dim=-1, keepdim=True) + self.eps)).clamp(min=self.eps)

        if self.mode == "multinomial":
            # Multinomial NLL = - sum_i Y_i * log(pi_i)
            nll = -torch.sum(y_active * torch.log(pi), dim=-1, keepdim=True)
        else:
            # Dirichlet-Multinomial NLL
            alpha_0 = self.concentration_alpha
            alpha = alpha_0 * pi
            nll = (
                torch.lgamma(yp_active + alpha_0)
                - torch.lgamma(torch.tensor(alpha_0, device=mu_children.device, dtype=mu_children.dtype))
                - torch.sum(torch.lgamma(y_active + alpha) - torch.lgamma(alpha), dim=-1, keepdim=True)
            )

        return nll.mean()

    def forward(
        self,
        mu_blocks: Dict[int, torch.Tensor],
        gt_blocks: Dict[int, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute hierarchical branch losses across active tree levels."""
        # 1. Macro: 64 -> 32
        loss_64_32 = self._compute_branch_nll(
            mu_children=mu_blocks[32],
            y_children=gt_blocks[32],
            parent_min_count=1.0,
        )

        # 2. Meso: 32 -> 16
        loss_32_16 = self._compute_branch_nll(
            mu_children=mu_blocks[16],
            y_children=gt_blocks[16],
            parent_min_count=1.0,
        )

        # 3. Micro: 16 -> 8 (Adaptive: only on congested parents)
        if (8 in mu_blocks) and (8 in gt_blocks):
            loss_16_8 = self._compute_branch_nll(
                mu_children=mu_blocks[8],
                y_children=gt_blocks[8],
                parent_min_count=self.dense_threshold_16,
            )
            total_tree_loss = (loss_64_32 + loss_32_16 + loss_16_8) / 3.0
            details = {
                "tree_64_32": loss_64_32.detach(),
                "tree_32_16": loss_32_16.detach(),
                "tree_16_8": loss_16_8.detach(),
                "tree_total": total_tree_loss.detach(),
            }
        else:
            total_tree_loss = 0.5 * (loss_64_32 + loss_32_16)
            details = {
                "tree_64_32": loss_64_32.detach(),
                "tree_32_16": loss_32_16.detach(),
                "tree_total": total_tree_loss.detach(),
            }

        return total_tree_loss, details


class LocalDensityContrastiveLoss(nn.Module):
    """Tree-Guided Local Density Contrastive Loss on P4 representations (Training-only)."""

    def __init__(
        self,
        in_dim: int = 32,
        proj_dim: int = 64,
        temperature: float = 0.10,
        q_sparse: float = 3.0,
        q_medium: float = 15.0,
        max_samples_per_class: int = 32,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.SiLU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.temperature = float(temperature)
        self.q_sparse = float(q_sparse)
        self.q_medium = float(q_medium)
        self.max_samples = int(max_samples_per_class)

    def forward(self, p4_features: torch.Tensor, gt_blocks16: torch.Tensor) -> torch.Tensor:
        """Supervised contrastive loss pulling same-density patches and repelling background."""
        # Pool P4 (B, C, H/4, W/4) into 16x16 patch tokens (B, C, H/16, W/16)
        feats = F.avg_pool2d(p4_features, kernel_size=4, stride=4)
        B, C, H, W = feats.shape
        feats_flat = feats.permute(0, 2, 3, 1).reshape(-1, C)  # (N, C)
        y16_flat = gt_blocks16.view(-1)                         # (N,)

        # Assign discrete density classes: 0 (Bg), 1 (Sparse), 2 (Med), 3 (Dense)
        labels = torch.zeros_like(y16_flat, dtype=torch.long)
        labels[y16_flat > 0] = 1
        labels[y16_flat > self.q_sparse] = 2
        labels[y16_flat > self.q_medium] = 3

        # Subsample to maintain efficient O(M^2) memory footprint
        sampled_indices = []
        for c in range(4):
            idx = (labels == c).nonzero(as_tuple=True)[0]
            if len(idx) > 0:
                perm = torch.randperm(len(idx), device=idx.device)[: self.max_samples]
                sampled_indices.append(idx[perm])

        if len(sampled_indices) < 2:
            return torch.tensor(0.0, device=p4_features.device, requires_grad=True)

        sel = torch.cat(sampled_indices)
        z = F.normalize(self.proj(feats_flat[sel]), dim=-1)  # (M, proj_dim)
        lbl = labels[sel]

        # Cosine similarity matrix
        sim = torch.matmul(z, z.T) / self.temperature
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Positive mask (same class, excluding self)
        pos_mask = (lbl.unsqueeze(1) == lbl.unsqueeze(0)).float()
        pos_mask.fill_diagonal_(0.0)

        # Exclude self from denominator
        self_mask = torch.eye(len(sel), device=z.device)
        exp_sim = torch.exp(sim) * (1.0 - self_mask)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-6)

        n_pos = pos_mask.sum(dim=1)
        valid = (n_pos > 0)
        if not valid.any():
            return torch.tensor(0.0, device=p4_features.device, requires_grad=True)

        loss = -(pos_mask[valid] * log_prob[valid]).sum(dim=1) / n_pos[valid]
        return loss.mean()


class HardZeroMiningLoss(nn.Module):
    """Hard-Zero Leaf False Positive Mining Loss on 16x16 empty blocks."""

    def __init__(self, topk_fraction: float = 0.10, beta: float = 1.0):
        super().__init__()
        self.topk_fraction = float(topk_fraction)
        self.beta = float(beta)

    def forward(self, mu16: torch.Tensor, y16: torch.Tensor) -> torch.Tensor:
        """Suppress top-rho false positive predicted mass on zero-count 16x16 blocks."""
        mu_flat = mu16.view(-1)
        y_flat = y16.view(-1)

        zero_mask = (y_flat == 0)
        n_zero = int(zero_mask.sum().item())
        if n_zero == 0:
            return torch.tensor(0.0, device=mu16.device, requires_grad=True)

        zero_pred = mu_flat[zero_mask]
        k = max(1, int(math.ceil(self.topk_fraction * n_zero)))
        topk_preds, _ = torch.topk(zero_pred, k=k, largest=True)

        return F.smooth_l1_loss(topk_preds, torch.zeros_like(topk_preds), beta=self.beta)


class MassCalibrationLoss(nn.Module):
    """Multi-scale exact L1 count calibration loss."""

    def __init__(self, block_weights: Optional[Dict[int, float]] = None):
        super().__init__()
        self.block_weights = block_weights or {16: 1.0, 32: 1.0, 64: 1.0}

    def forward(
        self,
        mu_blocks: Dict[int, torch.Tensor],
        gt_blocks: Dict[int, torch.Tensor],
        pred_total: torch.Tensor,
        gt_total: torch.Tensor,
    ) -> torch.Tensor:
        """Compute exact L1 error across spatial block scales and total global count."""
        loss = 0.0
        n_scales = len(self.block_weights) + 1

        for b, weight in self.block_weights.items():
            if b in mu_blocks and b in gt_blocks:
                gt_b = gt_blocks[b].unsqueeze(1) if gt_blocks[b].ndim == 3 else gt_blocks[b]
                loss += weight * F.l1_loss(mu_blocks[b], gt_b)

        loss += F.l1_loss(pred_total.view(-1), gt_total.view(-1)) / 100.0
        return loss / float(n_scales)


class HPCAdaptiveTreeCriterion(nn.Module):
    """Unified Final Objective for HPC-Lite:
    
    L = L_{TreeProb} + lambda_Z * L_{HardZero} + lambda_C * L_{LocalContrast} + lambda_M * L_{mass}
    """

    def __init__(
        self,
        tree_mode: str = "dirichlet_multinomial",
        dispersion_r: float = 10.0,
        dm_concentration_alpha: float = 10.0,
        dense_threshold_16: float = 10.0,
        hz_topk_fraction: float = 0.10,
        lambda_tree: float = 1.0,
        lambda_hard_zero: float = 0.25,
        lambda_contrast: float = 0.10,
        lambda_mass: float = 1.0,
    ):
        super().__init__()
        self.root_nb_loss = RootNegativeBinomialLoss(dispersion=dispersion_r)
        self.tree_branch_loss = AdaptiveHierarchicalCountTreeLoss(
            mode=tree_mode,
            concentration_alpha=dm_concentration_alpha,
            dense_threshold_16=dense_threshold_16,
        )
        self.hard_zero_loss = HardZeroMiningLoss(topk_fraction=hz_topk_fraction)
        self.contrastive_loss = LocalDensityContrastiveLoss(in_dim=32, proj_dim=64)
        self.mass_loss = MassCalibrationLoss()

        self.lambda_tree = float(lambda_tree)
        self.lambda_hard_zero = float(lambda_hard_zero)
        self.lambda_contrast = float(lambda_contrast)
        self.lambda_mass = float(lambda_mass)

    def forward(
        self,
        density_map: torch.Tensor,
        gt_blocks: Dict[int, torch.Tensor],
        gt_count: torch.Tensor,
        p4_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass computing adaptive probabilistic count tree loss."""
        # Exact hierarchical SumPool
        mu_8 = F.avg_pool2d(density_map, kernel_size=2, stride=2) * 4.0
        mu_16 = F.avg_pool2d(density_map, kernel_size=4, stride=4) * 16.0
        mu_32 = F.avg_pool2d(density_map, kernel_size=8, stride=8) * 64.0
        mu_64 = F.avg_pool2d(density_map, kernel_size=16, stride=16) * 256.0
        mu_total = density_map.sum(dim=(-2, -1), keepdim=True)

        mu_blocks = {8: mu_8, 16: mu_16, 32: mu_32, 64: mu_64}

        # 1. Root Count Likelihood
        l_root_nb = self.root_nb_loss(mu_total, gt_count)

        # 2. Adaptive Branching Tree Likelihood
        l_tree_branch, tree_details = self.tree_branch_loss(mu_blocks, gt_blocks)
        l_tree_prob = l_root_nb + l_tree_branch

        # 3. Hard-Zero Background Mining
        l_hz = self.hard_zero_loss(mu_16, gt_blocks[16])

        # 4. Tree-Guided Local Density Contrastive Learning
        if p4_features is not None and self.lambda_contrast > 0.0:
            l_contrast = self.contrastive_loss(p4_features, gt_blocks[16])
        else:
            l_contrast = torch.tensor(0.0, device=density_map.device)

        # 5. Mass Calibration
        l_mass = self.mass_loss(mu_blocks, gt_blocks, mu_total, gt_count)

        total_loss = (
            self.lambda_tree * l_tree_prob
            + self.lambda_hard_zero * l_hz
            + self.lambda_contrast * l_contrast
            + self.lambda_mass * l_mass
        )

        details = {
            "loss_total": total_loss.detach(),
            "loss_tree_prob": l_tree_prob.detach(),
            "loss_root_nb": l_root_nb.detach(),
            "loss_tree_branch": l_tree_branch.detach(),
            "loss_hz": l_hz.detach(),
            "loss_contrast": l_contrast.detach(),
            "loss_mass": l_mass.detach(),
            **tree_details,
        }

        return total_loss, details


# Backwards compatibility aliases
HierarchicalDirichletMultinomialLoss = AdaptiveHierarchicalCountTreeLoss
HPCLiteUnifiedCriterion = HPCAdaptiveTreeCriterion
