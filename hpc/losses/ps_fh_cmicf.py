"""Preconditioned Sobolev Finite-Horizon Cumulative Measure Integral Counting Field (PS-FH-CMICF) Losses.

Core components:
1. FractionalPrefixPreconditioner: Fixed non-learned fractional prefix preconditioner P_alpha = U Sigma^{-alpha} U^T.
2. balanced_sobolev_smooth_l1: Stratified foreground (Y>0) and background (Y=0) Smooth L1 on derivative recovered measure.
3. PSFHCMICFLoss: Complete objective L = L_PC + lambda_S * L_Sob + lambda_N * L_N + L_AL.
4. Target partition utilities: Block-vectorized decomposition of exact cell counts and local cumulative fields.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hpc.losses.micf import cell_counts_to_cumulative_field, discrete_mixed_difference


def partition_grid_into_blocks(
    x: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, int, int]:
    """Partition [B, C, H, W] or [B, H, W] into non-overlapping KxK blocks.

    Vectorized reshape/permute without Python loops.

    Returns:
        blocks: [B * nh * nw, C, K, K]
        nh: Number of block rows
        nw: Number of block columns
    """
    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:
        x = x.unsqueeze(1)
    elif x.ndim != 4:
        raise ValueError(f"Expected 2D, 3D, or 4D tensor, got ndim={x.ndim}")

    B, C, H, W = x.shape
    if H % k != 0 or W % k != 0:
        raise ValueError(f"Grid dimensions ({H}, {W}) must be divisible by k={k}")

    nh = H // k
    nw = W // k

    blocks = (
        x.view(B, C, nh, k, nw, k)
        .permute(0, 2, 4, 1, 3, 5)
        .contiguous()
        .view(B * nh * nw, C, k, k)
    )
    return blocks, nh, nw


class FractionalPrefixPreconditioner(nn.Module):
    r"""Fractional Prefix Preconditioner for Finite-Horizon Cumulative Charts.

    For a KxK chart, the 2D cumulative operator is T_K = A_K \otimes A_K, where
    A_K is the lower-triangular matrix of ones.
    Using SVD: T_K = U \Sigma V^T.
    The fixed preconditioner is:
        P_alpha = U \Sigma^{-alpha} U^T
    Default alpha = 0.5 reduces the condition number of the quadratic cumulative
    residual from kappa(T)^2 to kappa(T).
    """

    def __init__(
        self,
        k: int = 4,
        alpha: float = 0.5,
        sv_floor: float = 1e-8,
    ) -> None:
        super().__init__()
        self.k = int(k)
        self.alpha = float(alpha)
        self.sv_floor = float(sv_floor)

        # 1. 1D prefix operator A_K
        a_k = torch.tril(torch.ones(self.k, self.k, dtype=torch.float32))

        # 2. 2D prefix operator T_K = A_K \otimes A_K
        t_k = torch.kron(a_k, a_k)  # [K^2, K^2]

        # 3. SVD of T_K
        u, s, vh = torch.linalg.svd(t_k)
        s_clamped = s.clamp_min(self.sv_floor)

        # 4. P_alpha = U \Sigma^{-\alpha} U^T
        diag_inv = torch.diag(s_clamped ** (-self.alpha))
        p_alpha = u @ diag_inv @ u.t()
        self.register_buffer("P_alpha", p_alpha)

        # Diagnostics
        self.min_singular_value = float(s[-1].item())
        self.max_singular_value = float(s[0].item())
        self.prefix_condition_number = float((s[0] / s[-1]).item())
        self.quadratic_condition_number = float(
            (s[0] / s[-1]).item() ** (2.0 * (1.0 - self.alpha))
        )

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        """Apply P_alpha to the residual along the last two spatial dimensions.

        Args:
            residual: Tensor with trailing dimensions [..., K, K].

        Returns:
            Preconditioned residual of matching shape.
        """
        shape = residual.shape
        if shape[-2:] != (self.k, self.k):
            raise ValueError(
                f"Expected trailing spatial dimensions ({self.k}, {self.k}), got {shape[-2:]}"
            )

        res_flat = residual.reshape(-1, self.k * self.k)
        p = self.P_alpha.to(dtype=residual.dtype, device=residual.device)
        out_flat = res_flat @ p.t()
        return out_flat.view(shape)


def balanced_sobolev_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    r"""Stratified foreground / background Smooth L1 loss for sparse local counting measures.

    Guarantees that empty cells (Y=0) do not dominate the local derivative supervision:
        L_Sob = 0.5 * E_{Y>0}[rho(Y_hat - Y)] + 0.5 * E_{Y=0}[rho(Y_hat - Y)]

    Args:
        pred: Recovered cell counts \hat{Y} = \Delta_{xy} \hat{C}.
        target: Exact integer cell counts Y.
        beta: Smooth L1 threshold parameter.

    Returns:
        total: Balanced scalar loss tensor.
        stats: Dictionary containing stratum breakdown and fractions.
    """
    per = F.smooth_l1_loss(pred, target, beta=beta, reduction="none")

    pos_mask = target > 0
    zero_mask = ~pos_mask

    zero = pred.new_zeros(())

    if bool(pos_mask.any()):
        pos_loss = per[pos_mask].mean()
    else:
        pos_loss = zero

    if bool(zero_mask.any()):
        zero_loss = per[zero_mask].mean()
    else:
        zero_loss = zero

    if bool(pos_mask.any()) and bool(zero_mask.any()):
        total = 0.5 * (pos_loss + zero_loss)
    elif bool(pos_mask.any()):
        total = pos_loss
    else:
        total = zero_loss

    stats = {
        "sobolev_pos_loss": float(pos_loss.detach().item()),
        "sobolev_zero_loss": float(zero_loss.detach().item()),
        "positive_cell_fraction": float(pos_mask.float().mean().detach().item()),
        "zero_cell_fraction": float(zero_mask.float().mean().detach().item()),
    }
    return total, stats


class PSFHCMICFLoss(nn.Module):
    r"""Preconditioned Sobolev Finite-Horizon Cumulative Measure Loss.

    Objective:
        L = L_PC + lambda_S * L_Sob + lambda_N * L_N + L_AL

    Components:
    - L_PC: Preconditioned quadratic cumulative chart loss: ||P_alpha (C_hat - C) / N||_2^2
    - L_Sob: Stratified foreground/background Smooth L1 on recovered measure Y_hat = \Delta_{xy} C_hat
    - L_N: Global whole-crop total count Smooth L1: SmoothL1((N_hat - N) / N, 0)
    - L_AL: Augmented-Lagrangian validity constraint on negative recovered mass g = sum(ReLU(-Y_hat)) / N
    """

    def __init__(
        self,
        k: int = 4,
        precondition_alpha: float = 0.5,
        precondition_sv_floor: float = 1e-8,
        lambda_sobolev: float = 1.0,
        sobolev_beta: float = 1.0,
        lambda_count: float = 1.0,
        al_rho: float = 1.0,
        al_dual_init: float = 0.0,
        al_dual_max: float = 100.0,
        norm_eps: float = 1.0,
    ) -> None:
        super().__init__()
        self.k = int(k)
        self.precondition_alpha = float(precondition_alpha)
        self.precondition_sv_floor = float(precondition_sv_floor)

        self.lambda_sobolev = float(lambda_sobolev)
        self.sobolev_beta = float(sobolev_beta)
        self.lambda_count = float(lambda_count)

        self.al_rho = float(al_rho)
        self.al_dual_max = float(al_dual_max)
        self.register_buffer("al_lambda", torch.tensor(float(al_dual_init), dtype=torch.float32))

        self.norm_eps = float(norm_eps)

        self.preconditioner = FractionalPrefixPreconditioner(
            k=self.k,
            alpha=self.precondition_alpha,
            sv_floor=self.precondition_sv_floor,
        )

    def update_dual(self, epoch_constraint: float) -> float:
        """Augmented Lagrangian dual variable update step:
            lambda_{t+1} = clip(lambda_t + rho * g_t, 0, lambda_max)
        """
        current_val = float(self.al_lambda.item())
        new_val = min(max(current_val + self.al_rho * float(epoch_constraint), 0.0), self.al_dual_max)
        self.al_lambda.fill_(new_val)
        return new_val

    def forward(
        self,
        pred_c: torch.Tensor,
        target_c: torch.Tensor,
        target_y: torch.Tensor,
        pred_c_blocks: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute PS-FH-CMICF loss.

        Args:
            pred_c: Global predicted cumulative field [B, 1, H, W] or [B, H, W].
            target_c: Global target cumulative field [B, 1, H, W] or [B, H, W].
            target_y: Global target discrete cell counts [B, 1, H, W] or [B, H, W].
            pred_c_blocks: Optional pre-composed local charts [B * nh * nw, 1, K, K].
            return_components: Whether to return full diagnostic dictionary.
        """
        if pred_c.ndim == 3:
            pred_c = pred_c.unsqueeze(1)
        if target_c.ndim == 3:
            target_c = target_c.unsqueeze(1)
        if target_y.ndim == 3:
            target_y = target_y.unsqueeze(1)

        B, _, H, W = pred_c.shape

        # 1. Target block partitioning
        target_y_blocks, nh, nw = partition_grid_into_blocks(target_y, self.k)
        target_c_blocks = cell_counts_to_cumulative_field(target_y_blocks, orientation="TL")

        # 2. Predicted block retrieval
        if pred_c_blocks is None:
            # Fallback: recover local mass from pred_c, partition, and cumsum
            y_global = discrete_mixed_difference(pred_c)
            pred_y_blocks, _, _ = partition_grid_into_blocks(y_global, self.k)
            pred_c_blocks = cell_counts_to_cumulative_field(pred_y_blocks, orientation="TL")
        else:
            if pred_c_blocks.ndim == 3:
                pred_c_blocks = pred_c_blocks.unsqueeze(1)
            pred_y_blocks = discrete_mixed_difference(pred_c_blocks)

        # 3. Per-image total count normalization scale N_i
        target_n = target_c[:, 0, -1, -1]  # [B]
        scale = target_n.clamp_min(self.norm_eps)  # [B]
        scale_blocks = scale.repeat_interleave(nh * nw).view(-1, 1, 1, 1)  # [B * nh * nw, 1, 1, 1]

        # 4. Preconditioned Cumulative Chart Loss (L_PC)
        res_blocks = (pred_c_blocks - target_c_blocks) / scale_blocks
        p_res = self.preconditioner(res_blocks)
        pc_loss = (p_res ** 2).sum(dim=(-2, -1)).mean()

        # 5. Balanced Sobolev Mixed-Difference Loss (L_Sob)
        sobolev_loss, sob_stats = balanced_sobolev_smooth_l1(
            pred_y_blocks,
            target_y_blocks,
            beta=self.sobolev_beta,
        )

        # 6. Global Total Count Loss (L_N)
        pred_n = pred_c[:, 0, -1, -1]  # [B]
        count_err = (pred_n - target_n) / scale
        count_loss = F.smooth_l1_loss(
            count_err,
            torch.zeros_like(count_err),
            beta=self.sobolev_beta,
        )

        # 7. Augmented-Lagrangian Validity Loss (L_AL)
        # Reshape recovered mass per sample: [B, nh * nw * K * K]
        pred_y_per_sample = pred_y_blocks.view(B, -1)
        neg_mass_per_sample = F.relu(-pred_y_per_sample).sum(dim=-1)
        g_per_sample = neg_mass_per_sample / scale
        g = g_per_sample.mean()

        al_loss = self.al_lambda * g + 0.5 * self.al_rho * (g ** 2)

        # 8. Total objective
        total_loss = (
            pc_loss
            + self.lambda_sobolev * sobolev_loss
            + self.lambda_count * count_loss
            + al_loss
        )

        if return_components:
            components: Dict[str, Any] = {
                "loss": float(total_loss.item()),
                "ps_pc_loss": float(pc_loss.item()),
                "ps_sobolev_loss": float(sobolev_loss.item()),
                "sobolev_pos_loss": sob_stats["sobolev_pos_loss"],
                "sobolev_zero_loss": sob_stats["sobolev_zero_loss"],
                "ps_count_loss": float(count_loss.item()),
                "ps_constraint": float(g.item()),
                "ps_dual_lambda": float(self.al_lambda.item()),
                "ps_al_rho": float(self.al_rho),
                "ps_aug_lagrangian": float(al_loss.item()),
                "positive_cell_fraction": sob_stats["positive_cell_fraction"],
                "zero_cell_fraction": sob_stats["zero_cell_fraction"],
                "violation_rate": float((pred_y_blocks < 0).float().mean().item()),
            }
            return total_loss, components

        return total_loss

    def compute_gradient_diagnostics(
        self,
        pred_c: torch.Tensor,
        target_c: torch.Tensor,
        target_y: torch.Tensor,
        pred_c_blocks: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute gradient norms wrt predicted local cumulative charts C_blocks (Section 24).

        Diagnoses relative term influence without modifying optimizer state.
        """
        c_detached = pred_c_blocks.detach().clone().requires_grad_(True)
        B = target_c.shape[0]

        target_y_blocks, nh, nw = partition_grid_into_blocks(target_y, self.k)
        target_c_blocks = cell_counts_to_cumulative_field(target_y_blocks, orientation="TL")

        target_n = target_c[:, 0, -1, -1] if target_c.ndim == 4 else target_c[..., -1, -1]
        scale = target_n.clamp_min(self.norm_eps)
        scale_blocks = scale.repeat_interleave(nh * nw).view(-1, 1, 1, 1)

        # Term 1: PC
        res_blocks = (c_detached - target_c_blocks) / scale_blocks
        p_res = self.preconditioner(res_blocks)
        l_pc = (p_res ** 2).sum(dim=(-2, -1)).mean()
        g_pc = torch.autograd.grad(l_pc, c_detached, retain_graph=True, allow_unused=True)[0]
        norm_pc = float(g_pc.norm().item()) if g_pc is not None else 0.0

        # Term 2: Sobolev
        y_detached = discrete_mixed_difference(c_detached)
        l_sob, _ = balanced_sobolev_smooth_l1(y_detached, target_y_blocks, beta=self.sobolev_beta)
        l_sob_weighted = self.lambda_sobolev * l_sob
        g_sob = torch.autograd.grad(l_sob_weighted, c_detached, retain_graph=True, allow_unused=True)[0]
        norm_sob = float(g_sob.norm().item()) if g_sob is not None else 0.0

        # Term 3: Count (from composed corner)
        y_per_sample = y_detached.view(B, -1)
        pred_n_est = y_per_sample.sum(dim=-1)
        l_count = F.smooth_l1_loss((pred_n_est - target_n) / scale, torch.zeros_like(scale), beta=self.sobolev_beta)
        l_count_weighted = self.lambda_count * l_count
        g_count = torch.autograd.grad(l_count_weighted, c_detached, retain_graph=True, allow_unused=True)[0]
        norm_count = float(g_count.norm().item()) if g_count is not None else 0.0

        # Term 4: AL
        neg_mass = F.relu(-y_per_sample).sum(dim=-1)
        g_val = (neg_mass / scale).mean()
        l_al = self.al_lambda * g_val + 0.5 * self.al_rho * (g_val ** 2)
        g_al = torch.autograd.grad(l_al, c_detached, retain_graph=True, allow_unused=True)[0]
        norm_al = float(g_al.norm().item()) if g_al is not None else 0.0

        return {
            "grad_pc": norm_pc,
            "grad_sobolev": norm_sob,
            "grad_count": norm_count,
            "grad_al": norm_al,
        }
