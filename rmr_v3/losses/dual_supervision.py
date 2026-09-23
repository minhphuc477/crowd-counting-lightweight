from __future__ import annotations

import torch
from .auxiliary import count_invariant_cell_loss, mass_weighted_cell_loss
from rmr_core.losses import balanced_smooth_l1


def compute_dual_lattice_losses(
    y_fine: torch.Tensor,
    y_carrier: torch.Tensor,
    target_stride2: torch.Tensor,
    target_stride4: torch.Tensor,
    lambda_carrier_cell: float = 0.50,
    lambda_fine_cell: float = 0.25,
    cell_loss_mode: str = "mass_weighted",
    beta: float = 1.0,
    eps: float = 1e-3,
    alpha: float = 2.0,
    gamma: float = 1.25,
) -> dict[str, torch.Tensor]:
    """Supervise Stride 4 carrier with canonical v19 convex loss, and Stride 2 with sub-pixel loss.

    Prevents quadratic loss gradient starvation in dense regions by anchoring
    the carrier density on Stride 4, while allowing Stride 2 to resolve sparse heads.
    """
    if cell_loss_mode == "mass_weighted":
        loss_carrier = mass_weighted_cell_loss(
            y_carrier, target_stride4, beta=beta, eps=eps, alpha=alpha, gamma=gamma, stride=4
        )
        loss_fine = mass_weighted_cell_loss(
            y_fine, target_stride2, beta=beta, eps=eps, alpha=alpha, gamma=gamma, stride=2
        )
    elif cell_loss_mode in ("count_invariant", "ci_cell"):
        loss_carrier = count_invariant_cell_loss(
            y_carrier, target_stride4, beta=beta, alpha=alpha, stride=4
        )
        loss_fine = count_invariant_cell_loss(
            y_fine, target_stride2, beta=beta, alpha=alpha, stride=2
        )
    else:
        loss_carrier = balanced_smooth_l1(y_carrier, target_stride4, beta=beta, stride=4)
        loss_fine = balanced_smooth_l1(y_fine, target_stride2, beta=beta, stride=2)

    return {
        "cell_carrier": loss_carrier,
        "cell_fine": loss_fine,
        "cell_combined": float(lambda_carrier_cell) * loss_carrier + float(lambda_fine_cell) * loss_fine,
    }
