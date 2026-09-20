from __future__ import annotations

import torch
from .auxiliary import mass_weighted_cell_loss


def compute_dual_lattice_losses(
    y_fine: torch.Tensor,
    y_carrier: torch.Tensor,
    target_stride2: torch.Tensor,
    target_stride4: torch.Tensor,
    lambda_carrier_cell: float = 0.50,
    lambda_fine_cell: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Supervise Stride 4 carrier with canonical v19 convex loss, and Stride 2 with sub-pixel loss.

    Prevents quadratic loss gradient starvation in dense regions by anchoring
    the carrier density on Stride 4, while allowing Stride 2 to resolve sparse heads.
    """
    loss_carrier = mass_weighted_cell_loss(
        y_carrier, target_stride4, beta=1.0, eps=1e-3, alpha=2.0, gamma=1.25, stride=4
    )
    loss_fine = mass_weighted_cell_loss(
        y_fine, target_stride2, beta=1.0, eps=1e-3, alpha=2.0, gamma=1.25, stride=2
    )
    return {
        "cell_carrier": loss_carrier,
        "cell_fine": loss_fine,
        "cell_combined": float(lambda_carrier_cell) * loss_carrier + float(lambda_fine_cell) * loss_fine,
    }
