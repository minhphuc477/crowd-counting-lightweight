"""Routing supervision loss for the Shared Scale-Evidence Router (SSER).

L_route = (1 / |M_pos|) * Σ_{(x,y)∈M_pos} KL(q_scale(x,y) ‖ α(x,y))

where:
  q_scale ∈ Δ³  — soft target derived from local crowd geometry (d_NN)
  α       ∈ Δ³  — predicted routing weights (softmax output of SSER)
  M_pos         — set of grid cells with at least one annotated point

Notes:
  - KL(q ‖ α) = Σ_s q_s * log(q_s / α_s)  [forward KL, "zero-forcing"]
  - If no supervised cells exist (all-empty image), returns exact zero on the
    prediction device without touching the gradient graph.
  - Both q and α are clamped to ≥ 1e-8 before log to ensure numerical safety
    even if alpha becomes small during early training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoutingSupervisionLoss(nn.Module):
    """KL(q_scale ‖ α) routing geometry supervision (training only)."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        routes8: torch.Tensor,
        gt_route_q: torch.Tensor,
        gt_route_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL routing supervision loss.

        Args:
            routes8:      (B, 4, H/8, W/8) — predicted routing weights from SSER.
                          Must be a valid probability distribution along dim=1
                          (i.e. softmax output; verified via assert in debug builds).
            gt_route_q:   (B, 4, H/8, W/8) — soft target distributions.
                          Each column along dim=1 must sum to 1.0.
            gt_route_mask:(B, H/8, W/8) bool — True where supervision applies.

        Returns:
            Scalar loss tensor on routes8.device.
        """
        mask = gt_route_mask.to(device=routes8.device, dtype=torch.bool)
        if not mask.any():
            return routes8.new_zeros(())

        # Ensure float32 for stability (especially under AMP)
        alpha = routes8.float().clamp(min=self.eps)           # (B, 4, H/8, W/8)
        q     = gt_route_q.to(device=routes8.device, dtype=torch.float32).clamp(min=self.eps)

        # KL(q ‖ α) = sum_s [ q_s * (log q_s - log α_s) ]  → (B, H/8, W/8)
        kl_map = (q * (q.log() - alpha.log())).sum(dim=1)

        # Only average over supervised (positive-count) cells
        return kl_map[mask].mean()
