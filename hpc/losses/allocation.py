import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class LocalAllocationLoss(nn.Module):
    """Block-local spatial allocation cross-entropy on positive blocks.

    Extended for SR48 with:
      - Optional ``block_weights`` for special-block emphasis.
      - ``return_details`` for entropy/KL diagnostics.
    """

    def __init__(self, block_size: int = 16, output_stride: int = 4, eps: float = 1e-8):
        super().__init__()
        if block_size % output_stride != 0:
            raise ValueError("block_size must be divisible by output_stride")
        self.block_size = int(block_size)
        self.output_stride = int(output_stride)
        self.k_dim = self.block_size // self.output_stride
        self.k_cells = self.k_dim * self.k_dim
        self.eps = float(eps)

    def forward(
        self,
        d_map: torch.Tensor,
        z_map: torch.Tensor,
        y_16: torch.Tensor,
        block_weights: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute allocation loss.

        Args:
            d_map: (B, 1, H/4, W/4) or (B, H/4, W/4) predicted mass map.
            z_map: same shape as d_map — block-constrained allocation targets.
            y_16: (B, H/16, W/16) exact integer block counts.
            block_weights: optional (B, H/16, W/16) float weights for special blocks.
                           If None, equal weighting. Typical: ``1 + special_mask16.float()``.
            return_details: if True, return a diagnostics dict (alloc_ce, target_entropy, kl).

        Returns:
            loss: scalar tensor
            details: dict (empty unless return_details=True)
        """
        d = d_map.squeeze(1) if d_map.ndim == 4 and d_map.shape[1] == 1 else d_map
        z = z_map.squeeze(1) if z_map.ndim == 4 and z_map.shape[1] == 1 else z_map
        y = y_16.squeeze(1) if y_16.ndim == 4 and y_16.shape[1] == 1 else y_16

        if d.ndim != 3 or z.ndim != 3 or y.ndim != 3:
            raise ValueError("Expected d/z/y as BxHxW tensors after optional channel squeeze")

        # Operate in float32 under AMP
        d = d.float()
        z = z.to(device=d.device, dtype=torch.float32)
        y = y.to(device=d.device, dtype=torch.float32)

        bsz, h_out, w_out = d.shape
        kd = self.k_dim
        if h_out % kd != 0 or w_out % kd != 0:
            raise ValueError(
                f"d_map spatial shape {(h_out, w_out)} not divisible by local block {kd}"
            )
        if z.shape != d.shape:
            raise ValueError(f"z_map shape {tuple(z.shape)} != d_map shape {tuple(d.shape)}")

        h_blk, w_blk = h_out // kd, w_out // kd
        if y.shape != (bsz, h_blk, w_blk):
            raise ValueError(f"y block shape {tuple(y.shape)} != expected {(bsz, h_blk, w_blk)}")

        d_blocks = (
            d.reshape(bsz, h_blk, kd, w_blk, kd)
            .permute(0, 1, 3, 2, 4)
            .reshape(bsz, h_blk, w_blk, self.k_cells)
        )
        z_blocks = (
            z.reshape(bsz, h_blk, kd, w_blk, kd)
            .permute(0, 1, 3, 2, 4)
            .reshape(bsz, h_blk, w_blk, self.k_cells)
        )

        pos_mask = y > 0
        if not pos_mask.any():
            empty = d.new_zeros(())
            details = (
                {"alloc_ce": empty, "alloc_target_entropy": empty, "alloc_kl": empty}
                if return_details
                else {}
            )
            return empty, details

        d_pos = d_blocks[pos_mask]
        z_pos = z_blocks[pos_mask]
        y_pos = y[pos_mask]

        if not torch.allclose(z_pos.sum(-1), y_pos, atol=1e-4, rtol=0.0):
            raise ValueError("Allocation target does not conserve block counts")

        mu_pos = d_pos.sum(dim=-1, keepdim=True)
        p_pos = (d_pos + self.eps) / (mu_pos + self.k_cells * self.eps)
        ce_per_block = -(z_pos * torch.log(p_pos.clamp_min(self.eps))).sum(dim=-1) / y_pos.clamp_min(1.0)

        # Optional block weights (special-block emphasis)
        if block_weights is not None:
            w = block_weights.to(device=d.device, dtype=torch.float32)
            w = w.squeeze(1) if w.ndim == 4 and w.shape[1] == 1 else w
            if w.shape != (bsz, h_blk, w_blk):
                raise ValueError(
                    f"block_weights shape {tuple(w.shape)} != block grid {(bsz, h_blk, w_blk)}"
                )
            w_pos = w[pos_mask]
            loss = (w_pos * ce_per_block).sum() / w_pos.sum().clamp_min(self.eps)
        else:
            loss = ce_per_block.mean()

        if not return_details:
            return loss, {}

        # Diagnostics: target entropy and excess KL
        q_pos = z_pos / y_pos.unsqueeze(-1).clamp_min(1.0)
        target_entropy = -(q_pos * torch.log(q_pos.clamp_min(self.eps))).sum(dim=-1).mean()
        # KL = CE - H(q): the part expected to approach zero
        alloc_kl = (ce_per_block.mean() - target_entropy).clamp_min(0.0)

        return loss, {
            "alloc_ce": loss.detach(),
            "alloc_target_entropy": target_entropy.detach(),
            "alloc_kl": alloc_kl.detach(),
        }
