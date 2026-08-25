import torch
import torch.nn as nn


class LocalAllocationLoss(nn.Module):
    """Block-local spatial allocation cross-entropy on positive blocks."""

    def __init__(self, block_size: int = 16, output_stride: int = 4, eps: float = 1e-8):
        super().__init__()
        if block_size % output_stride != 0:
            raise ValueError("block_size must be divisible by output_stride")
        self.block_size = int(block_size)
        self.output_stride = int(output_stride)
        self.k_dim = self.block_size // self.output_stride
        self.k_cells = self.k_dim * self.k_dim
        self.eps = float(eps)

    def forward(self, d_map: torch.Tensor, z_map: torch.Tensor, y_16: torch.Tensor) -> torch.Tensor:
        d = d_map.squeeze(1) if d_map.ndim == 4 and d_map.shape[1] == 1 else d_map
        z = z_map.squeeze(1) if z_map.ndim == 4 and z_map.shape[1] == 1 else z_map
        y = y_16.squeeze(1) if y_16.ndim == 4 and y_16.shape[1] == 1 else y_16

        if d.ndim != 3 or z.ndim != 3 or y.ndim != 3:
            raise ValueError("Expected d/z/y as BxHxW tensors after optional channel squeeze")

        # Compute this probabilistic normalization in float32 under AMP.
        d = d.float()
        z = z.to(device=d.device, dtype=torch.float32)
        y = y.to(device=d.device, dtype=torch.float32)

        bsz, h_out, w_out = d.shape
        kd = self.k_dim
        if h_out % kd != 0 or w_out % kd != 0:
            raise ValueError(f"d_map spatial shape {(h_out, w_out)} not divisible by local block {kd}")
        if z.shape != d.shape:
            raise ValueError(f"z_map shape {tuple(z.shape)} != d_map shape {tuple(d.shape)}")

        h_blk, w_blk = h_out // kd, w_out // kd
        if y.shape != (bsz, h_blk, w_blk):
            raise ValueError(f"y block shape {tuple(y.shape)} != expected {(bsz, h_blk, w_blk)}")

        d_blocks = d.reshape(bsz, h_blk, kd, w_blk, kd).permute(0, 1, 3, 2, 4).reshape(
            bsz, h_blk, w_blk, self.k_cells
        )
        z_blocks = z.reshape(bsz, h_blk, kd, w_blk, kd).permute(0, 1, 3, 2, 4).reshape(
            bsz, h_blk, w_blk, self.k_cells
        )

        pos_mask = y > 0
        if not pos_mask.any():
            return d.new_zeros(())

        d_pos = d_blocks[pos_mask]
        z_pos = z_blocks[pos_mask]
        y_pos = y[pos_mask]

        # Sanity: allocation target mass must equal exact local count up to float tolerance.
        if not torch.allclose(z_pos.sum(-1), y_pos, atol=1e-4, rtol=0.0):
            raise ValueError("Allocation target does not conserve block counts")

        mu_pos = d_pos.sum(dim=-1, keepdim=True)
        p_pos = (d_pos + self.eps) / (mu_pos + self.k_cells * self.eps)
        ce = -(z_pos * torch.log(p_pos.clamp_min(self.eps))).sum(dim=-1)
        return (ce / y_pos.clamp_min(1.0)).mean()
