from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .negative_binomial import sum_pool


class RobustConsistencyLoss(nn.Module):
    """Clean -> degraded multi-scale count consistency with detached clean teacher."""

    def __init__(self, block_sizes: List[int], output_stride: int = 4):
        super().__init__()
        self.block_sizes = list(block_sizes)
        self.output_stride = int(output_stride)

    def forward(
        self,
        d_clean: torch.Tensor,
        d_degraded: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if d_clean.shape != d_degraded.shape:
            raise ValueError(
                f"clean/degraded map shapes differ: {tuple(d_clean.shape)} vs {tuple(d_degraded.shape)}"
            )

        if valid_mask is not None:
            valid_mask = valid_mask.to(device=d_clean.device, dtype=torch.bool).reshape(-1)
            if valid_mask.numel() != d_clean.shape[0]:
                raise ValueError("valid_mask length must equal batch size")
            if not valid_mask.any():
                return d_clean.new_zeros(())
            d_clean = d_clean[valid_mask]
            d_degraded = d_degraded[valid_mask]

        scale_losses = []
        for b in self.block_sizes:
            mu_clean = sum_pool(d_clean, input_block_size=b, output_stride=self.output_stride)
            mu_deg = sum_pool(d_degraded, input_block_size=b, output_stride=self.output_stride)
            teacher_log = torch.log1p(mu_clean.float()).detach()
            student_log = torch.log1p(mu_deg.float())
            scale_losses.append(F.smooth_l1_loss(student_log, teacher_log))

        return torch.stack(scale_losses).mean() if scale_losses else d_clean.new_zeros(())
