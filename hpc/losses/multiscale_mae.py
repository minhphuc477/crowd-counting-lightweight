"""Multi-Scale Block MAE Auxiliary Loss (Direct Spatial Density Supervision).

Computes L1 counting errors at multiple spatial block partitions (e.g. 16x16, 32x32, 64x64 px).
For density map D at stride 4:
  - Block 16x16 px in image space corresponds to 4x4 cells in D.
  - SumPool_16(D) = F.avg_pool2d(D, kernel_size=4, stride=4) * 16.
  - Loss_16 = mean(|SumPool_16(D) - gt_blocks[16]|).

Total Loss:
  L_MS = sum_{B} w_B * Loss_B
"""
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleBlockMAELoss(nn.Module):
    """Multi-Scale Block-level Mean Absolute Error Loss.

    Args:
        block_sizes: Spatial block sizes in input image pixels (e.g., [16, 32, 64]).
        output_stride: Stride of the density map relative to the input image (default: 4).
        weights: Optional dictionary or list of weights per block size.
        scale_by_count: Whether to normalize block error by block count (default: False, pure MAE).
    """

    def __init__(
        self,
        block_sizes: List[int] = (16, 32, 64),
        output_stride: int = 4,
        weights: Optional[Dict[int, float]] = None,
    ):
        super().__init__()
        self.block_sizes = tuple(sorted(int(b) for b in block_sizes))
        self.output_stride = int(output_stride)

        if weights is None:
            # Equal weighting by default
            self.weights = {b: 1.0 / len(self.block_sizes) for b in self.block_sizes}
        else:
            self.weights = {b: float(weights[b]) for b in self.block_sizes}

    def forward(
        self,
        d_map: torch.Tensor,
        gt_blocks: Dict[int, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            d_map: (B, 1, H/4, W/4) non-negative density map.
            gt_blocks: dict mapping block_size B -> (B, H/B, W/B) exact integer block counts.

        Returns:
            total_loss: scalar tensor.
            loss_dict: dictionary of individual block scale losses.
        """
        total_loss = d_map.new_zeros(())
        loss_dict = {}

        for b in self.block_sizes:
            if b not in gt_blocks:
                continue

            k = b // self.output_stride  # cell pooling kernel size (e.g. 16//4 = 4)
            if k <= 0:
                continue

            # Pool predicted density map: SumPool_B(D) = AvgPool * (k*k)
            # Input d_map: (B_batch, 1, H4, W4) -> (B_batch, 1, H/B, W/B)
            pred_b = F.avg_pool2d(d_map, kernel_size=k, stride=k) * (k * k)
            pred_b = pred_b.squeeze(1)  # (B_batch, H/B, W/B)

            target_b = gt_blocks[b].to(device=d_map.device, dtype=d_map.dtype)

            # Block MAE
            l1_b = torch.mean(torch.abs(pred_b - target_b))
            w = self.weights.get(b, 1.0)
            total_loss = total_loss + w * l1_b
            loss_dict[f"loss_ms_mae_{b}"] = l1_b.detach()

        loss_dict["loss_ms_mae_total"] = total_loss.detach()
        return total_loss, loss_dict
