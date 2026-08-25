import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .negative_binomial import sum_pool


class HardNegativeMassLoss(nn.Module):
    """Per-image top-k false-mass penalty on GT-zero blocks."""

    def __init__(self, top_fraction: float = 0.10, block_size: int = 16):
        super().__init__()
        if not 0.0 <= top_fraction <= 1.0:
            raise ValueError("top_fraction must be in [0, 1]")
        self.top_fraction = float(top_fraction)
        self.block_size = int(block_size)

    def forward(self, d_map: torch.Tensor, gt_y16: torch.Tensor) -> torch.Tensor:
        if self.top_fraction == 0.0:
            return d_map.new_zeros(())

        pred_mu16 = sum_pool(d_map, input_block_size=self.block_size, output_stride=4)
        if pred_mu16.ndim == 4 and pred_mu16.shape[1] == 1:
            pred_mu16 = pred_mu16.squeeze(1)

        if gt_y16.ndim == 4 and gt_y16.shape[1] == 1:
            gt_y16 = gt_y16.squeeze(1)
        gt_y16 = gt_y16.to(device=pred_mu16.device)

        if gt_y16.shape != pred_mu16.shape:
            raise ValueError(
                f"Hard-negative GT shape {tuple(gt_y16.shape)} != pred shape {tuple(pred_mu16.shape)}"
            )

        losses = []
        for pred_i, gt_i in zip(pred_mu16, gt_y16):
            zero_vals = pred_i[gt_i.eq(0)]
            if zero_vals.numel() == 0:
                continue
            k = max(1, int(math.ceil(self.top_fraction * zero_vals.numel())))
            hard_vals = zero_vals.topk(k, largest=True).values
            losses.append(F.smooth_l1_loss(hard_vals.float(), torch.zeros_like(hard_vals).float()))

        return torch.stack(losses).mean() if losses else d_map.new_zeros(())


class WholeImageEmptyLoss(nn.Module):
    """Directly suppress total predicted mass for GT-empty images."""

    def __init__(self, use_warmup_log: bool = False):
        super().__init__()
        self.use_warmup_log = bool(use_warmup_log)

    def forward(self, d_map: torch.Tensor, gt_counts: torch.Tensor) -> torch.Tensor:
        pred_counts = d_map.sum(dim=(-1, -2, -3))
        gt_counts = gt_counts.to(device=pred_counts.device).reshape(-1)
        if gt_counts.numel() != pred_counts.numel():
            raise ValueError("gt_counts batch size does not match d_map")

        empty_mask = gt_counts.eq(0)
        if not empty_mask.any():
            return d_map.new_zeros(())
        pred_empty = pred_counts[empty_mask]
        return torch.log1p(pred_empty).mean() if self.use_warmup_log else pred_empty.mean()


class GlobalCountLoss(nn.Module):
    """Global image-level count conservation loss."""

    def __init__(self, mode: str = "log_smooth_l1"):
        super().__init__()
        if mode not in {"log_smooth_l1", "l1", "sqrt_normalized"}:
            raise ValueError(f"Unknown mode {mode}")
        self.mode = mode

    def forward(self, d_map: torch.Tensor, gt_counts: torch.Tensor) -> torch.Tensor:
        pred_counts = d_map.sum(dim=(-1, -2, -3)).float()
        gt_counts = gt_counts.to(device=pred_counts.device, dtype=torch.float32).reshape(-1)
        if gt_counts.shape != pred_counts.shape:
            raise ValueError(f"gt_counts shape {tuple(gt_counts.shape)} != {tuple(pred_counts.shape)}")

        if self.mode == "log_smooth_l1":
            return F.smooth_l1_loss(torch.log1p(pred_counts), torch.log1p(gt_counts))
        if self.mode == "l1":
            return F.l1_loss(pred_counts, gt_counts)

        residual = (pred_counts - gt_counts) / torch.sqrt(gt_counts + 1.0)
        return F.smooth_l1_loss(residual, torch.zeros_like(residual))
