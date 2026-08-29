from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HardZeroRegionLoss(nn.Module):
    def __init__(
        self,
        top_fraction: float = 0.10,
        min_k: int = 1,
        beta: float = 1.0,
    ):
        super().__init__()
        self.top_fraction = top_fraction
        self.min_k = min_k
        self.beta = beta

    def forward(
        self,
        pred_count16: torch.Tensor,
        target_count16: torch.Tensor,
        valid16: torch.Tensor | None = None,
    ) -> torch.Tensor:
        losses = []

        for b in range(pred_count16.shape[0]):
            zero_mask = target_count16[b] == 0

            if valid16 is not None:
                zero_mask = zero_mask & valid16[b].bool()

            values = pred_count16[b][zero_mask]
            if values.numel() == 0:
                continue

            k = max(
                self.min_k,
                int(math.ceil(self.top_fraction * values.numel()))
            )
            k = min(k, values.numel())

            hard = torch.topk(
                values, k=k, largest=True, sorted=False
            ).values

            losses.append(
                F.smooth_l1_loss(
                    hard,
                    torch.zeros_like(hard),
                    beta=self.beta,
                    reduction="mean",
                )
            )

        if not losses:
            return pred_count16.sum() * 0.0

        return torch.stack(losses).mean()
