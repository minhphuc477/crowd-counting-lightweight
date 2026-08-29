from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .negative_binomial import sum_pool
from .ot_tv import DMCountLoss


class TeacherMultiScaleMassLoss(nn.Module):
    def __init__(
        self,
        block_weights: Mapping[int, float] = {16: 0.25, 32: 0.50, 64: 1.0},
        output_stride: int = 4,
    ):
        super().__init__()
        self.block_weights = dict(block_weights)
        self.output_stride = output_stride

    def forward(
        self,
        density: torch.Tensor,
        gt_blocks: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        total = density.new_zeros(())
        weight_sum = 0.0

        for block_size, weight in self.block_weights.items():
            if block_size not in gt_blocks:
                continue

            pred = sum_pool(
                density,
                input_block_size=block_size,
                output_stride=self.output_stride,
            )
            if pred.shape[1] == 1:
                pred = pred.squeeze(1)

            target = gt_blocks[block_size].to(pred.device).float()
            if target.dim() == 4 and target.shape[1] == 1:
                target = target.squeeze(1)

            loss_b = torch.abs(pred - target).mean()
            total = total + float(weight) * loss_b
            weight_sum += float(weight)

        if weight_sum == 0:
            return density.new_zeros(())
        return total / weight_sum


class TeacherCriterion(nn.Module):
    """Universal Teacher Supervision Criterion.

    Supports:
      - Positive-weighted smooth allocation map loss (gt_z_alloc)
      - Multi-scale exact mass MAE (blocks 16, 32, 64)
      - Hard negative background mining (HNB)
      - Scaled Direct Count losses on density map and regression head
      - Consistency regularization
      - Optional DM-Count Optimal Transport & Total Variation
    """

    def __init__(
        self,
        lambda_map: float = 1.0,
        positive_cell_weight: float = 3.0,
        lambda_ms: float = 1.0,
        lambda_count_map: float = 0.50,
        lambda_count_reg: float = 0.25,
        lambda_consistency: float = 0.10,
        lambda_hn: float = 0.10,
        hard_negative_fraction: float = 0.10,
        lambda_ot: float = 0.0,
        lambda_tv: float = 0.0,
        count_scale: float = 100.0,
        ot_reg: float = 10.0,
        ot_iterations: int = 100,
        block_weights: Mapping[int, float] = {16: 0.25, 32: 0.50, 64: 1.0},
    ):
        super().__init__()
        self.lambda_map = lambda_map
        self.positive_cell_weight = positive_cell_weight
        self.lambda_ms = lambda_ms
        self.lambda_count_map = lambda_count_map
        self.lambda_count_reg = lambda_count_reg
        self.lambda_consistency = lambda_consistency
        self.lambda_hn = lambda_hn
        self.hard_negative_fraction = hard_negative_fraction
        self.lambda_ot = lambda_ot
        self.lambda_tv = lambda_tv
        self.count_scale = float(count_scale)

        self.ms_loss = TeacherMultiScaleMassLoss(block_weights=block_weights)
        if self.lambda_ot > 0 or self.lambda_tv > 0:
            self.dm_loss = DMCountLoss(
                reg=ot_reg,
                max_iter=ot_iterations,
                w_ot=1.0,
                w_tv=1.0,
            )
        else:
            self.dm_loss = None

    def weighted_map_loss(
        self,
        density: torch.Tensor,
        gt_z_alloc: torch.Tensor,
    ) -> torch.Tensor:
        target = gt_z_alloc.to(density.device).float()
        if target.dim() == 3:
            target = target.unsqueeze(1)
        assert target.shape == density.shape, (target.shape, density.shape)

        weight = 1.0 + self.positive_cell_weight * (target > 0).float()
        pointwise = F.smooth_l1_loss(density, target, reduction="none")
        return (weight * pointwise).mean()

    def hard_negative_loss(
        self,
        density: torch.Tensor,
        gt_blocks: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        if 16 not in gt_blocks:
            return density.new_zeros(())

        pred16 = sum_pool(density, input_block_size=16, output_stride=4)
        if pred16.shape[1] == 1:
            pred16 = pred16.squeeze(1)

        gt16 = gt_blocks[16].to(pred16.device).float()
        if gt16.dim() == 4 and gt16.shape[1] == 1:
            gt16 = gt16.squeeze(1)

        losses = []
        for b in range(pred16.shape[0]):
            vals = pred16[b][gt16[b] == 0]
            if vals.numel() == 0:
                continue
            k = max(1, int(round(self.hard_negative_fraction * vals.numel())))
            hard = vals.topk(k, largest=True).values
            losses.append(F.smooth_l1_loss(hard, torch.zeros_like(hard)))

        if not losses:
            return density.new_zeros(())
        return torch.stack(losses).mean()

    def forward(
        self,
        out: Dict[str, torch.Tensor],
        batch: Dict,
        crop_size: int = 448,
    ):
        density = out["density"]
        count_map = out["count_map"]
        count_reg = out["count_reg"]
        gt_count = batch["gt_count"].to(density.device).float()
        gt_blocks = {
            int(k): v.to(density.device)
            for k, v in batch["gt_blocks"].items()
        }

        # 1. Scaled Direct Count losses
        l_count_map = F.l1_loss(count_map, gt_count) / self.count_scale
        l_count_reg = F.l1_loss(count_reg, gt_count) / self.count_scale

        # 2. Multi-scale exact mass MAE
        l_ms = self.ms_loss(density, gt_blocks)

        # 3. Positive-weighted allocation map loss
        if self.lambda_map > 0 and "gt_z_alloc" in batch:
            l_map = self.weighted_map_loss(density, batch["gt_z_alloc"])
        else:
            l_map = density.new_zeros(())

        # 4. Hard negative background loss
        if self.lambda_hn > 0:
            l_hn = self.hard_negative_loss(density, gt_blocks)
        else:
            l_hn = density.new_zeros(())

        # 5. Consistency loss
        l_cons = F.l1_loss(count_reg, count_map) / self.count_scale

        # 6. Optional OT / TV
        if self.dm_loss is not None:
            gt_points = batch.get("gt_points", [None] * len(gt_count))
            l_ot, l_tv, _ = self.dm_loss(density, gt_points, crop_size=crop_size)
        else:
            l_ot = density.new_zeros(())
            l_tv = density.new_zeros(())

        total = (
            self.lambda_count_map * l_count_map
            + self.lambda_count_reg * l_count_reg
            + self.lambda_ms * l_ms
            + self.lambda_map * l_map
            + self.lambda_hn * l_hn
            + self.lambda_consistency * l_cons
            + self.lambda_ot * l_ot
            + self.lambda_tv * l_tv
        )

        details = {
            "teacher_total": total.detach(),
            "teacher_count_map": l_count_map.detach(),
            "teacher_count_reg": l_count_reg.detach(),
            "teacher_ms": l_ms.detach(),
            "teacher_map": l_map.detach(),
            "teacher_hn": l_hn.detach(),
            "teacher_consistency": l_cons.detach(),
            "teacher_ot": l_ot.detach(),
            "teacher_tv": l_tv.detach(),
            "teacher_bias_map": (count_map - gt_count).mean().detach(),
            "teacher_bias_reg": (count_reg - gt_count).mean().detach(),
        }
        return total, details
