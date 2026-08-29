from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn

from hpc.losses.count_tree import (
    AdaptiveProbabilisticCountTreeLoss,
    CountTreeConfig,
    build_predicted_count_pyramid,
)
from hpc.losses.hard_zero import HardZeroRegionLoss
from hpc.losses.supervised_contrastive import LocalDensityContrastiveLoss


@dataclass
class HPCLossConfig:
    tree: CountTreeConfig

    hard_zero_weight: float = 0.10
    local_contrast_weight: float = 0.05
    exact_count_weight: float = 0.0

    hard_zero_top_fraction: float = 0.10

    local_low_threshold: int = 1
    local_dense_threshold: int = 4

    local_max_samples: int = 256
    local_temperature: float = 0.10


class AdaptiveHPCLoss(nn.Module):
    def __init__(
        self,
        cfg: HPCLossConfig,
        feature_dim: int = 32,
    ):
        super().__init__()
        self.cfg = cfg

        self.tree_loss = AdaptiveProbabilisticCountTreeLoss(
            cfg.tree
        )

        self.hard_zero_loss = HardZeroRegionLoss(
            top_fraction=cfg.hard_zero_top_fraction
        )

        self.local_contrast = LocalDensityContrastiveLoss(
            feature_dim=feature_dim,
            low_threshold=cfg.local_low_threshold,
            dense_threshold=cfg.local_dense_threshold,
            max_samples=cfg.local_max_samples,
            temperature=cfg.local_temperature,
        )

    def forward(
        self,
        mass: torch.Tensor,
        p4: torch.Tensor,
        target_pyramid: Dict[int | str, torch.Tensor],
        valid16: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pred_pyramid = build_predicted_count_pyramid(
            mass,
            block_sizes=(8, 16, 32, 64),
            output_stride=4,
        )

        l_tree, tree_parts = self.tree_loss(
            target=target_pyramid,
            pred=pred_pyramid,
        )

        l_zero = self.hard_zero_loss(
            pred_count16=pred_pyramid[16],
            target_count16=target_pyramid[16],
            valid16=valid16,
        )

        l_contrast = self.local_contrast(
            p4=p4,
            y16=target_pyramid[16],
            valid16=valid16,
        )

        # Multi-scale exact count loss (for Stage A0 baseline or warmup)
        if self.cfg.exact_count_weight > 0.0:
            l_exact_parts = []
            for b in (16, 32, 64):
                if b in target_pyramid and b in pred_pyramid:
                    l_exact_parts.append(
                        torch.nn.functional.l1_loss(
                            pred_pyramid[b].to(torch.float32),
                            target_pyramid[b].to(torch.float32),
                        )
                    )
            if "N" in target_pyramid and "N" in pred_pyramid:
                pred_n = pred_pyramid["N"].to(torch.float32)
                tgt_n = target_pyramid["N"].to(torch.float32)
                l_exact_parts.append(
                    torch.mean(torch.abs(pred_n - tgt_n) / (tgt_n + 1.0))
                )
            l_exact = torch.stack(l_exact_parts).mean() if l_exact_parts else mass.to(torch.float32).sum() * 0.0
        else:
            l_exact = mass.to(torch.float32).sum() * 0.0

        total = (
            l_tree
            + self.cfg.hard_zero_weight * l_zero
            + self.cfg.local_contrast_weight * l_contrast
            + self.cfg.exact_count_weight * l_exact
        )

        logs = {
            **tree_parts,
            "hard_zero": l_zero,
            "local_contrast": l_contrast,
            "exact_count": l_exact,
            "total": total,
        }

        return total, logs
