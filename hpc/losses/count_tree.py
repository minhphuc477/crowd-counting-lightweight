from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .negative_binomial import negative_binomial_nll_mean_dispersion
from .dirichlet_multinomial import (
    dirichlet_multinomial_nll,
    multinomial_nll,
    normalize_positive_mass,
)


def pad_mass_map_for_image_multiple(
    mass: torch.Tensor,
    image_multiple: int = 64,
    output_stride: int = 4,
) -> torch.Tensor:
    feature_multiple = image_multiple // output_stride
    h, w = mass.shape[-2:]

    hp = ((h + feature_multiple - 1) // feature_multiple) * feature_multiple
    wp = ((w + feature_multiple - 1) // feature_multiple) * feature_multiple

    return F.pad(mass, (0, wp - w, 0, hp - h), value=0.0)


def sum_pool_mass(
    mass: torch.Tensor,
    block_size: int,
    output_stride: int = 4,
) -> torch.Tensor:
    if block_size % output_stride != 0:
        raise ValueError("block_size must be divisible by output_stride")

    k = block_size // output_stride
    x = F.avg_pool2d(mass, kernel_size=k, stride=k) * float(k * k)
    return x[:, 0]


def build_predicted_count_pyramid(
    mass: torch.Tensor,
    block_sizes=(8, 16, 32, 64),
    output_stride: int = 4,
) -> Dict[int | str, torch.Tensor]:
    mass_pad = pad_mass_map_for_image_multiple(
        mass,
        image_multiple=max(block_sizes),
        output_stride=output_stride,
    )

    out: Dict[int | str, torch.Tensor] = {}
    for block in block_sizes:
        out[block] = sum_pool_mass(
            mass_pad,
            block_size=block,
            output_stride=output_stride,
        )

    out["N"] = mass_pad.sum(dim=(1, 2, 3))
    return out


def group_four_children(child_map: torch.Tensor) -> torch.Tensor:
    if child_map.ndim != 3:
        raise ValueError("child_map must be [B,H,W]")

    b, h, w = child_map.shape
    if h % 2 != 0 or w % 2 != 0:
        raise ValueError("child map H,W must be even")

    hp, wp = h // 2, w // 2

    x = child_map.view(b, hp, 2, wp, 2)
    x = x.permute(0, 1, 3, 2, 4).contiguous()
    return x.view(b, hp, wp, 4)


@dataclass
class CountTreeConfig:
    root_dispersion: float = 50.0

    kappa_root64: float = 20.0
    kappa_64_32: float = 20.0
    kappa_32_16: float = 20.0
    kappa_16_8: float = 20.0
    kappa_flat16: float = 20.0

    dense_threshold_16: int = 4

    use_dirichlet_multinomial: bool = True

    w_root_nb: float = 1.0
    w_root64: float = 1.0
    w_64_32: float = 1.0
    w_32_16: float = 1.0
    w_16_8: float = 1.0

    # Ablation support
    w_flat_16: float = 0.0
    w_indep_nb: float = 0.0


class AdaptiveProbabilisticCountTreeLoss(nn.Module):
    def __init__(self, cfg: CountTreeConfig):
        super().__init__()
        self.cfg = cfg

    def _allocation_loss(
        self,
        target_children: torch.Tensor,
        pred_children: torch.Tensor,
        concentration: float,
        valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        probs = normalize_positive_mass(pred_children, dim=-1)

        if self.cfg.use_dirichlet_multinomial:
            return dirichlet_multinomial_nll(
                target_counts=target_children,
                probs=probs,
                concentration=concentration,
                valid_mask=valid_mask,
            )

        return multinomial_nll(
            target_counts=target_children,
            probs=probs,
            valid_mask=valid_mask,
        )

    def forward(self, target, pred):
        pieces = {}

        # 1. Root NB
        if self.cfg.w_root_nb > 0.0:
            l_root = negative_binomial_nll_mean_dispersion(
                target=target["N"],
                mean=pred["N"],
                dispersion=self.cfg.root_dispersion,
            )
        else:
            l_root = pred["N"].sum() * 0.0
        pieces["root_nb"] = l_root

        # 2. Root -> all 64 blocks
        b = target[64].shape[0]
        if self.cfg.w_root64 > 0.0:
            y64_flat = target[64].reshape(b, -1)
            m64_flat = pred[64].reshape(b, -1)
            l_root64 = self._allocation_loss(
                y64_flat,
                m64_flat,
                concentration=self.cfg.kappa_root64,
                valid_mask=target["N"] > 0,
            )
        else:
            l_root64 = pred["N"].sum() * 0.0
        pieces["root_to_64"] = l_root64

        # 3. 64 -> 32
        if self.cfg.w_64_32 > 0.0:
            y32_child = group_four_children(target[32])
            m32_child = group_four_children(pred[32])
            l_64_32 = self._allocation_loss(
                y32_child,
                m32_child,
                concentration=self.cfg.kappa_64_32,
                valid_mask=target[64] > 0,
            )
        else:
            l_64_32 = pred["N"].sum() * 0.0
        pieces["64_to_32"] = l_64_32

        # 4. 32 -> 16
        if self.cfg.w_32_16 > 0.0:
            y16_child = group_four_children(target[16])
            m16_child = group_four_children(pred[16])
            l_32_16 = self._allocation_loss(
                y16_child,
                m16_child,
                concentration=self.cfg.kappa_32_16,
                valid_mask=target[32] > 0,
            )
        else:
            l_32_16 = pred["N"].sum() * 0.0
        pieces["32_to_16"] = l_32_16

        # 5. Dense-only 16 -> 8
        if self.cfg.w_16_8 > 0.0:
            y8_child = group_four_children(target[8])
            m8_child = group_four_children(pred[8])
            dense16 = target[16] >= self.cfg.dense_threshold_16
            l_16_8 = self._allocation_loss(
                y8_child,
                m8_child,
                concentration=self.cfg.kappa_16_8,
                valid_mask=dense16,
            )
        else:
            l_16_8 = pred["N"].sum() * 0.0
        pieces["16_to_8_dense"] = l_16_8

        # 6. Flat DM at leaf 16 (Ablation A5: Flat DM vs Hierarchical DTM)
        if self.cfg.w_flat_16 > 0.0:
            y16_flat = target[16].reshape(b, -1)
            m16_flat = pred[16].reshape(b, -1)
            l_flat16 = self._allocation_loss(
                y16_flat,
                m16_flat,
                concentration=self.cfg.kappa_flat16,
                valid_mask=target["N"] > 0,
            )
        else:
            l_flat16 = pred["N"].sum() * 0.0
        pieces["flat_16"] = l_flat16

        # 7. Independent NB at scales 64, 32, 16 (Ablation A2: Independent NB vs DTM)
        if self.cfg.w_indep_nb > 0.0:
            l_indep_parts = []
            for blk in (16, 32, 64):
                if blk in target and blk in pred:
                    l_indep_parts.append(
                        negative_binomial_nll_mean_dispersion(
                            target=target[blk],
                            mean=pred[blk],
                            dispersion=self.cfg.root_dispersion,
                        )
                    )
            l_indep = torch.stack(l_indep_parts).mean() if l_indep_parts else pred["N"].sum() * 0.0
        else:
            l_indep = pred["N"].sum() * 0.0
        pieces["indep_nb"] = l_indep

        total = (
            self.cfg.w_root_nb * l_root
            + self.cfg.w_root64 * l_root64
            + self.cfg.w_64_32 * l_64_32
            + self.cfg.w_32_16 * l_32_16
            + self.cfg.w_16_8 * l_16_8
            + self.cfg.w_flat_16 * l_flat16
            + self.cfg.w_indep_nb * l_indep
        )

        pieces["tree_total"] = total
        return total, pieces
