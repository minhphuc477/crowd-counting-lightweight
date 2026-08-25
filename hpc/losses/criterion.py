from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .negative_binomial import HierarchicalNBLoss
from .allocation import LocalAllocationLoss
from .hard_negative import HardNegativeMassLoss, WholeImageEmptyLoss, GlobalCountLoss
from .robustness import RobustConsistencyLoss


class HPCLossCriterion(nn.Module):
    """Full HPC-Lite training criterion with curriculum implemented as multipliers."""

    def __init__(
        self,
        block_sizes: List[int],
        allocation_block: int = 16,
        quantiles: Optional[Dict[int, Tuple[float, float]]] = None,
        lambda_hnb: float = 1.0,
        lambda_alloc: float = 0.5,
        lambda_hn: float = 0.25,
        lambda_empty: float = 0.5,
        lambda_global: float = 1.0,
        lambda_rob: float = 0.1,
        hard_negative_fraction: float = 0.10,
        use_stratified_nb: bool = True,
        use_poisson: bool = False,
        global_count_mode: str = "log_smooth_l1",
        enable_curriculum: bool = True,
    ):
        super().__init__()
        self.block_sizes = [int(b) for b in block_sizes]
        self.allocation_block = int(allocation_block)
        if self.allocation_block not in self.block_sizes:
            raise ValueError("allocation_block must be present in block_sizes")

        self.lambda_hnb = float(lambda_hnb)
        self.lambda_alloc = float(lambda_alloc)
        self.lambda_hn = float(lambda_hn)
        self.lambda_empty = float(lambda_empty)
        self.lambda_global = float(lambda_global)
        self.lambda_rob = float(lambda_rob)
        self.enable_curriculum = bool(enable_curriculum)

        self.hnb_loss = HierarchicalNBLoss(
            block_sizes=self.block_sizes,
            quantiles=quantiles,
            use_stratified=use_stratified_nb,
            use_poisson=use_poisson,
        )
        self.alloc_loss = LocalAllocationLoss(
            block_size=self.allocation_block,
            output_stride=4,
        )
        self.hn_loss = HardNegativeMassLoss(
            top_fraction=hard_negative_fraction,
            block_size=self.allocation_block,
        )
        self.empty_loss = WholeImageEmptyLoss(use_warmup_log=False)
        self.global_loss = GlobalCountLoss(mode=global_count_mode)
        self.rob_loss = RobustConsistencyLoss(block_sizes=self.block_sizes, output_stride=4)

    def _effective_weights(self, progress: float) -> Dict[str, float]:
        progress = min(max(float(progress), 0.0), 1.0)
        # Multipliers preserve user-configured lambdas and make lambda=0 a true ablation.
        if not self.enable_curriculum or progress >= 0.30:
            factors = dict(hnb=1.0, global_=1.0, alloc=1.0, hn=1.0, empty=1.0, rob=1.0)
        elif progress < 0.10:
            # Old effective defaults: HNB=1, Global=1, Alloc=0.25, others=0.
            factors = dict(hnb=1.0, global_=1.0, alloc=0.5, hn=0.0, empty=0.0, rob=0.0)
        else:
            # Old effective defaults: HNB=1, Global=1, Alloc=0.5, HN=0.10, Empty=0.25.
            factors = dict(hnb=1.0, global_=1.0, alloc=1.0, hn=0.4, empty=0.5, rob=0.0)

        return {
            "hnb": self.lambda_hnb * factors["hnb"],
            "global": self.lambda_global * factors["global_"],
            "alloc": self.lambda_alloc * factors["alloc"],
            "hn": self.lambda_hn * factors["hn"],
            "empty": self.lambda_empty * factors["empty"],
            "rob": self.lambda_rob * factors["rob"],
        }

    def forward(
        self,
        d_map: torch.Tensor,
        gt_block_counts: Dict[int, torch.Tensor],
        gt_z_alloc: torch.Tensor,
        gt_counts: torch.Tensor,
        d_degraded: Optional[torch.Tensor] = None,
        degraded_mask: Optional[torch.Tensor] = None,
        progress: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        loss_dict: Dict[str, torch.Tensor] = {}

        l_hnb, hnb_details = self.hnb_loss(d_map, gt_block_counts)
        loss_dict.update(hnb_details)
        l_global = self.global_loss(d_map, gt_counts)

        y_alloc = gt_block_counts[self.allocation_block]
        l_alloc = self.alloc_loss(d_map, gt_z_alloc, y_alloc)
        l_hn = self.hn_loss(d_map, y_alloc)
        l_empty = self.empty_loss(d_map, gt_counts)

        if d_degraded is not None:
            l_rob = self.rob_loss(d_map, d_degraded, valid_mask=degraded_mask)
        else:
            l_rob = d_map.new_zeros(())

        weights = self._effective_weights(progress)
        total_loss = (
            weights["hnb"] * l_hnb
            + weights["global"] * l_global
            + weights["alloc"] * l_alloc
            + weights["hn"] * l_hn
            + weights["empty"] * l_empty
            + weights["rob"] * l_rob
        )

        for name, value in {
            "loss_hnb": l_hnb,
            "loss_global": l_global,
            "loss_alloc": l_alloc,
            "loss_hn": l_hn,
            "loss_empty": l_empty,
            "loss_rob": l_rob,
            "loss_total": total_loss,
        }.items():
            loss_dict[name] = value.detach()
        for name, value in weights.items():
            loss_dict[f"weight_{name}"] = d_map.new_tensor(value)

        return total_loss, loss_dict
