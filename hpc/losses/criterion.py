"""HPC-S-SR48 Training Criterion — Experiment 1 revision.

Changes from previous version (per HPC_Lite_Objective_Training_Revision_2000ep.md):
  - DirectCountL1Loss added as main objective (lambda_count=1.0, weight 1.0)
  - HNB demoted to probabilistic regularizer (lambda_hnb=0.35)
  - Allocation reduced (lambda_alloc=0.15)
  - Empty suppression raised (lambda_empty=0.25)
  - Global log kept small (lambda_global=0.10)
  - Rob reduced (lambda_rob=0.05)
  - NB dispersion frozen via learn_dispersion=False (forces mean to absorb count error)
  - Curriculum simplified: 3 phases at 0/5/15% — always multiplies user lambdas
  - Detailed count diagnostics logged: batch_count_mae, mean_pred_count, etc.
  - SSER route supervision preserved (lambda_route=0.1, active from 10%)
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .negative_binomial import HierarchicalNBLoss
from .allocation import LocalAllocationLoss
from .hard_negative import (
    HardNegativeMassLoss,
    WholeImageEmptyLoss,
    GlobalCountLoss,
    SpecialBlockCountLoss,
    DirectCountL1Loss,
)
from .robustness import RobustConsistencyLoss
from .routing import RoutingSupervisionLoss


class HPCLossCriterion(nn.Module):
    """Full HPC-S-SR48 training criterion — Experiment 1 objective revision.

    Total loss (all phases):
        L = λ_count * L_count   (direct MAE-aligned count, main objective)
          + λ_hnb   * L_HNB     (hierarchical NB, demoted to regularizer)
          + λ_alloc * L_alloc   (spatial allocation, spatial regularizer)
          + λ_hn    * L_HN      (hard negative mass suppression)
          + λ_empty * L_empty   (empty image suppression)
          + λ_global* L_global  (log-smooth-l1 count stability)
          + λ_rob   * L_rob     (robustness consistency)
          + λ_route * L_route   (SSER geometry routing supervision)

    Curriculum multipliers (phases defined as fraction of total steps):
        0–5%:    count=1, hnb=0.5, alloc=0, hn=0, empty=0.5, global=1, rob=0, route=0
        5–15%:   count=1, hnb=1,   alloc=0.5, hn=0.5, empty=1, global=1, rob=0, route=1
        15–100%: count=1, hnb=1,   alloc=1,   hn=1,   empty=1, global=1, rob=1, route=1

    Invariant: setting any lambda to 0 yields zero contribution at all progress values.
    """

    def __init__(
        self,
        block_sizes: List[int],
        allocation_block: int = 16,
        quantiles: Optional[Dict[int, Tuple[float, float]]] = None,
        # Loss weights (Experiment 1 recommended values)
        lambda_count: float = 1.0,
        count_scale: float = 100.0,
        lambda_hnb: float = 0.35,
        lambda_alloc: float = 0.15,
        lambda_hn: float = 0.10,
        lambda_empty: float = 0.25,
        lambda_global: float = 0.10,
        lambda_direct: float = 0.0,   # kept for backwards compat, not used in Exp1
        lambda_special: float = 0.0,  # kept for backwards compat
        lambda_rob: float = 0.05,
        lambda_kd: float = 0.0,
        lambda_route: float = 0.1,
        # Config
        hard_negative_fraction: float = 0.10,
        use_stratified_nb: bool = True,
        use_poisson: bool = False,
        global_count_mode: str = "log_smooth_l1",
        special_alloc_beta: float = 1.0,
        learn_dispersion: bool = False,   # Experiment 1: freeze NB dispersion
        enable_curriculum: bool = True,
    ):
        super().__init__()
        self.block_sizes = [int(b) for b in block_sizes]
        self.allocation_block = int(allocation_block)
        if self.allocation_block not in self.block_sizes:
            raise ValueError("allocation_block must be present in block_sizes")

        self.lambda_count   = float(lambda_count)
        self.lambda_hnb     = float(lambda_hnb)
        self.lambda_alloc   = float(lambda_alloc)
        self.lambda_hn      = float(lambda_hn)
        self.lambda_empty   = float(lambda_empty)
        self.lambda_global  = float(lambda_global)
        self.lambda_rob     = float(lambda_rob)
        self.lambda_kd      = float(lambda_kd)
        self.lambda_route   = float(lambda_route)
        self.special_alloc_beta = float(special_alloc_beta)
        self.enable_curriculum = bool(enable_curriculum)

        self.hnb_loss = HierarchicalNBLoss(
            block_sizes=self.block_sizes,
            quantiles=quantiles,
            use_stratified=use_stratified_nb,
            use_poisson=use_poisson,
            learn_dispersion=learn_dispersion,
        )
        self.alloc_loss = LocalAllocationLoss(
            block_size=self.allocation_block,
            output_stride=4,
        )
        self.hn_loss     = HardNegativeMassLoss(top_fraction=hard_negative_fraction,
                                                 block_size=self.allocation_block)
        self.empty_loss  = WholeImageEmptyLoss(use_warmup_log=False)
        self.global_loss = GlobalCountLoss(mode=global_count_mode)
        self.count_loss  = DirectCountL1Loss(count_scale=count_scale)
        self.rob_loss    = RobustConsistencyLoss(block_sizes=self.block_sizes, output_stride=4)
        self.routing_loss = RoutingSupervisionLoss()

    # ------------------------------------------------------------------
    # Curriculum schedule
    # ------------------------------------------------------------------
    def _effective_weights(self, progress: float) -> Dict[str, float]:
        """Return effective per-term weights = lambda * curriculum_factor.

        Phases:
            0–5%:    count stabilization (direct count + global + partial hnb/empty)
            5–15%:   introduce spatial learning (add alloc, hn, full empty, route)
            15–100%: full objective
        """
        p = min(max(float(progress), 0.0), 1.0)

        if not self.enable_curriculum or p >= 0.15:
            f = dict(count=1.0, hnb=1.0, alloc=1.0, hn=1.0,
                     empty=1.0, global_=1.0, rob=1.0, route=1.0)
        elif p < 0.05:
            f = dict(count=1.0, hnb=0.5, alloc=0.0, hn=0.0,
                     empty=0.5, global_=1.0, rob=0.0, route=0.0)
        else:  # 5–15%
            f = dict(count=1.0, hnb=1.0, alloc=0.5, hn=0.5,
                     empty=1.0, global_=1.0, rob=0.0, route=1.0)

        return {
            "count":  self.lambda_count  * f["count"],
            "hnb":    self.lambda_hnb    * f["hnb"],
            "alloc":  self.lambda_alloc  * f["alloc"],
            "hn":     self.lambda_hn     * f["hn"],
            "empty":  self.lambda_empty  * f["empty"],
            "global": self.lambda_global * f["global_"],
            "rob":    self.lambda_rob    * f["rob"],
            "route":  self.lambda_route  * f["route"],
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        d_map: torch.Tensor,
        gt_block_counts: Dict[int, torch.Tensor],
        gt_z_alloc: torch.Tensor,
        gt_counts: torch.Tensor,
        gt_special_mask16: Optional[torch.Tensor] = None,
        d_degraded: Optional[torch.Tensor] = None,
        degraded_mask: Optional[torch.Tensor] = None,
        teacher_map: Optional[torch.Tensor] = None,
        routes8: Optional[torch.Tensor] = None,
        gt_route_q: Optional[torch.Tensor] = None,
        gt_route_mask: Optional[torch.Tensor] = None,
        progress: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        loss_dict: Dict[str, torch.Tensor] = {}

        # --- Direct count loss (main objective) ---
        l_count = self.count_loss(d_map, gt_counts)

        # --- Count diagnostics (no gradient, always logged) ---
        with torch.no_grad():
            pred_counts_d = d_map.sum(dim=(-1, -2, -3)).float()
            gt_counts_d   = gt_counts.to(pred_counts_d.device, dtype=torch.float32).reshape(-1)
            loss_dict["batch_count_mae"]        = torch.mean(torch.abs(pred_counts_d - gt_counts_d))
            loss_dict["mean_pred_count"]        = pred_counts_d.mean()
            loss_dict["mean_gt_count"]          = gt_counts_d.mean()
            loss_dict["mean_signed_count_error"]= (pred_counts_d - gt_counts_d).mean()

        # --- HNB (hierarchical NB, probabilistic regularizer) ---
        l_hnb, hnb_details = self.hnb_loss(d_map, gt_block_counts)
        loss_dict.update(hnb_details)

        # --- Global log-count (stability for large residuals) ---
        l_global = self.global_loss(d_map, gt_counts)

        # --- Allocation (spatial regularizer) ---
        y_alloc = gt_block_counts[self.allocation_block]
        weights16 = None
        if gt_special_mask16 is not None:
            weights16 = 1.0 + self.special_alloc_beta * gt_special_mask16.float().to(d_map.device)

        l_alloc, alloc_details = self.alloc_loss(
            d_map, gt_z_alloc, y_alloc,
            block_weights=weights16,
            return_details=True,
        )
        loss_dict.update(alloc_details)

        # --- Hard negative + empty suppression ---
        l_hn    = self.hn_loss(d_map, y_alloc)
        l_empty = self.empty_loss(d_map, gt_counts)

        # --- Robustness consistency ---
        if d_degraded is not None:
            l_rob = self.rob_loss(d_map, d_degraded, valid_mask=degraded_mask)
        else:
            l_rob = d_map.new_zeros(())

        # --- SSER geometry routing supervision (training-only) ---
        if routes8 is not None and gt_route_q is not None and gt_route_mask is not None:
            l_route = self.routing_loss(routes8, gt_route_q, gt_route_mask)
        else:
            l_route = d_map.new_zeros(())

        # --- Weighted sum ---
        weights = self._effective_weights(progress)
        total_loss = (
            weights["count"]  * l_count
            + weights["hnb"]  * l_hnb
            + weights["alloc"]* l_alloc
            + weights["hn"]   * l_hn
            + weights["empty"]* l_empty
            + weights["global"]* l_global
            + weights["rob"]  * l_rob
            + weights["route"]* l_route
        )

        if not torch.isfinite(total_loss):
            raise FloatingPointError(f"Non-finite total loss: {total_loss.item():.6f}")

        # --- Fill loss_dict ---
        for name, val in {
            "loss_count":  l_count,
            "loss_hnb":    l_hnb,
            "loss_alloc":  l_alloc,
            "loss_hn":     l_hn,
            "loss_empty":  l_empty,
            "loss_global": l_global,
            "loss_rob":    l_rob,
            "loss_route":  l_route,
            "loss_total":  total_loss,
        }.items():
            loss_dict[name] = val.detach()

        for name, val in weights.items():
            loss_dict[f"weight_{name}"] = d_map.new_tensor(val)

        return total_loss, loss_dict
