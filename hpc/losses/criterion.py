"""HPC-S-SR48 Training Criterion — Point-Decomposition & Direct Count Revision.

Key Upgrades:
  1. DirectCountL1Loss (lambda_count=1.0): Anchors global image-level count conservation.
  2. PointMassDecompositionLoss (lambda_point=1.0): Per-person Voronoi unit-mass conservation (sum_{V_i} D_k = 1.0),
     spatial centroid compactness, and true-background suppression. Replaces flawed 16x16 allocation CE.
  3. HierarchicalNBLoss (lambda_hnb=0.25, frozen dispersion): Probabilistic multi-scale regularizer.
  4. HardNegativeMassLoss (lambda_hn=0.10): Suppresses false positive clusters.
  5. SSER Routing Loss (lambda_route=0.10): Supervision for multi-scale pyramid routing.
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .negative_binomial import HierarchicalNBLoss
from .hard_negative import (
    HardNegativeMassLoss,
    WholeImageEmptyLoss,
    GlobalCountLoss,
    DirectCountL1Loss,
)
from .robustness import RobustConsistencyLoss
from .routing import RoutingSupervisionLoss
from .point_mass import BoundedPointMassLoss
from .multiscale_mae import MultiScaleBlockMAELoss


class HPCLossCriterion(nn.Module):
    """Refined HPC-S Training Criterion with Person-Level Mass Conservation.

    Total loss:
        L = λ_count * L_count   (global count magnitude, L1 aligned with MAE)
          + λ_point * L_point   (per-person Voronoi unit mass |m_i - 1| + centroid compactness)
          + λ_hnb   * L_HNB     (hierarchical NB multi-scale regularizer, frozen dispersion)
          + λ_hn    * L_HN      (hard negative suppression on empty blocks)
          + λ_empty * L_empty   (whole-image empty suppression)
          + λ_global* L_global  (log-smooth-l1 count stability)
          + λ_rob   * L_rob     (robustness consistency)
          + λ_route * L_route   (SSER geometry routing supervision)
    """

    def __init__(
        self,
        block_sizes: List[int],
        allocation_block: int = 16,
        quantiles: Optional[Dict[int, Tuple[float, float]]] = None,
        # Loss weights
        lambda_count: float = 1.0,
        count_scale: float = 100.0,
        lambda_point: float = 0.0,
        lambda_ms_mae: float = 0.0,   # Multi-scale block MAE direct density supervision
        lambda_hnb: float = 0.25,
        lambda_alloc: float = 0.0,    # Deprecated in favor of point loss / ms_mae
        lambda_hn: float = 0.10,
        lambda_empty: float = 0.25,
        lambda_global: float = 0.10,
        lambda_rob: float = 0.05,
        lambda_kd: float = 0.0,
        lambda_route: float = 0.1,
        # Point Loss Config
        point_bg_distance_px: float = 32.0,
        point_dispersion_weight: float = 0.05,
        point_bg_weight: float = 0.5,
        # General Config
        hard_negative_fraction: float = 0.10,
        use_stratified_nb: bool = True,
        use_poisson: bool = False,
        global_count_mode: str = "log_smooth_l1",
        learn_dispersion: bool = False,
        enable_curriculum: bool = True,
    ):
        super().__init__()
        self.block_sizes = [int(b) for b in block_sizes]
        self.allocation_block = int(allocation_block)

        self.lambda_count   = float(lambda_count)
        self.lambda_point   = float(lambda_point)
        self.lambda_ms_mae  = float(lambda_ms_mae)
        self.lambda_hnb     = float(lambda_hnb)
        self.lambda_alloc   = float(lambda_alloc)
        self.lambda_hn      = float(lambda_hn)
        self.lambda_empty   = float(lambda_empty)
        self.lambda_global  = float(lambda_global)
        self.lambda_rob     = float(lambda_rob)
        self.lambda_kd      = float(lambda_kd)
        self.lambda_route   = float(lambda_route)
        self.enable_curriculum = bool(enable_curriculum)

        self.hnb_loss = HierarchicalNBLoss(
            block_sizes=self.block_sizes,
            quantiles=quantiles,
            use_stratified=use_stratified_nb,
            use_poisson=use_poisson,
            learn_dispersion=learn_dispersion,
        )
        self.point_loss = BoundedPointMassLoss(
            output_stride=4,
            min_radius_px=8.0,
            max_radius_px=24.0,
            bg_weight=point_bg_weight,
        )
        self.ms_mae_loss = MultiScaleBlockMAELoss(
            block_sizes=self.block_sizes,
            output_stride=4,
        )
        self.count_loss  = DirectCountL1Loss(count_scale=count_scale)
        self.hn_loss     = HardNegativeMassLoss(top_fraction=hard_negative_fraction,
                                                block_size=self.allocation_block)
        self.empty_loss  = WholeImageEmptyLoss(use_warmup_log=False)
        self.global_loss = GlobalCountLoss(mode=global_count_mode)
        self.rob_loss    = RobustConsistencyLoss(block_sizes=self.block_sizes, output_stride=4)
        self.routing_loss = RoutingSupervisionLoss()

    def _effective_weights(self, progress: float) -> Dict[str, float]:
        """Return effective per-term weights = lambda * curriculum_factor."""
        p = min(max(float(progress), 0.0), 1.0)

        if not self.enable_curriculum or p >= 0.10:
            f = dict(count=1.0, point=1.0, ms_mae=1.0, hnb=1.0, hn=1.0,
                     empty=1.0, global_=1.0, rob=1.0, route=1.0)
        elif p < 0.03:
            f = dict(count=1.0, point=0.5, ms_mae=0.5, hnb=0.5, hn=0.0,
                     empty=0.5, global_=1.0, rob=0.0, route=0.0)
        else:  # 3–10%
            f = dict(count=1.0, point=1.0, ms_mae=1.0, hnb=1.0, hn=0.5,
                     empty=1.0, global_=1.0, rob=0.0, route=1.0)

        return {
            "count":  self.lambda_count  * f["count"],
            "point":  self.lambda_point  * f["point"],
            "ms_mae": self.lambda_ms_mae * f["ms_mae"],
            "hnb":    self.lambda_hnb    * f["hnb"],
            "hn":     self.lambda_hn     * f["hn"],
            "empty":  self.lambda_empty  * f["empty"],
            "global": self.lambda_global * f["global_"],
            "rob":    self.lambda_rob    * f["rob"],
            "route":  self.lambda_route  * f["route"],
        }

    def forward(
        self,
        d_map: torch.Tensor,
        gt_block_counts: Dict[int, torch.Tensor],
        gt_counts: torch.Tensor,
        gt_points: Optional[List[torch.Tensor]] = None,
        gt_z_alloc: Optional[torch.Tensor] = None,
        gt_special_mask16: Optional[torch.Tensor] = None,
        d_degraded: Optional[torch.Tensor] = None,
        degraded_mask: Optional[torch.Tensor] = None,
        routes8: Optional[torch.Tensor] = None,
        gt_route_q: Optional[torch.Tensor] = None,
        gt_route_mask: Optional[torch.Tensor] = None,
        progress: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Compatibility with the original positional API:
        # criterion(d_map, gt_blocks, gt_z_alloc, gt_counts, ...).
        # New code should pass gt_counts by keyword, but old trainers/tests must
        # not silently reinterpret a spatial allocation map as image counts.
        if (
            isinstance(gt_counts, torch.Tensor)
            and gt_counts.ndim > 1
            and isinstance(gt_points, torch.Tensor)
            and gt_points.ndim == 1
            and gt_points.shape[0] == d_map.shape[0]
        ):
            legacy_gt_z_alloc = gt_counts
            gt_counts = gt_points
            gt_points = None
            if gt_z_alloc is None:
                gt_z_alloc = legacy_gt_z_alloc

        loss_dict: Dict[str, torch.Tensor] = {}

        # 1. Direct count loss (main global constraint)
        l_count = self.count_loss(d_map, gt_counts)

        # Count diagnostics
        with torch.no_grad():
            pred_counts_d = d_map.sum(dim=(-1, -2, -3)).float()
            gt_counts_d   = gt_counts.to(pred_counts_d.device, dtype=torch.float32).reshape(-1)
            loss_dict["batch_count_mae"]         = torch.mean(torch.abs(pred_counts_d - gt_counts_d))
            loss_dict["mean_pred_count"]         = pred_counts_d.mean()
            loss_dict["mean_gt_count"]           = gt_counts_d.mean()
            loss_dict["mean_signed_count_error"] = (pred_counts_d - gt_counts_d).mean()

        # 2. Point-Neighbor Mass Conservation Loss (person-level spatial constraint)
        if gt_points is not None and self.lambda_point > 0:
            l_point, point_details = self.point_loss(d_map, gt_points)
            loss_dict.update(point_details)
        else:
            l_point = d_map.new_zeros(())

        # 2b. Multi-Scale Block MAE Loss (direct multi-resolution density supervision)
        if self.lambda_ms_mae > 0:
            l_ms_mae, ms_details = self.ms_mae_loss(d_map, gt_block_counts)
            loss_dict.update(ms_details)
        else:
            l_ms_mae = d_map.new_zeros(())

        # 3. Hierarchical Negative Binomial Loss (multi-scale probabilistic regularizer)
        l_hnb, hnb_details = self.hnb_loss(d_map, gt_block_counts)
        loss_dict.update(hnb_details)

        # 4. Global log-count (smooth L1 stability)
        l_global = self.global_loss(d_map, gt_counts)

        # 5. Hard negative + empty suppression
        y_alloc = gt_block_counts[self.allocation_block]
        l_hn    = self.hn_loss(d_map, y_alloc)
        l_empty = self.empty_loss(d_map, gt_counts)

        # 6. Robustness consistency
        if d_degraded is not None:
            l_rob = self.rob_loss(d_map, d_degraded, valid_mask=degraded_mask)
        else:
            l_rob = d_map.new_zeros(())

        # 7. SSER geometry routing supervision
        if routes8 is not None and gt_route_q is not None and gt_route_mask is not None:
            l_route = self.routing_loss(routes8, gt_route_q, gt_route_mask)
        else:
            l_route = d_map.new_zeros(())

        # Weighted combination
        weights = self._effective_weights(progress)
        total_loss = (
            weights["count"]  * l_count
            + weights["point"]* l_point
            + weights["ms_mae"]* l_ms_mae
            + weights["hnb"]  * l_hnb
            + weights["hn"]   * l_hn
            + weights["empty"]* l_empty
            + weights["global"]* l_global
            + weights["rob"]  * l_rob
            + weights["route"]* l_route
        )

        if not torch.isfinite(total_loss):
            raise FloatingPointError(f"Non-finite total loss: {total_loss.item():.6f}")

        loss_dict.update({
            "loss_count":  l_count.detach(),
            "loss_point":  l_point.detach(),
            "loss_ms_mae": l_ms_mae.detach(),
            "loss_hnb":    l_hnb.detach(),
            "loss_hn":     l_hn.detach(),
            "loss_empty":  l_empty.detach(),
            "loss_global": l_global.detach(),
            "loss_rob":    l_rob.detach(),
            "loss_route":  l_route.detach(),
            "loss_total":  total_loss.detach(),
            "loss_alloc":  d_map.new_zeros(()),  # deprecated compatibility key
        })

        for name, val in weights.items():
            loss_dict[f"weight_{name}"] = d_map.new_tensor(val)

        return total_loss, loss_dict
