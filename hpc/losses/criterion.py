from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .negative_binomial import HierarchicalNBLoss
from .allocation import LocalAllocationLoss
from .hard_negative import HardNegativeMassLoss, WholeImageEmptyLoss, GlobalCountLoss, SpecialBlockCountLoss
from .robustness import RobustConsistencyLoss
from .routing import RoutingSupervisionLoss


class HPCLossCriterion(nn.Module):
    """Full HPC-S-SR48 training criterion with curriculum multipliers.

    Loss terms:
        L = λ_H * L_HNB  +  λ_A * L_alloc  +  λ_N * L_HN  +  λ_E * L_empty
          + λ_G * L_global_log  +  λ_D * L_direct  +  λ_S * L_special
          + λ_R * L_rob  +  λ_K * L_KD

    Curriculum multipliers are applied so lambda=0 disables a term at every phase.

    Phase schedule (§14):
        0–10%:   HNB=1, Alloc=0.5, HN=0,   Empty=0,   Global=1, Direct=0.5, Special=0,   Rob=0, KD=0
        10–30%:  HNB=1, Alloc=1.0, HN=0.4, Empty=0.5, Global=1, Direct=1.0, Special=0.5, Rob=0, KD=0.5 if enabled
        30–100%: HNB=1, Alloc=1.0, HN=1.0, Empty=1.0, Global=1, Direct=1.0, Special=1.0, Rob=1, KD=1.0 if enabled
    """

    def __init__(
        self,
        block_sizes: List[int],
        allocation_block: int = 16,
        quantiles: Optional[Dict[int, Tuple[float, float]]] = None,
        # Loss weights
        lambda_hnb: float = 1.0,
        lambda_alloc: float = 0.5,
        lambda_hn: float = 0.25,
        lambda_empty: float = 0.5,
        lambda_global: float = 0.5,
        lambda_direct: float = 0.5,
        lambda_special: float = 0.25,
        lambda_rob: float = 0.1,
        lambda_kd: float = 0.0,
        lambda_route: float = 0.1,   # ← SSER geometry supervision
        # Config
        hard_negative_fraction: float = 0.10,
        use_stratified_nb: bool = True,
        use_poisson: bool = False,
        global_count_mode: str = "log_smooth_l1",
        special_alloc_beta: float = 1.0,
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
        self.lambda_direct = float(lambda_direct)
        self.lambda_special = float(lambda_special)
        self.lambda_rob = float(lambda_rob)
        self.lambda_kd = float(lambda_kd)
        self.lambda_route = float(lambda_route)
        self.special_alloc_beta = float(special_alloc_beta)
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
        self.direct_loss = GlobalCountLoss(mode="sqrt_normalized")
        self.special_loss = SpecialBlockCountLoss(
            block_size=self.allocation_block,
            output_stride=4,
        )
        self.rob_loss = RobustConsistencyLoss(block_sizes=self.block_sizes, output_stride=4)
        self.routing_loss = RoutingSupervisionLoss()

    def _effective_weights(self, progress: float) -> Dict[str, float]:
        progress = min(max(float(progress), 0.0), 1.0)
        if not self.enable_curriculum or progress >= 0.30:
            f = dict(hnb=1.0, alloc=1.0, hn=1.0, empty=1.0, global_=1.0,
                     direct=1.0, special=1.0, rob=1.0, kd=1.0, route=1.0)
        elif progress < 0.10:
            f = dict(hnb=1.0, alloc=0.5, hn=0.0, empty=0.0, global_=1.0,
                     direct=0.5, special=0.0, rob=0.0, kd=0.0, route=0.0)
        else:  # 10–30%
            f = dict(hnb=1.0, alloc=1.0, hn=0.4, empty=0.5, global_=1.0,
                     direct=1.0, special=0.5, rob=0.0, kd=0.5, route=1.0)
        return {
            "hnb":     self.lambda_hnb     * f["hnb"],
            "alloc":   self.lambda_alloc   * f["alloc"],
            "hn":      self.lambda_hn      * f["hn"],
            "empty":   self.lambda_empty   * f["empty"],
            "global":  self.lambda_global  * f["global_"],
            "direct":  self.lambda_direct  * f["direct"],
            "special": self.lambda_special * f["special"],
            "rob":     self.lambda_rob     * f["rob"],
            "kd":      self.lambda_kd      * f["kd"],
            "route":   self.lambda_route   * f["route"],
        }

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

        # --- Core count losses ---
        l_hnb, hnb_details = self.hnb_loss(d_map, gt_block_counts)
        loss_dict.update(hnb_details)
        l_global = self.global_loss(d_map, gt_counts)
        l_direct = self.direct_loss(d_map, gt_counts)

        # --- Allocation with special-block weighting ---
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
        l_hn = self.hn_loss(d_map, y_alloc)
        l_empty = self.empty_loss(d_map, gt_counts)

        # --- Special-block count loss ---
        if gt_special_mask16 is not None:
            l_special = self.special_loss(d_map, y_alloc, gt_special_mask16)
        else:
            l_special = d_map.new_zeros(())

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

        # --- Knowledge distillation (optional, training-only) ---
        if teacher_map is not None and self.lambda_kd > 0:
            with torch.no_grad():
                teacher_map_detach = teacher_map.detach()
            l_kd_terms = []
            from .negative_binomial import sum_pool
            import torch.nn.functional as F
            for B in self.block_sizes:
                mu_s = sum_pool(d_map, B, output_stride=4)
                mu_t = sum_pool(teacher_map_detach, B, output_stride=4)
                l_kd_terms.append(F.smooth_l1_loss(
                    torch.log1p(mu_s.float()),
                    torch.log1p(mu_t.float()),
                ))
            l_kd = torch.stack(l_kd_terms).mean()
        else:
            l_kd = d_map.new_zeros(())

        # --- Weighted sum ---
        weights = self._effective_weights(progress)
        total_loss = (
            weights["hnb"]     * l_hnb
            + weights["alloc"] * l_alloc
            + weights["hn"]    * l_hn
            + weights["empty"] * l_empty
            + weights["global"]* l_global
            + weights["direct"]* l_direct
            + weights["special"]* l_special
            + weights["rob"]   * l_rob
            + weights["kd"]    * l_kd
            + weights["route"] * l_route
        )

        for name, value in {
            "loss_hnb": l_hnb,
            "loss_alloc": l_alloc,
            "loss_hn": l_hn,
            "loss_empty": l_empty,
            "loss_global": l_global,
            "loss_direct": l_direct,
            "loss_special": l_special,
            "loss_rob": l_rob,
            "loss_kd": l_kd,
            "loss_route": l_route,
            "loss_total": total_loss,
        }.items():
            loss_dict[name] = value.detach()
        for name, value in weights.items():
            loss_dict[f"weight_{name}"] = d_map.new_tensor(value)

        return total_loss, loss_dict
