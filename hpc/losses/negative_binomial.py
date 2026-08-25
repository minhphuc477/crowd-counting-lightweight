import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_MAX_DISPERSION = 1e4


def _inv_softplus(y: float) -> float:
    y = max(float(y), 1e-12)
    return y + math.log(-math.expm1(-y))


def sum_pool(x: torch.Tensor, input_block_size: int, output_stride: int = 4) -> torch.Tensor:
    """Exact non-overlapping sum pooling over blocks measured in input pixels."""
    if input_block_size % output_stride != 0:
        raise ValueError(
            f"input_block_size ({input_block_size}) must be divisible by output_stride ({output_stride})"
        )
    k = input_block_size // output_stride
    if x.shape[-2] % k != 0 or x.shape[-1] % k != 0:
        raise ValueError(
            f"Output map {tuple(x.shape[-2:])} must be divisible by pooling kernel {k}"
        )
    return F.avg_pool2d(x, kernel_size=k, stride=k) * float(k * k)


def nb_nll(y: torch.Tensor, mu: torch.Tensor, r: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Negative-Binomial NLL parameterized by mean mu and dispersion r.

    Var(Y)=mu+mu^2/r. Computed in float32 even under autocast.
    """
    y = y.float()
    mu = mu.float().clamp_min(eps)
    r = r.float().clamp(min=eps, max=_MAX_DISPERSION)
    if torch.any(y < 0):
        raise ValueError("Negative-Binomial targets must be non-negative")

    log_r_plus_mu = torch.log(r + mu)
    log_prob = (
        torch.lgamma(y + r)
        - torch.lgamma(r)
        - torch.lgamma(y + 1.0)
        + r * (torch.log(r) - log_r_plus_mu)
        + y * (torch.log(mu) - log_r_plus_mu)
    )
    return -log_prob


def poisson_nll(y: torch.Tensor, mu: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Poisson NLL baseline, including the parameter-independent log(y!) term."""
    y = y.float()
    mu = mu.float().clamp_min(eps)
    if torch.any(y < 0):
        raise ValueError("Poisson targets must be non-negative")
    return -(y * torch.log(mu) - mu - torch.lgamma(y + 1.0))


class HierarchicalNBLoss(nn.Module):
    """Hierarchical NB loss with optional density-stratified risk reduction."""

    def __init__(
        self,
        block_sizes: List[int],
        quantiles: Optional[Dict[int, Tuple[float, float]]] = None,
        use_stratified: bool = True,
        use_poisson: bool = False,
        max_dispersion: float = _MAX_DISPERSION,
    ):
        super().__init__()
        if not block_sizes:
            raise ValueError("block_sizes cannot be empty")
        self.block_sizes = [int(b) for b in block_sizes]
        self.use_stratified = bool(use_stratified)
        self.use_poisson = bool(use_poisson)
        self.max_dispersion = float(max_dispersion)

        # Config files often deserialize dictionary keys as strings.
        self.quantiles: Dict[int, Tuple[float, float]] = {}
        for k, q in (quantiles or {}).items():
            q50, q90 = float(q[0]), float(q[1])
            if q50 > q90:
                raise ValueError(f"Invalid quantiles for block {k}: q50={q50} > q90={q90}")
            self.quantiles[int(k)] = (q50, q90)

        if not self.use_poisson:
            self.raw_dispersions = nn.ParameterDict({
                str(b): nn.Parameter(torch.tensor(_inv_softplus(10.0), dtype=torch.float32))
                for b in self.block_sizes
            })

    def get_dispersion(self, block_size: int) -> torch.Tensor:
        if self.use_poisson:
            raise RuntimeError("Dispersion is undefined in Poisson mode")
        raw = self.raw_dispersions[str(block_size)]
        return (F.softplus(raw) + 1e-4).clamp_max(self.max_dispersion)

    def init_dispersion_from_stats(self, block_size: int, mean: float, var: float) -> None:
        """Method-of-moments initialization with a finite Poisson-limit cap."""
        if self.use_poisson:
            return
        mean, var = float(mean), float(var)
        if mean < 0 or var < 0 or not math.isfinite(mean) or not math.isfinite(var):
            raise ValueError(f"Invalid mean/var: mean={mean}, var={var}")

        if var > mean and (var - mean) > 1e-8:
            r0 = (mean * mean) / (var - mean)
        else:
            # NB approaches Poisson as r -> infinity. A finite cap is enough numerically.
            r0 = self.max_dispersion
        r0 = min(max(r0, 1e-3), self.max_dispersion)

        raw_val = _inv_softplus(max(r0 - 1e-4, 1e-8))
        with torch.no_grad():
            self.raw_dispersions[str(block_size)].fill_(raw_val)

    def forward(
        self,
        d_map: torch.Tensor,
        gt_block_counts: Dict[int, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        loss_dict: Dict[str, torch.Tensor] = {}
        scale_losses = []

        for b in self.block_sizes:
            if b not in gt_block_counts:
                raise KeyError(f"Missing GT block counts for scale {b}")

            pred_mu = sum_pool(d_map, input_block_size=b, output_stride=4)
            if pred_mu.ndim == 4 and pred_mu.shape[1] == 1:
                pred_mu = pred_mu.squeeze(1)

            gt_y = gt_block_counts[b]
            if gt_y.ndim == 4 and gt_y.shape[1] == 1:
                gt_y = gt_y.squeeze(1)
            gt_y = gt_y.to(device=pred_mu.device, dtype=torch.float32)

            if gt_y.shape != pred_mu.shape:
                raise ValueError(
                    f"Scale {b}: GT shape {tuple(gt_y.shape)} != prediction shape {tuple(pred_mu.shape)}"
                )

            if self.use_poisson:
                pointwise_loss = poisson_nll(gt_y, pred_mu)
            else:
                r_b = self.get_dispersion(b)
                pointwise_loss = nb_nll(gt_y, pred_mu, r_b)
                loss_dict[f"dispersion_{b}"] = r_b.detach()

            if self.use_stratified and b in self.quantiles:
                q50, q90 = self.quantiles[b]
                masks = [
                    gt_y == 0,
                    (gt_y > 0) & (gt_y <= q50),
                    (gt_y > q50) & (gt_y <= q90),
                    gt_y > q90,
                ]
                group_losses = [pointwise_loss[m].mean() for m in masks if m.any()]
                scale_loss = torch.stack(group_losses).mean() if group_losses else pointwise_loss.mean()
            else:
                scale_loss = pointwise_loss.mean()

            if not torch.isfinite(scale_loss):
                raise FloatingPointError(f"Non-finite NB/Poisson loss at block scale {b}")
            scale_losses.append(scale_loss)
            loss_dict[f"hnb_scale_{b}"] = scale_loss.detach()

        total_loss = torch.stack(scale_losses).mean() if scale_losses else d_map.new_zeros(())
        return total_loss, loss_dict
