"""Elementwise sample-level importance weighting without batch cross-talk leakage."""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable
import torch

from .config import RMRv3LossConfig


def compute_elementwise_dense_scaling(
    outputs: dict[str, Any],
    target_y: torch.Tensor,
    cfg: RMRv3LossConfig,
    compute_losses_fn: Callable[..., dict[str, torch.Tensor]],
    points: list[torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Elementwise sample-level importance weighting without batch cross-talk leakage.

    Guarantees strict sample isolation:
    Each sample in a batch is evaluated independently with cfg_single, and its
    loss is weighted by its individual ground-truth count density boost:
        boost_i = alpha * clamp((N_i - thresh) / norm, min=0, max=max_boost)
        weight_i = 1.0 + boost_i
    Sample 0 receives identically zero gradient with respect to Sample 1's loss.
    """
    b_sz = target_y.shape[0]
    if b_sz == 0:
        return {}
    cfg_single = replace(cfg, elementwise_dense_scaling=False, density_loss_scaling=False)

    total_gt = target_y.float().sum(dim=(-2, -1)).view(-1)  # [B]
    dense_boost = float(cfg.dense_loss_alpha) * torch.clamp(
        (total_gt - float(cfg.dense_loss_thresh)) / float(cfg.dense_loss_norm),
        min=0.0,
        max=float(cfg.dense_loss_max_boost),
    )
    sample_weights = (1.0 + dense_boost).detach()  # [B]

    sample_losses = []
    for i in range(b_sz):
        out_i = {}
        for k, v in outputs.items():
            if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == b_sz:
                out_i[k] = v[i : i + 1]
            elif isinstance(v, list):
                out_i[k] = [
                    item[i : i + 1]
                    if isinstance(item, torch.Tensor) and item.ndim > 0 and item.shape[0] == b_sz
                    else item
                    for item in v
                ]
            else:
                out_i[k] = v
        tgt_i = target_y[i : i + 1]
        pts_i = [points[i]] if points is not None and i < len(points) else None
        l_i = compute_losses_fn(out_i, tgt_i, cfg_single, points=pts_i)
        sample_losses.append(l_i)

    aggregated = {}
    for k in sample_losses[0].keys():
        tensors = [sl[k] for sl in sample_losses]
        stacked = torch.stack(tensors)
        aggregated[k] = (sample_weights * stacked).mean() if k == "total" else stacked.mean()

    aggregated["dense_loss_scale"] = sample_weights.mean()
    return aggregated
