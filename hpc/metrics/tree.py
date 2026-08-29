"""Diagnostics for hierarchical NTPC count allocation."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from hpc.losses.ntpc import (
    NTPCConfig,
    dm_from_mass,
    group_2x2_flat,
    probs_from_positive_mass,
    sum_pool_mass_pyramid,
)


_PARENT_GROUPS = (
    ("0", lambda n: n == 0),
    ("1", lambda n: n == 1),
    ("2_4", lambda n: (n >= 2) & (n <= 4)),
    ("5_9", lambda n: (n >= 5) & (n <= 9)),
    ("ge10", lambda n: n >= 10),
)


def _level_raw(
    prefix: str,
    parent_count: torch.Tensor,
    child_gt: torch.Tensor,
    child_mass: torch.Tensor,
    kappa: float,
    eps: float,
) -> dict[str, float]:
    parent = parent_count.float().reshape(-1)
    child_arity = int(child_gt.shape[-1])
    if child_arity <= 0 or int(child_mass.shape[-1]) != child_arity:
        raise ValueError(f"{prefix}: invalid child arity")
    gt = child_gt.float().reshape(-1, child_arity)
    predicted = child_mass.float().reshape(-1, child_arity)
    if not (len(parent) == len(gt) == len(predicted)):
        raise ValueError(f"{prefix}: inconsistent parent/child diagnostic shapes")
    node_nll = dm_from_mass(gt, predicted, kappa, eps=eps)
    pi = probs_from_positive_mass(predicted, tiny=eps)
    empirical = gt / parent.unsqueeze(-1).clamp_min(1.0)
    probability_l1 = (pi - empirical).abs().sum(dim=-1)
    active = parent > 0
    out = {
        f"{prefix}_active_sum": float(active.sum()),
        f"{prefix}_parent_sum": float(parent.numel()),
        f"{prefix}_nll_sum": float(node_nll[active].sum()),
    }
    for label, selector in _PARENT_GROUPS:
        mask = selector(parent)
        out[f"{prefix}_parent_{label}_count"] = float(mask.sum())
        out[f"{prefix}_parent_{label}_nll_sum"] = float(node_nll[mask].sum())
        out[f"{prefix}_parent_{label}_prob_l1_sum"] = float(probability_l1[mask].sum())
    return out


@torch.no_grad()
def tree_allocation_raw_diagnostics(
    mass: torch.Tensor,
    targets: Mapping[int | str, torch.Tensor],
    cfg: NTPCConfig,
    levels: tuple[str, ...] | None = None,
) -> dict[str, float]:
    """Return additive raw statistics suitable for epoch-wide aggregation."""
    available = tuple(level for level in (4, 8, 16, 32, 64) if level in targets)
    predicted = sum_pool_mass_pyramid(mass.float(), block_sizes=available)
    batch = int(mass.shape[0])
    out: dict[str, float] = {"tree_images": float(batch)}
    if 64 in targets and (levels is None or "root_64" in levels):
        out.update(_level_raw(
            "tree_root_64",
            targets["N"].float(),
            targets[64].float().flatten(1).reshape(batch, 1, -1),
            predicted[64].flatten(1).reshape(batch, 1, -1),
            cfg.kappa_root64,
            cfg.eps,
        ))
    for parent_level, child_level, kappa in (
        (64, 32, cfg.kappa_64_32),
        (32, 16, cfg.kappa_32_16),
        (16, 8, cfg.kappa_16_8),
        (8, 4, cfg.kappa_8_4),
    ):
        if parent_level not in targets or child_level not in targets:
            continue
        level_name = f"{parent_level}_{child_level}"
        if levels is not None and level_name not in levels:
            continue
        out.update(_level_raw(
            f"tree_{level_name}",
            targets[parent_level].float().flatten(1),
            group_2x2_flat(targets[child_level].float()),
            group_2x2_flat(predicted[child_level]),
            kappa,
            cfg.eps,
        ))
    return out


def finalize_tree_diagnostics(raw: Mapping[str, float]) -> dict[str, float]:
    """Convert additive tree statistics into interpretable epoch metrics."""
    images = max(float(raw.get("tree_images", 0.0)), 1.0)
    result: dict[str, float] = {}
    prefixes = sorted({key.removesuffix("_active_sum") for key in raw if key.endswith("_active_sum")})
    for prefix in prefixes:
        active = float(raw[f"{prefix}_active_sum"])
        parents = float(raw[f"{prefix}_parent_sum"])
        result[f"{prefix}_active_parents_per_image"] = active / images
        result[f"{prefix}_zero_parent_fraction"] = 1.0 - active / max(parents, 1.0)
        result[f"{prefix}_nll_per_active_parent"] = float(raw[f"{prefix}_nll_sum"]) / max(active, 1.0)
        for label, _ in _PARENT_GROUPS:
            count = float(raw[f"{prefix}_parent_{label}_count"])
            result[f"{prefix}_parent_{label}_count"] = count
            result[f"{prefix}_parent_{label}_mean_nll"] = float(
                raw[f"{prefix}_parent_{label}_nll_sum"]
            ) / max(count, 1.0)
            result[f"{prefix}_parent_{label}_mean_prob_l1"] = float(
                raw[f"{prefix}_parent_{label}_prob_l1_sum"]
            ) / max(count, 1.0)
    return result
