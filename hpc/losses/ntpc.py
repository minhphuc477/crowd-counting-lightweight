"""Losses for Neural Tree-Polya Crowd Counting (NTPC).

The proposed core is ``N -> 64 -> 32 -> 16``.  R5 adds only a
dense-parent ``16 -> 8`` auxiliary term; stride-4 supervision is a depth-study
variant and is never silently enabled by the main method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .negative_binomial import negative_binomial_nll_mean_dispersion, poisson_nll


def block_sum(x: torch.Tensor, k: int) -> torch.Tensor:
    """Non-overlapping exact sum pooling via reshape."""
    had_channel = x.ndim == 4
    if not had_channel:
        x = x.unsqueeze(1)
    if x.ndim != 4:
        raise ValueError(f"Expected 3D/4D tensor, got {tuple(x.shape)}")
    batch, channels, height, width = x.shape
    if height % k or width % k:
        raise ValueError(f"Shape ({height}, {width}) not divisible by factor {k}")
    out = x.reshape(batch, channels, height // k, k, width // k, k).sum((3, 5))
    return out if had_channel else out.squeeze(1)


def sum_pool_mass_pyramid(
    mass: torch.Tensor,
    block_sizes: Tuple[int, ...] = (4, 8, 16, 32, 64),
    stride: int = 4,
) -> Dict[int, torch.Tensor]:
    """Build exactly the requested count maps from one positive mass map."""
    if mass.ndim == 3:
        mass = mass.unsqueeze(1)
    if mass.ndim != 4 or mass.shape[1] != 1:
        raise ValueError(f"Expected mass shape (B,1,H,W), got {tuple(mass.shape)}")
    result: Dict[int, torch.Tensor] = {}
    for block_size in block_sizes:
        if block_size % stride:
            raise ValueError(f"Block size {block_size} must be divisible by stride {stride}")
        factor = block_size // stride
        if factor < 1:
            raise ValueError(f"Block size {block_size} is smaller than stride {stride}")
        pooled = mass if factor == 1 else block_sum(mass, factor)
        result[int(block_size)] = pooled.squeeze(1)
    return result


mass_pyramid = sum_pool_mass_pyramid


def group_2x2_flat(x: torch.Tensor) -> torch.Tensor:
    """Group a child grid into ``[TL, TR, BL, BR]`` for every parent."""
    if x.ndim == 4 and x.shape[1] == 1:
        x = x.squeeze(1)
    if x.ndim != 3:
        raise ValueError(f"Expected (B,H,W), got {tuple(x.shape)}")
    batch, height, width = x.shape
    if height % 2 or width % 2:
        raise ValueError("Grid height and width must be even")
    children = torch.stack(
        (x[:, 0::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 0::2], x[:, 1::2, 1::2]),
        dim=-1,
    )
    return children.reshape(batch, -1, 4)


group_four_children = group_2x2_flat


def probs_from_positive_mass(mass: torch.Tensor, tiny: float = 1e-8) -> torch.Tensor:
    mass = mass.float().clamp_min(tiny)
    return mass / mass.sum(dim=-1, keepdim=True).clamp_min(tiny)


mass_to_prob = probs_from_positive_mass


def alpha_from_mass(mass: torch.Tensor, kappa: float, tiny: float = 1e-8) -> torch.Tensor:
    return float(kappa) * probs_from_positive_mass(mass, tiny=tiny)


def dm_nll_none(y: torch.Tensor, alpha: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Dirichlet-Multinomial NLL; empty parents contribute exactly zero."""
    y = y.float()
    alpha = alpha.float().clamp_min(eps)
    if torch.any(y < 0):
        raise ValueError("Dirichlet-Multinomial targets must be non-negative")
    n = y.sum(dim=-1)
    alpha0 = alpha.sum(dim=-1)
    log_prob = (
        torch.lgamma(n + 1.0)
        - torch.lgamma(y + 1.0).sum(dim=-1)
        + torch.lgamma(alpha0)
        - torch.lgamma(n + alpha0)
        + (torch.lgamma(y + alpha) - torch.lgamma(alpha)).sum(dim=-1)
    )
    return torch.where(n == 0, torch.zeros_like(n), -log_prob)


def dm_from_mass(y: torch.Tensor, child_mass: torch.Tensor, kappa: float) -> torch.Tensor:
    return dm_nll_none(y, alpha_from_mass(child_mass, kappa))


def multinomial_nll_none(y: torch.Tensor, pi: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    y = y.float()
    pi = pi.float().clamp_min(eps)
    pi = pi / pi.sum(dim=-1, keepdim=True).clamp_min(eps)  # renormalize: clamp can push sum > 1
    n = y.sum(dim=-1)
    log_prob = (
        torch.lgamma(n + 1.0)
        - torch.lgamma(y + 1.0).sum(dim=-1)
        + (y * torch.log(pi)).sum(dim=-1)
    )
    return torch.where(n == 0, torch.zeros_like(n), -log_prob)


def tree_level_dm_nll(
    child_gt_map: torch.Tensor,
    child_pred_map: torch.Tensor,
    kappa: float,
) -> torch.Tensor:
    """Joint node NLL at one tree level, returned per image."""
    return dm_from_mass(
        group_2x2_flat(child_gt_map),
        group_2x2_flat(child_pred_map),
        kappa,
    ).sum(dim=1)


tree_level_nll_per_image = tree_level_dm_nll


def root_to_64_nll(y64: torch.Tensor, mu64: torch.Tensor, kappa: float) -> torch.Tensor:
    return dm_from_mass(y64.float().flatten(1), mu64.float().flatten(1), kappa)


root_grid_nll_per_image = root_to_64_nll


_MODE_ALIASES = {
    "r4_dtm_tree": "r4_dtm_tree16",
    "r4_full_ntpc": "r5_full_ntpc",
}
_VALID_MODES = {
    "r0_exact",
    "r1_deterministic",
    "r2_flat_dm",
    "r3_multinomial_tree",
    "r4_dtm_tree16",
    "r4_dtm_tree8",
    "r4_dtm_tree4",
    "r5_full_ntpc",
}


@dataclass
class NTPCConfig:
    mode: str = "r4_dtm_tree16"
    root_loss: str = "nb"  # nb | poisson | l1
    root_dispersion: float = 50.0
    kappa_root64: float = 20.0
    kappa_64_32: float = 20.0
    kappa_32_16: float = 20.0
    kappa_16_8: float = 20.0
    kappa_8_4: float = 20.0
    kappa_flat16: float = 20.0
    dense_threshold_16: float = 2.0
    w_root_nb: float = 1.0
    w_root64: float = 1.0
    w_64_32: float = 1.0
    w_32_16: float = 1.0
    w_16_8: float = 1.0
    w_8_4: float = 1.0
    w_flat_16: float = 1.0
    w_exact_regression: float = 1.0
    w_deterministic_alloc: float = 1.0
    eps: float = 1e-8

    def __post_init__(self) -> None:
        self.mode = _MODE_ALIASES.get(self.mode, self.mode)
        if self.mode not in _VALID_MODES:
            raise ValueError(f"Unsupported NTPC mode: {self.mode}")
        if self.root_loss not in {"nb", "poisson", "l1"}:
            raise ValueError(f"Unsupported root_loss: {self.root_loss}")
        if self.root_dispersion <= 0 or self.eps <= 0:
            raise ValueError("root_dispersion and eps must be positive")
        for name in (
            "kappa_root64", "kappa_64_32", "kappa_32_16",
            "kappa_16_8", "kappa_8_4", "kappa_flat16",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


class NTPCLoss(nn.Module):
    """Exact R0-R5 objectives with fail-fast target validation."""

    def __init__(self, cfg: NTPCConfig | None = None):
        super().__init__()
        self.cfg = cfg or NTPCConfig()

    def _required_blocks(self) -> Tuple[int, ...]:
        if self.cfg.mode == "r0_exact":
            return (16, 32, 64)   # regional L1 at these three levels
        if self.cfg.mode == "r2_flat_dm":
            return (16,)
        if self.cfg.mode == "r5_full_ntpc" or self.cfg.mode == "r4_dtm_tree8":
            return (8, 16, 32, 64)
        if self.cfg.mode == "r4_dtm_tree4":
            return (4, 8, 16, 32, 64)
        return (16, 32, 64)

    @property
    def is_exact_joint_nll(self) -> bool:
        """True iff the loss configuration is the exact factorized joint NLL."""
        return (
            self.cfg.mode == "r4_dtm_tree16"
            and self.cfg.root_loss == "nb"
            and self.cfg.w_root_nb == 1.0
            and self.cfg.w_root64 == 1.0
            and self.cfg.w_64_32 == 1.0
            and self.cfg.w_32_16 == 1.0
        )

    def _validate_targets(self, targets: Dict[int | str, torch.Tensor]) -> None:
        """Fail-fast validation: key existence, integer values, non-negative, count conservation, and parent-child tree consistency."""
        missing = [key for key in ("N", *self._required_blocks()) if key not in targets]
        if missing:
            raise KeyError(f"Mode {self.cfg.mode} requires targets {missing}")

        N = targets["N"].float().reshape(-1)
        if not torch.isfinite(N).all():
            raise ValueError("N contains NaN/Inf values")
        if (N < 0).any():
            raise ValueError("N must be non-negative")
        if not torch.allclose(N, N.round(), atol=1e-4, rtol=0):
            raise ValueError("N must contain integer counts")

        for key in self._required_blocks():
            t = targets[key].float()
            if not torch.allclose(t, t.round(), atol=1e-4, rtol=0):
                raise ValueError(
                    f"Target level {key} contains non-integer values; "
                    "DTM requires exact integer counts."
                )
            if (t < 0).any():
                raise ValueError(f"Target level {key} contains negative values.")
            s = t.flatten(1).sum(1)
            if not torch.allclose(s, N, atol=1e-3, rtol=0):
                raise ValueError(
                    f"Target level {key} violates count conservation: "
                    f"sum={s.tolist()} != N={N.tolist()}"
                )

        def _assert_parent_child(parent: torch.Tensor, child: torch.Tensor, name: str) -> None:
            grouped = group_2x2_flat(child.float()).sum(dim=-1)
            expected = parent.float().flatten(1)
            if grouped.shape != expected.shape:
                raise ValueError(
                    f"{name}: parent/child shape mismatch: {expected.shape} vs {grouped.shape}"
                )
            if not torch.allclose(grouped, expected, atol=1e-3, rtol=0):
                diff = (grouped - expected).abs().max().item()
                raise ValueError(
                    f"{name}: child counts do not reconstruct parent counts; max diff = {diff:.4f}"
                )

        required = set(self._required_blocks())
        if {32, 64} <= required:
            _assert_parent_child(targets[64], targets[32], "64->32")
        if {16, 32} <= required:
            _assert_parent_child(targets[32], targets[16], "32->16")
        if 8 in required and 16 in required:
            _assert_parent_child(targets[16], targets[8], "16->8")
        if 4 in required and 8 in required:
            _assert_parent_child(targets[8], targets[4], "8->4")

    def _root_nll(self, target: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        if self.cfg.root_loss == "nb":
            return negative_binomial_nll_mean_dispersion(
                target, mean, self.cfg.root_dispersion, eps=self.cfg.eps, reduction="none"
            )
        if self.cfg.root_loss == "poisson":
            return poisson_nll(target, mean, eps=self.cfg.eps)
        return (mean.float() - target.float()).abs()

    @staticmethod
    def _zero(mass: torch.Tensor) -> torch.Tensor:
        return mass.sum() * 0.0

    @staticmethod
    def _deterministic_split(
        parent_gt: torch.Tensor,
        child_gt: torch.Tensor,
        child_mass: torch.Tensor,
        zero: torch.Tensor,
    ) -> torch.Tensor:
        parent = parent_gt.float().flatten(1)
        target_children = group_2x2_flat(child_gt.float())
        pi = probs_from_positive_mass(group_2x2_flat(child_mass.float()))
        expected = parent.unsqueeze(-1) * pi
        valid = parent > 0
        if not valid.any():
            return zero
        per_parent = (expected - target_children).abs().sum(dim=-1)
        return (per_parent[valid] / parent[valid].clamp_min(1.0)).mean()

    def forward(
        self,
        mass: torch.Tensor,
        target_pyramid: Dict[int | str, torch.Tensor],
        return_components: bool = False,
    ):
        self._validate_targets(target_pyramid)
        mass = mass.float()
        needed = tuple(sorted(set(self._required_blocks()) | {4, 8, 16, 32, 64}))
        pred = sum_pool_mass_pyramid(mass, block_sizes=needed, stride=4)
        pred_n = mass.flatten(1).sum(dim=1)
        target_n = target_pyramid["N"].to(mass.device, torch.float32).reshape(-1)
        if pred_n.shape != target_n.shape:
            raise ValueError(f"Root target shape {tuple(target_n.shape)} != {tuple(pred_n.shape)}")
        zero = self._zero(mass)
        names = (
            "root_magnitude", "root_nb", "root_to_64", "64_to_32", "32_to_16",
            "16_to_8", "16_to_8_dense", "8_to_4", "flat_16",
            "multinomial_tree", "deterministic_alloc", "exact_regression",
        )
        components = {name: zero for name in names}

        if self.cfg.mode == "r0_exact":
            # Root-NB: matched to R1–R5 so ablation is fair.
            root = self.cfg.w_root_nb * self._root_nll(target_n, pred_n).mean()
            components["root_magnitude"] = root
            if self.cfg.root_loss == "nb":
                components["root_nb"] = root
            # Regional L1: mean over three spatial levels (scale-invariant magnitude).
            # L_R0 = L_Root-NB + w_exact_reg * mean(L1_64 + L1_32 + L1_16) / 3
            regional = (
                F.l1_loss(pred[64], target_pyramid[64].to(mass.device).float())
                + F.l1_loss(pred[32], target_pyramid[32].to(mass.device).float())
                + F.l1_loss(pred[16], target_pyramid[16].to(mass.device).float())
            ) / 3.0
            components["exact_regression"] = self.cfg.w_exact_regression * regional
            total = root + components["exact_regression"]
            return self._finish(total, components, return_components)

        root = self.cfg.w_root_nb * self._root_nll(target_n, pred_n).mean()
        components["root_magnitude"] = root
        if self.cfg.root_loss == "nb":
            components["root_nb"] = root

        if self.cfg.mode == "r1_deterministic":
            root_parent = target_n.reshape(-1, 1)
            y64_flat = target_pyramid[64].to(mass.device).float().flatten(1)
            pi64 = probs_from_positive_mass(pred[64].flatten(1))
            valid_root = target_n > 0
            if valid_root.any():
                root_alloc = (
                    (root_parent * pi64 - y64_flat).abs().sum(dim=-1)[valid_root]
                    / target_n[valid_root].clamp_min(1.0)
                ).mean()
            else:
                root_alloc = zero
            allocation = root_alloc
            allocation = allocation + self._deterministic_split(
                target_pyramid[64].to(mass.device), target_pyramid[32].to(mass.device), pred[32], zero
            )
            allocation = allocation + self._deterministic_split(
                target_pyramid[32].to(mass.device), target_pyramid[16].to(mass.device), pred[16], zero
            )
            components["deterministic_alloc"] = self.cfg.w_deterministic_alloc * allocation
            return self._finish(root + components["deterministic_alloc"], components, return_components)

        if self.cfg.mode == "r2_flat_dm":
            flat = dm_from_mass(
                target_pyramid[16].to(mass.device).float().flatten(1),
                pred[16].flatten(1),
                self.cfg.kappa_flat16,
            ).mean()
            components["flat_16"] = self.cfg.w_flat_16 * flat
            return self._finish(root + components["flat_16"], components, return_components)

        if self.cfg.mode == "r3_multinomial_tree":
            root64 = multinomial_nll_none(
                target_pyramid[64].to(mass.device).float().flatten(1),
                probs_from_positive_mass(pred[64].flatten(1)),
            )
            split64 = multinomial_nll_none(
                group_2x2_flat(target_pyramid[32].to(mass.device).float()),
                probs_from_positive_mass(group_2x2_flat(pred[32])),
            ).sum(1)
            split32 = multinomial_nll_none(
                group_2x2_flat(target_pyramid[16].to(mass.device).float()),
                probs_from_positive_mass(group_2x2_flat(pred[16])),
            ).sum(1)
            components["multinomial_tree"] = (root64 + split64 + split32).mean()
            return self._finish(root + components["multinomial_tree"], components, return_components)

        root64 = self.cfg.w_root64 * root_to_64_nll(
            target_pyramid[64].to(mass.device), pred[64], self.cfg.kappa_root64
        ).mean()
        split64 = self.cfg.w_64_32 * tree_level_dm_nll(
            target_pyramid[32].to(mass.device), pred[32], self.cfg.kappa_64_32
        ).mean()
        split32 = self.cfg.w_32_16 * tree_level_dm_nll(
            target_pyramid[16].to(mass.device), pred[16], self.cfg.kappa_32_16
        ).mean()
        components["root_to_64"] = root64
        components["64_to_32"] = split64
        components["32_to_16"] = split32
        total = root + root64 + split64 + split32

        if self.cfg.mode == "r5_full_ntpc":
            child_gt = group_2x2_flat(target_pyramid[8].to(mass.device).float())
            child_mass = group_2x2_flat(pred[8])
            parent_gt = target_pyramid[16].to(mass.device).float().flatten(1)
            dense = parent_gt >= self.cfg.dense_threshold_16
            node_nll = dm_from_mass(child_gt, child_mass, self.cfg.kappa_16_8)
            dense_loss = node_nll[dense].mean() if dense.any() else zero
            components["16_to_8_dense"] = self.cfg.w_16_8 * dense_loss
            total = total + components["16_to_8_dense"]
        elif self.cfg.mode in {"r4_dtm_tree8", "r4_dtm_tree4"}:
            fine = self.cfg.w_16_8 * tree_level_dm_nll(
                target_pyramid[8].to(mass.device), pred[8], self.cfg.kappa_16_8
            ).mean()
            components["16_to_8"] = fine
            total = total + fine

        if self.cfg.mode == "r4_dtm_tree4":
            finest = self.cfg.w_8_4 * tree_level_dm_nll(
                target_pyramid[4].to(mass.device), pred[4], self.cfg.kappa_8_4
            ).mean()
            components["8_to_4"] = finest
            total = total + finest
        return self._finish(total, components, return_components)

    @staticmethod
    def _finish(total: torch.Tensor, components: Dict[str, torch.Tensor], return_components: bool):
        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite NTPC loss")
        logs = {name: value.detach() for name, value in components.items()}
        logs["total"] = total.detach()
        if return_components:
            return total, logs, components
        return total, logs


FullNTPCLoss = NTPCLoss
