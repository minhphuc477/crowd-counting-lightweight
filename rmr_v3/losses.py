from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Callable

import torch
import torch.nn.functional as F

from rmr_core.operators import (
    RegionSet,
    regional_sum,
)
from rmr_core.losses import (
    balanced_smooth_l1,
    count_magnitude_loss,
    flat_dm16_loss,
    multiscale_dm_loss,
    negative_binomial_nll_mean_dispersion,
)


def hurdle_focal_bce_loss(
    pi_logit: torch.Tensor,
    target_region: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal BCE loss for Hurdle head occupancy prediction.

    Trains π_R = P(region is occupied) from binary labels derived from GT counts.
    Focal weighting down-weights easy background regions (typical imbalanced setting).

    Args:
        pi_logit:      [B,1,M] raw occupancy logit z_π from hurdle_head_layer.
        target_region: [B,1,M] GT regional count sums (from regional_sum on target_y).
        gamma:         Focal exponent (default 2.0).

    Returns:
        Scalar focal BCE loss.
    """
    if pi_logit.ndim == 2:
        pi_logit = pi_logit.unsqueeze(1)
    if target_region.ndim == 2:
        target_region = target_region.unsqueeze(1)

    # Binary occupancy label: 1 if any count in region, 0 if empty background.
    y_bin = (target_region > 0.5).float()

    # Standard BCE with logits
    bce = F.binary_cross_entropy_with_logits(
        pi_logit.float(),
        y_bin,
        reduction="none",
    )

    # Focal weight: (1 - p_t)^gamma
    with torch.no_grad():
        p_t = torch.where(y_bin > 0.5, torch.sigmoid(pi_logit.float()), 1.0 - torch.sigmoid(pi_logit.float()))
        focal_weight = (1.0 - p_t.clamp(min=1e-6, max=1.0 - 1e-6)).pow(float(gamma))

    return (focal_weight * bce).mean()


def truncated_nb_nll_loss(
    mu_count: torch.Tensor,
    dispersion: torch.Tensor,
    target_region: torch.Tensor,
) -> torch.Tensor:
    """Truncated NB NLL computed only on occupied regions (target >= 1).

    In the Hurdle model, the NB component only models the count distribution
    conditional on the region being occupied (y >= 1). Computing NLL only on
    occupied regions avoids gradient conflict from zero-inflation.

    Args:
        mu_count:      [B,1,M] predicted NB mean count.
        dispersion:    [B,1,M] predicted NB dispersion (r).
        target_region: [B,1,M] GT regional count sums.

    Returns:
        Scalar truncated NB NLL, averaged over occupied regions.
        Returns 0.0 with autograd connectivity if no occupied region exists in batch.
    """
    if target_region.ndim == 2:
        target_region = target_region.unsqueeze(1)
    if mu_count.ndim == 2:
        mu_count = mu_count.unsqueeze(1)
    if dispersion.ndim == 2:
        dispersion = dispersion.unsqueeze(1)

    occ_mask = (target_region > 0.5)  # [B,1,M]
    if not occ_mask.any():
        return (mu_count * 0.0 + dispersion * 0.0).sum()

    per_region_nll = negative_binomial_nll_mean_dispersion(
        target_region,
        mu_count,
        dispersion=dispersion,
        reduction="none",
    )
    # Average only over occupied regions
    return per_region_nll[occ_mask].mean()


def scale_balanced_regional_nb_nll(
    target_region: torch.Tensor,
    mean_region: torch.Tensor,
    dispersion_region: torch.Tensor,
    regions: RegionSet,
) -> torch.Tensor:
    """Average proper NB NLL within scale, then average scales."""

    if target_region.shape != mean_region.shape:
        raise ValueError(
            f"target/mean mismatch: "
            f"{target_region.shape} vs {mean_region.shape}"
        )

    if dispersion_region.shape != mean_region.shape:
        raise ValueError(
            f"dispersion/mean mismatch: "
            f"{dispersion_region.shape} vs {mean_region.shape}"
        )

    per_region = negative_binomial_nll_mean_dispersion(
        target_region,
        mean_region,
        dispersion=dispersion_region,
        reduction="none",
    )

    losses = []

    for sid in torch.unique(regions.scale_id):
        if int(sid.item()) < 0:
            continue

        mask = regions.scale_id == sid

        if bool(mask.any()):
            losses.append(
                per_region[..., mask].mean()
            )

    if not losses:
        raise RuntimeError(
            "No valid regional scales for NB loss"
        )

    return torch.stack(losses).mean()


def bayesian_loss(
    prob_y0: torch.Tensor,
    points_list: list[torch.Tensor] | None,
    sigma: float = 8.0,
    background_ratio: float = 0.1,
    stride: int = 4,
) -> torch.Tensor:
    """Bayesian Loss for point supervision (Ma et al. ICCV 2019).
    
    Computes continuous spatial allocation loss without artificial block boundaries.
    """
    b, _, h, w = prob_y0.shape
    device = prob_y0.device
    dtype = prob_y0.dtype

    # Create grid of center coordinates in image pixels
    y_coords = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * float(stride)
    x_coords = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * float(stride)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    grid_xy = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # [M, 2]

    losses = []
    tau = float(background_ratio)

    for i in range(b):
        y_flat = prob_y0[i, 0].flatten().float()  # [M]
        total_pred = y_flat.sum()

        pts = points_list[i] if points_list is not None and i < len(points_list) else None
        if pts is None or pts.numel() == 0:
            losses.append(total_pred)
            continue

        pts = pts.to(device=device, dtype=torch.float32)
        n = pts.shape[0]

        # Pairwise squared distances: [N, M]
        diff = pts.unsqueeze(1) - grid_xy.unsqueeze(0)  # [N, M, 2]
        dist_sq = (diff ** 2).sum(dim=-1)  # [N, M]

        # Gaussian likelihood: [N, M]
        p_y_given_x = torch.exp(-dist_sq / (2.0 * sigma * sigma))

        # Denominator with background likelihood: [1, M]
        denom = p_y_given_x.sum(dim=0, keepdim=True) + tau

        # Posterior probability: [N, M] and [1, M]
        post_person = p_y_given_x / denom.clamp_min(1e-8)
        post_bg = tau / denom.clamp_min(1e-8)

        # Predicted count assigned to each person and background
        c_hat_person = torch.matmul(post_person, y_flat)  # [N]
        c_hat_bg = torch.matmul(post_bg, y_flat).squeeze(0)  # scalar

        person_err = torch.abs(c_hat_person - 1.0).sum()
        bg_err = c_hat_bg

        sample_loss = (person_err + bg_err) / float(max(n, 1))
        losses.append(sample_loss)

    return torch.stack(losses).mean().to(dtype)


def sinkhorn_ot_loss(
    prob_y0: torch.Tensor,
    points_list: list[torch.Tensor] | None,
    reg: float = 10.0,
    num_iters: int = 20,
    stride: int = 4,
) -> torch.Tensor:
    """Pure-PyTorch Log-Domain Sinkhorn Optimal Transport Loss.
    
    Measures Wasserstein transportation distance from normalized Y0 to ground truth points.
    Numerically stabilized with bounded dual variables and clamped log-domain scaling.
    """
    b, _, h, w = prob_y0.shape
    device = prob_y0.device
    dtype = prob_y0.dtype

    y_coords = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * float(stride)
    x_coords = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * float(stride)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    grid_xy = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)

    diag = math.sqrt((h * stride) ** 2 + (w * stride) ** 2)
    grid_norm = grid_xy / max(diag, 1.0)

    losses = []
    eps = max(1.0 / float(reg), 1e-3)

    for i in range(b):
        y_flat = prob_y0[i, 0].flatten().float()
        total_pred = y_flat.sum()

        pts = points_list[i] if points_list is not None and i < len(points_list) else None
        if pts is None or pts.numel() == 0:
            losses.append(total_pred)
            continue

        pts = pts.to(device=device, dtype=torch.float32)
        n = pts.shape[0]
        pts_norm = pts / max(diag, 1.0)

        # Normalized mass distributions
        p = (y_flat + 1e-5) / (total_pred + 1e-5 * y_flat.numel())
        q = torch.full((n,), 1.0 / float(n), device=device, dtype=torch.float32)

        # Cost matrix normalized in [0, 1]
        diff = pts_norm.unsqueeze(1) - grid_norm.unsqueeze(0)  # [N, M, 2]
        c = (diff ** 2).sum(dim=-1).clamp(0.0, 1.0)
        c_scaled = c / eps

        log_p = torch.log(p.clamp_min(1e-8))
        log_q = torch.log(q.clamp_min(1e-8))
        u = torch.zeros_like(q)
        v = torch.zeros_like(p)

        # Log-domain stabilized Sinkhorn iterations in scaled dual coordinates
        for _ in range(num_iters):
            u = log_q - torch.logsumexp(-c_scaled + v.unsqueeze(0), dim=1)
            v = log_p - torch.logsumexp(-c_scaled + u.unsqueeze(1), dim=0)

        log_gamma = u.unsqueeze(1) + v.unsqueeze(0) - c_scaled
        gamma = torch.exp(log_gamma)
        ot_cost = (gamma * c).sum()

        count_err = torch.abs(total_pred - float(n)) / float(max(n, 1))
        sample_loss = ot_cost + 0.1 * count_err
        losses.append(sample_loss)

    return torch.stack(losses).mean().to(dtype)


@dataclass
class RMRv3LossConfig:
    lambda_count: float = 1.0
    lambda_flat_dm16: float = 1.0
    lambda_cell: float = 0.25
    lambda_region_nb: float = 0.20

    # RMR-v7 Hurdle head loss weights
    lambda_hurdle: float = 0.0   # weight for Focal BCE occupancy loss (0 = disabled)
    lambda_trunc_nb: float = 0.0  # weight for Truncated NB NLL on occupied regions (0 = disabled)

    allocation_loss_type: str = "flat_dm16"  # "flat_dm16" | "bayesian" | "ot_sinkhorn"
    bayesian_sigma: float = 8.0
    bayesian_background_ratio: float = 0.1
    ot_reg: float = 10.0
    ot_num_iters: int = 20

    use_multiscale_dm: bool = False
    use_hierarchical_dm: bool = False  # backward compatibility alias
    dm_block_sizes_px: tuple[int, ...] = (16, 32, 64)
    dm_weights: tuple[float, ...] = (0.50, 0.30, 0.20)
    dm_kappas: tuple[float, ...] = (20.0, 20.0, 20.0)

    count_loss_mode: str = "nb"
    count_nb_dispersion: float = 50.0

    kappa_flat16: float = 20.0
    normalize_flat_dm16: bool = True

    cell_beta: float = 1.0

    # RMR-v8 Stage 2: cell loss mode
    # "balanced":      balanced smooth-L1 (v7 default — backward compatible)
    # "mass_weighted": weight per-pixel loss by GT density mass, emphasizing
    #                  dense crowd regions and reducing 200x dilution effect.
    cell_loss_mode: str = "balanced"
    cell_mass_weight_eps: float = 1e-3  # avoid division-by-zero in mass weighting
    cell_mass_weight_alpha: float = 1.0  # mass boost scaling factor

    # RMR-v9/v10: control which density map is supervised by the Dirichlet-Multinomial allocation loss.
    # "y0" (default): supervise the initial pre-solver density map (backward compatible, v7/v8 default).
    # "y":  supervise the final post-solver density map.
    # "dual": supervise both 0.5 * L(y) + 0.5 * L(y0), anchoring y0 to GT while optimizing y end-to-end.
    dm_target: str = "y0"
    dm_strict: bool = True

    # ── RMR-v11 additions ────────────────────────────────────────────────────
    cell_mass_weight_gamma: float = 1.0  # exponent on normalized crowd mass distribution
    lambda_curvature: float = 0.0        # weight for curvature power loss (0 = disabled)
    lambda_hard_bg: float = 0.0          # weight for top-k hard background loss (0 = disabled)
    hard_bg_ratio: float = 0.10          # fraction of worst false alarm background pixels to penalize
    lambda_fg_gate: float = 0.0          # weight for foreground gate BCE loss (0 = disabled)

    # ── RMR-v12 Density-Gated Curvature additions ────────────────────────────
    curvature_gate_threshold: float = 0.0  # density threshold to activate curvature loss (0 = disabled / full image)
    curvature_gate_kernel: int = 5         # kernel size for local density pooling (covers 20x20 px at stride 4)
    curvature_gate_mode: str = "none"      # "none" | "hard" | "soft"
    curvature_gate_scale: float = 0.02     # temperature for soft sigmoid transition

    # ── RMR-v13 Physical Scale Alignment Loss additions ─────────────────────
    lambda_scale_align: float = 0.0      # weight for physical scale alignment loss (0 = disabled)
    scale_align_tau_dense: float = 0.12  # local density threshold for fine scale (16x16)
    scale_align_tau_sparse: float = 0.03 # local density threshold for coarse scale (64x64)
    scale_align_kernel: int = 5          # kernel size for local density estimation
    scale_align_mask_bg: bool = True     # RMR-v14: mask out background pixels from scale alignment KL loss

    # ── RMR-v20 High-Density Sample-Level Loss Scaling (0 params) ────────────
    # Amplifies gradients on high-density crowd crops (>100 count, upper 30% of distribution)
    # so the regional and fine heads receive adequate gradient signal without drowning in
    # the massive volume of low-density crops.
    density_loss_scaling: bool = False
    dense_loss_thresh: float = 100.0
    dense_loss_norm: float = 150.0
    dense_loss_alpha: float = 1.0
    dense_loss_max_boost: float = 2.0

    # ── RMR-v21 Elementwise Sample-Level Loss Scaling (0 params) ─────────────
    # True sample-level importance weighting without batch cross-talk leakage:
    # weights each crop's loss L_i by w_i = 1.0 + dense_boost_i BEFORE taking
    # the batch mean, completely isolating background crops from stadium crops.
    elementwise_dense_scaling: bool = False

    def __post_init__(self) -> None:
        if self.use_hierarchical_dm and not self.use_multiscale_dm:
            self.use_multiscale_dm = True
        if self.dm_target not in ("y", "y0", "dual"):
            raise ValueError(f"dm_target must be 'y', 'y0', or 'dual', got '{self.dm_target}'")
        if self.count_loss_mode not in ("nb", "log1p", "l1"):
            raise ValueError(f"count_loss_mode must be 'nb', 'log1p', or 'l1', got '{self.count_loss_mode}'")
        if self.cell_loss_mode not in ("balanced", "mass_weighted"):
            raise ValueError(f"cell_loss_mode must be 'balanced' or 'mass_weighted', got '{self.cell_loss_mode}'")
        if self.allocation_loss_type not in ("flat_dm16", "bayesian", "ot_sinkhorn"):
            raise ValueError(f"allocation_loss_type must be 'flat_dm16', 'bayesian', or 'ot_sinkhorn', got '{self.allocation_loss_type}'")
        if self.cell_mass_weight_gamma <= 0.0:
            raise ValueError(f"cell_mass_weight_gamma must be strictly positive, got {self.cell_mass_weight_gamma}")
        if self.curvature_gate_mode not in ("none", "hard", "soft"):
            raise ValueError(f"curvature_gate_mode must be 'none', 'hard', or 'soft', got '{self.curvature_gate_mode}'")
        if self.lambda_curvature < 0.0:
            raise ValueError(f"lambda_curvature must be non-negative, got {self.lambda_curvature}")
        if self.lambda_hard_bg < 0.0:
            raise ValueError(f"lambda_hard_bg must be non-negative, got {self.lambda_hard_bg}")
        if not (0.0 < self.hard_bg_ratio <= 1.0):
            raise ValueError(f"hard_bg_ratio must be in (0.0, 1.0], got {self.hard_bg_ratio}")
        if self.lambda_fg_gate < 0.0:
            raise ValueError(f"lambda_fg_gate must be non-negative, got {self.lambda_fg_gate}")
        if self.lambda_scale_align < 0.0:
            raise ValueError(f"lambda_scale_align must be non-negative, got {self.lambda_scale_align}")
        if self.scale_align_tau_dense <= self.scale_align_tau_sparse:
            raise ValueError(
                f"scale_align_tau_dense ({self.scale_align_tau_dense}) must be > scale_align_tau_sparse ({self.scale_align_tau_sparse})"
            )
        if self.scale_align_kernel <= 0 or self.scale_align_kernel % 2 == 0:
            raise ValueError(
                f"scale_align_kernel must be a positive odd integer, got {self.scale_align_kernel}"
            )


    @classmethod
    def from_dict(cls, d: dict | None) -> "RMRv3LossConfig":
        if not d:
            return cls()
        kwargs: dict = {}
        use_multi = bool(d.get("use_multiscale_dm", d.get("use_hierarchical_dm", False)))
        kwargs["use_multiscale_dm"] = use_multi
        kwargs["use_hierarchical_dm"] = use_multi

        for k, v in d.items():
            if k in ("use_multiscale_dm", "use_hierarchical_dm"):
                continue
            if hasattr(cls, k) and not k.startswith("_"):
                if isinstance(getattr(cls, k), (int, float, bool, str, tuple)):
                    if isinstance(getattr(cls, k), tuple) and isinstance(v, (list, tuple)):
                        kwargs[k] = tuple(v)
                    elif isinstance(getattr(cls, k), bool):
                        kwargs[k] = bool(v)
                    elif isinstance(getattr(cls, k), int):
                        kwargs[k] = int(v)
                    elif isinstance(getattr(cls, k), float):
                        kwargs[k] = float(v)
                    elif isinstance(getattr(cls, k), str):
                        kwargs[k] = str(v)
                    else:
                        kwargs[k] = v
                else:
                    kwargs[k] = v
        return cls(**kwargs)


def curvature_power_loss(
    y: torch.Tensor,
    target: torch.Tensor,
    eps: float = 0.01,
    threshold: float = 0.0,
    kernel_size: int = 5,
    mode: str = "none",
    smooth_scale: float = 0.02,
) -> torch.Tensor:
    """Curvature-preserving square-root power loss for high-density crowds (RMR-v11/v12).

    Computes L_curv = mean(|sqrt(y + eps) - sqrt(y_gt + eps)|^2).
    In the square-root domain, the gradient for under-counted dense clusters is amplified
    by up to (1 + 1/sqrt(eps)) = 11x, counteracting gradient starvation while eps=0.01
    strictly bounds the gradient magnitude within [-9.0, +1.0] for optimizer stability.

    In RMR-v12, density-gated mode ('hard' or 'soft') restricts curvature loss to high-density
    regions where local crowd density exceeds `threshold`. This eliminates mass over-inflation
    on isolated heads in moderate crowds while preserving the anti-saturation gradient on dense clumps.
    """
    if target.ndim == 3:
        target = target.unsqueeze(1)
    if y.ndim == 3:
        y = y.unsqueeze(1)
    work_dtype = y.dtype if y.dtype in (torch.float32, torch.float64) else torch.float32
    y_f = y.to(dtype=work_dtype).clamp_min(0.0)
    t_f = target.to(dtype=work_dtype).clamp_min(0.0)
    diff = torch.sqrt(y_f + float(eps)) - torch.sqrt(t_f + float(eps))
    diff_sq = diff.square()

    if mode == "none" or float(threshold) <= 0.0:
        return torch.mean(diff_sq)

    # Compute local density via average pooling (covers (kernel_size * stride)^2 pixels)
    pad = int(kernel_size) // 2
    local_density = F.avg_pool2d(t_f, kernel_size=int(kernel_size), stride=1, padding=pad)

    if mode == "hard":
        gate = (local_density >= float(threshold)).float()
    elif mode == "soft":
        gate = torch.sigmoid((local_density - float(threshold)) / float(smooth_scale))
    else:
        raise ValueError(f"Unknown curvature gate mode: '{mode}'. Expected 'none', 'hard', or 'soft'.")

    gate_sum = gate.sum()
    return (gate * diff_sq).sum() / gate_sum.clamp_min(1.0)


def topk_hard_background_loss(
    y: torch.Tensor,
    target: torch.Tensor,
    ratio: float = 0.05,
    bg_threshold: float = 1e-5,
) -> torch.Tensor:
    """Top-K Hard Negative Background Mining Loss (RMR-v11).

    Extracts background pixels where target <= bg_threshold, and applies
    a quadratic penalty to the top `ratio` fraction of highest predicted false alarms.
    Exerts sharp quadratic repulsion on textured pavement, trees, and architectural facades.
    """
    if target.ndim == 3:
        target = target.unsqueeze(1)
    if y.ndim == 3:
        y = y.unsqueeze(1)
    y_f = y.float()
    t_f = target.float()
    bg_mask = (t_f <= float(bg_threshold))
    if not bg_mask.any():
        return (y_f * 0.0).sum()

    bg_preds = torch.clamp_min(y_f[bg_mask], 0.0)
    num_bg = bg_preds.numel()
    k = min(num_bg, max(1, int(float(ratio) * num_bg)))

    topk_vals, _ = torch.topk(bg_preds, k=k, largest=True, sorted=False)
    return torch.mean(topk_vals.square())


def mass_weighted_cell_loss(
    y: torch.Tensor,
    target: torch.Tensor,
    beta: float = 1.0,
    eps: float = 1e-3,
    alpha: float = 1.0,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Mass-weighted cell allocation loss (RMR-v8 Stage 2 / RMR-v11).

    Computes per-pixel smooth-L1 loss weighted by ground truth density mass.
    To prevent background collapse (where empty background regions receive zero
    penalty and false positives explode), every pixel receives a baseline weight of
    1.0, and pixels containing crowd mass receive an additive boost proportional to
    their density share:
        p(i) = target(i) / (sum(target) + eps)   # relative crowd mass distribution
        p_gamma(i) = (p(i)^gamma) / sum(p^gamma) # optional power shaping (gamma=1.25 in v11)
        raw_weight(i) = 1.0 + alpha * (H * W) * p_gamma(i)
        weight(i) = raw_weight(i) / mean(raw_weight)

    Properties:
    1. If target == 0 everywhere (empty background crop), p(i) == 0, raw_weight(i) == 1.0,
       weight(i) == 1.0. The loss reduces identically to standard smooth-L1.
    2. Background pixels (target == 0) always receive non-zero loss weight >= 1.0 / (1.0 + alpha),
       actively penalizing false positive hallucinations.
    3. Crowd pixels (target > 0) receive elevated loss weight up to (1.0 + alpha * H * W / N_crowd),
       ending the ~200x dilution in dense scenes.
    4. Normalized mean weight is identically 1.0, preserving overall gradient magnitude.

    Args:
        y:      [B, 1, H, W] predicted density map (float32).
        target: [B, 1, H, W] GT density map (float32).
        beta:   smooth-L1 threshold (default 1.0).
        eps:    small constant for numerical stability.
        alpha:  relative crowd boost weight (default 1.0).
        gamma:  power exponent on mass distribution (default 1.0).

    Returns:
        Scalar mass-weighted cell loss.
    """
    if target.ndim == 3:
        target = target.unsqueeze(1)
    if y.ndim == 3:
        y = y.unsqueeze(1)

    work_dtype = y.dtype if y.dtype in (torch.float32, torch.float64) else torch.float32
    y_f = y.to(dtype=work_dtype)
    t_f = target.to(dtype=work_dtype)

    # Spatial mass per image: [B, 1, 1, 1]
    total_mass = t_f.sum(dim=(-2, -1), keepdim=True)

    # Normalized mass distribution p in [0, 1]
    p = torch.where(total_mass > float(eps), t_f / total_mass.clamp_min(float(eps)), torch.zeros_like(t_f))

    if abs(float(gamma) - 1.0) > 1e-5:
        p_pow = p.pow(float(gamma))
        sum_p_pow = p_pow.sum(dim=(-2, -1), keepdim=True).clamp_min(float(eps))
        p = p_pow / sum_p_pow

    # Baseline 1.0 + mass boost
    hw = float(t_f.shape[-2] * t_f.shape[-1])
    raw_weight = 1.0 + float(alpha) * hw * p

    # Normalize so mean weight across spatial dimensions is 1.0
    norm_factor = raw_weight.mean(dim=(-2, -1), keepdim=True).clamp_min(float(eps))
    weights = raw_weight / norm_factor

    # Per-pixel smooth-L1
    per_pixel = F.smooth_l1_loss(y_f, t_f, beta=float(beta), reduction="none")

    return (weights * per_pixel).mean()


def physical_scale_alignment_loss(
    scale_weights: torch.Tensor,
    target_y: torch.Tensor,
    tau_dense: float = 0.12,
    tau_sparse: float = 0.03,
    kernel_size: int = 5,
    eps: float = 1e-6,
    mask_background: bool = True,
) -> torch.Tensor:
    """Physical Scale Alignment Loss (RMR-v13 / RMR-v14).

    Supervises the scale router's spatial scale probabilities pi(x, y) with a
    continuous physics-based prior derived from local crowd density:
    - High local density (heads close together) -> fine scale (Scale 0, e.g. 32x32 px)
    - Medium local density                      -> intermediate scale (Scale 1, e.g. 64x64 px)
    - Low local density                         -> coarse scale (Scale 2, e.g. 128x128 px)

    In RMR-v14: When mask_background=True, the loss is computed strictly on foreground
    regions where local density >= tau_sparse. This prevents empty background pixels
    from being artificially forced into Scale 2, eliminating phantom background count
    accumulations on repetitive textures (e.g. IMG_113 pavement).

    The loss computes the Kullback-Leibler divergence KL(pi* || pi).

    Args:
        scale_weights:   [B, K, H, W] predicted scale probabilities from ScaleRoutingHead.
        target_y:        [B, 1, H, W] or [B, H, W] GT density map.
        tau_dense:       Density threshold for dense crowd (Scale 0).
        tau_sparse:      Density threshold for sparse crowd (Scale 2).
        kernel_size:     Pooling kernel size for local density computation.
        eps:             Epsilon for numerical safety in log.
        mask_background: If True, only penalize pixels with local density >= tau_sparse.

    Returns:
        Scalar non-negative KL divergence loss.
    """
    if target_y.ndim == 3:
        target_y = target_y.unsqueeze(1)

    b, k, h, w = scale_weights.shape
    if k < 2:
        return (scale_weights * 0.0).sum()

    work_dtype = scale_weights.dtype if scale_weights.dtype in (torch.float32, torch.float64) else torch.float32
    t_f = target_y.to(dtype=work_dtype).clamp_min(0.0)
    pad = int(kernel_size) // 2
    local_density = F.avg_pool2d(t_f, kernel_size=int(kernel_size), stride=1, padding=pad)

    delta_tau = max(float(tau_dense) - float(tau_sparse), 1e-6)
    s = torch.clamp((local_density - float(tau_sparse)) / delta_tau, 0.0, 1.0)

    # General piece-wise linear barycentric target on Delta^{K-1}
    # Density s in [0, 1] maps from coarsest scale (K-1) at s=0 to finest scale (0) at s=1
    u = s * float(k - 1)  # [B, 1, H, W] in [0, K-1]
    target_pi_list = []
    for scale_idx in range(k):
        center = float(k - 1 - scale_idx)
        weight_k = torch.clamp(1.0 - torch.abs(u - center), min=0.0)
        target_pi_list.append(weight_k)
    target_pi = torch.cat(target_pi_list, dim=1).to(dtype=work_dtype)  # [B, K, H, W]

    pred_pi = scale_weights.to(dtype=work_dtype).clamp(min=float(eps), max=1.0)
    target_log_target = torch.where(
        target_pi > 1e-6,
        target_pi * torch.log(target_pi.clamp_min(1e-6)),
        torch.zeros_like(target_pi),
    )
    target_log_pred = target_pi * torch.log(pred_pi)
    kl_per_pixel = (target_log_target - target_log_pred).sum(dim=1)

    if mask_background:
        fg_mask = (local_density >= float(tau_sparse)).float().squeeze(1)  # [B, H, W]
        fg_sum = fg_mask.sum().clamp_min(1.0)
        return (kl_per_pixel * fg_mask).sum() / fg_sum

    return kl_per_pixel.mean()


class TargetSupervisionRouter:
    """Routes target supervision across terminal measure y, initial carrier y0, or symmetric dual.

    Eliminates duplicated 3-way branching across count, cell, curvature, and hard background losses.
    """

    def __init__(self, target_mode: str = "y") -> None:
        if target_mode not in ("dual", "y0", "y"):
            raise ValueError(f"Unknown target supervision mode: '{target_mode}'. Expected 'dual', 'y0', or 'y'.")
        self.mode = target_mode

    def dispatch(
        self,
        fn: Any,
        y: torch.Tensor,
        y0: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Dispatch loss computation according to target supervision mode."""
        if self.mode == "dual":
            l_y = fn(y, *args, **kwargs)
            l_y0 = fn(y0, *args, **kwargs)
            return 0.5 * l_y + 0.5 * l_y0, {"y": l_y, "y0": l_y0}
        elif self.mode == "y0":
            l_y0 = fn(y0, *args, **kwargs)
            return l_y0, {"y0": l_y0}
        else:
            l_y = fn(y, *args, **kwargs)
            return l_y, {"y": l_y}


def _compute_elementwise_dense_scaling(
    outputs: dict[str, Any],
    target_y: torch.Tensor,
    cfg: RMRv3LossConfig,
    points: list[torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Elementwise sample-level importance weighting without batch cross-talk leakage."""
    b_sz = target_y.shape[0]
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
        l_i = compute_rmr_v3_losses(out_i, tgt_i, cfg_single, points=pts_i)
        sample_losses.append(l_i)

    aggregated = {}
    for k in sample_losses[0].keys():
        tensors = [sl[k] for sl in sample_losses]
        stacked = torch.stack(tensors)
        if k == "total":
            # True elementwise weighted mean: (1 / B) * sum(w_i * L_i)
            aggregated[k] = (sample_weights * stacked).mean()
        else:
            aggregated[k] = stacked.mean()

    aggregated["dense_loss_scale"] = sample_weights.mean()
    return aggregated


def _compute_core_losses(
    target_float: torch.Tensor,
    target_region: torch.Tensor,
    y: torch.Tensor,
    y0: torch.Tensor,
    regions: RegionSet,
    mean_region: torch.Tensor,
    dispersion_region: torch.Tensor,
    cfg: RMRv3LossConfig,
    points: list[torch.Tensor] | None,
    router: TargetSupervisionRouter,
) -> dict[str, torch.Tensor]:
    """Compute primary losses: Count loss, Allocation loss, Cell loss, and Regional NB loss."""
    losses: dict[str, torch.Tensor] = {}

    def _compute_count_loss(density_map: torch.Tensor) -> torch.Tensor:
        return count_magnitude_loss(
            density_map,
            target_float,
            mode=cfg.count_loss_mode,
            dispersion=cfg.count_nb_dispersion,
        )

    def _compute_cell_loss(density_map: torch.Tensor) -> torch.Tensor:
        if cfg.cell_loss_mode == "mass_weighted":
            return mass_weighted_cell_loss(
                density_map,
                target_float,
                beta=cfg.cell_beta,
                eps=cfg.cell_mass_weight_eps,
                alpha=float(cfg.cell_mass_weight_alpha),
                gamma=float(cfg.cell_mass_weight_gamma),
            )
        return balanced_smooth_l1(
            density_map,
            target_float,
            beta=cfg.cell_beta,
        )

    # Count loss supervision
    loss_count, aux_count = router.dispatch(_compute_count_loss, y, y0)
    losses["count"] = loss_count
    if "y" in aux_count and "y0" in aux_count:
        losses["count_y"] = aux_count["y"]
        losses["count_y0"] = aux_count["y0"]

    # Cell loss supervision
    loss_cell, aux_cell = router.dispatch(_compute_cell_loss, y, y0)
    losses["cell"] = loss_cell
    if "y" in aux_cell and "y0" in aux_cell:
        losses["cell_y"] = aux_cell["y"]
        losses["cell_y0"] = aux_cell["y0"]

    # Allocation loss supervision (DM16 / Bayesian / OT)
    def _compute_single_allocation(inp: torch.Tensor) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        comps: dict[int, torch.Tensor] = {}
        if cfg.allocation_loss_type == "bayesian":
            loss_val = bayesian_loss(
                inp,
                points,
                sigma=cfg.bayesian_sigma,
                background_ratio=cfg.bayesian_background_ratio,
                stride=4,
            )
        elif cfg.allocation_loss_type == "ot_sinkhorn":
            loss_val = sinkhorn_ot_loss(
                inp,
                points,
                reg=cfg.ot_reg,
                num_iters=cfg.ot_num_iters,
                stride=4,
            )
        elif cfg.use_multiscale_dm or cfg.use_hierarchical_dm:
            loss_val, comps = multiscale_dm_loss(
                inp,
                target_float,
                block_sizes_px=tuple(int(x) for x in cfg.dm_block_sizes_px),
                weights=tuple(float(x) for x in cfg.dm_weights),
                kappas=tuple(float(x) for x in cfg.dm_kappas),
                stride=4,
                normalize_by_count=cfg.normalize_flat_dm16,
                strict=cfg.dm_strict,
                return_components=True,
            )
        else:
            loss_val = flat_dm16_loss(
                inp,
                target_float,
                kappa=cfg.kappa_flat16,
                normalize_by_count=cfg.normalize_flat_dm16,
                strict=cfg.dm_strict,
            )
            comps[16] = loss_val
        return loss_val, comps

    dm_components: dict[int, torch.Tensor] = {}
    if cfg.dm_target == "dual":
        loss_alloc_y, dm_components = _compute_single_allocation(y)
        loss_alloc_y0, _ = _compute_single_allocation(y0)
        loss_allocation = 0.5 * loss_alloc_y + 0.5 * loss_alloc_y0
        losses["allocation_y"] = loss_alloc_y
        losses["allocation_y0"] = loss_alloc_y0
    elif cfg.dm_target == "y":
        loss_allocation, dm_components = _compute_single_allocation(y)
    else:  # "y0"
        loss_allocation, dm_components = _compute_single_allocation(y0)

    losses["allocation"] = loss_allocation
    losses["flat_dm16"] = loss_allocation  # backward compatibility alias
    for bs, val in dm_components.items():
        losses[f"dm_{bs}"] = val

    losses["region_nb"] = scale_balanced_regional_nb_nll(
        target_region,
        mean_region,
        dispersion_region,
        regions,
    )

    losses["total"] = (
        cfg.lambda_count * losses["count"]
        + cfg.lambda_flat_dm16 * loss_allocation
        + cfg.lambda_cell * losses["cell"]
        + cfg.lambda_region_nb * losses["region_nb"]
    )
    return losses


def _compute_auxiliary_losses(
    losses: dict[str, torch.Tensor],
    outputs: dict[str, Any],
    target_float: torch.Tensor,
    target_region: torch.Tensor,
    mean_region: torch.Tensor,
    dispersion_region: torch.Tensor,
    y: torch.Tensor,
    y0: torch.Tensor,
    cfg: RMRv3LossConfig,
    zero_val: torch.Tensor,
    router: TargetSupervisionRouter,
) -> dict[str, torch.Tensor]:
    """Compute auxiliary losses: Curvature, Hard Background, FG Gate, Hurdle, and Scale Alignment."""
    # Curvature Power Loss (RMR-v11/v12)
    if cfg.lambda_curvature > 0.0:
        def _compute_curv(dmap: torch.Tensor) -> torch.Tensor:
            return curvature_power_loss(
                dmap,
                target_float,
                threshold=cfg.curvature_gate_threshold,
                kernel_size=cfg.curvature_gate_kernel,
                mode=cfg.curvature_gate_mode,
                smooth_scale=cfg.curvature_gate_scale,
            )

        loss_curv, _ = router.dispatch(_compute_curv, y, y0)
        losses["curvature"] = loss_curv
        losses["total"] = losses["total"] + cfg.lambda_curvature * loss_curv
    else:
        losses["curvature"] = zero_val

    # Top-K Hard Background Mining Loss (RMR-v11)
    if cfg.lambda_hard_bg > 0.0:
        def _compute_hard_bg(dmap: torch.Tensor) -> torch.Tensor:
            return topk_hard_background_loss(dmap, target_float, ratio=cfg.hard_bg_ratio)

        loss_hard_bg, _ = router.dispatch(_compute_hard_bg, y, y0)
        losses["hard_bg"] = loss_hard_bg
        losses["total"] = losses["total"] + cfg.lambda_hard_bg * loss_hard_bg
    else:
        losses["hard_bg"] = zero_val

    # Decoupled Foreground Gating Loss (RMR-v11)
    fg_logit = outputs.get("fg_logit", None)
    if fg_logit is not None and cfg.lambda_fg_gate > 0.0:
        t_bin = (target_float > 0.0).float()
        if t_bin.ndim == 3:
            t_bin = t_bin.unsqueeze(1)
        t_dilated = F.max_pool2d(t_bin, kernel_size=3, stride=1, padding=1)
        fg_bce = F.binary_cross_entropy_with_logits(fg_logit.float(), t_dilated)
        losses["fg_bce"] = fg_bce
        losses["total"] = losses["total"] + cfg.lambda_fg_gate * fg_bce
    else:
        losses["fg_bce"] = zero_val

    # Hurdle losses (RMR-v7)
    hurdle_logit = outputs.get("hurdle_logit", None)
    if hurdle_logit is not None:
        if cfg.lambda_hurdle > 0.0:
            losses["hurdle_bce"] = hurdle_focal_bce_loss(
                hurdle_logit.float(),
                target_region,
            )
            losses["total"] = losses["total"] + cfg.lambda_hurdle * losses["hurdle_bce"]
        else:
            losses["hurdle_bce"] = zero_val

        if cfg.lambda_trunc_nb > 0.0:
            losses["trunc_nb"] = truncated_nb_nll_loss(
                mean_region,
                dispersion_region,
                target_region,
            )
            losses["total"] = losses["total"] + cfg.lambda_trunc_nb * losses["trunc_nb"]
        else:
            losses["trunc_nb"] = zero_val
    else:
        losses["hurdle_bce"] = zero_val
        losses["trunc_nb"] = zero_val

    # Physical Scale Alignment Loss (RMR-v13/v14/v19)
    scale_weights = outputs.get("pi_scale", outputs.get("scale_weights", None))
    if scale_weights is not None and cfg.lambda_scale_align > 0.0:
        losses["scale_align"] = physical_scale_alignment_loss(
            scale_weights=scale_weights,
            target_y=target_float,
            tau_dense=cfg.scale_align_tau_dense,
            tau_sparse=cfg.scale_align_tau_sparse,
            kernel_size=cfg.scale_align_kernel,
            mask_background=cfg.scale_align_mask_bg,
        )
        losses["total"] = losses["total"] + cfg.lambda_scale_align * losses["scale_align"]
    else:
        losses["scale_align"] = zero_val

    # High-Density Sample-Level Loss Rescaling (RMR-v20)
    if cfg.density_loss_scaling or cfg.elementwise_dense_scaling:
        total_gt = target_float.sum(dim=(-2, -1))
        dense_boost = float(cfg.dense_loss_alpha) * torch.clamp(
            (total_gt - float(cfg.dense_loss_thresh)) / float(cfg.dense_loss_norm),
            min=0.0,
            max=float(cfg.dense_loss_max_boost),
        )
        sample_scale = (1.0 + dense_boost).detach().mean()
        losses["total"] = losses["total"] * sample_scale
        losses["dense_loss_scale"] = sample_scale

    return losses


def compute_rmr_v3_losses(
    outputs: dict[str, Any],
    target_y: torch.Tensor,
    cfg: RMRv3LossConfig | None = None,
    points: list[torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute composite multi-task loss for RMRv3 model predictions."""
    if cfg is None:
        cfg = RMRv3LossConfig()

    if target_y.ndim == 3:
        target_y = target_y.unsqueeze(1)

    # Elementwise High-Density Sample-Level Loss Scaling (RMR-v21)
    if cfg.elementwise_dense_scaling and target_y.shape[0] > 1:
        return _compute_elementwise_dense_scaling(outputs, target_y, cfg, points=points)

    y = outputs["y"].float()
    y0 = outputs["y0"].float()
    target_float = target_y.float()
    zero_val = y.sum() * 0.0

    regions: RegionSet = outputs["regions"]
    mean_region = outputs["b_region"].float()
    dispersion_region = outputs["region_dispersion"].float()

    target_region = regional_sum(
        target_float,
        regions.boxes,
        out_dtype=torch.float32,
    )

    router = TargetSupervisionRouter(cfg.dm_target)
    losses = _compute_core_losses(
        target_float=target_float,
        target_region=target_region,
        y=y,
        y0=y0,
        regions=regions,
        mean_region=mean_region,
        dispersion_region=dispersion_region,
        cfg=cfg,
        points=points,
        router=router,
    )

    return _compute_auxiliary_losses(
        losses=losses,
        outputs=outputs,
        target_float=target_float,
        target_region=target_region,
        mean_region=mean_region,
        dispersion_region=dispersion_region,
        y=y,
        y0=y0,
        cfg=cfg,
        zero_val=zero_val,
        router=router,
    )

