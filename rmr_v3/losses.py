from __future__ import annotations

from dataclasses import dataclass
import math

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
        return (mu_count * 0.0).sum()

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

    # RMR-v9: control which density map is supervised by the Dirichlet-Multinomial allocation loss.
    # "y0" (default): supervise the initial pre-solver density map (backward compatible, v7/v8 default).
    # "y":  supervise the final post-solver density map.
    #       Aligns Flat-DM16 with the terminal spatial output, closing the supervision gap between
    #       the allocation loss and the solver's reconciled measure (spec Section 7.2).
    dm_target: str = "y0"
    dm_strict: bool = True

    def __post_init__(self) -> None:
        if self.use_hierarchical_dm and not self.use_multiscale_dm:
            self.use_multiscale_dm = True
        if self.dm_target not in ("y", "y0"):
            raise ValueError(f"dm_target must be 'y' or 'y0', got '{self.dm_target}'")
        if self.count_loss_mode not in ("nb", "log1p", "l1"):
            raise ValueError(f"count_loss_mode must be 'nb', 'log1p', or 'l1', got '{self.count_loss_mode}'")
        if self.cell_loss_mode not in ("balanced", "mass_weighted"):
            raise ValueError(f"cell_loss_mode must be 'balanced' or 'mass_weighted', got '{self.cell_loss_mode}'")
        if self.allocation_loss_type not in ("flat_dm16", "bayesian", "ot_sinkhorn"):
            raise ValueError(f"allocation_loss_type must be 'flat_dm16', 'bayesian', or 'ot_sinkhorn', got '{self.allocation_loss_type}'")


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


def mass_weighted_cell_loss(
    y: torch.Tensor,
    target: torch.Tensor,
    beta: float = 1.0,
    eps: float = 1e-3,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Mass-weighted cell allocation loss (RMR-v8 Stage 2).

    Computes per-pixel smooth-L1 loss weighted by ground truth density mass.
    To prevent background collapse (where empty background regions receive zero
    penalty and false positives explode), every pixel receives a baseline weight of
    1.0, and pixels containing crowd mass receive an additive boost proportional to
    their density share:
        p(i) = target(i) / (sum(target) + eps)   # relative crowd mass distribution
        raw_weight(i) = 1.0 + alpha * (H * W) * p(i)
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

    Returns:
        Scalar mass-weighted cell loss.
    """
    if target.ndim == 3:
        target = target.unsqueeze(1)
    if y.ndim == 3:
        y = y.unsqueeze(1)

    y_f = y.float()
    t_f = target.float()

    # Spatial mass per image: [B, 1, 1, 1]
    total_mass = t_f.sum(dim=(-2, -1), keepdim=True)

    # Normalized mass distribution p in [0, 1]
    p = torch.where(total_mass > float(eps), t_f / total_mass.clamp_min(float(eps)), torch.zeros_like(t_f))

    # Baseline 1.0 + mass boost
    hw = float(t_f.shape[-2] * t_f.shape[-1])
    raw_weight = 1.0 + float(alpha) * hw * p

    # Normalize so mean weight across spatial dimensions is 1.0
    norm_factor = raw_weight.mean(dim=(-2, -1), keepdim=True).clamp_min(float(eps))
    weights = raw_weight / norm_factor

    # Per-pixel smooth-L1
    per_pixel = F.smooth_l1_loss(y_f, t_f, beta=float(beta), reduction="none")

    return (weights * per_pixel).mean()



def compute_rmr_v3_losses(
    outputs: dict,
    target_y: torch.Tensor,
    cfg: RMRv3LossConfig | None = None,
    points: list[torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    if cfg is None:
        cfg = RMRv3LossConfig()

    y = outputs["y"].float()
    y0 = outputs["y0"].float()
    target_float = target_y.float()

    regions: RegionSet = outputs["regions"]

    mean_region = outputs["b_region"].float()
    dispersion_region = outputs["region_dispersion"].float()

    target_region = regional_sum(
        target_float,
        regions.boxes,
        out_dtype=torch.float32,
    )

    losses: dict[str, torch.Tensor] = {}

    losses["count"] = count_magnitude_loss(
        y,
        target_float,
        mode=cfg.count_loss_mode,
        dispersion=cfg.count_nb_dispersion,
    )

    # ── Allocation loss target selection (RMR-v9: dm_target="y" supervises post-solver output) ──
    # "y0" (default, backward compatible): supervise the initial density map before SIRT.
    dm_input = y if cfg.dm_target == "y" else y0

    dm_components: dict[int, torch.Tensor] = {}
    if cfg.allocation_loss_type == "bayesian":
        loss_allocation = bayesian_loss(
            dm_input,
            points,
            sigma=cfg.bayesian_sigma,
            background_ratio=cfg.bayesian_background_ratio,
            stride=4,
        )
    elif cfg.allocation_loss_type == "ot_sinkhorn":
        loss_allocation = sinkhorn_ot_loss(
            dm_input,
            points,
            reg=cfg.ot_reg,
            num_iters=cfg.ot_num_iters,
            stride=4,
        )
    elif cfg.use_multiscale_dm or cfg.use_hierarchical_dm:
        loss_allocation, dm_components = multiscale_dm_loss(
            dm_input,
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
        loss_allocation = flat_dm16_loss(
            dm_input,
            target_float,
            kappa=cfg.kappa_flat16,
            normalize_by_count=cfg.normalize_flat_dm16,
            strict=cfg.dm_strict,
        )
        dm_components[16] = loss_allocation

    losses["allocation"] = loss_allocation
    losses["flat_dm16"] = loss_allocation  # backward compatibility alias
    for bs, val in dm_components.items():
        losses[f"dm_{bs}"] = val

    # ── Stage 2: Cell allocation loss (mode-selectable) ───────────────────────
    if cfg.cell_loss_mode == "mass_weighted":
        # Weight per-pixel smooth-L1 by GT density mass.
        # Dense crowd pixels receive proportionally more gradient than background,
        # fixing the ~200x dilution effect in balanced smooth-L1.
        losses["cell"] = mass_weighted_cell_loss(
            y,
            target_float,
            beta=cfg.cell_beta,
            eps=cfg.cell_mass_weight_eps,
            alpha=float(cfg.cell_mass_weight_alpha),
        )
    else:
        # "balanced": uniform smooth-L1 (v7 default — backward compatible)
        losses["cell"] = balanced_smooth_l1(
            y,
            target_float,
            beta=cfg.cell_beta,
        )

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

    # ── RMR-v7 Hurdle losses (opt-in; skipped when lambda=0 or logit absent) ──
    hurdle_logit = outputs.get("hurdle_logit", None)
    if hurdle_logit is not None:
        if cfg.lambda_hurdle > 0.0:
            losses["hurdle_bce"] = hurdle_focal_bce_loss(
                hurdle_logit.float(),
                target_region,
            )
            losses["total"] = losses["total"] + cfg.lambda_hurdle * losses["hurdle_bce"]
        else:
            losses["hurdle_bce"] = torch.tensor(0.0, device=y.device)

        if cfg.lambda_trunc_nb > 0.0:
            losses["trunc_nb"] = truncated_nb_nll_loss(
                mean_region,
                dispersion_region,
                target_region,
            )
            losses["total"] = losses["total"] + cfg.lambda_trunc_nb * losses["trunc_nb"]
        else:
            losses["trunc_nb"] = torch.tensor(0.0, device=y.device)
    else:
        losses["hurdle_bce"] = torch.tensor(0.0, device=y.device)
        losses["trunc_nb"] = torch.tensor(0.0, device=y.device)

    return losses

