from __future__ import annotations

from dataclasses import dataclass

import torch

from rmr_core.operators import (
    RegionSet,
    regional_sum,
)
from rmr_v2.losses import (
    balanced_smooth_l1,
    count_magnitude_loss,
    flat_dm16_loss,
    hierarchical_dm_loss,
    multiscale_dm_loss,
    negative_binomial_nll_mean_dispersion,
)


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
    import math
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

        log_p = torch.log(p.clamp_min(1e-8))
        log_q = torch.log(q.clamp_min(1e-8))
        u = torch.zeros_like(q)
        v = torch.zeros_like(p)

        # Log-domain stabilized Sinkhorn iterations
        for _ in range(num_iters):
            u = log_q - torch.logsumexp((-c + v.unsqueeze(0)) / eps, dim=1)
            v = log_p - torch.logsumexp((-c + u.unsqueeze(1)) / eps, dim=0)

        log_gamma = (u.unsqueeze(1) + v.unsqueeze(0) - c) / eps
        gamma = torch.exp(torch.clamp(log_gamma, max=0.0))
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

    def __post_init__(self) -> None:
        if self.use_hierarchical_dm and not self.use_multiscale_dm:
            self.use_multiscale_dm = True


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

    dm_components: dict[int, torch.Tensor] = {}
    if cfg.allocation_loss_type == "bayesian":
        loss_allocation = bayesian_loss(
            y0,
            points,
            sigma=cfg.bayesian_sigma,
            background_ratio=cfg.bayesian_background_ratio,
            stride=4,
        )
    elif cfg.allocation_loss_type == "ot_sinkhorn":
        loss_allocation = sinkhorn_ot_loss(
            y0,
            points,
            reg=cfg.ot_reg,
            num_iters=cfg.ot_num_iters,
            stride=4,
        )
    elif cfg.use_multiscale_dm or cfg.use_hierarchical_dm:
        loss_allocation, dm_components = multiscale_dm_loss(
            y0,
            target_float,
            block_sizes_px=tuple(int(x) for x in cfg.dm_block_sizes_px),
            weights=tuple(float(x) for x in cfg.dm_weights),
            kappas=tuple(float(x) for x in cfg.dm_kappas),
            stride=4,
            normalize_by_count=cfg.normalize_flat_dm16,
            return_components=True,
        )
    else:
        loss_allocation = flat_dm16_loss(
            y0,
            target_float,
            kappa=cfg.kappa_flat16,
            normalize_by_count=cfg.normalize_flat_dm16,
        )
        dm_components[16] = loss_allocation

    losses["allocation"] = loss_allocation
    losses["flat_dm16"] = loss_allocation  # backward compatibility alias
    for bs, val in dm_components.items():
        losses[f"dm_{bs}"] = val

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

    return losses
