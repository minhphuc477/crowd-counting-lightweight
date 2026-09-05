from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F

from .operators import RegionSet, regional_sum


def balanced_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """Equalize empty and non-empty cell contributions.

    This is deliberately a simple shared carrier loss, not a paper contribution.
    """
    per = F.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    pos = target > 0
    neg = ~pos
    terms = []
    if pos.any():
        terms.append(per[pos].mean())
    if neg.any():
        terms.append(per[neg].mean())
    if not terms:
        return per.mean()
    return torch.stack(terms).mean()


_MAX_DISPERSION = 1e4


def negative_binomial_nll_mean_dispersion(
    target: torch.Tensor,
    mean: torch.Tensor,
    dispersion: float | torch.Tensor = 50.0,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    """Negative-Binomial NLL with Var(Y) = mu + mu^2 / r evaluated in float32."""
    y = target.to(device=mean.device, dtype=torch.float32)
    mu = mean.to(dtype=torch.float32).clamp_min(eps)
    r = torch.as_tensor(dispersion, device=mean.device, dtype=torch.float32)

    if torch.any(r <= 0) or torch.any(r > _MAX_DISPERSION) or not torch.isfinite(r).all():
        raise ValueError(
            f"Negative-Binomial dispersion parameter r must be in (0, {_MAX_DISPERSION}], got {dispersion}"
        )
    if torch.any(y < 0):
        raise ValueError("Negative-Binomial targets must be non-negative")

    log_r_plus_mu = torch.log(r + mu)
    nll = -(
        torch.lgamma(y + r)
        - torch.lgamma(r)
        - torch.lgamma(y + 1.0)
        + r * (torch.log(r) - log_r_plus_mu)
        + y * (torch.log(mu) - log_r_plus_mu)
    )
    if reduction == "none":
        return nll
    if reduction == "sum":
        return nll.sum()
    if reduction == "mean":
        return nll.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")


def count_magnitude_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mode: str = "nb",
    dispersion: float = 50.0,
) -> torch.Tensor:
    """Total crop count loss using Negative Binomial NLL or log1p smooth L1."""
    pn = pred.sum(dim=(-2, -1)).view(-1)
    tn = target.sum(dim=(-2, -1)).view(-1)
    if mode == "nb":
        return negative_binomial_nll_mean_dispersion(tn, pn, dispersion=dispersion, reduction="mean")
    elif mode == "log1p":
        return F.smooth_l1_loss(torch.log1p(pn), torch.log1p(tn), reduction="mean", beta=0.2)
    elif mode == "l1":
        return F.l1_loss(pn, tn, reduction="mean")
    raise ValueError(f"Unsupported count loss mode: {mode}")


# Backward-compatible alias
global_count_loss = count_magnitude_loss


def block_sum_2d(x: torch.Tensor, k: int = 4) -> torch.Tensor:
    """Sum non-overlapping k x k blocks via reshape."""
    had_channel = x.ndim == 4
    if not had_channel:
        x = x.unsqueeze(1)
    b, c, h, w = x.shape
    h_trim = (h // k) * k
    w_trim = (w // k) * k
    if h != h_trim or w != w_trim:
        x = x[:, :, :h_trim, :w_trim]
        h, w = h_trim, w_trim
    out = x.reshape(b, c, h // k, k, w // k, k).sum((3, 5))
    return out if had_channel else out.squeeze(1)


def probs_from_positive_mass(mass: torch.Tensor, tiny: float = 1e-8) -> torch.Tensor:
    mass = mass.float().clamp_min(tiny)
    return mass / mass.sum(dim=-1, keepdim=True).clamp_min(tiny)


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


def flat_dm16_loss(
    pred_map: torch.Tensor,
    target_map: torch.Tensor,
    kappa: float = 20.0,
    stride: int = 4,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Flat Dirichlet-Multinomial-16 allocation loss on 16px blocks (4x4 stride-4 cells)."""
    k = max(1, 16 // stride)
    n16 = block_sum_2d(pred_map, k).flatten(1)
    y16 = block_sum_2d(target_map, k).flatten(1)
    pi = probs_from_positive_mass(n16, tiny=eps)
    alpha = float(kappa) * pi
    per_image_nll = dm_nll_none(y16, alpha, eps=eps)
    return per_image_nll.mean()


def scale_balanced_region_rate_loss(
    pred_region: torch.Tensor,
    target_region: torch.Tensor,
    regions: RegionSet,
    beta: float = 0.1,
) -> torch.Tensor:
    """Scale-balanced loss on per-cell RATE, not raw count.

    P0 #2 fix — normalize by area before computing loss:
        pred_rate   = b_R   / |R|    [predicted count/cell for each region]
        target_rate = N_R   / |R|    [GT count/cell for each region]
        L = (1/S) sum_s mean_{R in scale s} SmoothL1(pred_rate_R, target_rate_R, beta)

    Training on rate eliminates area-proportional gradient imbalance.
    """
    if pred_region.shape != target_region.shape:
        raise ValueError(f"shape mismatch: {pred_region.shape} vs {target_region.shape}")
    area = regions.area.to(dtype=pred_region.dtype).view(1, 1, -1).clamp_min(1.0)
    pred_rate = pred_region / area
    target_rate = target_region / area
    losses = []
    for sid in torch.unique(regions.scale_id):
        mask = regions.scale_id == sid
        if mask.any():
            losses.append(
                F.smooth_l1_loss(
                    pred_rate[..., mask],
                    target_rate[..., mask],
                    reduction="mean",
                    beta=beta,
                )
            )
    return torch.stack(losses).mean()


@dataclass
class LossConfig:
    lambda_count: float = 1.0
    lambda_flat_dm16: float = 1.0
    lambda_cell: float = 0.25
    lambda_region_head: float = 0.20
    lambda_region_map: float = 0.20
    lambda_deep_supervision: float = 0.0
    count_loss_mode: str = "nb"
    nb_dispersion: float = 50.0
    kappa_flat16: float = 20.0
    cell_beta: float = 1.0
    region_beta: float = 0.1

    # Backwards-compatibility property:
    @property
    def lambda_global(self) -> float:
        return self.lambda_count

    @lambda_global.setter
    def lambda_global(self, val: float) -> None:
        self.lambda_count = val


def compute_losses(
    outputs: dict,
    target_y: torch.Tensor,
    variant: str,
    cfg: LossConfig = LossConfig(),
) -> dict[str, torch.Tensor]:
    """Losses for all matched RQ variants (RMR-v2).

    Variant semantics:
      direct:          L_count + L_FlatDM16 + L_cell
      region_loss:     direct + training-only regional rate loss on output density map (B1)
      region_aux:      direct + auxiliary regional evidence head loss (B2)
      local_refine:    direct + purely local learned inference refinement (B3a)
      learned_project: region_aux + learned regional-membership projector (B3b)
      rmr:             region_aux + exact-adjoint reconciliation (B5-P)
    """
    y = outputs["y"]
    regions: RegionSet | None = outputs.get("regions")
    losses: dict[str, torch.Tensor] = {}

    losses["cell"] = balanced_smooth_l1(y, target_y, beta=cfg.cell_beta)
    losses["count"] = count_magnitude_loss(
        y, target_y, mode=cfg.count_loss_mode, dispersion=cfg.nb_dispersion
    )
    losses["global"] = losses["count"]  # backward-compatible key

    if cfg.lambda_flat_dm16 > 0:
        losses["flat_dm16"] = flat_dm16_loss(
            y, target_y, kappa=cfg.kappa_flat16
        )
    else:
        losses["flat_dm16"] = y.new_tensor(0.0)

    if variant in {"region_loss", "region_aux", "learned_project", "rmr"}:
        if regions is None:
            raise ValueError(f"Variant {variant} requires regions in outputs")
        target_region = regional_sum(target_y, regions.boxes)

        if variant == "region_loss":
            pred_region = regional_sum(y, regions.boxes)
            losses["region_map"] = scale_balanced_region_rate_loss(
                pred_region, target_region, regions, beta=cfg.region_beta
            )

        if variant in {"region_aux", "learned_project", "rmr"}:
            b_region = outputs["b_region"]
            losses["region_head"] = scale_balanced_region_rate_loss(
                b_region, target_region, regions, beta=cfg.region_beta
            )

    # Optional weak deep supervision on intermediate positive measures for iterative variants.
    iterates = outputs.get("iterates", [])
    if variant in {"local_refine", "learned_project", "rmr"} and len(iterates) > 2:
        mids = iterates[1:-1]
        if mids:
            losses["deep"] = torch.stack([
                balanced_smooth_l1(m, target_y, beta=cfg.cell_beta) for m in mids
            ]).mean()

    total = (
        cfg.lambda_count * losses["count"]
        + cfg.lambda_flat_dm16 * losses["flat_dm16"]
        + cfg.lambda_cell * losses["cell"]
    )
    if "region_map" in losses:
        total = total + cfg.lambda_region_map * losses["region_map"]
    if "region_head" in losses:
        total = total + cfg.lambda_region_head * losses["region_head"]
    if "deep" in losses and cfg.lambda_deep_supervision > 0:
        total = total + cfg.lambda_deep_supervision * losses["deep"]

    losses["total"] = total
    return losses
