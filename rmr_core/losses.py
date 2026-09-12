from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F

from rmr_core.operators import RegionSet, regional_sum


def balanced_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """Equalize empty and non-empty cell contributions."""
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
    if not torch.isfinite(y).all() or torch.any(y < 0):
        raise ValueError("Negative-Binomial targets must be finite non-negative numbers")

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
    pn = pred.float().sum(dim=(-2, -1)).view(-1)
    tn = target.float().sum(dim=(-2, -1)).view(-1)
    if mode == "nb":
        return negative_binomial_nll_mean_dispersion(tn, pn, dispersion=dispersion, reduction="mean")
    elif mode == "log1p":
        return F.smooth_l1_loss(torch.log1p(pn), torch.log1p(tn), reduction="mean", beta=0.2)
    elif mode == "l1":
        return F.l1_loss(pn, tn, reduction="mean")
    raise ValueError(f"Unsupported count loss mode: {mode}")


global_count_loss = count_magnitude_loss


def block_sum_2d(x: torch.Tensor, k: int = 4, strict: bool = True) -> torch.Tensor:
    """Sum non-overlapping k x k blocks via reshape.

    If strict=True, raises ValueError when height or width is not divisible by k.
    If strict=False, trailing edge cells are trimmed.
    """
    had_channel = x.ndim == 4
    if not had_channel:
        x = x.unsqueeze(1)
    b, c, h, w = x.shape
    if h % k != 0 or w % k != 0:
        if strict:
            raise ValueError(
                f"Input spatial dimensions ({h}, {w}) must be divisible by block size k={k}"
            )
        h_trim = (h // k) * k
        w_trim = (w // k) * k
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


def flat_dm_block_loss(
    pred_map: torch.Tensor,
    target_map: torch.Tensor,
    block_px: int = 16,
    kappa: float = 20.0,
    stride: int = 4,
    eps: float = 1e-8,
    normalize_by_count: bool = True,
    strict: bool = True,
) -> torch.Tensor:
    """Flat Dirichlet-Multinomial allocation loss on arbitrary block_px sizes."""
    if block_px % stride != 0:
        raise ValueError(f"block_px ({block_px}) must be divisible by stride ({stride})")

    k = block_px // stride
    h_pred, w_pred = pred_map.shape[-2:]
    h_tgt, w_tgt = target_map.shape[-2:]

    if h_pred % k != 0 or w_pred % k != 0:
        if strict:
            raise ValueError(
                f"FlatDM{block_px} requires grid dimensions divisible by block size k={k}: grid {pred_map.shape[-2:]}"
            )
    if h_tgt % k != 0 or w_tgt % k != 0:
        if strict:
            raise ValueError(
                f"FlatDM{block_px} requires grid dimensions divisible by block size k={k}: grid {target_map.shape[-2:]}"
            )

    if not strict and (h_pred < k or w_pred < k or h_tgt < k or w_tgt < k):
        return pred_map.new_tensor(0.0)

    pred_block = block_sum_2d(pred_map.float(), k=k, strict=strict).flatten(1)
    target_block = block_sum_2d(target_map.float(), k=k, strict=strict).flatten(1)

    if pred_block.shape[-1] == 0:
        return pred_map.new_tensor(0.0)

    pi = probs_from_positive_mass(pred_block, tiny=eps)
    alpha = float(kappa) * pi

    per_image = dm_nll_none(target_block, alpha, eps=eps)
    if normalize_by_count:
        count = target_block.sum(-1).clamp_min(1.0)
        per_image = per_image / count

    return per_image.mean()


def multiscale_dm_loss(
    pred_map: torch.Tensor,
    target_map: torch.Tensor,
    block_sizes_px: tuple[int, ...] = (16, 32, 64),
    weights: tuple[float, ...] | None = None,
    kappas: tuple[float, ...] | None = None,
    stride: int = 4,
    eps: float = 1e-8,
    normalize_by_count: bool = True,
    strict: bool = True,
    return_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Multi-Scale Dirichlet-Multinomial allocation loss across independent block granularities.

    Computes a weighted sum of Flat Dirichlet-Multinomial partition losses across multiple
    block scales (e.g. 16px, 32px, 64px), capturing local-to-regional count allocation
    without imposing a strict conditional tree-factored probability structure.
    """
    if not block_sizes_px:
        raise ValueError("block_sizes_px must not be empty")

    if weights is None:
        if len(block_sizes_px) == 3 and block_sizes_px == (16, 32, 64):
            weights = (0.50, 0.30, 0.20)
        else:
            weights = tuple(1.0 / len(block_sizes_px) for _ in block_sizes_px)
    if kappas is None:
        kappas = tuple(20.0 for _ in block_sizes_px)

    if not (len(block_sizes_px) == len(weights) == len(kappas)):
        raise ValueError(
            f"length mismatch: block_sizes_px={len(block_sizes_px)}, weights={len(weights)}, kappas={len(kappas)}"
        )

    if any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative")

    wsum = float(sum(weights))
    if wsum <= 0:
        raise ValueError("sum(weights) must be > 0")

    terms = []
    components: dict[int, torch.Tensor] = {}
    for block_px, w, kappa in zip(block_sizes_px, weights, kappas):
        if w == 0:
            continue
        li = flat_dm_block_loss(
            pred_map,
            target_map,
            block_px=int(block_px),
            kappa=float(kappa),
            stride=stride,
            eps=eps,
            normalize_by_count=normalize_by_count,
            strict=strict,
        )
        components[int(block_px)] = li
        terms.append((float(w) / wsum) * li)

    total = torch.stack(terms).sum()
    if return_components:
        return total, components
    return total


# Backward compatibility alias
hierarchical_dm_loss = multiscale_dm_loss


def flat_dm16_loss(
    pred_map: torch.Tensor,
    target_map: torch.Tensor,
    kappa: float = 20.0,
    stride: int = 4,
    eps: float = 1e-8,
    normalize_by_count: bool = True,
    strict: bool = True,
) -> torch.Tensor:
    """Flat Dirichlet-Multinomial-16 allocation loss on 16px blocks (backward compatible)."""
    return flat_dm_block_loss(
        pred_map,
        target_map,
        block_px=16,
        kappa=kappa,
        stride=stride,
        eps=eps,
        normalize_by_count=normalize_by_count,
        strict=strict,
    )


def scale_balanced_region_rate_loss(
    pred_region: torch.Tensor,
    target_region: torch.Tensor,
    regions: RegionSet,
    beta: float = 0.1,
) -> torch.Tensor:
    """Scale-balanced loss on per-cell RATE, not raw count."""
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
    normalize_flat_dm16: bool = True

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
    cfg: LossConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Losses for all matched RQ variants (RMR-v2)."""
    if cfg is None:
        cfg = LossConfig()
    y = outputs["y"]
    y0 = outputs["y0"]
    regions: RegionSet | None = outputs.get("regions")
    losses: dict[str, torch.Tensor] = {}

    losses["cell"] = balanced_smooth_l1(y, target_y, beta=cfg.cell_beta)
    losses["count"] = count_magnitude_loss(
        y, target_y, mode=cfg.count_loss_mode, dispersion=cfg.nb_dispersion
    )
    losses["global"] = losses["count"]

    if cfg.lambda_flat_dm16 > 0:
        losses["flat_dm16"] = flat_dm16_loss(
            y0,
            target_y,
            kappa=cfg.kappa_flat16,
            normalize_by_count=cfg.normalize_flat_dm16,
        )
    else:
        losses["flat_dm16"] = y0.new_tensor(0.0)

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
