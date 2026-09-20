from __future__ import annotations

import torch
import torch.nn.functional as F


def push_forward_stride2_to_stride4(y_stride2: torch.Tensor) -> torch.Tensor:
    """Exact discrete Radon measure push-forward from Stride 2 to Stride 4 lattice.

    Computes 2x2 cell box summation:
        y^{(4)}_{i,j} = sum_{u=0}^1 sum_{v=0}^1 y^{(2)}_{2i+u, 2j+v}
    Preserves exact total mass: sum(y^{(4)}) == sum(y^{(2)}) to machine precision.
    Trainable parameters: exactly 0.
    """
    orig_ndim = y_stride2.ndim
    if orig_ndim == 2:
        y4d = y_stride2.unsqueeze(0).unsqueeze(0)
    elif orig_ndim == 3:
        y4d = y_stride2.unsqueeze(1)
    elif orig_ndim == 4:
        y4d = y_stride2
    else:
        raise ValueError(f"push_forward_stride2_to_stride4 expects 2D, 3D, or 4D tensor, got ndim={orig_ndim}")

    h, w = y4d.shape[-2:]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h > 0 or pad_w > 0:
        y4d = F.pad(y4d.float(), (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    else:
        y4d = y4d.float()

    # Non-overlapping 2x2 sum pooling: 4.0 * AvgPool2d(2, 2)
    carrier = 4.0 * F.avg_pool2d(y4d, kernel_size=2, stride=2, count_include_pad=False)
    carrier = carrier.to(dtype=y_stride2.dtype)

    if orig_ndim == 2:
        return carrier.squeeze(0).squeeze(0)
    elif orig_ndim == 3:
        return carrier.squeeze(1)
    return carrier


def pullback_stride4_to_stride2_rn(
    y_carrier: torch.Tensor,
    y_fine_prior: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Radon-Nikodym measure prolongation from Stride 4 carrier to Stride 2 fine lattice.

    Redistributes coarse carrier mass onto fine cells proportionally to fine prior:
        y^{(2)}_u = y^{(4)}_k * (y^{(2, 0)}_u / sum_{v in sub(k)} y^{(2, 0)}_v)
    with uniform fallback (0.25) when carrier block mass <= eps.
    Guarantees strict local and global mass conservation (Theorem 1) and
    support preservation (Theorem 2).
    """
    y4 = y_carrier.float()
    y2_0 = y_fine_prior.float()

    prior_carrier = push_forward_stride2_to_stride4(y2_0)
    h, w = y2_0.shape[-2:]
    pad_h = h % 2
    pad_w = w % 2
    target_size = (h + pad_h, w + pad_w)

    y4_up = F.interpolate(y4, size=target_size, mode="nearest")[..., :h, :w]
    prior_carrier_up = F.interpolate(prior_carrier, size=target_size, mode="nearest")[..., :h, :w]

    zero_prior_mask = (prior_carrier_up <= float(eps))
    denom = torch.where(zero_prior_mask, torch.ones_like(prior_carrier_up), prior_carrier_up)
    weights = torch.where(zero_prior_mask, torch.full_like(y2_0, 0.25), y2_0 / denom)
    fine_prolong = y4_up * weights
    return fine_prolong.to(dtype=y_carrier.dtype)


def check_mass_conservation(
    y_fine: torch.Tensor,
    y_carrier: torch.Tensor,
    eps: float = 1e-6,
) -> bool:
    """Verify discrete mass conservation between fine and carrier lattices."""
    m_fine = y_fine.double().sum(dim=(-2, -1))
    m_carrier = y_carrier.double().sum(dim=(-2, -1))
    diff = (m_fine - m_carrier).abs()
    max_abs = diff.max().item()
    max_rel = (diff / m_carrier.clamp_min(1e-6)).max().item()
    return (max_abs < eps) or (max_rel < eps)
