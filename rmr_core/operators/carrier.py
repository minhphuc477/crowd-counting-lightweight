"""Carrier modulation and deficit-conditioned carrier seeding operators for RMR.

Provides:
1. compute_deficit_carrier_seed: Backprojects positive regional deficit (b_solver - Ay > 0)
   into pixel space to break the multiplicative zero-support trap on dense clusters
   without leaking mass into empty background.
2. modulate_carrier_field: Combines base carrier with shifted carrier, scale routing,
   crest discovery flux, and deficit carrier seeding.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .prefix_sums import regional_sum
from .regions import RegionSet

__all__ = [
    "compute_deficit_carrier_seed",
    "modulate_carrier_field",
]


def compute_deficit_carrier_seed(
    y: torch.Tensor,
    b_solver: torch.Tensor,
    regions: RegionSet,
    coverage: torch.Tensor | None = None,
    output_stride: int = 4,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute deficit-conditioned carrier seeding field psi_deficit(u).

    When dense crowd clusters cause y0 to under-count and Softplus suppresses adjacent
    cells to near-zero, the Radon-Nikodym update Delta y = (y / cov) * A* r cannot inject
    mass into zero-support cells (0 * r = 0).

    This operator backprojects the POSITIVE regional deficit:
        deficit_R = ReLU(b_solver - Ay_t)
        psi_deficit(u) = A*(ReLU(b_solver - Ay_t) / area_R) / (cov_1(u) + eps)

    Invariants:
    1. Zero background leakage: On background / non-deficit regions (b_solver <= Ay),
       deficit_R == 0 identically, guaranteeing psi_deficit(u) == 0.0 with 0 false positives.
    2. Dense deficit discovery: On dense clusters where b_solver > Ay, psi_deficit(u) > 0,
       providing a non-zero carrier seed m_carrier(u) = y(u) + eps_seed * psi_deficit(u).
    3. Self-terminating: As Ay -> b_solver, deficit -> 0 and psi_deficit -> 0 automatically.
    """
    from .adjoint import regional_adjoint

    if b_solver.ndim == 2:
        b_solver = b_solver.unsqueeze(1)

    b, _, h, w = y.shape
    y32 = y.float()
    q = regional_sum(y32, regions.boxes, out_dtype=torch.float32)
    b32 = b_solver.float()

    deficit = torch.relu(b32 - q)
    if not (deficit > 0.0).any():
        return torch.zeros((b, 1, h, w), device=y.device, dtype=y.dtype)

    area = regions.area.float().view(1, 1, -1).clamp_min(1.0)
    eff_area = area * ((float(output_stride) / 4.0) ** 2)
    norm_deficit = deficit / eff_area.clamp_min(1.0)

    psi = regional_adjoint(norm_deficit, regions.boxes, h, w, out_dtype=torch.float32)
    if coverage is not None:
        psi = psi / coverage.float().clamp_min(eps)

    return psi.to(dtype=y.dtype)


def modulate_carrier_field(
    y: torch.Tensor,
    y_initial: torch.Tensor | None = None,
    scale_routing_weights: torch.Tensor | None = None,
    carrier_energy: torch.Tensor | None = None,
    resonant_lambda: float = 0.0,
    crest_discovery_flux: bool = False,
    crest_kappa_0: float = 2.0,
    crest_eps_seed: float = 0.005,
    shifted_carrier: bool = False,
    shifted_carrier_eps: float = 0.02,
    scale_seeded_carrier: bool = False,
    scale_seed_eps: float = 0.02,
    deficit_carrier_seeding: bool = False,
    deficit_seed_eps: float = 0.02,
    b_solver: torch.Tensor | None = None,
    regions: RegionSet | None = None,
    coverage: torch.Tensor | None = None,
    output_stride: int = 4,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Modulate continuous Radon carrier field with prior texture, shifts, and deficit seeds.

    Returns:
        (m_carrier, m_base): Modulated carrier for update and base carrier for rate normalization.
    """
    b, _, h, w = y.shape
    y32 = y.float()
    m_carrier = y32
    psi_crest = None

    if carrier_energy is not None:
        E = carrier_energy.float()
        if E.shape[-2:] != (h, w):
            E = F.interpolate(E, size=(h, w), mode="bilinear", align_corners=False)
        E_pool = F.avg_pool2d(E, kernel_size=9, stride=1, padding=4, count_include_pad=False)
        if resonant_lambda > 0.0:
            phi = (E / (E_pool + 1e-4)).clamp(0.2, 4.0)
            m_carrier = y32 * ((1.0 - float(resonant_lambda)) + float(resonant_lambda) * phi)
        if crest_discovery_flux:
            E_sq_pool = F.avg_pool2d(E.square(), kernel_size=9, stride=1, padding=4, count_include_pad=False)
            E_var = (E_sq_pool - E_pool.square()).clamp_min(1e-8)
            z_E = (E - E_pool) / (torch.sqrt(E_var) + 1e-4)
            psi_crest = torch.relu(z_E - float(crest_kappa_0)).square()

    if shifted_carrier and y_initial is not None:
        m_carrier = m_carrier + float(shifted_carrier_eps) * y_initial.float()
    if scale_seeded_carrier and scale_routing_weights is not None:
        m_carrier = m_carrier + float(scale_seed_eps) * scale_routing_weights[:, 0:1].float().detach()
    if crest_discovery_flux and psi_crest is not None:
        m_carrier = m_carrier + float(crest_eps_seed) * psi_crest

    if deficit_carrier_seeding and b_solver is not None and regions is not None:
        psi_deficit = compute_deficit_carrier_seed(
            y, b_solver, regions, coverage=coverage, output_stride=output_stride, eps=eps
        )
        if (psi_deficit > 0.0).any():
            m_carrier = m_carrier + float(deficit_seed_eps) * psi_deficit.float()

    m_base = m_carrier
    return m_carrier, m_base
