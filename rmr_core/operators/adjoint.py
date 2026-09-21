from __future__ import annotations

import torch
import torch.nn.functional as F

from .prefix_sums import regional_sum
from .regions import RegionSet, partition_regions_by_scale


def regional_adjoint(
    values: torch.Tensor,
    boxes: torch.Tensor,
    height: int,
    width: int,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Exact adjoint A^T of rectangular summation.

    values: [B,C,M] or [B,M]
    boxes:  [M,4]
    returns [B,C,H,W]

    Uses a 2-D difference buffer followed by cumulative sums.
    Forces FP32 accumulation during autocast to maintain exact precision.
    """
    if values.ndim == 2:
        values = values.unsqueeze(1)
    elif values.ndim != 3:
        raise ValueError(f"values must be [B,C,M] or [B,M], got {tuple(values.shape)}")
    b, c, m = values.shape
    if boxes.shape != (m, 4):
        raise ValueError(f"boxes must be [{m},4], got {tuple(boxes.shape)}")

    boxes = boxes.to(device=values.device, dtype=torch.long)
    y1, x1, y2, x2 = boxes.unbind(dim=-1)
    hp, wp = height + 1, width + 1

    y1 = y1.clamp(0, height)
    x1 = x1.clamp(0, width)
    y2 = y2.clamp(0, height)
    x2 = x2.clamp(0, width)

    orig_dtype = values.dtype if out_dtype is None else out_dtype
    work = values.float() if values.dtype in (torch.float16, torch.bfloat16) else values
    diff = work.new_zeros((b, c, hp * wp))

    def scatter(y: torch.Tensor, x: torch.Tensor, src: torch.Tensor) -> None:
        yc = y.clamp(0, hp - 1)
        xc = x.clamp(0, wp - 1)
        idx = (yc * wp + xc).view(1, 1, -1).expand(b, c, -1)
        diff.scatter_add_(dim=-1, index=idx, src=src)

    scatter(y1, x1, work)
    scatter(y1, x2, -work)
    scatter(y2, x1, -work)
    scatter(y2, x2, work)

    diff = diff.view(b, c, hp, wp)
    field = diff.cumsum(dim=-2).cumsum(dim=-1)
    res = field[..., :height, :width]
    return res.to(orig_dtype) if res.dtype != orig_dtype else res


def multiplicative_gated_adjoint(
    values: torch.Tensor,
    boxes: torch.Tensor,
    y_current: torch.Tensor,
    height: int,
    width: int,
    rho0: float = 0.02,
    gate_floor: float = 0.0,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Multiplicative Gated SIRT adjoint step with recovery floor.

    Suppresses the correction signal on near-zero pixels via a tanh gate,
    preventing background pixels from being lifted off zero during repeated
    SIRT iterations (the "background lift" degradation observed at T>=2 with
    the plain additive adjoint).

    A small gate_floor > 0 (default: 0.02) prevents the "zero-absorbing state"
    where a false-negative zero prediction in y_0 can never receive a positive
    correction from regional evidence.

    The gate is:
        gate(i) = (1.0 - floor) * tanh(|y_current(i)| / rho0) + floor

    Where:
        - At y ~ 0: gate = floor (default 0.02, suppressing background lift by 98%
          while allowing false-negative regions to recover).
        - At y >> rho0: gate = 1.0 (full correction for crowd clusters).

    Args:
        values:     [B, C, M]  residual values to scatter (same as regional_adjoint).
        boxes:      [M, 4]     half-open box coordinates (y1, x1, y2, x2).
        y_current:  [B, 1, H, W]  current density iterate for gate computation.
        height:     output height H.
        width:      output width W.
        rho0:       gate threshold; default 0.02 matches the empirical mean cell density prior.
        gate_floor: lower floor for gate to prevent permanent zero traps (default: 0.02).
        out_dtype:  output dtype (default: values.dtype).

    Returns:
        [B, C, H, W] multiplicatively gated correction field.
    """
    # Compute the standard additive adjoint field in fp32
    field_additive = regional_adjoint(values, boxes, height, width, out_dtype=torch.float32)

    # Gate: (1 - floor) * tanh(|y| / rho0) + floor
    tanh_gate = torch.tanh(y_current.float().abs() / float(rho0))  # [B, 1, H, W]
    floor_val = float(max(0.0, min(gate_floor, 1.0)))
    gate = (1.0 - floor_val) * tanh_gate + floor_val

    # Broadcast gate over C channels if needed
    gated = gate * field_additive  # [B, C, H, W]

    orig_dtype = values.dtype if out_dtype is None else out_dtype
    return gated.to(orig_dtype) if gated.dtype != orig_dtype else gated


def weighted_coverage(
    weight: torch.Tensor,
    regions: RegionSet,
    height: int,
    width: int,
    eps: float = 1e-6,
    scale_routing_weights: torch.Tensor | None = None,
    scale_partitions: list[tuple[int, torch.Tensor | None, torch.Tensor | None]] | None = None,
) -> torch.Tensor:
    """Compute D_{c,w} diagonal field = A^T w, optionally modulated by spatial scale routing weights."""
    if weight.ndim == 2:
        weight = weight.unsqueeze(1)
    if scale_routing_weights is not None:
        b, k_scales = scale_routing_weights.shape[:2]
        if scale_routing_weights.shape[-2:] != (height, width):
            scale_routing_weights = F.interpolate(
                scale_routing_weights, size=(height, width), mode="bilinear", align_corners=False
            )
        cov_total = torch.zeros((b, 1, height, width), device=weight.device, dtype=torch.float32)
        if scale_partitions is None:
            scale_partitions = partition_regions_by_scale(regions, k_scales, device=weight.device)
        for k, mask_k, boxes_k in scale_partitions:
            if mask_k is None or boxes_k is None:
                continue
            weight_k = weight[:, :, mask_k].float()
            cov_k = regional_adjoint(weight_k, boxes_k, height, width, out_dtype=torch.float32)
            pi_k = scale_routing_weights[:, k:k+1, :, :].float()
            cov_total = cov_total + pi_k * cov_k
        return cov_total.clamp_min(float(eps))

    cov = regional_adjoint(
        weight.float(),
        regions.boxes,
        height,
        width,
        out_dtype=torch.float32,
    )
    return cov.clamp_min(float(eps))


def weighted_normalized_adjoint_field(
    y: torch.Tensor,
    b_region: torch.Tensor,
    weight: torch.Tensor,
    regions: RegionSet,
    *,
    weighted_cov: torch.Tensor | None = None,
    residual_clip: float = 0.0,
    eps: float = 1e-6,
    solver_mode: str = "additive",
    density_gate_rho: float = 0.02,
    density_gate_floor: float = 0.02,
    scale_routing_weights: torch.Tensor | None = None,
    scale_partitions: list[tuple[int, torch.Tensor | None, torch.Tensor | None]] | None = None,
    adjoint_mode: str = "flat",
    b_variance: torch.Tensor | None = None,
    morozov_gamma: float = 0.0,
    hybrid_recovery_alpha: float = 0.0,
    output_stride: int = 4,
    area_normalized: bool = False,
    carrier_energy: torch.Tensor | None = None,
    resonant_lambda: float = 0.0,
    anscombe_morozov: bool = False,
    crest_discovery_flux: bool = False,
    crest_kappa_0: float = 2.0,
    crest_eps_seed: float = 0.005,
    asymmetric_morozov: bool = False,
    morozov_gamma_under: float = 0.20,
    morozov_rho: float = 0.30,
) -> torch.Tensor:
    """Compute normalized adjoint correction field with CRCDF and A-SAM."""
    _, _, h, w = y.shape

    if b_region.ndim == 2:
        b_region = b_region.unsqueeze(1)
    if weight.ndim == 2:
        weight = weight.unsqueeze(1)
    if b_variance is not None and b_variance.ndim == 2:
        b_variance = b_variance.unsqueeze(1)

    y32 = y.float()
    b32 = b_region.float()
    weight32 = weight.float()

    q = regional_sum(y32, regions.boxes, out_dtype=torch.float32)
    delta = q - b32

    # Morozov Discrepancy Shrinkage (Symmetric or Asymmetric SNR-Adaptive A-SAM)
    if morozov_gamma > 0.0 and b_variance is not None:
        if anscombe_morozov:
            c = 0.375
            g_q = 2.0 * torch.sqrt(q.clamp_min(0.0) + c)
            g_b = 2.0 * torch.sqrt(b32.clamp_min(0.0) + c)
            g_delta = g_q - g_b
            if asymmetric_morozov:
                gamma_under = float(morozov_gamma_under) / (1.0 + float(morozov_rho) * torch.sqrt(b32.clamp_min(0.0)))
                gamma_eff = torch.where(g_delta > 0.0, float(morozov_gamma), gamma_under)
            else:
                gamma_eff = float(morozov_gamma)
            g_shrunk = torch.sign(g_delta) * torch.clamp_min(g_delta.abs() - gamma_eff, 0.0)
            scale_symm = 0.5 * (torch.sqrt(q.clamp_min(0.0) + c) + torch.sqrt(b32.clamp_min(0.0) + c))
            delta = g_shrunk * scale_symm
        else:
            sigma_b = torch.sqrt(b_variance.float().clamp_min(1e-12))
            if asymmetric_morozov:
                gamma_under = float(morozov_gamma_under) / (1.0 + float(morozov_rho) * torch.sqrt(b32.clamp_min(0.0)))
                gamma_eff = torch.where(delta > 0.0, float(morozov_gamma), gamma_under)
            else:
                gamma_eff = float(morozov_gamma)
            deadband = gamma_eff * sigma_b
            delta = torch.sign(delta) * torch.clamp_min(delta.abs() - deadband, 0.0)

    area = regions.area.float().view(1, 1, -1)
    eff_area = area * ((float(output_stride) / 4.0) ** 2) if area_normalized else area

    alpha_recov = float(max(0.0, min(1.0, hybrid_recovery_alpha)))
    use_hybrid = (adjoint_mode == "radon_nikodym" and alpha_recov > 0.0)

    # Carrier Texture Modulation & Crest Discovery Flux (CRCDF)
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

    # Radon-Nikodym Measure-Modulated Adjoint vs Flat Lebesgue Adjoint
    if adjoint_mode == "radon_nikodym":
        if carrier_energy is not None and (resonant_lambda > 0.0 or (crest_discovery_flux and psi_crest is not None)):
            m_base = m_carrier + (float(crest_eps_seed) * psi_crest if (crest_discovery_flux and psi_crest is not None) else 0.0)
            q_m = regional_sum(m_base, regions.boxes, out_dtype=torch.float32)
            eff_q = q_m + float(eps) * eff_area.clamp_min(1.0)
        else:
            eff_q = q + float(eps) * eff_area.clamp_min(1.0)
        rate_residual = delta / eff_q.clamp_min(float(eps))
    else:
        rate_residual = delta / area.clamp_min(1.0)

    weighted_residual = weight32 * rate_residual
    weighted_residual_leb = (weight32 * (delta / area.clamp_min(1.0))) if use_hybrid else None

    def _scatter_residual(w_res: torch.Tensor) -> torch.Tensor:
        if scale_routing_weights is not None:
            b_sz, k_scales = scale_routing_weights.shape[:2]
            sc_weights = scale_routing_weights
            if sc_weights.shape[-2:] != (h, w):
                sc_weights = F.interpolate(
                    sc_weights, size=(h, w), mode="bilinear", align_corners=False
                )
            back_tot = torch.zeros((b_sz, 1, h, w), device=y.device, dtype=torch.float32)
            partitions = scale_partitions
            if partitions is None:
                partitions = partition_regions_by_scale(regions, k_scales, device=y.device)
            for k, mask_k, boxes_k in partitions:
                if mask_k is None or boxes_k is None:
                    continue
                res_k = w_res[:, :, mask_k]
                if solver_mode == "multiplicative":
                    bk_k = multiplicative_gated_adjoint(
                        res_k,
                        boxes_k,
                        y32,
                        h,
                        w,
                        rho0=density_gate_rho,
                        gate_floor=density_gate_floor,
                        out_dtype=torch.float32,
                    )
                else:
                    bk_k = regional_adjoint(
                        res_k,
                        boxes_k,
                        h,
                        w,
                        out_dtype=torch.float32,
                    )
                pi_k = sc_weights[:, k:k+1, :, :].float()
                back_tot = back_tot + pi_k * bk_k
            return back_tot
        else:
            if solver_mode == "multiplicative":
                return multiplicative_gated_adjoint(
                    w_res,
                    regions.boxes,
                    y32,
                    h,
                    w,
                    rho0=density_gate_rho,
                    gate_floor=density_gate_floor,
                    out_dtype=torch.float32,
                )
            else:
                return regional_adjoint(
                    w_res,
                    regions.boxes,
                    h,
                    w,
                    out_dtype=torch.float32,
                )

    back = _scatter_residual(weighted_residual)

    if adjoint_mode == "radon_nikodym":
        if crest_discovery_flux and psi_crest is not None:
            m_eff = m_carrier + float(crest_eps_seed) * psi_crest * (back < 0.0).float()
        else:
            m_eff = m_carrier
        if use_hybrid and weighted_residual_leb is not None:
            back_leb = _scatter_residual(weighted_residual_leb)
            back = (1.0 - alpha_recov) * (m_eff * back) + alpha_recov * back_leb
        else:
            back = m_eff * back

    if weighted_cov is None:
        weighted_cov = weighted_coverage(
            weight32,
            regions,
            h,
            w,
            eps=eps,
            scale_routing_weights=scale_routing_weights,
            scale_partitions=scale_partitions,
        )

    field = back / weighted_cov.float().clamp_min(eps)

    if residual_clip > 0:
        field = field.clamp(
            -float(residual_clip),
            float(residual_clip),
        )

    return field


def weighted_regional_energy(
    y: torch.Tensor,
    b_region: torch.Tensor,
    weight: torch.Tensor,
    regions: RegionSet,
) -> torch.Tensor:
    """Per-sample weighted regional energy.

        E = 1/2 sum_R w_R * (Ay-b)^2 / area_R
    """
    if b_region.ndim == 2:
        b_region = b_region.unsqueeze(1)
    if weight.ndim == 2:
        weight = weight.unsqueeze(1)

    q = regional_sum(
        y.float(),
        regions.boxes,
        out_dtype=torch.float32,
    )

    delta = q - b_region.float()

    area = regions.area.float().view(1, 1, -1)

    energy = 0.5 * (
        weight.float()
        * delta.square()
        / area.clamp_min(1.0)
    ).sum(dim=(-2, -1))

    return energy


def center_scatter(
    values: torch.Tensor,
    boxes: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Sparse learned-projection control: place each region residual at its center.

    values: [B,1,M]
    returns [B,1,H,W] with collision averaging.
    """
    if values.ndim != 3 or values.shape[1] != 1:
        raise ValueError("center_scatter expects values [B,1,M]")
    b, _, m = values.shape
    boxes = boxes.to(device=values.device, dtype=torch.long)
    y = ((boxes[:, 0] + boxes[:, 2] - 1) // 2).long().clamp(0, height - 1)
    x = ((boxes[:, 1] + boxes[:, 3] - 1) // 2).long().clamp(0, width - 1)
    idx = (y * width + x).view(1, 1, m).expand(b, 1, -1)
    out = values.new_zeros((b, 1, height * width))
    cnt = values.new_zeros((b, 1, height * width))
    out.scatter_add_(-1, idx, values)
    cnt.scatter_add_(-1, idx, torch.ones_like(values))
    out = out / cnt.clamp_min(1.0)
    return out.view(b, 1, height, width)
