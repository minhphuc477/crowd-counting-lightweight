from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from rmr_core.operators import (
    RegionSet,
    charbonnier_tv_step,
    compute_deficit_carrier_seed,
    partition_regions_by_scale,
    regional_adjoint,
    regional_sum,
    weighted_coverage,
    weighted_normalized_adjoint_field,
    weighted_regional_energy,
)
from .solver_ops import (
    anscombe_discrepancy,
    apply_trust_region_clamping,
    compute_adaptive_relaxation_factor,
    compute_adaptive_tau,
    compute_barzilai_borwein_step,
    density_gated_anscombe_discrepancy,
    laplacian_tv_diffusion,
    perona_malik_anisotropic_diffusion,
    prepare_anscombe_target,
    proximal_firm_threshold,
    proximal_soft_threshold,
)


def unrolled_sirt_solver(
    y0: torch.Tensor,
    b_solver: torch.Tensor,
    weight_solver: torch.Tensor,
    regions: RegionSet,
    iterations: int = 2,
    omega: float = 1.0,
    solver_strength: float = 1.0,
    residual_clip: float = 0.0,
    eps: float = 1e-6,
    solver_mode: str = "additive",
    density_gate_rho: float = 0.02,
    density_gate_floor: float = 0.02,
    proximal_tau: float = 0.0,
    proximal_mode: str = "soft",
    proximal_mu: float = 3.0,
    tv_lambda: float = 0.0,
    tv_type: str = "laplacian",
    tv_eps_c: float = 0.1,
    laplace_kernel: torch.Tensor | None = None,
    scale_routing_weights: torch.Tensor | None = None,
    trust_region_kappa: float = 0.0,
    trust_region_floor: float = 0.005,
    adjoint_mode: str = "flat",
    b_variance: torch.Tensor | None = None,
    morozov_gamma: float = 0.0,
    use_barzilai_borwein: bool = False,
    use_alternating_bb: bool = False,
    bb_clamp_min: float = 0.2,
    bb_clamp_max: float = 2.0,
    cyclic_bb_length: int = 1,
    use_scale_entropy_trust: bool = False,
    use_nesterov_momentum: bool = False,
    adaptive_relaxation: bool = False,
    adaptive_relax_sparse: float = 0.70,
    adaptive_relax_dense_boost: float = 0.50,
    adaptive_relax_threshold: float = 0.05,
    adaptive_relax_scale: float = 0.02,
    hybrid_recovery_alpha: float = 0.0,
    density_gated_diffusion: bool = False,
    diffusion_dense_threshold: float = 0.15,
    diffusion_gate_beta: float = 0.03,
    output_stride: int = 4,
    use_anscombe: bool = False,
    anscombe_c: float = 0.375,
    adaptive_tau: bool = False,
    adaptive_tau_rho0: float = 0.05,
    anisotropic_diffusion: bool = False,
    pm_kappa: float = 0.05,
    density_gated_anscombe: bool = False,
    anscombe_tau_dense: float = 0.08,
    area_normalized_adjoint: bool = False,
    carrier_energy: torch.Tensor | None = None,
    resonant_lambda: float = 0.0,
    anscombe_morozov: bool = False,
    crest_discovery_flux: bool = False,
    crest_kappa_0: float = 2.0,
    crest_eps_seed: float = 0.005,
    asymmetric_morozov: bool = False,
    morozov_gamma_under: float = 0.20,
    morozov_rho: float = 0.30,
    shifted_carrier: bool = False,
    shifted_carrier_eps: float = 0.02,
    density_adaptive_trust: bool = False,
    trust_dense_tau: float = 0.10,
    trust_dense_kappa: float = 0.80,
    asymmetric_trust: bool = False,
    trust_pos_kappa: float = 1.0,
    spatial_morozov: bool = False,
    morozov_gamma_scales: tuple[float, ...] = (0.25, 0.50, 0.75),
    scale_routed_trust: bool = False,
    scale_seeded_carrier: bool = False,
    scale_seed_eps: float = 0.02,
    trust_region_floor_dense: float = 0.025,
    evidence_anchored_trust: bool = False,
    eatr_alpha: float = 0.5,
    regional_rate: torch.Tensor | None = None,
    morozov_rho_cap: float = 0.0,
    multiscale_morozov: bool = False,
    morozov_scale_beta: float = 1.0,
    deficit_carrier_seeding: bool = False,
    deficit_seed_eps: float = 0.02,
    density_adaptive_morozov: bool = False,
    morozov_density_beta: float = 2.0,
    compute_energy: bool = True,
    omega_steps: torch.Tensor | Sequence[float] | None = None,
    gamma_steps: torch.Tensor | Sequence[float] | None = None,
) -> dict[str, Any]:

    """Execute unrolled Proximal Reliability-Weighted SIRT measure reconciliation."""
    if b_solver.ndim == 2:
        b_solver = b_solver.unsqueeze(1)
    if weight_solver.ndim == 2:
        weight_solver = weight_solver.unsqueeze(1)
    if b_variance is not None and b_variance.ndim == 2:
        b_variance = b_variance.unsqueeze(1)

    b, _, h, w = y0.shape
    if scale_routing_weights is not None:
        scale_routing_weights = scale_routing_weights.detach()
        if scale_routing_weights.shape[-2:] != (h, w):
            scale_routing_weights = F.interpolate(
                scale_routing_weights, size=(h, w), mode="bilinear", align_corners=False
            )
    area_scale = (float(output_stride) / 4.0) ** 2
    strength = min(max(float(solver_strength), 0.0), 1.0)
    effective_omega = float(omega) * strength
    effective_tv_lambda = float(tv_lambda) * strength
    effective_tau = float(proximal_tau) * area_scale
    effective_rho = float(density_gate_rho) * area_scale
    effective_diff_thresh = float(diffusion_dense_threshold) * area_scale
    effective_diff_beta = float(diffusion_gate_beta) * area_scale
    effective_relax_thresh = float(adaptive_relax_threshold) * area_scale
    effective_relax_scale = float(adaptive_relax_scale) * area_scale
    effective_trust_floor = float(trust_region_floor) * area_scale
    tau_step = (effective_omega * effective_tau) / max(int(iterations), 1)
    tv_step = effective_tv_lambda / max(int(iterations), 1)

    if iterations <= 0 or (effective_omega == 0.0 and effective_tv_lambda == 0.0 and tau_step == 0.0):
        return {
            "y": y0,
            "iterates": [y0],
            "residual_fields": [],
            "energy_trace": [],
            "step_omegas": [],
            "effective_omega": effective_omega,
            "effective_tv_lambda": effective_tv_lambda,
        }


    # Pre-partition regions by scale once before the T-iteration solver loop.
    # This completely eliminates dynamic boolean masking and tensor slicing allocations inside the loop.
    scale_partitions = None
    if scale_routing_weights is not None:
        k_scales = scale_routing_weights.shape[1]
        scale_partitions = partition_regions_by_scale(regions, k_scales, device=y0.device)

    # Compute weighted coverage field: D_w = A^T w (optionally scale-routed)
    cov_w = weighted_coverage(
        weight_solver,
        regions,
        h,
        w,
        eps=eps,
        scale_routing_weights=scale_routing_weights,
        scale_partitions=scale_partitions,
    )

    scale_confidence = None
    if use_scale_entropy_trust and scale_routing_weights is not None:
        k_scales = scale_routing_weights.shape[1]
        if k_scales > 1:
            pi_safe = scale_routing_weights.clamp_min(1e-7)
            entropy = -(pi_safe * torch.log(pi_safe)).sum(dim=1, keepdim=True)
            max_entropy = math.log(float(k_scales))
            scale_confidence = (1.0 - (entropy / max_entropy)).clamp(0.0, 1.0)
            if scale_confidence.shape[-2:] != (h, w):
                scale_confidence = F.interpolate(
                    scale_confidence, size=(h, w), mode="bilinear", align_corners=False
                )

    eatr_floor = None
    if evidence_anchored_trust and regional_rate is not None:
        cov_1 = weighted_coverage(torch.ones_like(weight_solver), regions, h, w, eps=eps)
        r_rate = regional_rate.unsqueeze(1) if regional_rate.ndim == 2 else regional_rate
        rho_back = regional_adjoint(r_rate.float().detach(), regions.boxes, h, w, out_dtype=torch.float32)
        rho_spatial = rho_back / cov_1.clamp_min(eps)
        if scale_routing_weights is not None:
            rho_spatial = rho_spatial * scale_routing_weights[:, 0:1, :, :].float()
        eatr_floor = float(eatr_alpha) * rho_spatial

    y_curr = y0
    y_prev = y0
    theta_curr = 1.0
    iterates: list[torch.Tensor] = [y0]
    residual_fields: list[torch.Tensor] = []
    energy_trace: list[dict[str, torch.Tensor]] = []
    step_omegas: list[torch.Tensor | float] = []

    prev_y: torch.Tensor | None = None
    prev_field: torch.Tensor | None = None
    cached_bb_omega: float | torch.Tensor | None = None
    energy_after: torch.Tensor | None = None

    for iter_idx in range(iterations):
        # Nesterov momentum extrapolation
        if use_nesterov_momentum and iter_idx > 0:
            theta_next = (1.0 + math.sqrt(1.0 + 4.0 * (theta_curr ** 2))) / 2.0
            beta = (theta_curr - 1.0) / theta_next
            z_state = y_curr + beta * (y_curr - y_prev)
            z_state = torch.clamp_min(z_state, 0.0)
            theta_curr = theta_next
        else:
            z_state = y_curr

        if compute_energy:
            if iter_idx == 0 or energy_after is None:
                energy_before = weighted_regional_energy(
                    y_curr, b_solver, weight_solver, regions,
                ).detach()
            else:
                energy_before = energy_after

        # ── Step 1: Adjoint discrepancy scatter evaluated at extrapolated state z ──
        is_anscombe = use_anscombe or (adjoint_mode == "anscombe_vst")
        if is_anscombe:
            b_target = prepare_anscombe_target(
                z_state=z_state,
                b_solver=b_solver,
                regions=regions,
                area_scale=area_scale,
                area_normalized_adjoint=area_normalized_adjoint,
                density_gated_anscombe=density_gated_anscombe,
                anscombe_c=anscombe_c,
                b_variance=b_variance,
                morozov_gamma=float(morozov_gamma),
                anscombe_tau_dense=float(anscombe_tau_dense),
                asymmetric_morozov=asymmetric_morozov,
                morozov_gamma_under=float(morozov_gamma_under),
                morozov_rho=float(morozov_rho),
                eps=eps,
            )
            adj_mode, b_var, m_gamma = "radon_nikodym", None, 0.0
            ansc_morozov, asym_morozov, spat_morozov, mscale_morozov, dadapt_morozov = (
                False, False, False, False, False
            )

        else:
            b_target, adj_mode, b_var = b_solver, adjoint_mode, b_variance
            if gamma_steps is not None:
                m_gamma = gamma_steps[iter_idx - 1]
            elif isinstance(morozov_gamma, torch.Tensor):
                m_gamma = morozov_gamma
            else:
                m_gamma = float(morozov_gamma)
            ansc_morozov, asym_morozov, spat_morozov, mscale_morozov, dadapt_morozov = (
                anscombe_morozov, asymmetric_morozov, spatial_morozov, multiscale_morozov, density_adaptive_morozov
            )

        psi_deficit = None
        if deficit_carrier_seeding and b_target is not None and regions is not None:
            psi_deficit = compute_deficit_carrier_seed(
                z_state, b_target, regions, coverage=cov_w,
                scale_routing_weights=scale_routing_weights,
                scale_partitions=scale_partitions,
                output_stride=output_stride, eps=eps,
            )

        field = weighted_normalized_adjoint_field(
            z_state, b_target, weight_solver, regions,
            weighted_cov=cov_w, residual_clip=residual_clip, eps=eps,
            solver_mode=solver_mode, density_gate_rho=float(effective_rho),
            density_gate_floor=float(density_gate_floor),
            scale_routing_weights=scale_routing_weights, scale_partitions=scale_partitions,
            adjoint_mode=adj_mode, b_variance=b_var, morozov_gamma=m_gamma,
            hybrid_recovery_alpha=float(hybrid_recovery_alpha), output_stride=output_stride,
            area_normalized=area_normalized_adjoint, carrier_energy=carrier_energy,
            resonant_lambda=float(resonant_lambda), anscombe_morozov=ansc_morozov,
            crest_discovery_flux=crest_discovery_flux, crest_kappa_0=float(crest_kappa_0),
            crest_eps_seed=float(crest_eps_seed), asymmetric_morozov=asym_morozov,
            morozov_gamma_under=float(morozov_gamma_under), morozov_rho=float(morozov_rho),
            shifted_carrier=shifted_carrier, shifted_carrier_eps=float(shifted_carrier_eps),
            y_initial=y0, spatial_morozov=spat_morozov, morozov_gamma_scales=morozov_gamma_scales,
            scale_seeded_carrier=scale_seeded_carrier, scale_seed_eps=float(scale_seed_eps),
            morozov_rho_cap=float(morozov_rho_cap),
            multiscale_morozov=mscale_morozov, morozov_scale_beta=float(morozov_scale_beta),
            deficit_carrier_seeding=deficit_carrier_seeding, deficit_seed_eps=float(deficit_seed_eps),
            density_adaptive_morozov=dadapt_morozov, morozov_density_beta=float(morozov_density_beta),
            psi_deficit=psi_deficit,
        )

        # Adaptive Barzilai-Borwein step size or learned step size
        if omega_steps is not None:
            current_omega: float | torch.Tensor = omega_steps[iter_idx - 1] * strength
        else:
            current_omega = effective_omega

        can_use_bb = (
            omega_steps is None
            and (use_barzilai_borwein or use_alternating_bb)
            and prev_y is not None
            and prev_field is not None
            and effective_omega > 0.0
        )
        if can_use_bb:
            if (
                cyclic_bb_length > 1
                and cached_bb_omega is not None
                and ((iter_idx - 1) % cyclic_bb_length != 0)
            ):
                current_omega = cached_bb_omega
            else:
                current_omega = compute_barzilai_borwein_step(
                    z_state=z_state,
                    prev_y=prev_y,
                    field=field,
                    prev_field=prev_field,
                    effective_omega=effective_omega,
                    use_alternating_bb=use_alternating_bb,
                    iter_idx=iter_idx,
                    bb_clamp_min=bb_clamp_min,
                    bb_clamp_max=bb_clamp_max,
                )
                cached_bb_omega = current_omega

        # Density-Adaptive Over-Relaxation (RMR-v20)
        if adaptive_relaxation:
            omega_mod = compute_adaptive_relaxation_factor(
                z_state=z_state,
                effective_relax_thresh=effective_relax_thresh,
                effective_relax_scale=effective_relax_scale,
                adaptive_relax_sparse=adaptive_relax_sparse,
                adaptive_relax_dense_boost=adaptive_relax_dense_boost,
            )
            current_omega = current_omega * omega_mod

        step_omegas.append(current_omega)
        # Gradient descent on 0.5 * ||Ay - b||^2: negative gradient step direction is -field.
        # delta_y > 0 adds mass (when under-counting: field < 0 => delta_y > 0).
        # delta_y < 0 removes mass (when over-counting: field > 0 => delta_y < 0).
        delta_y = -current_omega * field

        delta_y = apply_trust_region_clamping(
            delta_y=delta_y,
            z_state=z_state,
            trust_region_kappa=trust_region_kappa,
            effective_trust_floor=effective_trust_floor,
            area_scale=area_scale,
            scale_routed_trust=scale_routed_trust,
            scale_routing_weights=scale_routing_weights,
            trust_pos_kappa=trust_pos_kappa,
            trust_region_floor_dense=trust_region_floor_dense,
            density_adaptive_trust=density_adaptive_trust,
            trust_dense_tau=trust_dense_tau,
            trust_dense_kappa=trust_dense_kappa,
            deficit_carrier_seeding=deficit_carrier_seeding,
            psi_deficit=psi_deficit,
            scale_seeded_carrier=scale_seeded_carrier,
            scale_seed_eps=scale_seed_eps,
            eatr_floor=eatr_floor,
            asymmetric_trust=asymmetric_trust,
            scale_confidence=scale_confidence,
        )


        prev_y = y_curr.detach()   # Track actual iterate (not Nesterov extrapolate) for BB-1 correctness
        prev_field = field.detach()

        y_step = z_state.float() + delta_y

        # ── Step 2: Proximal thresholding L1-shrinkage ────────────────────
        tau_current: float | torch.Tensor = tau_step
        if adaptive_tau and tau_step > 0.0:
            tau_current = compute_adaptive_tau(
                tau_step,
                z_state,
                stride=output_stride,
                mode="density_adaptive",
                rho0=float(adaptive_tau_rho0),
                pool_kernel=5,
            )

        if proximal_mode == "firm":
            y_next = proximal_firm_threshold(y_step, tau=tau_current, mu=float(proximal_mu))
        elif proximal_mode == "soft":
            y_next = proximal_soft_threshold(y_step, tau=tau_current)
        elif proximal_mode in ("none", "clamp"):
            y_next = torch.clamp_min(y_step, 0.0)
        else:
            raise ValueError(
                f"Unknown proximal_mode: '{proximal_mode}'. Must be 'firm', 'soft', or 'none'."
            )

        # Step 3: Total Variation diffusion
        if tv_step > 0.0:
            if tv_type == "charbonnier":
                y_next = charbonnier_tv_step(y_next, tv_step, float(tv_eps_c))
            elif tv_type == "perona_malik" or anisotropic_diffusion:
                y_next = perona_malik_anisotropic_diffusion(y_next, tv_step, kappa=float(pm_kappa))
            else:
                y_next = laplacian_tv_diffusion(
                    y_next,
                    tv_step,
                    kernel=laplace_kernel,
                    density_gated=density_gated_diffusion,
                    diffusion_dense_threshold=effective_diff_thresh,
                    diffusion_gate_beta=effective_diff_beta,
                )

        y_next = y_next.to(dtype=y_curr.dtype)
        # Zero-host-sync numerical divergence guard: restore y_curr on NaN/Inf
        y_next = torch.where(torch.isfinite(y_next), y_next, y_curr)

        if compute_energy:
            energy_after = weighted_regional_energy(
                y_next, b_solver, weight_solver, regions,
            ).detach()
            energy_trace.append({"before": energy_before, "after": energy_after})
        residual_fields.append(field.detach())
        iterates.append(y_next)
        y_prev = y_curr
        y_curr = y_next

    return {
        "y": y_curr,
        "iterates": iterates,
        "residual_fields": residual_fields,
        "energy_trace": energy_trace,
        "step_omegas": step_omegas,
        "effective_omega": effective_omega,
        "effective_tv_lambda": effective_tv_lambda,
    }

