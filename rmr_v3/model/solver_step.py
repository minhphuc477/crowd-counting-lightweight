from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from rmr_core.operators import (
    RegionSet,
    closed_form_mcbpro_transport,
    closed_form_radon_nikodym_transport,
)
from rmr_core.types import RMRModelOutput
from ..solver import unrolled_sirt_solver
from .config import RMRv3Config


def solve_inverse_measure(
    cfg: RMRv3Config,
    y0: torch.Tensor,
    z0: torch.Tensor,
    regional_evidence: dict[str, Any],
    regions: RegionSet,
    scale_weights: torch.Tensor | None,
    solver_strength: float | None,
    default_solver_strength: float,
    uniform_reliability: bool,
    p16: torch.Tensor,
    fg_logit: torch.Tensor | None,
    pi_scale: torch.Tensor | None,
    pi_aspect: torch.Tensor | None,
    trust_gate: nn.Linear | None,
    laplace_kernel: torch.Tensor,
    carrier_energy: torch.Tensor | None = None,
    compute_energy: bool = True,
    omega_steps: torch.Tensor | None = None,
    gamma_steps: torch.Tensor | None = None,
) -> RMRModelOutput:
    """Solve the unrolled inverse problem on Radon measures using SIRT and return RMRModelOutput."""
    mu_count = regional_evidence["mu_count"]
    dispersion = regional_evidence["dispersion"]
    b_solver = regional_evidence["b_solver"]
    b_variance = regional_evidence["b_variance"]
    weight = regional_evidence["weight"]
    weight_solver = regional_evidence["weight_solver"]
    hurdle_logit = regional_evidence["hurdle_logit"]

    base_out: dict[str, Any] = {
        "y0": y0,
        "z0": z0,
        "regions": regions,
        "b_region": mu_count,
        "b_solver": b_solver,
        "region_rate": regional_evidence["rate"],
        "region_dispersion": dispersion,
        "region_log_dispersion": regional_evidence["log_dispersion"],
        "region_weight": weight,
        "solver_region_weight": weight_solver,
        "region_precision": regional_evidence["precision"],
        "region_rate_variance": regional_evidence["rate_variance"],
        "region_count_variance": regional_evidence["count_variance"],
        "solver_count_variance": b_variance,
        "uniform_reliability": uniform_reliability,
    }

    if scale_weights is not None:
        base_out["scale_weights"] = scale_weights
    if pi_scale is not None:
        base_out["pi_scale"] = pi_scale
    if pi_aspect is not None:
        base_out["pi_aspect"] = pi_aspect
    if hurdle_logit is not None:
        base_out["hurdle_logit"] = hurdle_logit
    if fg_logit is not None:
        base_out["fg_logit"] = fg_logit

    if not cfg.enable_solver or cfg.solver_mode == "direct":
        base_out.update({
            "y": y0,
            "iterates": [y0],
            "residual_fields": [],
            "energy_trace": [],
            "solver_strength": 0.0,
        })
        return RMRModelOutput(**base_out)

    effective_strength = default_solver_strength if solver_strength is None else solver_strength

    y0_solver_input = y0.detach() if getattr(cfg, "detach_y0_for_solver", False) else y0

    if cfg.solver_mode == "closed_form_rn":
        trust_alpha = float(getattr(cfg, "rn_trust_alpha", 0.15))
        trust_alpha_up_val = getattr(cfg, "rn_trust_alpha_up", None)
        trust_alpha_up = float(trust_alpha_up_val) if trust_alpha_up_val is not None else None
        num_iters = max(1, int(getattr(cfg, "rn_iterations", 1)))
        y_curr = y0_solver_input
        s_accum = torch.ones_like(y0_solver_input)
        iterates = [y0]
        for _ in range(num_iters):
            y_next, s_step = closed_form_radon_nikodym_transport(
                y0=y_curr,
                b_solver=b_solver,
                weight_solver=weight_solver,
                regions=regions,
                eps=cfg.eps,
                scale_weights=scale_weights,
                trust_alpha=trust_alpha,
                trust_alpha_up=trust_alpha_up,
                return_multiplier=True,
            )
            s_accum = s_accum * s_step
            y_curr = y_next
            iterates.append(y_curr)

        y_rn = y_curr
        if effective_strength < 1.0:
            s_field = (1.0 - effective_strength) + effective_strength * s_accum
            y_final = (1.0 - effective_strength) * y0 + effective_strength * y_rn
        else:
            s_field = s_accum
            y_final = y_rn
        iterates[-1] = y_final
        base_out.update({
            "y": y_final,
            "iterates": iterates,
            "residual_fields": [],
            "energy_trace": [],
            "solver_strength": effective_strength,
            "rn_multiplier": s_field,
        })
        return RMRModelOutput(**base_out)

    if cfg.solver_mode == "mcbpro":
        deficit_seeding = bool(getattr(cfg, "mcbpro_deficit_seeding", True))
        valley_power = float(getattr(cfg, "mcbpro_valley_power", 2.0))
        reg_threshold = float(getattr(cfg, "mcbpro_reg_threshold", 0.001))
        fg_mask = torch.sigmoid(fg_logit) if fg_logit is not None else None

        y_mcbpro, s_field = closed_form_mcbpro_transport(
            y0=y0_solver_input,
            b_solver=b_solver,
            weight_solver=weight_solver,
            regions=regions,
            eps=cfg.eps,
            scale_weights=scale_weights,
            fg_mask=fg_mask,
            deficit_seeding=deficit_seeding,
            valley_power=valley_power,
            reg_threshold=reg_threshold,
            return_multiplier=True,
        )

        if effective_strength < 1.0:
            y_final = (1.0 - effective_strength) * y0 + effective_strength * y_mcbpro
        else:
            y_final = y_mcbpro

        base_out.update({
            "y": y_final,
            "iterates": [y0, y_final],
            "residual_fields": [],
            "energy_trace": [],
            "solver_strength": effective_strength,
            "rn_multiplier": s_field,
        })
        return RMRModelOutput(**base_out)

    solver_res = unrolled_sirt_solver(
        y0=y0_solver_input,
        b_solver=b_solver,
        weight_solver=weight_solver,
        regions=regions,
        iterations=cfg.iterations,
        omega=cfg.omega,
        solver_strength=effective_strength,
        residual_clip=cfg.residual_clip,
        eps=cfg.eps,
        solver_mode=cfg.solver_mode,
        density_gate_rho=cfg.density_gate_rho,
        density_gate_floor=cfg.density_gate_floor,
        proximal_tau=cfg.proximal_tau,
        proximal_mode=cfg.proximal_mode,
        proximal_mu=cfg.proximal_mu,
        tv_lambda=cfg.tv_lambda,
        tv_type=cfg.tv_type,
        tv_eps_c=cfg.tv_eps_c,
        laplace_kernel=laplace_kernel,
        scale_routing_weights=scale_weights,
        trust_region_kappa=cfg.trust_region_kappa,
        trust_region_floor=cfg.trust_region_floor,
        adjoint_mode=cfg.adjoint_mode,
        b_variance=b_variance,
        morozov_gamma=cfg.morozov_gamma,
        use_barzilai_borwein=cfg.use_barzilai_borwein,
        use_alternating_bb=cfg.use_alternating_bb,
        bb_clamp_min=cfg.bb_clamp_min,
        bb_clamp_max=cfg.bb_clamp_max,
        cyclic_bb_length=cfg.cyclic_bb_length,
        use_scale_entropy_trust=cfg.use_scale_entropy_trust,
        use_nesterov_momentum=cfg.use_nesterov_momentum,
        adaptive_relaxation=cfg.adaptive_relaxation,
        adaptive_relax_sparse=cfg.adaptive_relax_sparse,
        adaptive_relax_dense_boost=cfg.adaptive_relax_dense_boost,
        adaptive_relax_threshold=cfg.adaptive_relax_threshold,
        adaptive_relax_scale=cfg.adaptive_relax_scale,
        hybrid_recovery_alpha=cfg.hybrid_recovery_alpha,
        density_gated_diffusion=cfg.density_gated_diffusion,
        diffusion_dense_threshold=cfg.diffusion_dense_threshold,
        diffusion_gate_beta=cfg.diffusion_gate_beta,
        output_stride=2 if getattr(cfg, "subpixel_stride2", False) else (4 if getattr(cfg, "subpixel_dm", False) else int(cfg.output_stride)),
        use_anscombe=getattr(cfg, "use_anscombe_sirt", False) or (cfg.adjoint_mode == "anscombe_vst"),
        anscombe_c=getattr(cfg, "anscombe_c", 0.375),
        adaptive_tau=getattr(cfg, "adaptive_tau", False),
        adaptive_tau_rho0=getattr(cfg, "adaptive_tau_rho0", 0.05),
        anisotropic_diffusion=getattr(cfg, "anisotropic_diffusion", False),
        pm_kappa=getattr(cfg, "pm_kappa", 0.05),
        density_gated_anscombe=getattr(cfg, "density_gated_anscombe", False),
        anscombe_tau_dense=getattr(cfg, "anscombe_tau_dense", 0.08),
        area_normalized_adjoint=getattr(cfg, "area_normalized_adjoint", False),
        carrier_energy=carrier_energy,
        resonant_lambda=float(cfg.resonant_adjoint_lambda) if getattr(cfg, "resonant_adjoint", False) else 0.0,
        anscombe_morozov=getattr(cfg, "anscombe_morozov", False),
        crest_discovery_flux=getattr(cfg, "crest_discovery_flux", False),
        crest_kappa_0=getattr(cfg, "crest_kappa_0", 2.0),
        crest_eps_seed=getattr(cfg, "crest_eps_seed", 0.005),
        asymmetric_morozov=getattr(cfg, "asymmetric_morozov", False),
        morozov_gamma_under=getattr(cfg, "morozov_gamma_under", 0.20),
        morozov_rho=getattr(cfg, "morozov_rho", 0.30),
        shifted_carrier=getattr(cfg, "shifted_carrier", False),
        shifted_carrier_eps=getattr(cfg, "shifted_carrier_eps", 0.02),
        density_adaptive_trust=getattr(cfg, "density_adaptive_trust", False),
        trust_dense_tau=getattr(cfg, "trust_dense_tau", 0.10),
        trust_dense_kappa=getattr(cfg, "trust_dense_kappa", 0.80),
        asymmetric_trust=getattr(cfg, "asymmetric_trust", False),
        trust_pos_kappa=getattr(cfg, "trust_pos_kappa", 1.0),
        spatial_morozov=getattr(cfg, "spatial_morozov", False),
        morozov_gamma_scales=getattr(cfg, "morozov_gamma_scales", (0.25, 0.50, 0.75)),
        scale_routed_trust=getattr(cfg, "scale_routed_trust", False),
        scale_seeded_carrier=getattr(cfg, "scale_seeded_carrier", False),
        scale_seed_eps=getattr(cfg, "scale_seed_eps", 0.02),
        trust_region_floor_dense=getattr(cfg, "trust_region_floor_dense", 0.025),
        evidence_anchored_trust=getattr(cfg, "evidence_anchored_trust", False),
        eatr_alpha=getattr(cfg, "eatr_alpha", 0.5),
        regional_rate=regional_evidence.get("rate"),
        morozov_rho_cap=getattr(cfg, "morozov_rho_cap", 0.0),
        multiscale_morozov=getattr(cfg, "multiscale_morozov", False),
        morozov_scale_beta=getattr(cfg, "morozov_scale_beta", 1.0),
        deficit_carrier_seeding=getattr(cfg, "deficit_carrier_seeding", False),
        deficit_seed_eps=float(getattr(cfg, "deficit_seed_eps", 0.02)),
        density_adaptive_morozov=getattr(cfg, "density_adaptive_morozov", False),
        morozov_density_beta=float(getattr(cfg, "morozov_density_beta", 2.0)),
        compute_energy=compute_energy,
        omega_steps=omega_steps,
        gamma_steps=gamma_steps,
    )

    y = solver_res["y"]
    strength = solver_res["effective_omega"] / max(cfg.omega, 1e-8)

    solver_trust_alpha = None
    if trust_gate is not None:
        feat_global = p16.mean(dim=(-2, -1))
        solver_trust_alpha = torch.sigmoid(trust_gate(feat_global)).view(-1, 1, 1, 1)
        y = (1.0 - solver_trust_alpha) * y0 + solver_trust_alpha * y
        base_out["solver_trust_alpha"] = solver_trust_alpha

    base_out.update({
        "y": y,
        "iterates": solver_res["iterates"],
        "residual_fields": solver_res["residual_fields"],
        "energy_trace": solver_res["energy_trace"],
        "solver_strength": strength,
    })
    return RMRModelOutput(**base_out)
