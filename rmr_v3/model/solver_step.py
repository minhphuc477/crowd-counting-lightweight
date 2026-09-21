from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from rmr_core.operators import RegionSet
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

    if not cfg.enable_solver:
        base_out.update({
            "y": y0,
            "iterates": [y0],
            "residual_fields": [],
            "energy_trace": [],
            "solver_strength": 0.0,
        })
        return RMRModelOutput(**base_out)

    effective_strength = default_solver_strength if solver_strength is None else solver_strength

    solver_res = unrolled_sirt_solver(
        y0=y0,
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
        output_stride=cfg.output_stride,
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
        resonant_lambda=getattr(cfg, "resonant_adjoint_lambda", 0.5),
        anscombe_morozov=getattr(cfg, "anscombe_morozov", False),
        crest_discovery_flux=getattr(cfg, "crest_discovery_flux", False),
        crest_kappa_0=getattr(cfg, "crest_kappa_0", 2.0),
        crest_eps_seed=getattr(cfg, "crest_eps_seed", 0.005),
        asymmetric_morozov=getattr(cfg, "asymmetric_morozov", False),
        morozov_gamma_under=getattr(cfg, "morozov_gamma_under", 0.20),
        morozov_rho=getattr(cfg, "morozov_rho", 0.30),
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
