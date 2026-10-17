from __future__ import annotations

import torch
import torch.nn.functional as F


def apply_trust_region_clamping(
    delta_y: torch.Tensor,
    z_state: torch.Tensor,
    trust_region_kappa: float,
    effective_trust_floor: float,
    area_scale: float,
    scale_routed_trust: bool = False,
    scale_routing_weights: torch.Tensor | None = None,
    trust_pos_kappa: float = 1.0,
    trust_region_floor_dense: float = 0.025,
    density_adaptive_trust: bool = False,
    trust_dense_tau: float = 0.10,
    trust_dense_kappa: float = 0.80,
    deficit_carrier_seeding: bool = False,
    psi_deficit: torch.Tensor | None = None,
    scale_seeded_carrier: bool = False,
    scale_seed_eps: float = 0.02,
    eatr_floor: torch.Tensor | None = None,
    asymmetric_trust: bool = False,
    scale_confidence: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply trust region bounds to displacement delta_y with scale and deficit adaptation."""
    if trust_region_kappa <= 0.0:
        return delta_y

    eff_kappa = float(trust_region_kappa)
    if scale_routed_trust and scale_routing_weights is not None:
        pi_fine = scale_routing_weights[:, 0:1, :, :].float().detach()
        eff_pos_kappa = eff_kappa + (float(trust_pos_kappa) - eff_kappa) * pi_fine
        fl_sparse = float(effective_trust_floor)
        fl_dense = float(trust_region_floor_dense) * area_scale
        eff_floor = fl_sparse + (fl_dense - fl_sparse) * pi_fine
    elif density_adaptive_trust:
        z_local = F.avg_pool2d(
            z_state.float(), kernel_size=5, stride=1, padding=2, count_include_pad=False
        )
        dense_gate = torch.sigmoid((z_local - float(trust_dense_tau)) / 0.03)
        eff_pos_kappa = eff_kappa + (float(trust_dense_kappa) - eff_kappa) * dense_gate
        eff_floor = float(effective_trust_floor)
    else:
        eff_pos_kappa = eff_kappa
        eff_floor = float(effective_trust_floor)

    base_bound = torch.clamp_min(z_state.float(), eff_floor)
    bound_removal = eff_kappa * base_bound
    if deficit_carrier_seeding and psi_deficit is not None:
        deficit_ratio = torch.clamp(
            psi_deficit.float().detach() / (base_bound + 1e-4), max=1.0
        )
        bound_removal = bound_removal * (1.0 - 0.75 * deficit_ratio)

    seed_allowance: float | torch.Tensor = 0.0
    if scale_seeded_carrier and scale_routing_weights is not None:
        seed_supp = float(scale_seed_eps) * scale_routing_weights[:, 0:1, :, :].float().detach()
        base_bound_pos = torch.clamp_min(
            torch.maximum(z_state.float(), seed_supp), eff_floor
        )
        seed_allowance = seed_supp
    else:
        base_bound_pos = base_bound

    if deficit_carrier_seeding and psi_deficit is not None:
        deficit_supp = psi_deficit.float().detach()
        base_bound_pos = torch.clamp_min(
            torch.maximum(base_bound_pos, deficit_supp), eff_floor
        )
        seed_allowance = seed_allowance + deficit_supp

    if eatr_floor is not None:
        base_bound_pos = torch.maximum(base_bound_pos, eatr_floor)

    if asymmetric_trust:
        pos_k = float(trust_pos_kappa)
        if isinstance(eff_kappa, torch.Tensor):
            eff_pos_kappa = torch.clamp_min(eff_kappa, pos_k)
        else:
            eff_pos_kappa = max(float(eff_kappa), pos_k)

    bound_addition = eff_pos_kappa * base_bound_pos + seed_allowance

    if scale_confidence is not None:
        conf_mod = 0.25 + 0.75 * scale_confidence
        bound_removal = bound_removal * conf_mod
        bound_addition = bound_addition * conf_mod

    return torch.clamp(delta_y, min=-bound_removal, max=bound_addition)
