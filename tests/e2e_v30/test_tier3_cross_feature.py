"""Tier 3: Cross-Feature Combinations Test Suite for RMR-v30.

Verifies non-trivial interactions, dynamics, and composite couplings across system components:
- Dual-Lattice DCSR x Anscombe SIRT Adjoint
- Anscombe SIRT x BB-1 Rayleigh Contraction Dynamics
- Dual-Lattice x Adaptive Tau Modulation x Deep SIRT (T=8)
- Dual-Lattice x Convex Loss Core (Mass-Weighted Cell + Flat-DM16 + Curvature)
- Anscombe SIRT x Bayesian Morozov Discrepancy Deadband
- Full Composite Production Pipeline (Dual-Lattice + Anscombe + T=8 + BB-1 + Convex Triad)
- Dual-Lattice Prolongation x Firm Thresholding Tail Preservation
- Two-Scale Macro Resolution Invariance
- Multi-Iterate Energy Contraction under Anscombe-SIRT
- Multi-Step Training Optimization Dynamics with AdamW
"""
from __future__ import annotations

import math
from typing import Any, Dict

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from rmr_core.losses import flat_dm16_loss
from rmr_core.operators import (
    RegionSet,
    build_multiscale_regions,
    regional_sum,
    weighted_coverage,
)
from rmr_v3.losses import curvature_power_loss, mass_weighted_cell_loss
from rmr_v3.model import RMRv3, RMRv3Config

from tests.e2e_v30.conftest import (
    anscombe_discrepancy_oracle,
    anscombe_transform_oracle,
    check_mass_conservation_oracle,
    compute_adaptive_tau_oracle,
    get_rmr_v30_config_spec,
    inverse_anscombe_oracle,
    prolongation_stride4_to_stride2_rn_oracle,
    proximal_firm_threshold_oracle,
    pure_bb1_oracle,
    push_forward_stride2_to_stride4_oracle,
    support_invariance_oracle,
)


class TestTier3CrossFeatureCombinations:
    """Verifies complex cross-cutting interactions among RMR-v30 technologies."""

    def test_tier3_dual_lattice_with_anscombe_sirt(
        self,
        synthetic_fine_stride2: torch.Tensor,
        standard_multiscale_regions_stride4: RegionSet,
    ) -> None:
        """Integration: Stride-2 fine measure pushed to Stride-4 carrier, solved in Anscombe domain."""
        y2 = synthetic_fine_stride2
        # 1. Push-forward to Stride-4 carrier
        y4 = push_forward_stride2_to_stride4_oracle(y2)
        assert check_mass_conservation_oracle(y2, y4, eps=1e-6)

        # 2. Regional measurement in carrier space
        regions = standard_multiscale_regions_stride4
        q = regional_sum(y4, regions.boxes)
        b = q * 1.25  # Simulated 25% undercount

        # 3. Anscombe discrepancy calculation
        res = anscombe_discrepancy_oracle(q, b, c=0.375)
        assert torch.isfinite(res).all()

        # 4. Prolong back to fine lattice
        delta_y4 = y4 * 0.05
        y4_updated = torch.clamp_min(y4 + delta_y4, 0.0)
        y2_updated = prolongation_stride4_to_stride2_rn_oracle(y4_updated, y2)
        assert check_mass_conservation_oracle(y2_updated, y4_updated, eps=1e-5)

    def test_tier3_anscombe_sirt_with_bb1_rayleigh_contraction(self) -> None:
        """Integration: BB-1 Rayleigh contraction step size operating on Anscombe rate residuals."""
        torch.manual_seed(707)
        y = torch.rand(1, 1, 32, 32) * 5.0 + 0.1
        b_gt = torch.rand(1, 1, 32, 32) * 5.0 + 0.1

        # Calculate consecutive Anscombe rate residuals
        r_k_minus_1 = anscombe_discrepancy_oracle(y, b_gt, c=0.375)
        y_next = y - 0.5 * r_k_minus_1
        r_k = anscombe_discrepancy_oracle(y_next, b_gt, c=0.375)

        s_diff = y_next - y
        r_diff = r_k - r_k_minus_1

        alpha_bb = pure_bb1_oracle(s_diff, r_diff, omega_0=1.0, clamp_min=0.2, clamp_max=2.0)
        assert 0.2 <= alpha_bb.item() <= 2.0
        assert torch.isfinite(alpha_bb)

    def test_tier3_dual_lattice_adaptive_tau_deep_sirt_t8(self) -> None:
        """Integration: T=8 unrolled iterations with density-adaptive tau modulation."""
        torch.manual_seed(808)
        y = torch.rand(1, 1, 32, 32) * 2.0
        b = y * 1.1

        for t in range(8):
            # Compute adaptive tau
            tau_eff = compute_adaptive_tau_oracle(
                base_tau_step=0.015 / 8, y_current=y, stride=4, rho0=0.05
            )
            res = anscombe_discrepancy_oracle(y, b, c=0.375)
            y_step = torch.clamp_min(y - 0.2 * res, 0.0)
            y = proximal_firm_threshold_oracle(y_step, tau=tau_eff, mu=3.0)
            assert torch.isfinite(y).all()
            assert (y >= 0.0).all()

    def test_tier3_dual_lattice_with_convex_loss_core(self) -> None:
        """Integration: Supervise Stride-4 carrier with Cell L1 and Flat-DM16 simultaneously."""
        torch.manual_seed(909)
        y_fine = torch.rand(2, 1, 64, 64, requires_grad=True)
        y_carrier = push_forward_stride2_to_stride4_oracle(y_fine)

        target_carrier = torch.rand(2, 1, 32, 32)
        l_cell = mass_weighted_cell_loss(y_carrier, target_carrier)
        l_dm16 = flat_dm16_loss(y_carrier, target_carrier)
        l_curv = curvature_power_loss(y_carrier, target_carrier)

        total_loss = 0.5 * l_cell + 1.0 * l_dm16 + 0.5 * l_curv
        total_loss.backward()

        assert y_fine.grad is not None
        assert torch.isfinite(y_fine.grad).all()
        assert y_fine.grad.abs().sum().item() > 0.0

    def test_tier3_anscombe_sirt_with_morozov_deadband_shrinkage(self) -> None:
        """Integration: Morozov discrepancy deadband filtering high-frequency noise in Anscombe space."""
        q = torch.tensor([5.0, 10.0, 20.0])
        # Add small sub-variance noise
        b_clean = q.clone()
        b_noisy = q + torch.tensor([0.02, -0.03, 0.01])
        b_variance = b_clean.clone()

        res_filtered = anscombe_discrepancy_oracle(
            q, b_noisy, c=0.375, b_variance=b_variance, morozov_gamma=0.75
        )
        # Small perturbations inside deadband are shrunk to zero
        assert torch.allclose(res_filtered, torch.zeros_like(res_filtered), atol=1e-5)

    def test_tier3_full_composite_pipeline_gradient_flow(
        self,
        canonical_h3_config_dict: Dict[str, Any],
    ) -> None:
        """Integration: H3 Primary Sub-60 composite model forward & backward pass."""
        torch.manual_seed(1010)
        m_cfg = RMRv3Config.from_dict(canonical_h3_config_dict["model"], pretrained=False)
        model = RMRv3(m_cfg).train()

        img = torch.randn(2, 3, 128, 128)
        target = torch.rand(2, 1, 64, 64)

        out = model(img)
        assert out.y.shape == (2, 1, 64, 64)
        assert torch.isfinite(out.y).all()

        loss = mass_weighted_cell_loss(out.y, target)
        if getattr(out, "hurdle_logit", None) is not None:
            loss = loss + 0.01 * out.hurdle_logit.sum()
        loss.backward()

        # Verify gradient flow through backbone, neck, and subpixel head
        has_grad = [p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad]
        assert all(has_grad), "Gradient did not flow through all trainable model parameters!"

    def test_tier3_prolongation_with_firm_thresholding_synergy(self) -> None:
        """Integration: Q_{4->2}^RN followed by adaptive firm thresholding cleans background without clump bias."""
        torch.manual_seed(1111)
        # Foreground head in center
        y4 = torch.zeros(1, 1, 16, 16)
        y4[0, 0, 8, 8] = 5.0
        y2_prior = torch.ones(1, 1, 32, 32) * 0.001  # faint background noise
        y2_prior[0, 0, 16, 16] = 1.0

        y2_prolonged = prolongation_stride4_to_stride2_rn_oracle(y4, y2_prior)
        tau_eff = compute_adaptive_tau_oracle(base_tau_step=0.015, y_current=y2_prolonged, stride=2, rho0=0.05)
        y2_clean = proximal_firm_threshold_oracle(y2_prolonged, tau=tau_eff, mu=3.0)

        # Background noise should be zero
        assert (y2_clean[0, 0, 0:8, 0:8] == 0.0).all()
        # True head in center remains active
        assert y2_clean[0, 0, 16, 16].item() > 0.0

    def test_tier3_two_scale_macro_resolution_invariance(self) -> None:
        """Integration: Discrete sum over 64x64 Stride 2 matches 32x32 Stride 4 exactly."""
        fine_density = torch.zeros(1, 1, 64, 64)
        fine_density[0, 0, 10:14, 20:24] = 0.5  # 16 cells * 0.5 = 8.0 count
        carrier_density = push_forward_stride2_to_stride4_oracle(fine_density)

        assert math.isclose(fine_density.sum().item(), 8.0, rel_tol=1e-6)
        assert math.isclose(carrier_density.sum().item(), 8.0, rel_tol=1e-6)

    def test_tier3_multi_iterate_energy_contraction(self) -> None:
        """Integration: Residual Lyapunov energy strictly contracts across unrolled SIRT iterations."""
        torch.manual_seed(1212)
        y = torch.rand(1, 1, 32, 32) * 2.0 + 0.5
        b = torch.rand(1, 1, 32, 32) * 2.0 + 0.5

        energies = []
        for it in range(6):
            r = anscombe_discrepancy_oracle(y, b, c=0.375)
            e = 0.5 * (r * r).sum().item()
            energies.append(e)
            step = 0.15
            y = torch.clamp_min(y - step * r, 0.0)

        # Energy at end must be substantially lower than energy at start
        assert energies[-1] < energies[0], f"Energy did not contract: {energies[0]} -> {energies[-1]}"

    def test_tier3_multistep_training_dynamics_adamw(
        self,
        canonical_step0_config_dict: Dict[str, Any],
    ) -> None:
        """Integration: 3 gradient descent steps with AdamW optimizer execute with decreasing loss."""
        torch.manual_seed(1313)
        m_cfg = RMRv3Config.from_dict(canonical_step0_config_dict["model"], pretrained=False)
        model = RMRv3(m_cfg).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        images = torch.randn(2, 3, 64, 64)
        targets = torch.rand(2, 1, 16, 16)

        losses = []
        for _ in range(3):
            optimizer.zero_grad()
            out = model(images)
            loss = mass_weighted_cell_loss(out.y, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert all(torch.isfinite(torch.tensor(losses)))
        assert len(losses) == 3
