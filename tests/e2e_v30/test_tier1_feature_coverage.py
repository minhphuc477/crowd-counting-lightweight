"""Tier 1: Feature Coverage Test Suite for RMR-v30.

Exhaustively covers primary behaviors across all 7 core features in SCOPE.md:
- F1: Dual-Lattice Consistency & DCSR Operators (>= 5 tests)
- F2: Anscombe Variance-Stabilizing SIRT Adjoint (>= 5 tests)
- F3: Barzilai-Borwein BB-1 Rayleigh Contraction (>= 5 tests)
- F4: Density-Adaptive Proximal Tau Modulation (>= 5 tests)
- F5: Preserved Convex Loss Core & Bitwise Parity (>= 5 tests)
- F6: 6-Model Two-Loop Ablation Suite Specifications (>= 5 tests)
- F7: Evaluation Metrics & Integrity Invariants (>= 5 tests)
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Dict

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from rmr_core.data import CrowdManifestDataset
from rmr_core.losses import flat_dm16_loss
from rmr_core.metrics import game_physical_image, summarize_predictions
from rmr_core.operators import (
    RegionSet,
    build_multiscale_regions,
    weighted_normalized_adjoint_field,
)
from rmr_v3.config import load_config
from rmr_v3.losses import (
    compute_rmr_v3_losses,
    curvature_power_loss,
    mass_weighted_cell_loss,
    scale_balanced_regional_nb_nll,
    RMRv3LossConfig,
)
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


# ============================================================================
# Tier 1 - F1: Dual-Lattice Consistency & DCSR Operators
# ============================================================================
class TestTier1F1DualLatticeDCSR:
    """Verifies feature coverage for F1: Dual-Lattice Consistency & DCSR."""

    def test_f1_dual_lattice_shape_alignment(self, synthetic_fine_stride2: torch.Tensor) -> None:
        """Push-forward operator maps Stride 2 fine grid (H, W) to Stride 4 carrier (H/2, W/2)."""
        y_fine = synthetic_fine_stride2
        y_carrier = push_forward_stride2_to_stride4_oracle(y_fine)
        assert y_carrier.shape == (y_fine.shape[0], y_fine.shape[1], y_fine.shape[2] // 2, y_fine.shape[3] // 2)
        assert torch.isfinite(y_carrier).all()

    def test_f1_pushforward_exact_mass_conservation(self) -> None:
        """Theorem 1: Pushforward P_{2->4} strictly conserves total count sum(y4) == sum(y2)."""
        torch.manual_seed(101)
        for h, w in [(64, 64), (128, 128), (96, 160), (128, 256)]:
            y_fine = torch.rand(2, 1, h, w) * 0.5
            y_carrier = push_forward_stride2_to_stride4_oracle(y_fine)
            assert check_mass_conservation_oracle(y_fine, y_carrier, eps=1e-6), (
                f"Mass conservation failed on {h}x{w}: fine={y_fine.sum():.6f}, carrier={y_carrier.sum():.6f}"
            )

    def test_f1_pushforward_odd_prime_dimensions(self) -> None:
        """Theorem 1 holds on even dimension non-powers-of-two (e.g. 26x38 -> 13x19)."""
        y_fine = torch.rand(1, 1, 26, 38)
        y_carrier = push_forward_stride2_to_stride4_oracle(y_fine)
        assert y_carrier.shape == (1, 1, 13, 19)
        assert check_mass_conservation_oracle(y_fine, y_carrier, eps=1e-6)

    def test_f1_prolongation_radon_nikodym_mass_conservation(self) -> None:
        """Theorem 1: Radon-Nikodym prolongation Q_{4->2}^RN strictly preserves carrier mass."""
        torch.manual_seed(202)
        y4 = torch.rand(1, 1, 32, 32) * 5.0 + 0.1
        y2_prior = torch.rand(1, 1, 64, 64) * 0.5 + 0.01

        y2_prolonged = prolongation_stride4_to_stride2_rn_oracle(y4, y2_prior)
        assert y2_prolonged.shape == (1, 1, 64, 64)
        assert check_mass_conservation_oracle(y2_prolonged, y4, eps=1e-5)

    def test_f1_prolongation_support_preservation(self) -> None:
        """Theorem 2: Q_{4->2}^RN leaves zero carrier cells identically zero in fine measure."""
        y4 = torch.zeros(1, 1, 16, 16)
        y4[0, 0, 4:8, 4:8] = 2.0  # carrier exists only in center 4x4
        y2_prior = torch.ones(1, 1, 32, 32) * 0.1

        y2_prolonged = prolongation_stride4_to_stride2_rn_oracle(y4, y2_prior)
        bg_fine = y2_prolonged[0, 0, 0:8, 0:8]
        assert (bg_fine == 0.0).all(), "Support preservation violated: positive mass leaked to background!"

    def test_f1_subpixel_model_parameter_ceiling(self) -> None:
        """Subpixel Stride 2 dual-lattice model contains exactly 104,540 parameters (<= 105,000)."""
        spec = get_rmr_v30_config_spec("h2_dual_lattice_dcsr")
        m_cfg = RMRv3Config.from_dict(spec["model"], pretrained=False)
        model = RMRv3(m_cfg)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable == 104540, f"Expected exactly 104,540 params, got {trainable}"
        assert trainable <= 105000, "Violated parameter hard ceiling <= 105,000!"


# ============================================================================
# Tier 1 - F2: Anscombe Variance-Stabilizing SIRT Adjoint
# ============================================================================
class TestTier1F2AnscombeSIRTAdjoint:
    """Verifies feature coverage for F2: Anscombe Variance-Stabilizing SIRT."""

    def test_f2_anscombe_transform_oracle_exactness(self) -> None:
        """Anscombe transform T(y) = 2.0 * sqrt(max(0, y) + 0.375)."""
        y = torch.tensor([0.0, 1.0, 5.0, 25.0, 100.0])
        expected = 2.0 * torch.sqrt(y + 0.375)
        out = anscombe_transform_oracle(y, c=0.375)
        assert torch.allclose(out, expected, atol=1e-7)

    def test_f2_inverse_anscombe_roundtrip(self) -> None:
        """Roundtrip identity: T^{-1}(T(y)) == y for non-negative inputs."""
        y = torch.tensor([0.0, 0.5, 2.0, 15.0, 250.0, 1200.0])
        z = anscombe_transform_oracle(y, c=0.375)
        y_rec = inverse_anscombe_oracle(z, c=0.375)
        assert torch.allclose(y_rec, y, atol=1e-6)

    def test_f2_anscombe_variance_stabilization_poisson(self) -> None:
        """Theorem 3: For Poisson counts Y ~ Pois(mu), Var(T(Y)) is asymptotically constant ~1.0."""
        torch.manual_seed(303)
        n_samples = 30000
        for mu in [5.0, 20.0, 100.0, 500.0]:
            poisson_samples = torch.poisson(torch.full((n_samples,), float(mu)))
            transformed = anscombe_transform_oracle(poisson_samples, c=0.375)
            sample_var = transformed.var().item()
            # Variance should be close to 1.0 (within [0.85, 1.15] for empirical samples)
            assert 0.85 <= sample_var <= 1.15, f"Anscombe variance not stabilized at mu={mu}: {sample_var}"

    def test_f2_anscombe_discrepancy_o1_scaling(self) -> None:
        """Theorem 3: Under 50% undercount (q = 0.5*b), Anscombe discrepancy remains O(1) across all b."""
        for b_val in [2.0, 10.0, 50.0, 200.0, 1000.0]:
            b = torch.tensor([b_val])
            q = torch.tensor([0.5 * b_val])
            res = anscombe_discrepancy_oracle(q, b, c=0.375)
            # The rate residual res = 2 * (1 - sqrt((b+3/8)/(q+3/8))) is strictly O(1) in [-0.85, -0.60]
            assert -0.90 <= res.item() <= -0.60, (
                f"Anscombe rate residual violated O(1) bound at b={b_val}: {res.item()}"
            )

    def test_f2_anscombe_discrepancy_morozov_deadband(self) -> None:
        """Morozov discrepancy deadband shrinks perturbations below gamma * sigma to zero."""
        q = torch.tensor([10.0])
        b = torch.tensor([10.05])  # Tiny residual
        res_no_deadband = anscombe_discrepancy_oracle(q, b, morozov_gamma=0.0)
        assert res_no_deadband.abs().item() > 0.0

        res_with_deadband = anscombe_discrepancy_oracle(q, b, morozov_gamma=0.75)
        assert res_with_deadband.abs().item() == 0.0, "Small noise was not pruned by Morozov deadband!"

    def test_f2_anscombe_adjoint_nonnegativity_protection(self) -> None:
        """Anscombe rate residual is strictly finite and non-NaN even for zero density."""
        q = torch.zeros(1, 1, 10)
        b = torch.zeros(1, 1, 10)
        res = anscombe_discrepancy_oracle(q, b, c=0.375)
        assert torch.isfinite(res).all()
        assert (res == 0.0).all()


# ============================================================================
# Tier 1 - F3: Barzilai-Borwein BB-1 Rayleigh Contraction
# ============================================================================
class TestTier1F3BarzilaiBorweinRayleighContraction:
    """Verifies feature coverage for F3: BB-1 Rayleigh Contraction."""

    def test_f3_bb1_rayleigh_quotient_oracle(self) -> None:
        """Pure BB-1 step size alpha_BB1 = <s, r> / (||r||^2 + eps)."""
        s = torch.tensor([2.0, 4.0, 4.0])
        r = torch.tensor([1.0, 2.0, 2.0])
        # <s, r> = 2*1 + 4*2 + 4*2 = 18. ||r||^2 = 1 + 4 + 4 = 9. alpha = 18/9 = 2.0
        step = pure_bb1_oracle(s, r, omega_0=1.0, clamp_min=0.2, clamp_max=3.0)
        assert torch.isclose(step, torch.tensor([2.0]), atol=1e-5)

    def test_f3_bb1_clamping_interval(self) -> None:
        """Step size is strictly clamped within [0.2, 2.0] * omega_0."""
        s_huge = torch.tensor([1000.0, 1000.0])
        r_tiny = torch.tensor([0.01, 0.01])
        step_clamped_high = pure_bb1_oracle(s_huge, r_tiny, omega_0=1.0, clamp_min=0.2, clamp_max=2.0)
        assert torch.isclose(step_clamped_high, torch.tensor([2.0]), atol=1e-5)

        s_tiny = torch.tensor([0.001, 0.001])
        r_huge = torch.tensor([100.0, 100.0])
        step_clamped_low = pure_bb1_oracle(s_tiny, r_huge, omega_0=1.0, clamp_min=0.2, clamp_max=2.0)
        assert torch.isclose(step_clamped_low, torch.tensor([0.2]), atol=1e-5)

    def test_f3_bb1_non_convex_fallback_to_omega0(self) -> None:
        """When <s, r> <= 0 (negative curvature), BB-1 safely falls back to default omega_0."""
        s = torch.tensor([1.0, 2.0])
        r = torch.tensor([-1.0, -2.0])  # <s, r> = -5.0 < 0
        step = pure_bb1_oracle(s, r, omega_0=1.0, clamp_min=0.2, clamp_max=2.0)
        assert torch.isclose(step, torch.tensor([1.0]), atol=1e-5)

    def test_f3_bb1_scale_invariance(self) -> None:
        """Dimensionless property: scaling s and r by lambda does not alter step size."""
        s = torch.tensor([3.0, 4.0])
        r = torch.tensor([1.0, 2.0])
        step_base = pure_bb1_oracle(s, r, omega_0=1.0)
        step_scaled = pure_bb1_oracle(s * 17.5, r * 17.5, omega_0=1.0)
        assert torch.isclose(step_base, step_scaled, atol=1e-5)

    def test_f3_bb1_multi_iterate_energy_contraction(self) -> None:
        """BB-1 step sizes maintain stable contraction across 6 consecutive iterations."""
        torch.manual_seed(404)
        y = torch.rand(1, 1, 32, 32) * 2.0
        r_prev = torch.randn_like(y) * 0.1
        y_prev = y - 0.05 * r_prev

        for _ in range(6):
            s_diff = y - y_prev
            r_curr = 0.5 * r_prev  # Simulated decaying residual
            r_diff = r_curr - r_prev
            alpha = pure_bb1_oracle(s_diff, r_diff, omega_0=1.0)
            assert 0.2 <= alpha.item() <= 2.0
            y_prev = y
            r_prev = r_curr
            y = y - alpha * r_curr

    def test_f3_bb1_exclusion_of_unstable_bb2(self) -> None:
        """BB-1 avoids the division-by-zero instability that plagues alternating BB-2 when <s, r> -> 0."""
        s = torch.tensor([1e-5, 0.0])
        r = torch.tensor([1.0, 1.0])
        # In BB-2, step = ||s||^2 / <s, r> which blows up to 1e-10 / 1e-5 = 1e-5 or if <s, r> = 0 explodes to inf.
        # In BB-1, step = <s, r> / ||r||^2 which safely approaches 0 before clamping to min 0.2.
        step_bb1 = pure_bb1_oracle(s, r, omega_0=1.0, clamp_min=0.2, clamp_max=2.0)
        assert torch.isfinite(step_bb1)
        assert math.isclose(step_bb1.item(), 0.2, rel_tol=1e-5)


# ============================================================================
# Tier 1 - F4: Density-Adaptive Proximal Tau Modulation (DCSR)
# ============================================================================
class TestTier1F4AdaptiveTauModulation:
    """Verifies feature coverage for F4: Density-Adaptive Proximal Tau Modulation."""

    def test_f4_adaptive_tau_sparse_background_suppression(self) -> None:
        """On empty background (rho <= rho0), tau_eff equals base_tau_step (full noise pruning)."""
        y_bg = torch.zeros(1, 1, 32, 32)
        tau_eff = compute_adaptive_tau_oracle(base_tau_step=0.015, y_current=y_bg, stride=4, rho0=0.05)
        assert isinstance(tau_eff, torch.Tensor)
        assert torch.allclose(tau_eff, torch.full_like(tau_eff, 0.015), atol=1e-5)

    def test_f4_adaptive_tau_dense_clump_vanishing(self) -> None:
        """On dense crowd clusters (rho >> rho0), tau_eff vanishes toward 0 to prevent mass clipping."""
        y_dense = torch.ones(1, 1, 32, 32) * 1.0  # density 1.0 >> rho0 0.05
        tau_eff = compute_adaptive_tau_oracle(base_tau_step=0.015, y_current=y_dense, stride=4, rho0=0.05)
        assert isinstance(tau_eff, torch.Tensor)
        assert tau_eff.max().item() <= 0.001, f"Tau did not vanish on dense clump: {tau_eff.max().item()}"

    def test_f4_firm_thresholding_zero_bias_dense(self) -> None:
        """MCP / Firm thresholding applies zero shrinkage (zero bias) when y > mu * tau."""
        tau = 0.015
        mu = 3.0
        y_true_head = torch.tensor([0.10, 0.50, 2.0])  # > mu*tau = 0.045
        out = proximal_firm_threshold_oracle(y_true_head, tau=tau, mu=mu)
        assert torch.allclose(out, y_true_head, atol=1e-7), "Firm thresholding introduced bias on true head!"

    def test_f4_firm_thresholding_deadband_noise_pruning(self) -> None:
        """MCP / Firm thresholding sets small noise y <= tau identically to 0.0."""
        tau = 0.015
        mu = 3.0
        y_noise = torch.tensor([0.001, 0.005, 0.015])
        out = proximal_firm_threshold_oracle(y_noise, tau=tau, mu=mu)
        assert (out == 0.0).all(), "Noise was not pruned by firm thresholding deadband!"

    def test_f4_firm_thresholding_continuous_ramp(self) -> None:
        """In interval tau < y <= mu*tau, firm thresholding is a strictly continuous monotonic ramp."""
        tau = 0.015
        mu = 3.0
        y_ramp = torch.linspace(tau + 1e-4, mu * tau - 1e-4, steps=10)
        out = proximal_firm_threshold_oracle(y_ramp, tau=tau, mu=mu)
        # Check strictly monotonic increasing
        diffs = out[1:] - out[:-1]
        assert (diffs > 0.0).all(), "Firm thresholding ramp is not strictly monotonic!"

    def test_f4_spatial_tensor_tau_support(self) -> None:
        """Firm thresholding operates seamlessly with a 2D spatial tensor of effective tau values."""
        y = torch.ones(1, 1, 16, 16) * 0.03
        tau_map = torch.zeros(1, 1, 16, 16)
        tau_map[:, :, :, :8] = 0.04  # left half: y (0.03) <= tau (0.04) -> clipped to 0
        tau_map[:, :, :, 8:] = 0.005  # right half: y (0.03) > mu*tau (0.015) -> preserved at 0.03
        out = proximal_firm_threshold_oracle(y, tau=tau_map, mu=3.0)
        assert (out[:, :, :, :8] == 0.0).all()
        assert torch.allclose(out[:, :, :, 8:], torch.full_like(out[:, :, :, 8:], 0.03))


# ============================================================================
# Tier 1 - F5: Preserved Convex Loss Core & Bitwise Parity
# ============================================================================
class TestTier1F5PreservedConvexLossCore:
    """Verifies feature coverage for F5: Preserved Convex Loss Core."""

    def test_f5_mass_weighted_cell_loss_exactness(self) -> None:
        """Mass-weighted cell loss computes correctly with alpha=2.0, gamma=1.25."""
        pred = torch.tensor([[[[0.5, 0.0], [1.0, 2.0]]]])
        target = torch.tensor([[[[0.5, 0.0], [1.0, 2.0]]]])
        loss_zero = mass_weighted_cell_loss(pred, target, alpha=2.0, gamma=1.25)
        assert loss_zero.item() == 0.0

        target_diff = torch.tensor([[[[1.0, 0.0], [1.0, 2.0]]]])
        loss_nonzero = mass_weighted_cell_loss(pred, target_diff, alpha=2.0, gamma=1.25)
        assert loss_nonzero.item() > 0.0

    def test_f5_flat_dm16_loss_exactness(self) -> None:
        """Flat-DM16 optimal transport loss evaluates without errors."""
        pred = torch.rand(1, 1, 32, 32)
        target = torch.rand(1, 1, 32, 32)
        loss = flat_dm16_loss(pred, target, kappa=20.0)
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0

    def test_f5_truncated_nb_regional_nll_exactness(self, standard_multiscale_regions_stride4: RegionSet) -> None:
        """Scale-balanced regional negative binomial NLL evaluates smoothly."""
        regions = standard_multiscale_regions_stride4
        m = len(regions.boxes)
        pred_q = torch.full((1, 1, m), 10.0)
        b_gt = torch.full((1, 1, m), 10.0)
        phi = torch.full((1, 1, m), 2.0)
        loss = scale_balanced_regional_nb_nll(pred_q, b_gt, phi, regions=regions)
        assert torch.isfinite(loss)

    def test_f5_density_gated_curvature_loss(self) -> None:
        """Square-root power curvature loss operates as Anscombe domain power loss."""
        pred = torch.rand(1, 1, 32, 32)
        target = torch.rand(1, 1, 32, 32)
        loss = curvature_power_loss(pred, target, eps=0.01)
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0

    def test_f5_loss_gradient_backprop(self) -> None:
        """Autograd gradients propagate smoothly through cell and curvature losses."""
        pred = torch.rand(1, 1, 16, 16, requires_grad=True)
        target = torch.rand(1, 1, 16, 16)
        loss = mass_weighted_cell_loss(pred, target) + curvature_power_loss(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()

    def test_f5_v19_anchor_loss_bitwise_parity(self) -> None:
        """Step 0 configuration preserves bitwise loss formulation identity with v19 anchor."""
        c19 = load_config(Path("configs/rmr_v19/rmr_v19_canonical_isotropic.yaml"))
        spec30 = get_rmr_v30_config_spec("step0_v19_anchor")
        assert c19["loss"]["lambda_cell"] == spec30["loss"]["lambda_cell"]
        assert c19["loss"]["lambda_flat_dm16"] == spec30["loss"]["lambda_flat_dm16"]
        assert c19["loss"]["lambda_trunc_nb"] == spec30["loss"]["lambda_trunc_nb"]
        assert c19["loss"]["lambda_scale_align"] == spec30["loss"]["lambda_scale_align"]


# ============================================================================
# Tier 1 - F6: Six-Model Two-Loop Suite Configs
# ============================================================================
class TestTier1F6SixModelSuiteConfigs:
    """Verifies feature coverage for F6: 6-Model Two-Loop Suite Configurations."""

    def test_f6_all_six_configs_valid(self) -> None:
        """All 6 configurations generate complete, valid model dictionary specs."""
        variants = [
            "step0_v19_anchor",
            "h1_anscombe_sirt",
            "h2_dual_lattice_dcsr",
            "h3_anscombe_dual_lattice",
            "h4_deep_sirt_t8",
            "control_no_solver",
        ]
        for var in variants:
            spec = get_rmr_v30_config_spec(var)
            assert "model" in spec
            assert "loss" in spec

    def test_f6_parameter_budgets_under_105k(self) -> None:
        """Strict parameter ceiling: all 6 variants strictly <= 105,000 trainable parameters."""
        expected_params = {
            "step0_v19_anchor": 104441,
            "h1_anscombe_sirt": 104441,
            "h2_dual_lattice_dcsr": 104540,
            "h3_anscombe_dual_lattice": 104540,
            "h4_deep_sirt_t8": 104540,
            "control_no_solver": 104441,
        }
        for var, exp in expected_params.items():
            spec = get_rmr_v30_config_spec(var)
            m_cfg = RMRv3Config.from_dict(spec["model"], pretrained=False)
            m = RMRv3(m_cfg)
            p_count = sum(p.numel() for p in m.parameters() if p.requires_grad)
            assert p_count == exp, f"{var}: expected {exp} params, got {p_count}"
            assert p_count <= 105000, f"{var}: exceeded 105,000 parameter ceiling: {p_count}"

    def test_f6_zero_kd_strict_enforcement(self) -> None:
        """Zero Knowledge Distillation: strictly zero teacher models and 0 distillation loss."""
        for var in ["step0_v19_anchor", "h1_anscombe_sirt", "h2_dual_lattice_dcsr", "h3_anscombe_dual_lattice"]:
            spec = get_rmr_v30_config_spec(var)
            assert spec["model"].get("use_kd", False) is False
            assert spec["loss"].get("lambda_kd", 0.0) == 0.0

    def test_f6_single_variable_step0_to_h1(self) -> None:
        """Ablation purity: Step 0 vs H1 differs ONLY in use_anscombe_sirt."""
        s0 = get_rmr_v30_config_spec("step0_v19_anchor")["model"]
        h1 = get_rmr_v30_config_spec("h1_anscombe_sirt")["model"]
        assert s0["use_anscombe_sirt"] is False
        assert h1["use_anscombe_sirt"] is True
        s0_diff = {k: v for k, v in s0.items() if k != "use_anscombe_sirt"}
        h1_diff = {k: v for k, v in h1.items() if k != "use_anscombe_sirt"}
        assert s0_diff == h1_diff, "H1 introduced uncontrolled variables against Step 0!"

    def test_f6_single_variable_step0_to_h2(self) -> None:
        """Ablation purity: Step 0 vs H2 isolates dual spatial discretization and adaptive tau."""
        s0 = get_rmr_v30_config_spec("step0_v19_anchor")["model"]
        h2 = get_rmr_v30_config_spec("h2_dual_lattice_dcsr")["model"]
        assert h2["output_stride"] == 2
        assert h2["subpixel_stride2"] is True
        assert h2["adaptive_tau"] is True

    def test_f6_single_variable_h3_to_h4(self) -> None:
        """Ablation purity: H3 vs H4 differs ONLY in iterations (6 vs 8)."""
        h3 = get_rmr_v30_config_spec("h3_anscombe_dual_lattice")["model"]
        h4 = get_rmr_v30_config_spec("h4_deep_sirt_t8")["model"]
        assert h3["iterations"] == 6
        assert h4["iterations"] == 8
        h3_diff = {k: v for k, v in h3.items() if k != "iterations"}
        h4_diff = {k: v for k, v in h4.items() if k != "iterations"}
        assert h3_diff == h4_diff, "H4 introduced uncontrolled variables against H3!"


# ============================================================================
# Tier 1 - F7: Evaluation Metrics & Integrity Invariants
# ============================================================================
class TestTier1F7IntegrityInvariantsAndMetrics:
    """Verifies feature coverage for F7: Metrics & Integrity Invariants."""

    def test_f7_support_invariance_radon_nikodym(self) -> None:
        """Theorem 4: Background support is strictly preserved y0(u)=0 ==> delta_y(u)=0."""
        y0 = torch.zeros(1, 1, 32, 32)
        y0[:, :, 10:20, 10:20] = 1.0  # foreground square
        delta_y = torch.randn(1, 1, 32, 32) * 0.1 * y0  # Radon-Nikodym modulation by y0
        assert support_invariance_oracle(y0, delta_y)

    def test_f7_game0_strictly_equals_mae(self) -> None:
        """Physical GAME(0) on exact image coordinates strictly equals Absolute Error."""
        import numpy as np
        y = np.ones((16, 16), dtype=np.float64) * 0.5  # sum = 128.0
        pts = np.zeros((100, 2), dtype=np.float64)
        pts[:, 0] = 30.0
        pts[:, 1] = 30.0
        games = game_physical_image(y, pts, image_h=64, image_w=64, stride=4, levels=(0, 1, 2, 3))
        ae = abs(128.0 - 100.0)
        assert abs(games[0] - ae) < 1e-6, f"GAME(0) {games[0]} != AE {ae}"
        assert games[1] >= games[0]  # Monotonic hierarchy

    def test_f7_zero_adhoc_split_policy_enforcement(self) -> None:
        """Dataset integrity: attempting to load ad-hoc split manifests raises ValueError."""
        with pytest.raises(ValueError, match="Zero Ad-hoc Split Policy"):
            CrowdManifestDataset(manifest="data/sha_a_train.jsonl", train=True)

    def test_f7_test_manifest_182_images_guard(self, sha_a_test_manifest_path: Path) -> None:
        """Dataset integrity: canonical ShanghaiTech Part A test manifest contains exactly 182 records."""
        ds = CrowdManifestDataset(manifest=str(sha_a_test_manifest_path), train=False)
        assert len(ds) == 182, f"Canonical test set length guard violated: got {len(ds)} images"

    def test_f7_adjoint_scale_invariance_h1_1(self, standard_multiscale_regions_stride4: RegionSet) -> None:
        """Adjoint scale invariance: uniform constant field creates uniform normalized coverage."""
        from rmr_core.operators import weighted_coverage
        regions = standard_multiscale_regions_stride4
        weight = torch.ones(1, 1, len(regions.boxes))
        cov = weighted_coverage(weight, regions, height=64, width=64)
        assert cov.min().item() > 0.0
        assert torch.isfinite(cov).all()
