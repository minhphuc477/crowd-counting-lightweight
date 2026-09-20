"""Tests for RMR-v30 Milestone 1: Dual-Lattice DCSR & Anscombe SIRT Architecture."""
from __future__ import annotations

import math
import pytest
import torch
import torch.nn.functional as F

from rmr_v3.solver_ops import (
    anscombe_transform,
    anscombe_discrepancy,
    compute_adaptive_tau,
    proximal_firm_threshold,
)
from rmr_v3.model.dual_lattice import (
    push_forward_stride2_to_stride4,
    pullback_stride4_to_stride2_rn,
    check_mass_conservation,
)
from rmr_v3.losses.dual_supervision import compute_dual_lattice_losses
from rmr_v3.solver import unrolled_sirt_solver
from rmr_v3.model import RMRv3, RMRv3Config
from rmr_core.operators import build_multiscale_regions


class TestAnscombeOperators:
    """Mathematical verification of Anscombe VST and Discrepancy."""

    def test_anscombe_transform_exactness(self):
        """T(y) = 2 * sqrt(max(0, y) + 3/8)"""
        y = torch.tensor([0.0, 1.0, 9.625, -5.0])
        t = anscombe_transform(y, c=0.375)
        expected = torch.tensor([
            2.0 * math.sqrt(0.375),
            2.0 * math.sqrt(1.375),
            2.0 * math.sqrt(10.0),
            2.0 * math.sqrt(0.375),  # clamped to 0
        ])
        assert torch.allclose(t, expected, atol=1e-6)

    def test_anscombe_discrepancy_bounded_dynamics(self):
        """Theorem 3: Anscombe discrepancy remains O(1) across dense and sparse regimes."""
        # 50% undercount in sparse regime (q=1, b=2)
        q_sparse = torch.tensor([1.0])
        b_sparse = torch.tensor([2.0])
        r_sparse = anscombe_discrepancy(q_sparse, b_sparse, c=0.375)

        # 50% undercount in dense regime (q=250, b=500)
        q_dense = torch.tensor([250.0])
        b_dense = torch.tensor([500.0])
        r_dense = anscombe_discrepancy(q_dense, b_dense, c=0.375)

        # In dense regime, r_dense = 2 * (1 - sqrt(500.375 / 250.375)) ≈ 2 * (1 - 1.414) ≈ -0.828
        assert torch.isfinite(r_dense).all()
        assert torch.isfinite(r_sparse).all()
        assert -1.5 < r_dense.item() < -0.5
        assert -1.5 < r_sparse.item() < -0.5

    def test_anscombe_discrepancy_morozov_deadband(self):
        """Discrepancy within noise deadband is shrunk to zero."""
        q = torch.tensor([10.0])
        b = torch.tensor([10.1])  # discrepancy is very small
        b_var = torch.tensor([10.0])
        # With high Morozov threshold, small noise is completely killed
        r = anscombe_discrepancy(q, b, c=0.375, b_variance=b_var, morozov_gamma=2.0)
        assert r.item() == 0.0


class TestDensityAdaptiveTau:
    """Verification of spatially density-conditioned proximal thresholding."""

    def test_adaptive_tau_behavior(self):
        """tau_eff vanishes in dense clumps and remains active on background."""
        base_tau = 0.015
        y = torch.zeros(1, 1, 32, 32)
        # Place a dense clump in the center (density 0.50 per cell)
        y[0, 0, 12:20, 12:20] = 0.50

        tau_map = compute_adaptive_tau(
            base_tau, y, stride=4, mode="density_adaptive", rho0=0.05, pool_kernel=5
        )
        assert isinstance(tau_map, torch.Tensor)
        assert tau_map.shape == (1, 1, 32, 32)

        # Background should have full threshold
        bg_tau = tau_map[0, 0, 0, 0].item()
        assert math.isclose(bg_tau, base_tau, rel_tol=1e-5)

        # Center dense clump should have heavily attenuated threshold
        center_tau = tau_map[0, 0, 16, 16].item()
        assert center_tau < base_tau * 0.25  # at least 4x reduction

    def test_proximal_firm_threshold_with_spatial_tensor(self):
        """Firm thresholding handles spatial tensor tau and preserves peaks."""
        y = torch.tensor([[[[0.005, 0.020, 0.200]]]])  # noise, transition, true peak
        tau = torch.tensor([[[[0.010, 0.010, 0.010]]]])  # tau=0.010, mu=3.0 -> mu*tau=0.030

        out = proximal_firm_threshold(y, tau=tau, mu=3.0)
        # y=0.005 <= tau -> exactly 0.0
        assert out[0, 0, 0, 0].item() == 0.0
        # y=0.200 > mu*tau -> exactly 0.200 (ZERO shrinkage)
        assert math.isclose(out[0, 0, 0, 2].item(), 0.200, rel_tol=1e-5)
        # y=0.020 in ramp (tau < y <= mu*tau): slope = 3/2 = 1.5, ramp = 1.5 * (0.02 - 0.01) = 0.015
        assert math.isclose(out[0, 0, 0, 1].item(), 0.015, rel_tol=1e-5)


class TestDualLatticeOperators:
    """Theorems 1 & 2: Push-forward, Prolongation, and Mass Conservation."""

    def test_push_forward_exact_mass_conservation(self):
        """Theorem 1: sum(y4) == sum(y2) to machine precision."""
        torch.manual_seed(42)
        for h, w in [(64, 64), (128, 128), (96, 160)]:
            y2 = torch.rand(2, 1, h, w) * 5.0
            y4 = push_forward_stride2_to_stride4(y2)
            assert y4.shape == (2, 1, h // 2, w // 2)
            assert check_mass_conservation(y2, y4, eps=1e-6)

    def test_pullback_prolongation_support_preservation(self):
        """Theorem 2: y4=0 implies prolonged y2=0 unconditionally."""
        y4 = torch.zeros(1, 1, 16, 16)
        y4[0, 0, 4:12, 4:12] = 1.0  # carrier only in center
        y2_prior = torch.ones(1, 1, 32, 32) * 0.1

        y2_prolong = pullback_stride4_to_stride2_rn(y4, y2_prior)
        assert y2_prolong.shape == (1, 1, 32, 32)
        # Corner of coarse grid is 0 -> fine corner MUST be 0
        assert (y2_prolong[0, 0, 0:4, 0:4] == 0.0).all()
        # Mass conservation
        assert check_mass_conservation(y2_prolong, y4, eps=1e-5)

    def test_push_forward_and_pullback_odd_dimensions_mass_conservation(self):
        """Boundary zero-padding guarantees exact mass conservation on odd dimensions."""
        torch.manual_seed(42)
        for h, w in [(65, 65), (251, 352), (127, 201), (501, 703)]:
            y2 = torch.rand(2, 1, h, w).abs() * 3.0
            y4 = push_forward_stride2_to_stride4(y2)
            expected_h4 = (h + 1) // 2
            expected_w4 = (w + 1) // 2
            assert y4.shape == (2, 1, expected_h4, expected_w4)
            assert check_mass_conservation(y2, y4, eps=1e-5)

            # Prolongation test
            prolong = pullback_stride4_to_stride2_rn(y4, y2)
            assert prolong.shape == (2, 1, h, w)
            assert check_mass_conservation(prolong, y4, eps=1e-5)

    def test_dual_lattice_losses(self):
        """Dual lattice supervision computes finite loss and propagates gradients."""
        y_fine = torch.rand(2, 1, 64, 64, requires_grad=True)
        y_carrier = push_forward_stride2_to_stride4(y_fine)
        t_fine = torch.zeros(2, 1, 64, 64)
        t_fine[0, 0, 10, 10] = 1.0
        t_carrier = push_forward_stride2_to_stride4(t_fine)

        losses = compute_dual_lattice_losses(y_fine, y_carrier, t_fine, t_carrier)
        assert "cell_carrier" in losses
        assert "cell_fine" in losses
        assert "cell_combined" in losses
        assert torch.isfinite(losses["cell_combined"])

        losses["cell_combined"].backward()
        assert y_fine.grad is not None
        assert torch.isfinite(y_fine.grad).all()


class TestSolverAnscombeSupportInvariance:
    """Theorem 4: Background support preservation y0(u)=0 => yt(u)=0."""

    def test_support_invariance_radon_nikodym_anscombe(self):
        """Unrolled SIRT with Anscombe discrepancy preserves true background support."""
        torch.manual_seed(42)
        h, w = 32, 32
        regions = build_multiscale_regions(h, w, output_stride=4, region_sizes_px=(32,), overlap=0.5, include_full_image=False)

        # Initial prediction has non-zero heads in center, exactly 0.0 on background
        y0 = torch.zeros(1, 1, h, w)
        y0[0, 0, 14:18, 14:18] = 0.50

        # Regional target evidence demands mass everywhere (including empty regions)
        b_solver = torch.ones(1, 1, len(regions.boxes)) * 2.0
        weight_solver = torch.ones_like(b_solver)

        res = unrolled_sirt_solver(
            y0=y0,
            b_solver=b_solver,
            weight_solver=weight_solver,
            regions=regions,
            iterations=4,
            omega=1.0,
            use_anscombe=True,
            adjoint_mode="radon_nikodym",
            proximal_tau=0.015,
            proximal_mode="firm",
        )
        y_final = res["y"]

        # All pixels that were originally 0 MUST remain identically 0.0
        bg_mask = (y0 == 0.0)
        assert (y_final[bg_mask] == 0.0).all(), "Support invariance violated: background pixels became non-zero!"


class TestModelArchitectureAndBudgets:
    """Verification of parameter budgets and model forward passes."""

    def test_parameter_budgets_strict_invariants(self):
        """Verify 104,441 params (Stride 4) and 104,540 params (Stride 2 Dual-Lattice)."""
        import yaml
        from pathlib import Path

        # Check Stride 4 Anchor
        raw_anchor = yaml.safe_load(Path("configs/rmr_v29/rmr_v29_step0_v19_anchor.yaml").read_text())["model"]
        m_anchor = RMRv3(RMRv3Config.from_dict(raw_anchor))
        p_anchor = sum(p.numel() for p in m_anchor.parameters() if p.requires_grad)
        assert p_anchor == 104441, f"Stride 4: {p_anchor} != 104441"

        # Check Stride 2 Dual Lattice
        raw_dual = yaml.safe_load(Path("configs/rmr_v29/rmr_v29_h2_subpixel2.yaml").read_text())["model"]
        raw_dual["use_anscombe_sirt"] = True
        raw_dual["adaptive_tau"] = True
        m_dual = RMRv3(RMRv3Config.from_dict(raw_dual))
        p_dual = sum(p.numel() for p in m_dual.parameters() if p.requires_grad)
        assert p_dual == 104540, f"Stride 2 Dual: {p_dual} != 104540"
        assert p_dual <= 105000, "Exceeded 105,000 parameter budget!"

    def test_model_dual_lattice_forward_and_carrier_attachment(self):
        """Dual-lattice forward pass returns terminal y and pushed-forward y_carrier with exact mass conservation."""
        import yaml
        from pathlib import Path

        raw_dual = yaml.safe_load(Path("configs/rmr_v29/rmr_v29_h2_subpixel2.yaml").read_text())["model"]
        raw_dual["use_anscombe_sirt"] = True
        raw_dual["adaptive_tau"] = True
        m_dual = RMRv3(RMRv3Config.from_dict(raw_dual))
        m_dual.eval()

        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            out = m_dual(x)

        assert "y" in out
        assert "y_carrier" in out
        assert "y0_carrier" in out
        assert out.y.shape == (2, 1, 128, 128)
        assert out.y_carrier.shape == (2, 1, 64, 64)
        assert check_mass_conservation(out.y, out.y_carrier, eps=1e-5)

    def test_dual_supervision_loss_wired_and_nonzero(self):
        """compute_dual_lattice_losses is called by orchestration and produces non-zero loss."""
        import yaml
        from pathlib import Path
        from rmr_v3.losses.orchestration import compute_rmr_v3_losses
        from rmr_v3.losses.config import RMRv3LossConfig

        raw_dual = yaml.safe_load(Path("configs/rmr_v29/rmr_v29_h2_subpixel2.yaml").read_text())["model"]
        raw_dual["use_anscombe_sirt"] = True
        raw_dual["adaptive_tau"] = True
        m = RMRv3(RMRv3Config.from_dict(raw_dual))
        m.eval()

        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            out = m(x)

        # Target at stride-2 resolution (same as model output)
        target = torch.rand(2, 1, 128, 128).abs()
        loss_cfg = RMRv3LossConfig(lambda_carrier_cell=0.50, lambda_fine_cell=0.25)
        losses = compute_rmr_v3_losses(dict(out), target, loss_cfg)

        # Both carrier and fine losses must be present and finite
        assert "cell_carrier" in losses, "cell_carrier not in losses dict"
        assert "cell_fine" in losses, "cell_fine not in losses dict"
        assert torch.isfinite(losses["cell_carrier"]), "cell_carrier loss is NaN/Inf"
        assert torch.isfinite(losses["cell_fine"]), "cell_fine loss is NaN/Inf"
        # With non-trivial targets, the dual losses must be strictly positive
        assert losses["cell_carrier"].item() > 0.0, "cell_carrier loss is zero — supervision not active"
        assert losses["cell_fine"].item() > 0.0, "cell_fine loss is zero — supervision not active"

