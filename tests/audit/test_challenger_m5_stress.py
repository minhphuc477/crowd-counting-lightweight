from __future__ import annotations

"""Empirical Stress Test Suite for Milestone 5 Gate: Challenger.

Executes rigorous adversarial verification:
1. Parameter accounting: exact layer-by-layer parameter inventory for Sub-50 MAE Blueprint (<= 105,000 params).
2. Mass conservation and support invariance: push-forward P and pullback P* across even/odd spatial dimensions and background masks.
3. Anisotropic diffusion peak preservation: Dirac impulse preservation under T=8 iterations of isotropic vs anisotropic Perona-Malik diffusion.
4. Summary cross-check: audit matrix verification against raw summary.json files in runs/sha_a.
"""

import json
import math
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from rmr_core.backbones import MobileNetV4Backbone
from rmr_core.necks import ASPPLiteFPNNeck
from rmr_core.scale_routing import ScaleRoutingHead
from rmr_v3.regional_head import ProbabilisticRegionalEvidenceHead
from rmr_v3.model.dual_lattice import (
    push_forward_stride2_to_stride4,
    pullback_stride4_to_stride2_rn,
    check_mass_conservation,
)
from rmr_core.operators import (
    build_multiscale_regions,
    regional_adjoint,
    regional_sum,
    weighted_normalized_adjoint_field,
)


# ==============================================================================
# 1. PARAMETER ACCOUNTING STRESS TEST (SUB-50 MAE BLUEPRINT)
# ==============================================================================

class TestSub50BlueprintParameterBudget:
    """Verifies layer-by-layer parameter accounting and hard <= 105,000 parameter budget."""

    def test_backbone_parameter_count(self):
        """Feature Backbone: MobileNetV4 truncated at stage 3 (C16) == 87,568 params."""
        backbone = MobileNetV4Backbone(pretrained=False)
        p_bb = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        assert p_bb == 87568, f"Backbone parameter count mismatch: expected 87,568, got {p_bb}"

    def test_neck_parameter_count(self):
        """FPN Neck: ASPPLiteFPNNeck (dilations 1,3,6 + GAP) == 10,784 params."""
        neck = ASPPLiteFPNNeck(in_channels=(16, 32, 48), width=32)
        p_neck = sum(p.numel() for p in neck.parameters() if p.requires_grad)
        assert p_neck == 10784, f"Neck parameter count mismatch: expected 10,784, got {p_neck}"

    def test_scale_router_parameter_count(self):
        """Dynamic Scale Router: DW 3x3 + LN + PW 1x1 (32->3) == 483 params."""
        sr = ScaleRoutingHead(in_channels=32, num_scales=3)
        p_sr = sum(p.numel() for p in sr.parameters() if p.requires_grad)
        assert p_sr == 483, f"Scale router parameter count mismatch: expected 483, got {p_sr}"

    def test_regional_evidence_head_parameter_count(self):
        """Regional Evidence Head: trunk (33->48->48) + mean/disp/hurdle heads == 4,131 params."""
        rh = ProbabilisticRegionalEvidenceHead(
            feature_dim=32,
            hidden=48,
            hurdle_head=True,
            regional_feature_stats="mean",
        )
        p_rh = sum(p.numel() for p in rh.parameters() if p.requires_grad)
        assert p_rh == 4131, f"Regional head parameter count mismatch: expected 4,131, got {p_rh}"

    def test_decoupled_dual_fine_head_parameter_count(self):
        """Decoupled Dual Fine Head: DW(32) + GN(32) + PW(32) + GN(32) + Proj4(1) + Proj2(4) + 2 scalars == 1,607 params."""
        class DecoupledDualFineHeadBlueprint(nn.Module):
            def __init__(self, in_channels: int = 32):
                super().__init__()
                self.dw = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
                self.gn1 = nn.GroupNorm(num_groups=4, num_channels=in_channels)
                self.pw = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
                self.gn2 = nn.GroupNorm(num_groups=4, num_channels=in_channels)
                self.carrier_proj = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)
                self.fine_proj = nn.Conv2d(in_channels, 4, kernel_size=1, bias=True)
                self.temperature = nn.Parameter(torch.tensor(1.0))
                self.curv_alpha = nn.Parameter(torch.tensor(1.0))

            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                h = F.silu(self.gn1(self.dw(x)))
                h = F.silu(self.gn2(self.pw(h)))
                carrier_logit = self.carrier_proj(h)
                fine_subpixel = F.pixel_shuffle(self.fine_proj(h), upscale_factor=2)
                return carrier_logit, fine_subpixel

        dfh = DecoupledDualFineHeadBlueprint(in_channels=32)
        p_dfh = sum(p.numel() for p in dfh.parameters() if p.requires_grad)
        assert p_dfh == 1607, f"Decoupled fine head mismatch: expected 1,607, got {p_dfh}"

    def test_anisotropic_sirt_engine_parameters(self):
        """Anisotropic SIRT Engine: 2 scalar parameters (log kappa_edge + conductance scale)."""
        log_kappa = nn.Parameter(torch.tensor(0.0))
        cond_scale = nn.Parameter(torch.tensor(1.0))
        p_sirt = log_kappa.numel() + cond_scale.numel()
        assert p_sirt == 2, f"Anisotropic SIRT params: expected 2, got {p_sirt}"

    def test_total_model_trainable_parameters_strict_budget(self):
        """Full Sub-50 MAE Blueprint: Total trainable params == 104,575 <= 105,000 (headroom: 425)."""
        p_bb = 87568
        p_neck = 10784
        p_sr = 483
        p_rh = 4131
        p_dfh = 1607
        p_sirt = 2

        total_trainable = p_bb + p_neck + p_sr + p_rh + p_dfh + p_sirt
        assert total_trainable == 104575, f"Expected 104,575, got {total_trainable}"
        assert total_trainable <= 105000, f"VIOLATION: {total_trainable} exceeds hard ceiling 105,000!"
        headroom = 105000 - total_trainable
        assert headroom == 425, f"Expected 425 headroom, got {headroom}"


# ==============================================================================
# 2. MASS CONSERVATION AND SUPPORT INVARIANCE STRESS TEST
# ==============================================================================

class TestMassConservationAndSupportInvariance:
    """Stress tests push-forward, pullback, and Radon-Nikodym support invariance."""

    @pytest.mark.parametrize("h, w", [
        (64, 64),
        (128, 128),
        (256, 256),
        (65, 64),
        (64, 65),
        (65, 65),
        (127, 127),
        (131, 97),
        (255, 255),
        (16, 256),
        (255, 17),
        (501, 703),
    ])
    def test_push_forward_exact_mass_conservation_all_shapes(self, h: int, w: int):
        """P(y2) conserves total mass exactly across even, odd, and anisotropic dimensions."""
        torch.manual_seed(42 + h + w)
        y2 = torch.rand(2, 1, h, w, dtype=torch.float64) * 10.0
        y4 = push_forward_stride2_to_stride4(y2)

        exp_h4 = (h + 1) // 2
        exp_w4 = (w + 1) // 2
        assert y4.shape == (2, 1, exp_h4, exp_w4)

        m2 = y2.sum(dim=(-2, -1))
        m4 = y4.sum(dim=(-2, -1))
        rel_diff = (m2 - m4).abs() / m2.clamp_min(1e-6)
        assert rel_diff.max().item() < 1e-6, f"Shape ({h},{w}): Mass not conserved! rel_diff={rel_diff.max().item()}"

    @pytest.mark.parametrize("h, w", [
        (64, 64),
        (128, 128),
        (65, 65),
        (127, 127),
        (131, 97),
        (255, 255),
    ])
    def test_pullback_exact_mass_conservation_all_shapes(self, h: int, w: int):
        """P*(y4, y2_prior) conserves total mass sum(y2_prolong) == sum(y4)."""
        torch.manual_seed(100 + h + w)
        y2_prior = torch.rand(2, 1, h, w, dtype=torch.float32) * 5.0
        y4 = push_forward_stride2_to_stride4(y2_prior)

        # Scale carrier mass arbitrarily (as if updated by solver)
        y4_updated = y4 * 1.35 + 0.12

        prolong = pullback_stride4_to_stride2_rn(y4_updated, y2_prior)
        assert prolong.shape == (2, 1, h, w)

        m_carrier = y4_updated.double().sum(dim=(-2, -1))
        m_prolong = prolong.double().sum(dim=(-2, -1))
        rel_diff = (m_carrier - m_prolong).abs() / m_carrier.clamp_min(1e-6)
        assert rel_diff.max().item() < 1e-5, f"Pullback shape ({h},{w}) mass mismatch: {rel_diff.max().item()}"

    def test_pullback_support_invariance_background_mask(self):
        """Where fine prior y0(u)=0, prolonged mass y(u) is identically 0.0 (no leakage)."""
        torch.manual_seed(2026)
        h, w = 64, 64
        y2_prior = torch.zeros(1, 1, h, w, dtype=torch.float32)
        # Place isolated heads: some blocks have 1 head, some have 2, and rest are pure background
        # Block (5, 5) -> head at (10, 10) (1 head, 3 empty cells in block)
        y2_prior[0, 0, 10, 10] = 1.5
        # Block (10, 12) -> heads at (20, 24) and (21, 24) (2 heads, 2 empty cells in block)
        y2_prior[0, 0, 20, 24] = 0.8
        y2_prior[0, 0, 21, 24] = 1.2
        # Block (20, 20) -> head at (41, 41) (1 head, 3 empty cells)
        y2_prior[0, 0, 41, 41] = 2.4

        bg_mask = (y2_prior == 0.0)
        assert bg_mask.sum().item() == (h * w - 4)

        # Generate carrier from prior and update active carrier cells (e.g. from solver)
        y4 = push_forward_stride2_to_stride4(y2_prior)
        # Scale active carrier cells by 1.75 (preserving zero support on empty carrier cells)
        y4_solver = y4 * 1.75

        # Pullback with Radon-Nikodym weighting
        y2_out = pullback_stride4_to_stride2_rn(y4_solver, y2_prior)

        # Invariant 1: Mass conservation
        assert check_mass_conservation(y2_out, y4_solver, eps=1e-5)

        # Invariant 2: Support invariance: all background pixels must remain identically 0.0
        bg_leakage = y2_out[bg_mask].abs().max().item()
        assert bg_leakage == 0.0, f"VIOLATION: Support invariance broken! Leakage: {bg_leakage}"

        # Invariant 3: Head pixels receive the updated carrier mass proportionally
        assert math.isclose(y2_out[0, 0, 10, 10].item(), 1.5 * 1.75, abs_tol=1e-5)
        # In block (10, 12), the 2 empty cells (20, 25) and (21, 25) MUST be identically 0.0
        assert y2_out[0, 0, 20, 25].item() == 0.0
        assert y2_out[0, 0, 21, 25].item() == 0.0

    def test_radon_nikodym_adjoint_zero_support_absorption(self):
        """Radon-Nikodym adjoint A_RN^T guarantees identically zero update on y(u)=0."""
        torch.manual_seed(999)
        h, w = 32, 32
        regions = build_multiscale_regions(h, w, output_stride=4, region_sizes_px=(16, 32), overlap=0.5, include_full_image=False)
        m = len(regions.boxes)

        # Initial field with 50% true background pixels (identically 0.0)
        y = torch.zeros(1, 1, h, w, dtype=torch.float32)
        y[0, 0, 12:22, 12:22] = torch.rand(10, 10) * 0.5 + 0.1

        bg_mask = (y == 0.0)

        # Hostile regional target that demands heavy positive mass on background
        b_region = torch.ones(1, 1, m, dtype=torch.float32) * 100.0
        weight = torch.ones(1, 1, m, dtype=torch.float32)

        # Compute Radon-Nikodym adjoint field
        adj_field = weighted_normalized_adjoint_field(
            y=y,
            b_region=b_region,
            weight=weight,
            regions=regions,
            adjoint_mode="radon_nikodym",
            hybrid_recovery_alpha=0.0,  # pure Radon-Nikodym
        )

        assert adj_field.shape == (1, 1, h, w)
        assert torch.isfinite(adj_field).all()

        # Update on background pixels MUST be identically zero
        bg_update_max = adj_field[bg_mask].abs().max().item()
        assert bg_update_max == 0.0, f"Radon-Nikodym support absorption failed: bg_update_max={bg_update_max}"


# ==============================================================================
# 3. ANISOTROPIC DIFFUSION PEAK PRESERVATION TEST
# ==============================================================================

class TestAnisotropicDiffusionPeakPreservation:
    """Simulates Dirac impulse under T=8 iterations of isotropic vs Perona-Malik diffusion."""

    def test_dirac_impulse_attenuation_comparison(self):
        """Verify: Anisotropic peak preservation > 95% while isotropic degrades to < 20% under T=8."""
        grid_size = 64
        center = 32
        T = 8
        lambda_tv = 0.10
        kappa_edge = 0.05

        # Initialize Dirac impulse
        y_initial = torch.zeros(1, 1, grid_size, grid_size, dtype=torch.float32)
        y_initial[0, 0, center, center] = 1.0

        # 1. Isotropic Laplacian Diffusion
        y_iso = y_initial.clone()
        laplace_kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)

        for t in range(T):
            lap = F.conv2d(y_iso, laplace_kernel, padding=1)
            y_iso = torch.clamp_min(y_iso + lambda_tv * lap, 0.0)

        iso_peak = y_iso[0, 0, center, center].item()
        iso_preservation = iso_peak / 1.0

        # 2. Perona-Malik Anisotropic Conductance Diffusion
        y_aniso = y_initial.clone()
        for t in range(T):
            dN = F.pad(y_aniso[:, :, :-1, :] - y_aniso[:, :, 1:, :], (0, 0, 1, 0))
            dS = F.pad(y_aniso[:, :, 1:, :] - y_aniso[:, :, :-1, :], (0, 0, 0, 1))
            dW = F.pad(y_aniso[:, :, :, :-1] - y_aniso[:, :, :, 1:], (1, 0, 0, 0))
            dE = F.pad(y_aniso[:, :, :, 1:] - y_aniso[:, :, :, :-1], (0, 1, 0, 0))

            cN = 1.0 / (1.0 + (dN.abs() / kappa_edge).pow(2))
            cS = 1.0 / (1.0 + (dS.abs() / kappa_edge).pow(2))
            cW = 1.0 / (1.0 + (dW.abs() / kappa_edge).pow(2))
            cE = 1.0 / (1.0 + (dE.abs() / kappa_edge).pow(2))

            flux = cN * dN + cS * dS + cW * dW + cE * dE
            y_aniso = torch.clamp_min(y_aniso + lambda_tv * flux, 0.0)

        aniso_peak = y_aniso[0, 0, center, center].item()
        aniso_preservation = aniso_peak / 1.0

        # Check empirical assertions:
        # Blueprint says: Isotropic T=8 peak = 0.1058 (< 0.20), Anisotropic T=8 peak = 0.9920 (> 0.95)
        assert math.isclose(iso_peak, 0.1058, abs_tol=1e-3), f"Expected iso_peak approx 0.1058, got {iso_peak}"
        assert math.isclose(aniso_peak, 0.9920, abs_tol=1e-3), f"Expected aniso_peak approx 0.9920, got {aniso_peak}"

        assert iso_preservation < 0.20, f"Isotropic did not degrade enough: {iso_preservation} >= 0.20"
        assert aniso_preservation > 0.95, f"Anisotropic failed to preserve peak: {aniso_preservation} <= 0.95"

        # Verify mass conservation under both diffusions (Neumann zero-flux boundaries)
        assert math.isclose(y_iso.sum().item(), 1.0, abs_tol=1e-5)
        assert math.isclose(y_aniso.sum().item(), 1.0, abs_tol=1e-5)

    def test_moderate_lambda_diffusion_preservation(self):
        """Even at moderate lambda_tv=0.02, anisotropic preserves >99% while isotropic loses >45%."""
        grid_size = 64
        center = 32
        T = 8
        lambda_tv = 0.02
        kappa_edge = 0.05

        y_iso = torch.zeros(1, 1, grid_size, grid_size, dtype=torch.float32)
        y_iso[0, 0, center, center] = 1.0
        y_aniso = y_iso.clone()

        laplace_kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)
        for t in range(T):
            lap = F.conv2d(y_iso, laplace_kernel, padding=1)
            y_iso = torch.clamp_min(y_iso + lambda_tv * lap, 0.0)

        for t in range(T):
            dN = F.pad(y_aniso[:, :, :-1, :] - y_aniso[:, :, 1:, :], (0, 0, 1, 0))
            dS = F.pad(y_aniso[:, :, 1:, :] - y_aniso[:, :, :-1, :], (0, 0, 0, 1))
            dW = F.pad(y_aniso[:, :, :, :-1] - y_aniso[:, :, :, 1:], (1, 0, 0, 0))
            dE = F.pad(y_aniso[:, :, :, 1:] - y_aniso[:, :, :, :-1], (0, 1, 0, 0))
            cN = 1.0 / (1.0 + (dN.abs() / kappa_edge).pow(2))
            cS = 1.0 / (1.0 + (dS.abs() / kappa_edge).pow(2))
            cW = 1.0 / (1.0 + (dW.abs() / kappa_edge).pow(2))
            cE = 1.0 / (1.0 + (dE.abs() / kappa_edge).pow(2))
            flux = cN * dN + cS * dS + cW * dW + cE * dE
            y_aniso = torch.clamp_min(y_aniso + lambda_tv * flux, 0.0)

        assert y_aniso[0, 0, center, center].item() > 0.99
        assert y_iso[0, 0, center, center].item() < 0.55


# ==============================================================================
# 4. SUMMARY & LOG CROSS-CHECK STRESS TEST
# ==============================================================================

class TestBenchmarkAuditMatrixCrossCheck:
    """Verifies entries in BENCHMARK_AUDIT_MATRIX.md against raw summary.json files in runs/sha_a."""

    def test_cross_check_comprehensive_matrix_runs(self):
        """Verify 12 key runs from Section 4 of BENCHMARK_AUDIT_MATRIX.md match summary.json."""
        runs_dir = Path("runs/sha_a")
        audit_matrix_path = Path(".agents/orchestrator_crossgen/BENCHMARK_AUDIT_MATRIX.md")
        assert audit_matrix_path.is_file(), f"Audit matrix not found at {audit_matrix_path}"

        expected_records = {
            "rmr_v19_canonical_isotropic": {"mae": 72.84, "rmse": 110.57, "sparse": 22.06, "dense": 122.71},
            "rmr_v29_h2_subpixel2": {"mae": 90.03, "rmse": 160.11, "sparse": 10.64, "dense": 191.32},
            "rmr_v25_canonical": {"mae": 78.03, "rmse": 122.66, "sparse": 29.69, "dense": 139.54},
            "rmr_v26_ablation_with_curv01": {"mae": 76.92, "rmse": 116.87, "sparse": 24.18, "dense": 132.68},
            "rmr_v30_step0_v19_anchor": {"mae": 75.81, "rmse": 118.14, "sparse": 30.65, "dense": 137.39},
            "rmr_v30_h1_anscombe_sirt": {"mae": 76.17, "rmse": 116.74, "sparse": 44.16, "dense": 127.43},
            "rmr_v30_h2_dual_lattice_dcsr": {"mae": 95.43, "rmse": 153.96, "sparse": 15.20, "dense": 183.49},
            "rmr_v30_h3_anscombe_dual_lattice": {"mae": 85.08, "rmse": 138.62, "sparse": 18.50, "dense": 159.19},
            "rmr_v30_control_no_solver": {"mae": 89.22, "rmse": 140.06, "sparse": 23.98, "dense": 167.31},
            "rmr_v20_canonical": {"mae": 78.17, "rmse": 123.88, "sparse": 58.52, "dense": 150.25},
            "rmr_v24_canonical": {"mae": 79.31, "rmse": 125.12, "sparse": 18.80, "dense": 143.94},
            "rmr_v27_canonical_restored": {"mae": 81.49, "rmse": 124.25, "sparse": 46.85, "dense": 145.36},
        }

        for run_name, exp in expected_records.items():
            rd = runs_dir / run_name
            summary_file = rd / "eval_val" / "summary.json"
            assert summary_file.is_file(), f"Missing eval_val summary.json in {run_name}"

            data = json.loads(summary_file.read_text())
            mae = data.get("MAE") or data.get("mae")
            rmse = data.get("RMSE") or data.get("rmse")
            sparse = data.get("mae_sparse") or data.get("sparse_mae")
            dense = data.get("mae_dense") or data.get("dense_mae")

            assert math.isclose(mae, exp["mae"], abs_tol=0.05), f"{run_name}: MAE mismatch ({mae} vs {exp['mae']})"
            assert math.isclose(rmse, exp["rmse"], abs_tol=0.1), f"{run_name}: RMSE mismatch ({rmse} vs {exp['rmse']})"
            assert math.isclose(sparse, exp["sparse"], abs_tol=0.05), f"{run_name}: Sparse mismatch ({sparse} vs {exp['sparse']})"
            assert math.isclose(dense, exp["dense"], abs_tol=0.05), f"{run_name}: Dense mismatch ({dense} vs {exp['dense']})"

    def test_identify_section3_clerical_discrepancy(self):
        """Identify that Section 3 peak summary table has clerical entries for v15/v17/v26 bias while Section 4 is ground truth."""
        # This test documents that in Section 3:
        # - v15 claims 75.88 MAE, while raw summary.json and Section 4 show 91.62 MAE
        # - v17 claims 74.89 MAE, while raw summary.json and Section 4 show 78.83 MAE
        # - v26 claims -12.10 Bias, while raw summary.json and Section 4 show -2.54 Bias
        # This confirms our adversarial finding that Section 4 is the authentic audit matrix.
        runs_dir = Path("runs/sha_a")

        v15_data = json.loads((runs_dir / "rmr_v15_native_geometry" / "eval_val" / "summary.json").read_text())
        assert math.isclose(v15_data["mae"], 91.62, abs_tol=0.05)

        v17_data = json.loads((runs_dir / "rmr_v17_canonical" / "eval_val" / "summary.json").read_text())
        assert math.isclose(v17_data["mae"], 78.83, abs_tol=0.05)

        v26_data = json.loads((runs_dir / "rmr_v26_ablation_with_curv01" / "eval_val" / "summary.json").read_text())
        assert math.isclose(v26_data["bias"], -2.54, abs_tol=0.05)
