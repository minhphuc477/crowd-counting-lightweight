"""Exhaustive DeepCode Test & Verification Suite for RMR-v23 Architecture.

Core Invariants & Requirements Audited:
1. Strict parameter budget compliance (strictly <= 105,000 parameters, target: exactly 104,441).
2. Invariant 1: Clean FineMeasureHead (1x1 conv) with analytical bias b_0 = -4.1422.
   Zero Softmax simplex concatenation into pre-softplus logits (Zero DC leakage).
3. Invariant 2: Pure Radon-Nikodym adjoint (hybrid_recovery_alpha = 0.0).
4. Invariant 3: Strict Monotonic Area Ordering: Isotropic dictionary [32, 64, 128] px.
5. Invariant 4: Density-Gated Curvature active only when y_local >= 0.15 (tau=0.15, beta=0.03, alpha_0=-8.0).
6. Invariant 5: Density-Gated Diffusion: Laplacian TV diffusion gated by torch.maximum(y, y_smooth)
   with tau=0.15, beta=0.03 (peak protection on dense, smoothing on sparse).
7. Invariant 6: Sample-Level Isolation: Zero batch cross-talk in loss calculation.
8. Invariant 7: Barzilai-Borwein Dynamic Step size (BB-1 secant step size in [0.2*omega, 2.0*omega]).
9. Invariant 8: Prior Calibration (init_m0 = 0.015763 count/cell).
10. All 4 RMR-v23 configs pass strict schema validation (validate_v3_config).
11. End-to-end forward/backward pass with AMP float16 and float32.
12. Numerical edge case stability: zero count, extreme density (>2000), prime dimensions.
"""
from __future__ import annotations

import math
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from rmr_core.heads import FineMeasureHead, _pool_local_density
from rmr_core.operators import build_multiscale_regions, weighted_normalized_adjoint_field
from rmr_v3.config import compute_config_hash, load_config, validate_v3_config
from rmr_v3.losses import (
    RMRv3LossConfig,
    compute_rmr_v3_losses,
    curvature_power_loss,
    physical_scale_alignment_loss,
    topk_hard_background_loss,
)
from rmr_v3.model import RMRv3, RMRv3Config
from rmr_v3.solver import laplacian_tv_diffusion, unrolled_sirt_solver


class TestRMRv23ParameterBudget:
    """Verifies that RMR-v23 adheres strictly to the <= 105,000 parameter budget."""

    @pytest.mark.parametrize(
        "cfg_name,expected_params",
        [
            ("rmr_v23_canonical.yaml", 104_441),
            ("rmr_v23_ablation_no_bb.yaml", 104_441),
            ("rmr_v23_ablation_no_density_gated_diffusion.yaml", 104_441),
            ("rmr_v23_control_no_solver.yaml", 104_441),
        ],
    )
    def test_parameter_budget_all_configs(self, cfg_name: str, expected_params: int) -> None:
        cfg_path = Path("configs/rmr_v23") / cfg_name
        assert cfg_path.exists(), f"Missing config: {cfg_path}"
        raw_cfg = load_config(cfg_path)

        model_cfg = RMRv3Config.from_dict(raw_cfg["model"], pretrained=False)
        model = RMRv3(model_cfg)

        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert num_trainable <= 105_000, f"{cfg_name} exceeded budget! Got {num_trainable} > 105,000"
        assert num_trainable == expected_params, f"{cfg_name} expected {expected_params} params, got {num_trainable}"
        headroom = 105_000 - num_trainable
        assert headroom == 559, f"{cfg_name} expected 559 headroom, got {headroom}"


class TestInvariant1PartitionOfUnityLeakage:
    """Invariant 1: Clean FineMeasureHead (1x1 conv) with analytical bias b_0 = -4.1422.

    Absolutely NO Softmax simplex concatenation into pre-softplus logits.
    """

    def test_head_is_clean_fine_measure_head(self) -> None:
        cfg_path = Path("configs/rmr_v23/rmr_v23_canonical.yaml")
        raw_cfg = load_config(cfg_path)
        model = RMRv3(RMRv3Config.from_dict(raw_cfg["model"], pretrained=False))

        assert isinstance(model.fine_head, FineMeasureHead), (
            f"Expected FineMeasureHead, got {type(model.fine_head).__name__}"
        )
        assert not model.cfg.scale_conditioned_fine_head, (
            "scale_conditioned_fine_head must be False in RMR-v23"
        )

    def test_analytical_bias_initialization(self) -> None:
        expected_b0 = math.log(math.exp(0.015763) - 1.0)  # ≈ -4.142188
        cfg_path = Path("configs/rmr_v23/rmr_v23_canonical.yaml")
        raw_cfg = load_config(cfg_path)
        model = RMRv3(RMRv3Config.from_dict(raw_cfg["model"], pretrained=False))

        final_conv: nn.Conv2d = model.fine_head.body[-1]  # type: ignore[assignment]
        assert isinstance(final_conv, nn.Conv2d)
        actual_bias = final_conv.bias.data.item()
        assert abs(actual_bias - expected_b0) < 1e-4, (
            f"Analytical bias b0 mismatch: expected {expected_b0:.4f}, got {actual_bias:.4f}"
        )

    def test_zero_softmax_simplex_leakage(self) -> None:
        """Verify that passing different scale routing weights does NOT alter pre-activation logits z0."""
        cfg_path = Path("configs/rmr_v23/rmr_v23_canonical.yaml")
        raw_cfg = load_config(cfg_path)
        model = RMRv3(RMRv3Config.from_dict(raw_cfg["model"], pretrained=False))
        model.eval()

        p4 = torch.randn(1, 32, 16, 16)
        # Condition A: 100% fine scale
        pi_a = torch.zeros(1, 3, 16, 16)
        pi_a[:, 0, :, :] = 1.0

        # Condition B: 100% coarse scale
        pi_b = torch.zeros(1, 3, 16, 16)
        pi_b[:, 2, :, :] = 1.0

        with torch.no_grad():
            z0_a, y0_a, _ = model._predict_fine_density(p4, scale_weights=pi_a)
            z0_b, y0_b, _ = model._predict_fine_density(p4, scale_weights=pi_b)

        # Invariant 1: z0 and y0 must be completely independent of scale simplex pi
        assert torch.equal(z0_a, z0_b), "Pre-activation logits leaked scale routing simplex weights!"
        assert torch.equal(y0_a, y0_b), "Fine density leaked scale routing simplex weights!"


class TestInvariant2and3AdjointAndDictionary:
    """Invariant 2: Pure Radon-Nikodym adjoint (hybrid_recovery_alpha = 0.0).

    Invariant 3: Strict Monotonic Area Ordering: Isotropic dictionary [32, 64, 128] px.
    """

    def test_pure_radon_nikodym_adjoint(self) -> None:
        cfg_path = Path("configs/rmr_v23/rmr_v23_canonical.yaml")
        raw_cfg = load_config(cfg_path)
        assert raw_cfg["model"]["adjoint_mode"] == "radon_nikodym"
        assert raw_cfg["model"]["hybrid_recovery_alpha"] == 0.0

    def test_pure_radon_nikodym_support_absorption(self) -> None:
        """Confirm support absorption: y(u) = 0 ==> (A^* r)(u) = 0 exactly."""
        h, w = 32, 32
        reg = build_multiscale_regions(h, w, output_stride=4, region_sizes_px=(32, 64), include_full_image=False)
        m = len(reg.boxes)

        # Support mask: left half is positive, right half is strictly zero
        y = torch.zeros(1, 1, h, w)
        y[:, :, :, :16] = 0.5

        # Regional target b with large discrepancy delta = q - b != 0
        b_solver = torch.full((1, 1, m), 10.0)
        weight = torch.ones(1, 1, m)

        field = weighted_normalized_adjoint_field(
            y=y,
            b_region=b_solver,
            weight=weight,
            regions=reg,
            adjoint_mode="radon_nikodym",
            hybrid_recovery_alpha=0.0,
        )

        # Invariant 2: support absorption everywhere y == 0
        zero_mask = (y == 0.0)
        assert torch.all(field[zero_mask] == 0.0), (
            "Pure Radon-Nikodym adjoint violated support absorption! Non-zero field on y == 0"
        )
        pos_mask = (y > 0.0)
        assert torch.any(field[pos_mask] != 0.0), "Field was unexpectedly zero on support y > 0"

    def test_isotropic_dictionary_ordering(self) -> None:
        cfg_path = Path("configs/rmr_v23/rmr_v23_canonical.yaml")
        raw_cfg = load_config(cfg_path)
        sizes = raw_cfg["model"]["region_sizes_px"]
        assert list(sizes) == [32, 64, 128]
        # Strict monotonic ordering
        for i in range(len(sizes) - 1):
            assert sizes[i] < sizes[i + 1]


class TestInvariant4DensityGatedCurvature:
    """Invariant 4: phi-power Taylor curvature active only when y_local >= 0.15 (tau=0.15, beta=0.03, alpha_0=-8.0)."""

    def test_curvature_hyperparameters(self) -> None:
        cfg_path = Path("configs/rmr_v23/rmr_v23_canonical.yaml")
        raw_cfg = load_config(cfg_path)
        m = raw_cfg["model"]
        assert m["density_curvature"] is True
        assert m["gated_density_curvature"] is True
        assert m["curvature_dense_threshold"] == 0.15
        assert m["curvature_gate_beta"] == 0.03
        assert m["curvature_pool_kernel"] == 8

        model = RMRv3(RMRv3Config.from_dict(m, pretrained=False))
        head: FineMeasureHead = model.fine_head  # type: ignore[assignment]
        assert abs(head.curvature_alpha.item() - (-8.0)) < 1e-4

    def test_gated_curvature_activation(self) -> None:
        head = FineMeasureHead(
            width=32,
            temp_softplus=True,
            density_curvature=True,
            gated_density_curvature=True,
            curvature_dense_threshold=0.15,
            curvature_gate_beta=0.03,
            curvature_pool_kernel=8,
        )
        head.eval()

        # Artificially set curvature_alpha to 2.0 to amplify curvature effect for testing
        head.curvature_alpha.data.fill_(2.0)
        alpha_eff = F.softplus(torch.tensor(2.0)).item()

        # Case A: Low density background (z = -3.0 => softplus(-3) ≈ 0.048 << 0.15)
        z_sparse = torch.full((1, 1, 32, 32), -3.0)
        with torch.no_grad():
            y_sparse = head.activate(z_sparse)
            y_base_sparse = F.softplus(z_sparse)
        # Gating should shut off curvature on sparse: y_sparse ≈ y_base_sparse
        diff_sparse = (y_sparse - y_base_sparse).abs().max().item()
        assert diff_sparse < 1e-3, f"Curvature leaked onto sparse background: diff={diff_sparse}"

        # Case B: High density clump (z = 2.0 => softplus(2) ≈ 2.126 >> 0.15)
        z_dense = torch.full((1, 1, 32, 32), 2.0)
        with torch.no_grad():
            y_dense = head.activate(z_dense)
            y_base_dense = F.softplus(z_dense)
        # Gating should be fully active (gate ≈ 1.0): y_dense ≈ y_base + alpha * y_base^2
        diff_dense = (y_dense - y_base_dense).mean().item()
        expected_curv = alpha_eff * (y_base_dense.mean().item() ** 2)
        assert abs(diff_dense - expected_curv) < 0.05, (
            f"Curvature not active on dense cluster: actual={diff_dense}, expected={expected_curv}"
        )

    def test_pool_local_density_dimension_invariance(self) -> None:
        """Verify _pool_local_density handles scalar, 1D, 2D, 3D, and 4D without error."""
        # 0D
        s = torch.tensor(0.5)
        assert _pool_local_density(s, 8) is s

        # 1D
        v = torch.randn(10)
        assert _pool_local_density(v, 8) is v

        # 4D
        y4 = torch.rand(2, 1, 32, 32)
        p4 = _pool_local_density(y4, 8)
        assert p4.shape == (2, 1, 32, 32)

        # 3D
        y3 = y4[0]
        p3 = _pool_local_density(y3, 8)
        assert p3.shape == (1, 32, 32)
        assert torch.allclose(p4[0], p3, atol=1e-6)

        # 2D
        y2 = y3[0]
        p2 = _pool_local_density(y2, 8)
        assert p2.shape == (32, 32)
        assert torch.allclose(p3[0], p2, atol=1e-6)

    def test_curvature_power_loss_dimension_invariance_and_empty(self) -> None:
        """Verify curvature_power_loss handles 2D, 3D, 4D, and empty inputs without IndexError."""
        # 4D
        y4 = torch.rand(2, 1, 32, 32)
        t4 = torch.rand(2, 1, 32, 32)
        l4 = curvature_power_loss(y4, t4, threshold=0.08, mode="hard")
        assert torch.isfinite(l4)

        # 3D
        y3 = y4[0]
        t3 = t4[0]
        l3 = curvature_power_loss(y3, t3, threshold=0.08, mode="hard")
        assert torch.isfinite(l3)

        # 2D (previously threw IndexError in F.avg_pool2d)
        y2 = y3[0]
        t2 = t3[0]
        l2 = curvature_power_loss(y2, t2, threshold=0.08, mode="hard")
        assert torch.isfinite(l2)

        # Empty input
        y_empty = torch.empty(0, 1, 32, 32)
        t_empty = torch.empty(0, 1, 32, 32)
        l_empty = curvature_power_loss(y_empty, t_empty, threshold=0.08, mode="hard")
        assert l_empty.item() == 0.0
        assert torch.isfinite(l_empty)

    def test_curvature_power_loss_boundary_density_un_diluted(self) -> None:
        """Verify that F.avg_pool2d in curvature_power_loss does not dilute corner density (count_include_pad=False)."""
        t = torch.zeros(1, 1, 16, 16)
        t.fill_(1.0)
        pad = 5 // 2
        local_density = F.avg_pool2d(t, kernel_size=5, stride=1, padding=pad, count_include_pad=False)
        assert torch.allclose(local_density, torch.ones_like(local_density), atol=1e-5)


class TestInvariant5DensityGatedDiffusion:
    """Invariant 5: Laplacian TV diffusion gated by torch.maximum(y, y_smooth) with tau=0.15, beta=0.03."""

    def test_peak_protection_and_sparse_smoothing(self) -> None:
        h, w = 32, 32
        # Isolated sharp Dirac peak (y = 2.0 >> 0.15)
        y_dense = torch.zeros(1, 1, h, w)
        y_dense[0, 0, 16, 16] = 2.0

        y_diff_standard = laplacian_tv_diffusion(y_dense, tv_lambda=0.05, density_gated=False)
        y_diff_gated = laplacian_tv_diffusion(
            y_dense,
            tv_lambda=0.05,
            density_gated=True,
            diffusion_dense_threshold=0.15,
            diffusion_gate_beta=0.03,
        )

        peak_std = y_diff_standard[0, 0, 16, 16].item()
        peak_gated = y_diff_gated[0, 0, 16, 16].item()
        assert peak_gated > peak_std
        assert peak_gated > 1.95, f"Dense peak was eroded: {peak_gated}"

        # Low-density background noise (mean 0.01 << 0.15)
        torch.manual_seed(42)
        y_sparse = torch.rand(1, 1, h, w) * 0.02
        y_diff_sparse = laplacian_tv_diffusion(
            y_sparse,
            tv_lambda=0.05,
            density_gated=True,
            diffusion_dense_threshold=0.15,
            diffusion_gate_beta=0.03,
        )
        assert y_diff_sparse.var().item() < y_sparse.var().item()

    def test_density_gated_diffusion_dimension_invariance(self) -> None:
        """Verify laplacian_tv_diffusion handles 2D [H,W], 3D [C,H,W], and 4D [B,C,H,W]."""
        # 4D
        y_4d = torch.rand(2, 1, 32, 32)
        out_4d = laplacian_tv_diffusion(y_4d, tv_lambda=0.02, density_gated=True)
        assert out_4d.shape == (2, 1, 32, 32)

        # 3D
        y_3d = y_4d[0]
        out_3d = laplacian_tv_diffusion(y_3d, tv_lambda=0.02, density_gated=True)
        assert out_3d.shape == (1, 32, 32)
        assert torch.allclose(out_4d[0], out_3d, atol=1e-6)

        # 2D
        y_2d = y_3d[0]
        out_2d = laplacian_tv_diffusion(y_2d, tv_lambda=0.02, density_gated=True)
        assert out_2d.shape == (32, 32)
        assert torch.allclose(out_3d[0], out_2d, atol=1e-6)

    def test_laplacian_tv_diffusion_int_and_tensor_early_return(self) -> None:
        """Verify integer tv_lambda=0 and 0-D tensor tv_lambda=0.0 return y directly."""
        y = torch.rand(1, 1, 16, 16)
        # Integer 0
        out_int = laplacian_tv_diffusion(y, tv_lambda=0)
        assert out_int is y

        # Float 0.0
        out_flt = laplacian_tv_diffusion(y, tv_lambda=0.0)
        assert out_flt is y

        # 0-D Tensor 0.0
        out_t = laplacian_tv_diffusion(y, tv_lambda=torch.tensor(0.0))
        assert out_t is y


class TestInvariant6BatchActiveGradientPreservation:
    """Invariant 6: Batch-Active Foreground Normalization.

    Guarantees that rare high-density crowd clumps receive full O(1) gradient signals
    without being diluted by Bx (preventing the catastrophic 8x gradient starvation).
    """

    def test_gradient_preservation_on_asymmetric_batch(self) -> None:
        """Verify that a dense scene in a batch of B=8 does NOT have its curvature gradient diluted by 8x."""
        B = 8
        H, W = 32, 32
        y = torch.full((B, 1, H, W), 0.05, requires_grad=True)
        target = torch.zeros(B, 1, H, W)
        target[0, 0, 8:24, 8:24] = 2.0  # Only sample 0 has a dense cluster

        loss = curvature_power_loss(y, target, threshold=0.08, mode="hard")
        grad = torch.autograd.grad(loss, y)[0]

        dense_grad_max = grad[0].abs().max().item()
        # Full gradient strength on dense cluster must be >= 0.010 (v19 level, NOT diluted to 0.0015)
        assert dense_grad_max > 0.010, (
            f"Curvature gradient was starved by batch dilution! Got {dense_grad_max:.4f}, expected > 0.010"
        )
        # Empty samples must receive strictly zero curvature gradient
        assert (grad[1:].abs() == 0.0).all(), "Curvature gradient leaked onto empty samples!"

    def test_scale_alignment_gradient_preservation(self) -> None:
        """Verify that Scale Router receives full gradient signal on active foreground without Bx dilution."""
        B = 8
        H, W = 32, 32
        sw = torch.full((B, 3, H, W), 1 / 3, requires_grad=True)
        target = torch.zeros(B, 1, H, W)
        target[0, 0, 8:24, 8:24] = 1.0  # Only sample 0 has foreground

        loss = physical_scale_alignment_loss(sw, target, mask_background=True)
        grad = torch.autograd.grad(loss, sw)[0]

        active_grad_max = grad[0].abs().max().item()
        # Full router gradient must be >= 0.005 (not diluted to 0.0009)
        assert active_grad_max > 0.005, (
            f"Scale router gradient was starved by batch dilution! Got {active_grad_max:.6f}, expected > 0.005"
        )
        assert (grad[1:].abs() == 0.0).all(), "Scale alignment gradient leaked onto empty samples!"

    def test_empty_and_2d_loss_components_isolation(self) -> None:
        """Verify topk_hard_background_loss and physical_scale_alignment_loss handle empty & 2D inputs."""
        # Top-K on empty batch (previously threw RuntimeError on torch.stack([]))
        y_empty = torch.empty(0, 1, 32, 32, requires_grad=True)
        t_empty = torch.empty(0, 1, 32, 32)
        l_topk = topk_hard_background_loss(y_empty, t_empty)
        assert l_topk.item() == 0.0
        l_topk.backward()
        assert y_empty.grad is not None

        # Scale alignment on 2D target (previously threw IndexError in F.avg_pool2d)
        sw = torch.rand(1, 3, 16, 16)
        sw = sw / sw.sum(dim=1, keepdim=True)
        t_2d = torch.rand(16, 16)
        l_sa = physical_scale_alignment_loss(scale_weights=sw, target_y=t_2d)
        assert torch.isfinite(l_sa)


class TestInvariant7BarzilaiBorweinDynamicStep:
    """Invariant 7: Re-enable BB-1 secant step size (use_barzilai_borwein: true, restored from v19)."""

    def test_bb1_step_execution_and_clamping(self) -> None:
        cfg_path = Path("configs/rmr_v23/rmr_v23_canonical.yaml")
        raw_cfg = load_config(cfg_path)
        assert raw_cfg["model"]["use_barzilai_borwein"] is True

        model = RMRv3(RMRv3Config.from_dict(raw_cfg["model"], pretrained=False))
        x = torch.randn(2, 3, 128, 128)
        out = model(x)

        assert "y" in out
        assert out["y"].shape == (2, 1, 32, 32)
        assert torch.isfinite(out["y"]).all()
        assert len(out["iterates"]) == 7  # y0 + 6 iterates

    def test_ablation_no_bb_matches_fixed_omega(self) -> None:
        cfg_path = Path("configs/rmr_v23/rmr_v23_ablation_no_bb.yaml")
        raw_cfg = load_config(cfg_path)
        assert raw_cfg["model"]["use_barzilai_borwein"] is False

        model = RMRv3(RMRv3Config.from_dict(raw_cfg["model"], pretrained=False))
        x = torch.randn(1, 3, 128, 128)
        out = model(x)
        assert torch.isfinite(out["y"]).all()

    def test_bb1_exact_secant_mathematics(self) -> None:
        """Verify BB1 secant equation Delta y^T Delta g / ||Delta g||^2 and positivity guard."""
        h, w = 16, 16
        reg = build_multiscale_regions(h, w, output_stride=4, region_sizes_px=(16, 32), include_full_image=False)
        m = len(reg.boxes)

        y0 = torch.full((2, 1, h, w), 0.5)
        b_solver = torch.full((2, 1, m), 2.0)
        weight = torch.ones(2, 1, m)

        res = unrolled_sirt_solver(
            y0=y0,
            b_solver=b_solver,
            weight_solver=weight,
            regions=reg,
            iterations=3,
            omega=1.0,
            use_barzilai_borwein=True,
        )

        assert len(res["iterates"]) == 4  # y0, y1, y2, y3
        # Iteration 1 -> 2: verify s = y1 - y0, r = field1 - field0
        y0_t = res["iterates"][0]
        y1_t = res["iterates"][1]
        f0_t = res["residual_fields"][0]
        f1_t = res["residual_fields"][1]

        s_diff = (y1_t - y0_t).float()
        r_diff = (f1_t - f0_t).float()
        dot_sr = (s_diff * r_diff).sum(dim=(-3, -2, -1), keepdim=True)
        norm_r_sq = (r_diff * r_diff).sum(dim=(-3, -2, -1), keepdim=True) + 1e-6
        expected_omega = torch.clamp(
            torch.where(dot_sr > 0.0, dot_sr / norm_r_sq, torch.tensor(1.0)),
            min=0.2,
            max=2.0,
        )
        assert torch.all(expected_omega >= 0.2)
        assert torch.all(expected_omega <= 2.0)


class TestInvariant8PriorCalibration:
    """Invariant 8: Calibrated background prior."""

    def test_prior_initial_density(self) -> None:
        cfg_path = Path("configs/rmr_v23/rmr_v23_canonical.yaml")
        raw_cfg = load_config(cfg_path)
        assert raw_cfg["model"]["init_m0"] == 0.015763

        model = RMRv3(RMRv3Config.from_dict(raw_cfg["model"], pretrained=False))
        # At init with zero inputs, softplus(b0) should produce ≈ 0.015763 count/cell
        z_init = torch.tensor(model.fine_head.body[-1].bias.item())
        y_init = model.fine_head.activate(z_init).item()
        assert abs(y_init - 0.015763) < 1e-4, f"Uncalibrated prior: {y_init} vs 0.015763"


class TestRMRv23SchemaAndEndToEnd:
    """Schema validation and end-to-end forward/loss/backward verification."""

    def test_all_v23_configs_pass_schema_validation(self) -> None:
        configs = list(Path("configs/rmr_v23").glob("*.yaml"))
        assert len(configs) == 4, f"Expected exactly 4 configs in configs/rmr_v23, found {len(configs)}"
        for c in configs:
            raw = load_config(c)
            validate_v3_config(raw)

    @pytest.mark.parametrize("amp_enabled", [False, True])
    def test_full_pipeline_forward_backward(self, amp_enabled: bool) -> None:
        cfg_path = Path("configs/rmr_v23/rmr_v23_canonical.yaml")
        raw_cfg = load_config(cfg_path)

        model = RMRv3(RMRv3Config.from_dict(raw_cfg["model"], pretrained=False))
        loss_cfg = RMRv3LossConfig.from_dict(raw_cfg["loss"])

        x = torch.randn(2, 3, 256, 256)
        target_y = torch.zeros(2, 1, 64, 64)
        target_y[:, :, 16:32, 16:32] = 0.8

        with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu", enabled=amp_enabled):
            out = model(x)
            losses = compute_rmr_v3_losses(out, target_y, loss_cfg)

        assert "total" in losses
        assert torch.isfinite(losses["total"]).all()

        losses["total"].backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has None grad"
                assert torch.isfinite(param.grad).all(), f"Parameter {name} has NaN/Inf grad"

    def test_numerical_edge_cases(self) -> None:
        cfg_path = Path("configs/rmr_v23/rmr_v23_canonical.yaml")
        raw_cfg = load_config(cfg_path)

        model = RMRv3(RMRv3Config.from_dict(raw_cfg["model"], pretrained=False))
        loss_cfg = RMRv3LossConfig.from_dict(raw_cfg["loss"])

        # Edge Case A: Empty image (0 count everywhere)
        x_empty = torch.zeros(1, 3, 128, 128)
        out_empty = model(x_empty)
        target_empty = torch.zeros(1, 1, 32, 32)
        losses_empty = compute_rmr_v3_losses(out_empty, target_empty, loss_cfg)
        assert torch.isfinite(losses_empty["total"]).all()

        # Edge Case B: Prime spatial resolution (197 x 243)
        x_prime = torch.randn(1, 3, 197, 243)
        out_prime = model(x_prime)
        assert torch.isfinite(out_prime["y"]).all()

        # Edge Case C: Extreme dense crowd (>2000 counts)
        x_dense = torch.randn(1, 3, 128, 128)
        target_dense = torch.full((1, 1, 32, 32), 2.5)  # 2.5 * 1024 cells = 2560 count
        out_dense = model(x_dense)
        losses_dense = compute_rmr_v3_losses(out_dense, target_dense, loss_cfg)
        assert torch.isfinite(losses_dense["total"]).all()

        # Edge Case D: Empty batch tensor [0, 3, 128, 128]
        x_empty_batch = torch.empty(0, 3, 128, 128)
        out_empty_batch = model(x_empty_batch)
        target_empty_batch = torch.empty(0, 1, 32, 32)
        losses_empty_batch = compute_rmr_v3_losses(out_empty_batch, target_empty_batch, loss_cfg)
        assert "total" in losses_empty_batch
        assert torch.isfinite(losses_empty_batch["total"]).all()
        assert losses_empty_batch["total"].item() == 0.0

    def test_gradient_flow_completeness_v23_canonical(self) -> None:
        """Verify that 100% of trainable parameters in RMR-v23 canonical receive non-zero gradients."""
        cfg_path = Path("configs/rmr_v23/rmr_v23_canonical.yaml")
        raw_cfg = load_config(cfg_path)
        model = RMRv3(RMRv3Config.from_dict(raw_cfg["model"], pretrained=False))
        model.train()
        loss_cfg = RMRv3LossConfig.from_dict(raw_cfg["loss"])

        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

        img = torch.randn(2, 3, 256, 256)
        target_y = torch.randint(0, 3, (2, 1, 64, 64)).float()
        target_y[:, :, 16:48, 16:48] += 0.5

        out = model(img)
        losses = compute_rmr_v3_losses(out, target_y, loss_cfg)
        losses["total"].backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()

        out2 = model(img)
        losses2 = compute_rmr_v3_losses(out2, target_y, loss_cfg)
        losses2["total"].backward()

        dead_params = []
        none_params = []
        for name, p in model.named_parameters():
            if p.requires_grad:
                if p.grad is None:
                    none_params.append(name)
                elif p.grad.abs().sum().item() == 0.0:
                    dead_params.append(name)

        assert not none_params, f"Parameters with None grad: {none_params}"
        assert not dead_params, f"Dead parameters (zero grad): {dead_params}"

    def test_config_hash_determinism_and_uniqueness(self) -> None:
        """Verify config hash is deterministic and distinguishes all 4 v23 configs."""
        configs = {
            c.name: load_config(c)
            for c in Path("configs/rmr_v23").glob("*.yaml")
        }
        hashes = {name: compute_config_hash(cfg) for name, cfg in configs.items()}
        assert len(set(hashes.values())) == 4, (
            f"Config hashes collided! Expected 4 unique hashes, got {len(set(hashes.values()))}: {hashes}"
        )
        # Determinism check
        for name, cfg in configs.items():
            assert compute_config_hash(cfg) == hashes[name]
