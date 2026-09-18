"""Comprehensive Unit Test Suite for Mass-Conserved Perspective Modulation (MPE-v2).

Milestone M1 / Requirement R1:
1. Normalize the 1D elevation modulation field so that its spatial vertical mean
   across the image height is identically 1.0:
       M(v) = 1.0 + tanh(W v + b)
       M_bar = (1 / H) * sum_{i=1}^H M(v_i)
       P4_tilde(v) = P4(v) * (M(v) / M_bar)
2. Mathematical Invariant: (1 / H) * sum_{i=1}^H (P4_tilde(v_i) / P4(v_i)) == 1.000000 +- 1e-6.
   Total carrier mass is strictly conserved across any height H >= 2, batch size, and channel width.
3. Completely prevents false density inflation in the lower foreground while preserving 1D projective camera depth calibration.
4. Total trainable parameters: exactly 64 parameters (identity warm-start with W=0, b=0).
5. Full canonical model trainable parameters strictly <= 105,000 (target: exactly 104,505 params).
"""
from __future__ import annotations

import math
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from rmr_v3.config import load_config
from rmr_v3.model import MicroPerspectiveElevation, RMRv3, RMRv3Config


# ============================================================================
# 1. Parameter Count & Budget Tests
# ============================================================================
class TestMPEv2ParametersAndBudgets:
    """Verifies parameter budget compliance of MPE-v2 and the integrated canonical RMR-v3 model."""

    def test_mpe_v2_exact_parameter_count(self) -> None:
        """Requirement R1: MicroPerspectiveElevation must contain exactly 64 trainable parameters."""
        mpe = MicroPerspectiveElevation(channels=32)
        total_params = sum(p.numel() for p in mpe.parameters() if p.requires_grad)
        assert total_params == 64, f"Expected exactly 64 trainable params in MPE-v2, got {total_params}"
        assert mpe.proj.weight.shape == (32, 1), f"Unexpected weight shape: {mpe.proj.weight.shape}"
        assert mpe.proj.bias.shape == (32,), f"Unexpected bias shape: {mpe.proj.bias.shape}"
        assert mpe.proj.weight.numel() == 32
        assert mpe.proj.bias.numel() == 32

    def test_mpe_v2_no_extra_buffers(self) -> None:
        """MPE-v2 should not allocate untracked non-trainable buffers."""
        mpe = MicroPerspectiveElevation(channels=32)
        buffers = list(mpe.buffers())
        assert len(buffers) == 0, f"MPE-v2 should have 0 registered buffers, got {len(buffers)}"

    def test_canonical_model_parameter_ceiling(self) -> None:
        """Requirement R1: Canonical RMR-v3 model trainable parameters must be <= 105,000 and target exactly 104,505."""
        cfg_path = Path("configs/rmr_v25/rmr_v25_canonical.yaml")
        if not cfg_path.exists():
            pytest.skip(f"Config file not found: {cfg_path}")
        raw_cfg = load_config(cfg_path)
        model_cfg = RMRv3Config.from_dict(raw_cfg["model"], pretrained=False)
        model = RMRv3(model_cfg)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable_params <= 105_000, f"Parameter budget exceeded! Got {trainable_params} > 105,000"
        assert trainable_params == 104_505, f"Expected exactly 104,505 params in canonical model, got {trainable_params}"

        headroom = 105_000 - trainable_params
        assert headroom == 495, f"Expected 495 parameters headroom, got {headroom}"


# ============================================================================
# 2. Identity Warm-Start Tests
# ============================================================================
class TestMPEv2IdentityWarmStart:
    """Verifies that at initialization, MPE-v2 acts as an exact identity transformation."""

    def test_identity_warm_start_4d_tensor(self) -> None:
        """At init (W=0, b=0), M(v) = 1.0, M_bar = 1.0, out = x identically."""
        mpe = MicroPerspectiveElevation(channels=32)
        torch.manual_seed(42)
        x = torch.randn(4, 32, 64, 64)
        out = mpe(x)
        assert torch.allclose(out, x, atol=1e-7), "Identity warm-start violated for 4D tensor!"
        max_diff = (out - x).abs().max().item()
        assert max_diff == 0.0, f"Expected bit-exact identity output at init, got max diff = {max_diff}"

    def test_identity_warm_start_3d_tensor(self) -> None:
        """Verify identity warm-start on unbatched 3D tensor [C, H, W]."""
        mpe = MicroPerspectiveElevation(channels=32)
        torch.manual_seed(42)
        x = torch.randn(32, 48, 48)
        out = mpe(x)
        assert torch.allclose(out, x, atol=1e-7), "Identity warm-start violated for 3D tensor!"
        max_diff = (out - x).abs().max().item()
        assert max_diff == 0.0, f"Expected bit-exact identity output at init, got max diff = {max_diff}"

    @pytest.mark.parametrize("h,w", [(2, 2), (7, 11), (33, 27), (64, 64), (73, 89), (128, 128)])
    def test_identity_warm_start_diverse_dimensions(self, h: int, w: int) -> None:
        """Identity warm-start holds identically across arbitrary prime, odd, and power-of-two dimensions."""
        mpe = MicroPerspectiveElevation(channels=32)
        x = torch.randn(2, 32, h, w)
        out = mpe(x)
        assert torch.allclose(out, x, atol=1e-7)
        assert (out - x).abs().max().item() == 0.0


# ============================================================================
# 3. Exact Mass Conservation Invariant Tests
# ============================================================================
class TestMPEv2MassConservation:
    """Verifies the vertical mass invariant (1/H) sum(P_tilde / P) == 1.000000 +- 1e-6."""

    @pytest.mark.parametrize("h", [2, 3, 5, 7, 16, 33, 64, 73, 128, 256, 512])
    def test_mass_conservation_across_varying_heights(self, h: int) -> None:
        """Requirement R1: Mathematical Invariant holds to <= 1e-6 across varying heights."""
        torch.manual_seed(1000 + h)
        mpe = MicroPerspectiveElevation(channels=32)

        # Strongly perturb weights and biases away from zero to activate non-linear elevation
        nn.init.normal_(mpe.proj.weight, mean=0.0, std=1.0)
        nn.init.normal_(mpe.proj.bias, mean=0.0, std=1.0)

        # Strictly positive carrier to avoid zero-division in ratio
        carrier = torch.rand(4, 32, h, 48) + 0.1
        out = mpe(carrier)

        # Carrier ratio along vertical axis
        ratio = out / carrier
        vert_mean = ratio.mean(dim=-2)  # [4, 32, 48]

        max_err = (vert_mean - 1.0).abs().max().item()
        assert max_err < 1e-6, f"Mass conservation violated for H={h}: max error = {max_err:.2e} >= 1e-6"

    def test_mass_conservation_column_by_column(self) -> None:
        """Every individual horizontal column u in [1, W] independently satisfies exact vertical mass conservation."""
        torch.manual_seed(777)
        mpe = MicroPerspectiveElevation(channels=32)
        nn.init.normal_(mpe.proj.weight, mean=0.5, std=0.5)
        nn.init.normal_(mpe.proj.bias, mean=-0.2, std=0.5)

        h, w = 64, 32
        carrier = torch.rand(2, 32, h, w) + 0.05
        out = mpe(carrier)

        ratio = out / carrier
        for col_idx in range(w):
            col_mean = ratio[..., col_idx].mean(dim=-1)  # average along height
            diff = (col_mean - 1.0).abs().max().item()
            assert diff < 1e-6, f"Column {col_idx} violated mass conservation: diff = {diff:.2e}"

    @pytest.mark.parametrize("channels", [16, 32, 48, 64])
    def test_mass_conservation_varying_channels(self, channels: int) -> None:
        """Mass conservation invariant holds independently for different channel configurations."""
        torch.manual_seed(channels)
        mpe = MicroPerspectiveElevation(channels=channels)
        nn.init.normal_(mpe.proj.weight, mean=0.0, std=0.8)
        nn.init.normal_(mpe.proj.bias, mean=0.3, std=0.8)

        carrier = torch.rand(2, channels, 40, 20) + 0.1
        out = mpe(carrier)

        vert_mean = (out / carrier).mean(dim=-2)
        diff = (vert_mean - 1.0).abs().max().item()
        assert diff < 1e-6, f"Channels={channels} mass conservation error: {diff:.2e}"

    def test_minimal_height_h2(self) -> None:
        """Mass conservation holds for minimal non-trivial height H=2."""
        mpe = MicroPerspectiveElevation(channels=32)
        nn.init.normal_(mpe.proj.weight, std=2.0)
        nn.init.normal_(mpe.proj.bias, std=2.0)

        carrier = torch.rand(3, 32, 2, 10) + 0.2
        out = mpe(carrier)
        vert_mean = (out / carrier).mean(dim=-2)
        diff = (vert_mean - 1.0).abs().max().item()
        assert diff < 1e-6, f"Minimal height H=2 failed mass conservation: diff={diff:.2e}"


# ============================================================================
# 4. Gradient Flow & Differentiability Tests
# ============================================================================
class TestMPEv2GradientFlowAndDifferentiability:
    """Verifies gradient flow, backward propagation, and numerical differentiability."""

    def test_gradient_flow_to_all_parameters(self) -> None:
        """Gradients propagate smoothly to W, b, and input carrier P4."""
        mpe = MicroPerspectiveElevation(channels=32)
        # Move away from exact zero so bias receives gradient as well
        nn.init.normal_(mpe.proj.weight, mean=0.0, std=0.2)
        nn.init.normal_(mpe.proj.bias, mean=0.0, std=0.2)

        x = torch.randn(2, 32, 32, 32, requires_grad=True)
        out = mpe(x)

        target = torch.randn_like(out)
        loss = F.mse_loss(out, target)
        loss.backward()

        assert mpe.proj.weight.grad is not None, "Weight gradient is None"
        assert mpe.proj.bias.grad is not None, "Bias gradient is None"
        assert x.grad is not None, "Input tensor gradient is None"

        assert torch.isfinite(mpe.proj.weight.grad).all(), "Non-finite weight gradient"
        assert torch.isfinite(mpe.proj.bias.grad).all(), "Non-finite bias gradient"
        assert torch.isfinite(x.grad).all(), "Non-finite input gradient"

        assert mpe.proj.weight.grad.abs().sum().item() > 0.0, "Vanishing weight gradient"
        assert mpe.proj.bias.grad.abs().sum().item() > 0.0, "Vanishing bias gradient"
        assert x.grad.abs().sum().item() > 0.0, "Vanishing input gradient"

    def test_gradient_at_identity_warm_start(self) -> None:
        """At initialization (W=0, b=0), weight W receives active gradient dL/dW = P4 * v."""
        mpe = MicroPerspectiveElevation(channels=32)
        x = torch.ones(1, 32, 16, 16, requires_grad=True)
        out = mpe(x)

        # Apply a loss that penalizes higher values at bottom than top
        v_weights = torch.linspace(-1.0, 1.0, steps=16).view(1, 1, 16, 1).expand(1, 32, 16, 16)
        loss = (out * v_weights).sum()
        loss.backward()

        assert mpe.proj.weight.grad is not None
        assert mpe.proj.weight.grad.abs().sum().item() > 0.0, "Weight W did not receive gradient at warm-start!"

    def test_numerical_gradcheck(self) -> None:
        """Analytical autograd matches finite differences via torch.autograd.gradcheck."""
        channels = 4
        mpe = MicroPerspectiveElevation(channels=channels).to(dtype=torch.float64)
        nn.init.normal_(mpe.proj.weight, std=0.5)
        nn.init.normal_(mpe.proj.bias, std=0.5)

        x = torch.randn(1, channels, 8, 8, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(mpe, (x,), eps=1e-6, atol=1e-4)


# ============================================================================
# 5. Non-Negative Density Preservation Tests
# ============================================================================
class TestMPEv2NonNegativeDensityPreservation:
    """Verifies that MPE-v2 preserves non-negativity and support without creating background phantom counts."""

    def test_non_negative_preservation(self) -> None:
        """For non-negative input x >= 0, output out >= 0 strictly."""
        mpe = MicroPerspectiveElevation(channels=32)
        nn.init.normal_(mpe.proj.weight, std=1.0)
        nn.init.normal_(mpe.proj.bias, std=1.0)

        x = torch.rand(2, 32, 64, 64).abs()
        out = mpe(x)
        assert (out >= 0.0).all(), "MPE-v2 produced negative values from non-negative input!"

    def test_zero_support_absorption(self) -> None:
        """Where input x == 0 (e.g. background regions), output is identically 0 (zero phantom counts)."""
        mpe = MicroPerspectiveElevation(channels=32)
        nn.init.normal_(mpe.proj.weight, std=1.0)
        nn.init.normal_(mpe.proj.bias, std=1.0)

        x = torch.zeros(2, 32, 48, 48)
        # Inject sparse foreground spots
        x[:, :, 10:15, 10:15] = 1.5
        x[:, :, 30:35, 20:25] = 2.0

        out = mpe(x)

        # Background positions must remain exactly 0.0
        bg_mask = (x == 0.0)
        assert (out[bg_mask] == 0.0).all(), "MPE-v2 inflated background zero cells into non-zero phantom counts!"

        # Foreground positions must be strictly positive
        fg_mask = (x > 0.0)
        assert (out[fg_mask] > 0.0).all(), "MPE-v2 zeroed out true foreground cells!"

    def test_sign_preservation(self) -> None:
        """Since M(v) / M_bar > 0 strictly everywhere, sign(out) == sign(x)."""
        mpe = MicroPerspectiveElevation(channels=32)
        nn.init.normal_(mpe.proj.weight, std=1.0)
        nn.init.normal_(mpe.proj.bias, std=1.0)

        x = torch.randn(2, 32, 32, 32)
        out = mpe(x)
        assert torch.equal(torch.sign(out), torch.sign(x)), "MPE-v2 altered the sign of the carrier!"


# ============================================================================
# 6. Perspective Properties & Zero-Inflation Prevention
# ============================================================================
class TestMPEv2PerspectiveProperties:
    """Verifies camera depth elevation encoding and zero-inflation prevention."""

    def test_vertical_elevation_gradient(self) -> None:
        """Positive weight elevates lower foreground (v=+1) relative to upper background (v=-1)."""
        mpe = MicroPerspectiveElevation(channels=32)
        mpe.proj.weight.data.fill_(1.0)
        mpe.proj.bias.data.fill_(0.0)

        x = torch.ones(1, 32, 32, 32)
        out = mpe(x)

        top_val = out[0, :, 0, :].mean().item()
        bottom_val = out[0, :, -1, :].mean().item()

        assert bottom_val > top_val, f"Perspective modulation failed to elevate bottom: {bottom_val} <= {top_val}"

    def test_horizontal_invariance(self) -> None:
        """Modulation is identical across horizontal columns (variance across width is 0)."""
        mpe = MicroPerspectiveElevation(channels=32)
        nn.init.normal_(mpe.proj.weight, std=1.0)
        nn.init.normal_(mpe.proj.bias, std=1.0)

        x = torch.ones(1, 32, 64, 48)
        out = mpe(x)

        # Variance across width columns for every row must be negligible (< 1e-10)
        col_var = out.var(dim=-1).max().item()
        assert col_var < 1e-10, f"Horizontal invariance violated: max column variance = {col_var}"

    def test_zero_inflation_prevention_compared_to_unnormalized(self) -> None:
        """MPE-v2 prevents the net foreground inflation that affected v25 unnormalized MPE."""
        mpe_v2 = MicroPerspectiveElevation(channels=32)

        # Simulating unnormalized v25 with positive bias (which caused +13.45 net over-counting)
        weight = torch.full((32, 1), 0.5)
        bias = torch.full((32,), 0.5)
        mpe_v2.proj.weight.data.copy_(weight)
        mpe_v2.proj.bias.data.copy_(bias)

        h = 64
        v = torch.linspace(-1.0, 1.0, steps=h).view(h, 1)
        elevation_mod = F.linear(v, weight, bias).transpose(0, 1).unsqueeze(-1).unsqueeze(0)

        # Unnormalized v25 modulation multiplier:
        m_unnorm = 1.0 + torch.tanh(elevation_mod)
        unnorm_mean = m_unnorm.mean(dim=-2).mean().item()
        assert unnorm_mean > 1.10, f"Unnormalized modulation was expected to inflate mean (> 1.10), got {unnorm_mean}"

        # MPE-v2 normalized modulation multiplier:
        x = torch.ones(1, 32, h, 32)
        out_v2 = mpe_v2(x)
        v2_mean = out_v2.mean(dim=-2).mean().item()
        assert abs(v2_mean - 1.0) < 1e-6, f"MPE-v2 failed to maintain mean of 1.0: got {v2_mean}"


# ============================================================================
# 7. Full Canonical Model Integration Tests
# ============================================================================
class TestMPEv2FullModelIntegration:
    """Verifies end-to-end forward/backward integration within the canonical RMR-v3 model."""

    def test_canonical_model_forward_and_backward(self) -> None:
        """Canonical model forward and backward pass with MPE-v2 active."""
        cfg_path = Path("configs/rmr_v25/rmr_v25_canonical.yaml")
        if not cfg_path.exists():
            pytest.skip(f"Config file not found: {cfg_path}")
        raw_cfg = load_config(cfg_path)
        model_cfg = RMRv3Config.from_dict(raw_cfg["model"], pretrained=False)
        model = RMRv3(model_cfg)

        # Perturb MPE-v2 parameters
        model.perspective_elevation.proj.weight.data.normal_(0, 0.1)
        model.perspective_elevation.proj.bias.data.normal_(0, 0.1)

        x = torch.randn(2, 3, 256, 256)
        out = model(x)

        assert "y" in out, "Output dictionary missing terminal measure 'y'"
        assert "y0" in out, "Output dictionary missing initial measure 'y0'"
        assert out["y"].shape == (2, 1, 64, 64), f"Unexpected terminal measure shape: {out['y'].shape}"

        loss = out["y"].sum()
        loss.backward()

        w_grad = model.perspective_elevation.proj.weight.grad
        b_grad = model.perspective_elevation.proj.bias.grad
        assert w_grad is not None and torch.isfinite(w_grad).all()
        assert b_grad is not None and torch.isfinite(b_grad).all()
        assert w_grad.norm().item() > 0.0, "Zero gradient for MPE-v2 weight in full model"
        assert b_grad.norm().item() > 0.0, "Zero gradient for MPE-v2 bias in full model"
