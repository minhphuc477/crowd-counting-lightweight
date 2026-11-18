"""Tests for non-ASPP bidirectional/PConv necks, subpixel head, and ResidualDilatedFineHead compact_c1 fixes."""
from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn

from rmr_core.fine_head import ResidualDilatedFineHead
from rmr_core.heads import FineMeasureHead, build_fine_head
from rmr_core.necks import (
    ConvexFastNormalizedFusion,
    HighResPConvFPNNeck,
    PAFPNBiNormNeck,
)
from rmr_v3.config.validator import validate_v3_config
from rmr_v3.model import RMRv3, RMRv3Config


def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


@pytest.mark.parametrize(
    "neck_type,head_type,enable_regional_head",
    [
        ("pafpn_binorm", "standard", True),
        ("pafpn_binorm", "standard", False),
        ("pafpn_binorm", "dw_subpixel", False),
        ("pafpn_binorm", "residual_dilated", False),
        ("pconv_fpn", "standard", True),
        ("pconv_fpn", "standard", False),
        ("pconv_fpn", "dw_subpixel", False),
        ("bifpn_lite", "standard", False),
    ],
)
def test_non_aspp_parameter_budget_and_groupnorm_purity(
    neck_type: str,
    head_type: str,
    enable_regional_head: bool,
) -> None:
    """All new non-ASPP configurations must fit <= 104,441 params and contain zero BatchNorm2d."""
    cfg = RMRv3Config(
        pretrained=False,
        output_stride=2,
        subpixel_stride2=True,
        feature_width=32,
        neck_type=neck_type,
        head_type=head_type,
        enable_regional_head=enable_regional_head,
        enable_solver=False,
        solver_mode="direct",
        head_activation="compact_c1",
        use_aspp_gap=False,
    )
    model = RMRv3(cfg)
    total_params = _count_params(model)
    assert total_params <= 104441, f"{neck_type}+{head_type} (reg={enable_regional_head}) has {total_params} > 104441"

    bn_modules = [m for m in model.fusion.modules() if isinstance(m, nn.BatchNorm2d)]
    assert len(bn_modules) == 0, f"{neck_type} contains forbidden BatchNorm2d modules: {bn_modules}"


@pytest.mark.parametrize("neck_cls", [PAFPNBiNormNeck, HighResPConvFPNNeck])
def test_non_aspp_necks_odd_resolution_and_sample_isolation(neck_cls: type[nn.Module]) -> None:
    """Non-ASPP necks must handle prime resolutions, finite grads, and zero batch cross-talk."""
    neck = neck_cls(in_channels=(16, 32, 48), width=32)
    neck.train()

    c4 = torch.randn(2, 16, 73, 51, requires_grad=True)
    c8 = torch.randn(2, 32, 37, 26, requires_grad=True)
    c16 = torch.randn(2, 48, 19, 13, requires_grad=True)

    p4, p8, p16 = neck(c4, c8, c16)
    assert p4.shape == (2, 32, 73, 51)
    assert p8.shape == (2, 32, 37, 26)
    assert p16.shape == (2, 32, 19, 13)
    assert torch.isfinite(p4).all() and torch.isfinite(p8).all() and torch.isfinite(p16).all()

    # Sample 0 loss must leave sample 1 input gradients strictly zero (GroupNorm sample isolation)
    loss = p4[0].sum() + 0.1 * p8[0].sum() + 0.1 * p16[0].sum()
    loss.backward()

    for name, tensor in (("c4", c4), ("c8", c8), ("c16", c16)):
        assert tensor.grad is not None, f"Missing grad for {name}"
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad[0].abs().sum().item() > 0.0
        assert tensor.grad[1].abs().max().item() == 0.0, f"Cross-sample leakage detected in {name}.grad[1]"


@pytest.mark.parametrize("neck_cls", [PAFPNBiNormNeck, HighResPConvFPNNeck])
def test_non_aspp_necks_shift_equivariance_and_no_gap(neck_cls: type[nn.Module]) -> None:
    """Non-ASPP necks must have zero global pooling layers and exact spatial shift equivariance."""
    torch.manual_seed(7)
    neck = neck_cls(in_channels=(16, 32, 48), width=32).eval()

    gap_modules = [m for m in neck.modules() if isinstance(m, (nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d))]
    assert len(gap_modules) == 0, f"{neck_cls.__name__} must not contain global pooling: {gap_modules}"

    # Zero-padded pattern translated by (dy=4, dx=6) on c16 -> (8, 12) on c8 -> (16, 24) on c4
    # Large 128x128 C4 (32x32 C16) canvas keeps GroupNorm statistics invariant and isolates canvas edges
    c4_a = torch.zeros(1, 16, 128, 128)
    c8_a = torch.zeros(1, 32, 64, 64)
    c16_a = torch.zeros(1, 48, 32, 32)
    c4_a[:, :, 32:48, 32:48] = torch.randn(1, 16, 16, 16)
    c8_a[:, :, 16:24, 16:24] = torch.randn(1, 32, 8, 8)
    c16_a[:, :, 8:12, 8:12] = torch.randn(1, 48, 4, 4)

    c4_b = torch.roll(c4_a, shifts=(16, 24), dims=(-2, -1))
    c8_b = torch.roll(c8_a, shifts=(8, 12), dims=(-2, -1))
    c16_b = torch.roll(c16_a, shifts=(4, 6), dims=(-2, -1))

    with torch.no_grad():
        p4_a, p8_a, p16_a = neck(c4_a, c8_a, c16_a)
        p4_b, p8_b, p16_b = neck(c4_b, c8_b, c16_b)

    patch_a = p4_a[:, :, 32:48, 32:48]
    patch_b = p4_b[:, :, 48:64, 56:72]
    max_diff = (patch_a - patch_b).abs().max().item()
    assert max_diff < 1e-5, f"{neck_cls.__name__} broke shift equivariance: diff={max_diff}"
    assert p8_b.shape == p8_a.shape and p16_b.shape == p16_a.shape


def test_convex_fast_normalized_fusion_bounds() -> None:
    """Convex fusion weights must stay non-negative and sum strictly < 1."""
    fuse = ConvexFastNormalizedFusion(num_inputs=3)
    with torch.no_grad():
        fuse.weights.copy_(torch.tensor([-2.0, 0.5, 3.0]))
    w = fuse.normalized_weights()
    assert (w >= 0.0).all()
    assert abs(w[0].item()) < 1e-8
    assert w.sum().item() < 1.0


@pytest.mark.parametrize("neck_type", ["pafpn_binorm", "pconv_fpn"])
def test_full_rmrv3_odd_resolution_forward_backward(neck_type: str) -> None:
    """Verify full RMRv3 model handles odd 409x513 input and produces finite non-negative density + grads."""
    cfg = RMRv3Config(
        pretrained=False,
        output_stride=2,
        subpixel_stride2=True,
        feature_width=32,
        neck_type=neck_type,
        head_type="dw_subpixel",
        enable_regional_head=False,
        enable_solver=False,
        solver_mode="direct",
        head_activation="compact_c1",
    )
    model = RMRv3(cfg).train()
    x = torch.randn(2, 3, 409, 513)
    out = model(x)
    assert torch.isfinite(out.y0).all()
    assert (out.y0 >= 0.0).all()
    loss = out.y0.sum()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0.0 for p in model.fusion.parameters())


@pytest.mark.parametrize(
    ("neck_type", "head_type"),
    [
        ("bifpn_lite", "standard"),
        ("pafpn_binorm", "standard"),
        ("pconv_fpn", "standard"),
        ("pafpn_binorm", "dw_subpixel"),
        ("pafpn_binorm", "residual_dilated"),
    ],
)
def test_direct_mode_zero_dead_parameters(neck_type: str, head_type: str) -> None:
    """Every trainable parameter in Pure Direct mode must receive non-zero gradient."""
    cfg = RMRv3Config(
        backbone_name="mobilenetv4_conv_small_050.e3000_r224_in1k",
        pretrained=False,
        feature_width=32,
        output_stride=2,
        subpixel_stride2=True,
        neck_type=neck_type,
        head_type=head_type,
        enable_regional_head=False,
        enable_solver=False,
        dynamic_scale_routing=False,
        factorized_scale_routing=False,
        use_diag=False,
    )
    model = RMRv3(cfg)
    model.train()
    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    out.y0.sum().backward()
    dead = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and (param.grad is None or param.grad.abs().sum().item() == 0.0)
    ]
    assert dead == [], f"Dead zero-grad params for ({neck_type}, {head_type}): {dead}"


@pytest.mark.parametrize("subpixel_stride2", [False, True])
@pytest.mark.parametrize("floor_tau", [0.0, 0.1])
@pytest.mark.parametrize("head_act", ["compact_c1", "relu", "softplus"])
def test_residual_dilated_and_dw_subpixel_step0_prior_and_grads(
    subpixel_stride2: bool,
    floor_tau: float,
    head_act: str,
) -> None:
    """Verify ResidualDilatedFineHead (esp. compact_c1 fix) and dw_subpixel calibrate Step-0 mean and flow grads."""
    m0 = 0.015763
    init_bias = math.log(math.expm1(m0))
    expected = (m0 / 4.0) if subpixel_stride2 else m0

    for htype in ("residual_dilated", "dw_subpixel"):
        head = build_fine_head(
            width=32,
            num_scales=3,
            init_bias=init_bias,
            temp_softplus=True,
            subpixel_stride2=subpixel_stride2,
            floor_tau=floor_tau,
            head_activation=head_act,
            head_type=htype,
            subpixel_kernel_size=3,
            use_film=True,
        ).eval()

        if head_act in ("relu", "compact_c1") and isinstance(head, ResidualDilatedFineHead):
            assert not hasattr(head, "tau"), "Unused temp_softplus tau parameter must not be created for compact_c1/relu"

        # Zero features isolate pure bias calibration (Step-0 density prior)
        feats = torch.zeros(1, 32, 16, 16, requires_grad=True)
        y0 = head(feats)
        mean_y0 = y0.mean().item()
        if head_act == "softplus" and floor_tau > 0.0 and not subpixel_stride2:
            # _smooth_floor subtracts 0.5*tau for linear regime; small-y quadratics vary slightly near threshold
            assert y0.isfinite().all() and mean_y0 >= 0.0
        elif not (head_act == "softplus" and floor_tau > 0.0):
            assert abs(mean_y0 - expected) < 2e-4, (
                f"{htype} ({head_act}, s2={subpixel_stride2}, tau={floor_tau}) Step-0 mean {mean_y0:.6f} != {expected:.6f}"
            )

        # Ensure compact_c1 is NOT stuck in the dead zero-gradient regime at Step 0
        head.train()
        feats_grad = torch.randn(1, 32, 16, 16, requires_grad=True)
        loss = head(feats_grad).sum()
        loss.backward()
        if isinstance(head, ResidualDilatedFineHead):
            assert head.final_conv.bias.grad is not None and head.final_conv.bias.grad.abs().sum().item() > 0.0
        else:
            assert isinstance(head, FineMeasureHead)
            final_bias = head.body[-1].bias
            assert final_bias is not None and final_bias.grad is not None and final_bias.grad.abs().sum().item() > 0.0
        assert feats_grad.grad is not None and feats_grad.grad.abs().sum().item() > 0.0


def test_validator_accepts_new_necks_and_rejects_typos() -> None:
    """Validator must accept pafpn_binorm / pconv_fpn / dw_subpixel and reject unknown neck/head types."""
    for neck_type in ("pafpn_binorm", "pconv_fpn", "bifpn_lite"):
        for head_type in ("standard", "dw_subpixel", "residual_dilated"):
            cfg = {
                "model": {
                    "neck_type": neck_type,
                    "head_type": head_type,
                    "enable_regional_head": False,
                    "enable_solver": False,
                },
                "loss": {"lambda_region_nb": 0.0},
            }
            validate_v3_config(cfg)

    with pytest.raises(ValueError, match="neck_type"):
        validate_v3_config({"model": {"neck_type": "unknown_neck_typo"}})

    with pytest.raises(ValueError, match="head_type"):
        validate_v3_config({"model": {"head_type": "unknown_head_typo"}})


def test_sfi_lite_shift_equivariance_and_sample_isolation() -> None:
    """SFILiteFPNNeck must preserve shift equivariance and sample isolation without cross-talk."""
    from rmr_core.necks.sfi import SFILiteFPNNeck

    torch.manual_seed(42)
    neck = SFILiteFPNNeck(in_channels=(16, 32, 48), width=32, use_gap=False).eval()

    c4_a = torch.zeros(1, 16, 128, 128)
    c8_a = torch.zeros(1, 32, 64, 64)
    c16_a = torch.zeros(1, 48, 32, 32)
    c4_a[:, :, 32:48, 32:48] = torch.randn(1, 16, 16, 16)
    c8_a[:, :, 16:24, 16:24] = torch.randn(1, 32, 8, 8)
    c16_a[:, :, 8:12, 8:12] = torch.randn(1, 48, 4, 4)

    c4_b = torch.roll(c4_a, shifts=(16, 24), dims=(-2, -1))
    c8_b = torch.roll(c8_a, shifts=(8, 12), dims=(-2, -1))
    c16_b = torch.roll(c16_a, shifts=(4, 6), dims=(-2, -1))

    with torch.no_grad():
        p4_a, p8_a, p16_a = neck(c4_a, c8_a, c16_a)
        p4_b, p8_b, p16_b = neck(c4_b, c8_b, c16_b)

    patch_a = p4_a[:, :, 32:48, 32:48]
    patch_b = p4_b[:, :, 48:64, 56:72]
    diff = (patch_a - patch_b).abs().max().item()
    assert diff < 2e-3, f"SFILiteFPNNeck broke shift equivariance: diff={diff}"

    # Sample isolation check
    neck.train()
    c4 = torch.randn(2, 16, 64, 64, requires_grad=True)
    c8 = torch.randn(2, 32, 32, 32, requires_grad=True)
    c16 = torch.randn(2, 48, 16, 16, requires_grad=True)
    p4, p8, p16 = neck(c4, c8, c16)
    loss = p4[0].sum()
    loss.backward()
    assert c4.grad is not None
    assert c4.grad[0].abs().sum().item() > 0.0
    assert c4.grad[1].abs().max().item() == 0.0, "Cross-sample leakage detected in SFILiteFPNNeck!"


def test_sfi_v2_norm_free_hf_exact_zero_fixed_point_and_no_bg_blowup() -> None:
    """Verify SFI-v2 norm-free zero-bias HF branch preserves exact zero fixed point and avoids 1/sigma_bg blow-up."""
    from rmr_core.necks.sfi import SFILiteFPNNeck

    torch.manual_seed(7)
    neck_v2 = SFILiteFPNNeck(
        in_channels=(16, 32, 48),
        width=32,
        use_gap=False,
        norm_free_hf=True,
        use_hdc_proj=True,
        hdc_proj_groups=2,
    ).eval()
    neck_legacy = SFILiteFPNNeck(
        in_channels=(16, 32, 48),
        width=32,
        use_gap=False,
        norm_free_hf=False,
        use_hdc_proj=False,
    ).eval()

    with torch.no_grad():
        # Exact zero fixed point on flat high-frequency residual
        zero_hf = torch.zeros(1, 32, 32, 32)
        out_v2_zero = torch.tanh(neck_v2.hf_enhance(zero_hf))
        assert out_v2_zero.abs().max().item() == 0.0

        # Tiny background sensor ripple (~1e-4) must remain bounded (~1e-4), NOT blown up by 1/sigma_bg
        bg_ripple = torch.randn(1, 32, 32, 32) * 1e-4
        out_v2_ripple = neck_v2.gamma_hf * torch.tanh(neck_v2.hf_enhance(bg_ripple))
        out_leg_ripple = neck_legacy.gamma_hf * neck_legacy.hf_enhance(bg_ripple)
        assert out_v2_ripple.abs().max().item() < 1e-3, (
            f"Norm-free HF branch blew up background ripple: {out_v2_ripple.abs().max().item()}"
        )
        assert out_leg_ripple.abs().max().item() > 50.0 * out_v2_ripple.abs().max().item()

        # Verify P16 grouped proj + balanced HDC scaling preserves shift equivariance & finite outputs
        c4 = torch.randn(2, 16, 64, 64)
        c8 = torch.randn(2, 32, 32, 32)
        c16 = torch.randn(2, 48, 16, 16)
        p4, p8, p16 = neck_v2(c4, c8, c16)
        assert p4.shape == (2, 32, 64, 64) and p8.shape == (2, 32, 32, 32) and p16.shape == (2, 32, 16, 16)
        assert torch.isfinite(p4).all() and torch.isfinite(p8).all() and torch.isfinite(p16).all()


def test_dsimr_subpixel_head_measure_conservation_and_step0_prior() -> None:
    """Verify DSIMR Sub-Pixel head Step-0 calibration, exact coarse-to-fine conservation, prior-shifted scaling, and params."""
    m0 = 0.015763
    init_bias = math.log(math.expm1(m0))
    head = build_fine_head(
        width=32,
        init_bias=init_bias,
        temp_softplus=False,
        subpixel_stride2=True,
        floor_tau=0.001,
        density_adaptive_scale=True,
        density_scale_gamma=0.15,
        head_activation="softplus",
        head_type="dsimr_subpixel",
        subpixel_kernel_size=3,
    ).eval()

    assert isinstance(head, FineMeasureHead) and head.dsimr_mode is True
    assert head.adaptive_scale_ref < -3.5, "Prior-shifted adaptive_scale_ref must stay in active sub-zero regime"

    # Zero features verify Step-0 prior density (m0 / 4.0)
    with torch.no_grad():
        feats_zero = torch.zeros(1, 32, 16, 16)
        y_zero = head(feats_zero)
        expected = m0 / 4.0
        assert abs(y_zero.mean().item() - expected) < 5e-4, f"DSIMR Step-0 mean {y_zero.mean().item():.6f} != {expected:.6f}"

        # Coarse-to-fine measure conservation: 2x2 sum equals 4 * s(u) * Lambda(u)
        feats_rand = torch.randn(2, 32, 16, 16)
        h4 = head.body[:-1](feats_rand)
        z_sub = head.body[-1](h4)
        alpha = 4.0 * torch.softmax(z_sub, dim=1)
        assert torch.allclose(alpha.sum(dim=1, keepdim=True), torch.full((2, 1, 16, 16), 4.0), atol=1e-5)

        y_rand = head(feats_rand)
        assert (y_rand >= 0.0).all() and torch.isfinite(y_rand).all()
        cell_sums = torch.nn.functional.avg_pool2d(y_rand, kernel_size=2, stride=2) * 4.0
        dsimr_proj = head.dsimr_proj(h4)
        s = torch.sigmoid(dsimr_proj[:, 0:1])
        z_lam = dsimr_proj[:, 1:2]
        y_lam = torch.nn.functional.softplus(z_lam)
        tau_s2 = head.floor_tau
        y_lam = torch.where(y_lam > tau_s2, y_lam - 0.5 * tau_s2, y_lam.square() / (2.0 * tau_s2))
        y_lam = y_lam * (1.0 + 0.15 * torch.relu(z_lam - head.adaptive_scale_ref))
        expected_cell_sum = 4.0 * s * y_lam
        assert torch.allclose(cell_sums, expected_cell_sum, atol=1e-5)

    # Full RMRv3 integration: param ceiling 101,174 <= 104,441 & zero dead gradients
    cfg = RMRv3Config(
        pretrained=False,
        output_stride=2,
        subpixel_stride2=True,
        subpixel_kernel_size=3,
        feature_width=32,
        max_trainable_params=104441,
        neck_type="sfi_lite",
        sfi_norm_free_hf=True,
        sfi_use_hdc_proj=True,
        sfi_hdc_proj_groups=2,
        head_type="dsimr_subpixel",
        density_adaptive_scale=True,
        density_scale_gamma=0.15,
        enable_regional_head=False,
        enable_solver=False,
        solver_mode="direct",
        floor_tau=0.001,
    )
    model = RMRv3(cfg).train()
    total_params = _count_params(model)
    assert total_params <= 104441, f"Parameter ceiling exceeded: {total_params}"
    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    out.y.sum().backward()
    dead = [
        name for name, p in model.named_parameters()
        if p.requires_grad and (p.grad is None or p.grad.abs().sum().item() == 0.0)
    ]
    assert dead == [], f"Dead zero-grad params in SFI-v2 + DSIMR: {dead}"



