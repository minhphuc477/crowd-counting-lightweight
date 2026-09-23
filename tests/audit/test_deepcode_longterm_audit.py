"""tests/audit/test_deepcode_longterm_audit.py

Adversarial, non-superficial audit test suite for RMR long-term architectural
hardening, parameter integrity, resume compatibility, sample isolation,
and mathematical invariants.
"""

from __future__ import annotations

import glob
import pytest
import torch
import torch.nn as nn

from rmr_core.heads import FineMeasureHead
from rmr_core.operators import charbonnier_tv_step
from rmr_core.scale_routing import FactorizedRoutingHead
from rmr_core.types import RMRModelOutput
from rmr_v3.config.provenance import load_config
from rmr_v3.config.resume import validate_resume_compatibility
from rmr_v3.config.validator import validate_v3_config
from rmr_v3.model.architecture import RMRv3
from rmr_v3.model.config import RMRv3Config


def test_rmr_output_dynamic_attribute_safety():
    """RMRModelOutput should safely return None for optional fields without AttributeError."""
    dummy_y = torch.zeros(1, 1, 32, 32)
    dummy_y0 = torch.zeros(1, 1, 32, 32)
    output = RMRModelOutput(
        y=dummy_y,
        y0=dummy_y0,
        m0=torch.tensor([1.0]),
        count=torch.tensor([1.0]),
        raw_count=torch.tensor([1.0]),
    )

    # Optional fields should return None safely
    assert output.pi_aspect is None
    assert output.y_carrier is None
    assert output.y0_carrier is None
    assert output.carrier_energy is None

    # Truly invalid fields must still raise AttributeError
    with pytest.raises(AttributeError, match="has no attribute 'unregistered_dummy_attr'"):
        _ = output.unregistered_dummy_attr


def test_factorized_routing_head_boundary_guards():
    """FactorizedRoutingHead must fail-fast with ValueError on invalid scale/aspect counts."""
    # Invalid num_scales != 3
    with pytest.raises(ValueError, match="supports exactly 3 marginal scales"):
        FactorizedRoutingHead(in_channels=32, num_scales=4, num_aspects=2)

    # Invalid num_aspects != 2
    with pytest.raises(ValueError, match="supports exactly 3 marginal scales and 2 aspect ratios"):
        FactorizedRoutingHead(in_channels=32, num_scales=3, num_aspects=3)

    # Valid configuration should construct cleanly
    head = FactorizedRoutingHead(in_channels=32, num_scales=3, num_aspects=2)
    assert head.pw_scale.out_channels == 3
    assert head.pw_aspect.out_channels == 2


def test_curvature_alpha_init_configurability():
    """FineMeasureHead and RMRv3Config must allow configurable curvature initialization."""
    cfg_default = RMRv3Config(curvature_alpha_init=-8.0)
    assert cfg_default.curvature_alpha_init == -8.0

    cfg_custom = RMRv3Config(curvature_alpha_init=-4.0)
    assert cfg_custom.curvature_alpha_init == -4.0

    head_default = FineMeasureHead(width=32, density_curvature=True, curvature_alpha_init=-8.0)
    head_custom = FineMeasureHead(width=32, density_curvature=True, curvature_alpha_init=-4.0)

    assert head_default.curvature_alpha.item() == pytest.approx(-8.0, rel=1e-4)
    assert head_custom.curvature_alpha.item() == pytest.approx(-4.0, rel=1e-4)


def test_method_critical_fields_resume_validation():
    """validate_resume_compatibility must flag alterations in critical architecture and loss fields."""
    base_cfg = {
        "model": {
            "output_stride": 4,
            "feature_width": 32,
            "subpixel_stride2": False,
            "curvature_alpha_init": -8.0,
        },
        "loss": {
            "allocation_loss_type": "flat_dm16",
            "spectral_bandpass": True,
            "spectral_omega_0": 0.0,
            "spectral_omega_low": 0.02,
            "spectral_omega_high": 0.35,
        },
    }

    # 1. Compatible match returns None without error
    assert validate_resume_compatibility(base_cfg, base_cfg) is None

    # 2. Incompatible subpixel_stride2
    incompat_model = {
        "model": {**base_cfg["model"], "subpixel_stride2": True},
        "loss": base_cfg["loss"],
    }
    with pytest.raises(ValueError, match="Resume config mismatch for 'model.subpixel_stride2'"):
        validate_resume_compatibility(base_cfg, incompat_model)

    # 3. Incompatible spectral_bandpass
    incompat_loss = {
        "model": base_cfg["model"],
        "loss": {**base_cfg["loss"], "spectral_bandpass": False},
    }
    with pytest.raises(ValueError, match="Resume config mismatch for 'loss.spectral_bandpass'"):
        validate_resume_compatibility(base_cfg, incompat_loss)


def test_all_new_research_configs_parameter_and_validation():
    """All research configurations must pass strict validation and obey <= 105,000 parameter budget."""
    new_configs = [
        "configs/rmr_research/h11_seed123.yaml",
        "configs/rmr_research/h11_seed456.yaml",
        "configs/rmr_research/h11_shb_canonical.yaml",
        "configs/rmr_research/h11_abl_no_solver.yaml",
        "configs/rmr_research/h11_abl_no_resonant.yaml",
        "configs/rmr_research/h11_abl_no_spectral.yaml",
        "configs/rmr_research/h11_abl_no_ci_cell.yaml",
        "configs/rmr_research/h11_abl_soft_proximal.yaml",
        "configs/rmr_research/h11_abl_no_proximal.yaml",
        "configs/rmr_research/h11_abl_no_tv.yaml",
        "configs/rmr_research/h11_charbonnier_tv.yaml",
        "configs/rmr_research/h13_harmonious_frontier.yaml",
        "configs/rmr_research/h13_seed123.yaml",
        "configs/rmr_research/h13_seed456.yaml",
        "configs/rmr_research/h13_shb_canonical.yaml",
        "configs/rmr_research/h13_abl_no_cpcm.yaml",
        "configs/rmr_research/h13_abl_no_asam.yaml",
        "configs/rmr_research/h13_abl_no_bb.yaml",
    ]

    for cfg_path in new_configs:
        raw = load_config(cfg_path)
        validate_v3_config(raw)
        m_cfg = RMRv3Config.from_dict(raw.get("model", {}))
        m_cfg.pretrained = False  # Avoid downloading weights during fast unit tests
        model = RMRv3(m_cfg)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert (
            trainable_params <= 105000
        ), f"{cfg_path} parameter count {trainable_params} violates budget <= 105,000!"


def test_sample_isolation_batch_gradient_orthogonality():
    """Gradients of sample i loss with respect to sample j input (i != j) must be strictly zero."""
    m_cfg = RMRv3Config(
        output_stride=4,
        feature_width=32,
        pretrained=False,
        enable_solver=True,
        iterations=2,
    )
    model = RMRv3(m_cfg)
    model.eval()

    # Batch of 2 images with requires_grad
    x = torch.randn(2, 3, 128, 128, requires_grad=True)
    out = model(x)

    # Loss computed only on sample 0
    loss_0 = out.y[0].sum()
    loss_0.backward(retain_graph=True)

    # Gradient of sample 0 loss w.r.t sample 1 input must be exactly zero
    assert x.grad is not None
    grad_sample_1 = x.grad[1]
    assert torch.all(grad_sample_1 == 0.0), "Cross-sample gradient leakage detected!"


def test_arbitrary_odd_prime_resolution_forward():
    """Forward pass must succeed on arbitrary odd/prime resolutions without size mismatches."""
    m_cfg = RMRv3Config(
        output_stride=4,
        feature_width=32,
        pretrained=False,
        enable_solver=True,
        iterations=2,
    )
    model = RMRv3(m_cfg)
    model.eval()

    # Odd/prime test resolutions
    shapes = [(1, 3, 117, 233), (1, 3, 211, 307)]
    with torch.no_grad():
        for shape in shapes:
            inp = torch.randn(*shape)
            out = model(inp)
            assert out.y is not None
            assert out.y.ndim == 4
            expected_h = (shape[2] + 3) // 4
            expected_w = (shape[3] + 3) // 4
            assert out.y.shape[-2:] == (expected_h, expected_w)


def test_discrete_mass_conservation_tv_step():
    """Charbonnier TV step divergence must conserve total mass over the discrete spatial grid."""
    # Test mass conservation: sum(charbonnier_tv_step(y)) == sum(y) with zero flux boundary
    y = torch.ones(2, 1, 32, 32) * 5.0
    y[:, :, 10:20, 10:20] += 2.0
    lambda_tv = 0.01
    eps_c = 0.1
    y_diffused = charbonnier_tv_step(y, lambda_tv=lambda_tv, eps_c=eps_c)

    mass_orig = y.sum(dim=(-2, -1))
    mass_diff = y_diffused.sum(dim=(-2, -1))

    diff = torch.abs(mass_diff - mass_orig)
    assert torch.all(diff < 1e-4), f"TV step violated mass conservation: max diff = {diff.max().item()}"
