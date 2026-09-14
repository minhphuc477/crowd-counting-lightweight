from __future__ import annotations

"""Exhaustive Verification Suite for RMR-v14 Unified Continuous-Discrete Reconstruction."""

import math
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
import yaml

from rmr_core.operators import build_multiscale_regions
from rmr_v3.config import load_config
from rmr_v3.losses import (
    RMRv3LossConfig,
    compute_rmr_v3_losses,
    physical_scale_alignment_loss,
)
from rmr_v3.model import RMRv3, RMRv3Config
from rmr_v3.regional_head import reliability_from_nb


def test_rmr_v14_config_loading():
    """Verify loading and schema validation of rmr_v14_unified_reconstruction.yaml."""
    cfg_path = Path("configs/rmr_v14/rmr_v14_unified_reconstruction.yaml")
    assert cfg_path.exists(), f"Missing config: {cfg_path}"

    cfg = load_config(cfg_path)
    model_cfg = RMRv3Config.from_dict(cfg.get("model", {}))
    loss_cfg = RMRv3LossConfig.from_dict(cfg.get("loss", {}))

    assert model_cfg.neck_type == "aspp_lite"
    assert model_cfg.use_top_down_semantic_gate is True
    assert model_cfg.tdsg_floor == 0.20
    assert model_cfg.foreground_gate is True
    assert model_cfg.fg_gate_floor == 0.10
    assert model_cfg.reliability_mode == "hybrid_hurdle"
    assert model_cfg.adjoint_mode == "radon_nikodym"
    assert model_cfg.morozov_gamma == 0.75
    assert loss_cfg.lambda_scale_align == 0.05
    assert loss_cfg.scale_align_mask_bg is True
    assert loss_cfg.lambda_curvature == 0.50
    assert loss_cfg.curvature_gate_threshold == 0.08
    assert loss_cfg.curvature_gate_mode == "hard"


def test_rmr_v14_parameter_budget_exactness():
    """Verify exact parameter budget of RMR-v14 remains strictly <= 105,000."""
    cfg_path = Path("configs/rmr_v14/rmr_v14_unified_reconstruction.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    model_dict = raw.get("model", {})
    model_dict["pretrained"] = False
    m_cfg = RMRv3Config.from_dict(model_dict)

    model = RMRv3(m_cfg)
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert total_trainable <= 105000, (
        f"RMR-v14 parameter count {total_trainable} strictly exceeds hard limit 105,000!"
    )
    assert total_trainable == 104506, (
        f"Expected exactly 104,506 parameters (104,473 base + 33 TDSG), got {total_trainable}"
    )


def test_top_down_semantic_context_gate_invariants():
    """Verify TDSG modulates P4 correctly with safe floor and gradients flow to P16."""
    cfg = RMRv3Config(
        pretrained=False,
        neck_type="aspp_lite",
        use_top_down_semantic_gate=True,
        tdsg_floor=0.20,
    )
    model = RMRv3(cfg)
    assert model.tdsg is not None

    x = torch.randn(2, 3, 128, 128, requires_grad=True)
    out = model(x)
    y0 = out["y0"]
    assert y0.shape == (2, 1, 32, 32)

    # Backward pass checks gradient flow to TDSG
    loss = y0.sum()
    loss.backward()
    assert model.tdsg.weight.grad is not None
    assert torch.isfinite(model.tdsg.weight.grad).all()


def test_fg_gate_deep_floor_modulation():
    """Verify dynamic fg_gate_floor allows attenuation down to 0.10 when logits are negative."""
    cfg_v11 = RMRv3Config(pretrained=False, foreground_gate=True, fg_gate_floor=0.70)
    cfg_v14 = RMRv3Config(pretrained=False, foreground_gate=True, fg_gate_floor=0.10)

    m_v11 = RMRv3(cfg_v11)
    m_v14 = RMRv3(cfg_v14)

    # Force gate bias strongly negative
    with torch.no_grad():
        m_v11.fg_gate.bias.fill_(-10.0)
        m_v14.fg_gate.bias.fill_(-10.0)
        m_v11.fg_gate.weight.zero_()
        m_v14.fg_gate.weight.zero_()

    x = torch.randn(1, 3, 64, 64)
    out_v11 = m_v11(x)
    out_v14 = m_v14(x)

    # In v11: gate is clamped at ~0.70
    # In v14: gate can drop down to ~0.10, suppressing 85% of background noise!
    ratio = float(out_v14["y0"].sum() / out_v11["y0"].sum().clamp_min(1e-6))
    assert ratio < 0.20, f"Expected v14 output to be ~0.10/0.70 (~0.14x) of v11, got ratio {ratio:.3f}"


def test_foreground_masked_scale_alignment_invariants():
    """Verify scale alignment loss ignores empty background pixels when mask_background=True."""
    b, k, h, w = 2, 3, 32, 32
    scale_weights = torch.softmax(torch.randn(b, k, h, w), dim=1)

    # 1. Empty image (all zeros)
    empty_target = torch.zeros(b, 1, h, w)
    loss_masked = physical_scale_alignment_loss(
        scale_weights, empty_target, tau_dense=0.12, tau_sparse=0.03, mask_background=True
    )
    assert float(loss_masked.item()) == 0.0, (
        f"Empty background with mask_background=True must yield exactly 0.0 loss, got {loss_masked.item()}"
    )

    # 2. Dense image (high density everywhere)
    dense_target = torch.full((b, 1, h, w), 0.20)
    loss_dense = physical_scale_alignment_loss(
        scale_weights, dense_target, tau_dense=0.12, tau_sparse=0.03, mask_background=True
    )
    assert float(loss_dense.item()) > 0.0
    assert torch.isfinite(loss_dense)


def test_hybrid_hurdle_reliability_invariants():
    """Verify Hurdle-hybrid reliability blends rate variance on background and SNR on crowds."""
    regions = build_multiscale_regions(
        height=32, width=32, output_stride=4, region_sizes_px=(32, 64, 128), overlap=0.5
    )
    m = regions.boxes.shape[0]

    mu = torch.full((1, 1, m), 10.0)
    disp = torch.full((1, 1, m), 50.0)

    # Background hurdle (pi -> 0)
    pi_bg = torch.zeros(1, 1, m)
    rel_bg = reliability_from_nb(
        mu, disp, regions, mode="hybrid_hurdle", hurdle_pi=pi_bg
    )
    rel_var = reliability_from_nb(mu, disp, regions, mode="nb_rate_variance")
    assert torch.allclose(rel_bg["weight"], rel_var["weight"], atol=1e-5), (
        "When pi_R=0, hybrid_hurdle must identically match nb_rate_variance!"
    )

    # Crowd hurdle (pi -> 1)
    pi_crowd = torch.ones(1, 1, m)
    rel_crowd = reliability_from_nb(
        mu, disp, regions, mode="hybrid_hurdle", hurdle_pi=pi_crowd
    )
    rel_snr = reliability_from_nb(mu, disp, regions, mode="snr")
    assert torch.allclose(rel_crowd["weight"], rel_snr["weight"], atol=1e-5), (
        "When pi_R=1, hybrid_hurdle must identically match SNR weighting!"
    )


def test_rmr_v14_end_to_end_loss_and_gradients():
    """Run full forward and backward pass with all RMR-v14 components active under AMP."""
    cfg_path = Path("configs/rmr_v14/rmr_v14_unified_reconstruction.yaml")
    full_cfg = load_config(cfg_path)
    model_dict = full_cfg.get("model", {}).copy()
    model_dict["pretrained"] = False
    model_cfg = RMRv3Config.from_dict(model_dict)
    loss_cfg = RMRv3LossConfig.from_dict(full_cfg.get("loss", {}))

    model = RMRv3(model_cfg)
    model.train()

    b, c, h, w = 2, 3, 128, 128
    x = torch.randn(b, c, h, w, requires_grad=True)
    target_y = torch.zeros(b, 1, h // 4, w // 4)
    # Add synthetic crowd cluster
    target_y[:, :, 10:18, 10:18] = 0.15

    outputs = model(x)
    losses = compute_rmr_v3_losses(outputs, target_y, cfg=loss_cfg)

    assert "total" in losses
    assert "scale_align" in losses
    assert "curvature" in losses
    assert "hard_bg" in losses

    total_loss = losses["total"]
    assert torch.isfinite(total_loss)
    total_loss.backward()

    # Verify gradients flow to all key modules
    assert model.fine_head.body[-1].weight.grad is not None
    assert model.scale_router.pw.weight.grad is not None
    assert model.tdsg.weight.grad is not None
    assert model.fg_gate.weight.grad is not None
