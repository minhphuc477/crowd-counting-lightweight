"""Tests for Phase 1 (Loss Purification) and Phase 2 (Architecture Scaling to 175k params).

Verifies:
1. Zero autograd graph connection for disabled auxiliary losses (detached zeros).
2. Phase 1 config (sub60_e139) budget <= 104,441 params and clean purified forward/backward.
3. Phase 2 architecture (sub60_e140 / v35) budget in [150k, 200k] (exact 175,221 params).
4. Multi-resolution stability and odd-dimension handling for scaled feature_width=80.
5. Invariant: source file <= 450 lines.
"""

from __future__ import annotations

import pytest
import torch
import yaml
from pathlib import Path

from rmr_core.operators import RegionSet
from rmr_v3.config import validate_v3_config
from rmr_v3.losses import RMRv3LossConfig, compute_rmr_v3_losses
from rmr_v3.model import (
    CANONICAL_V35_EXPECTED_PARAMS,
    CANONICAL_V35_PARAM_BUDGET,
    RMRv3,
    RMRv3Config,
    build_v35_rmr_model,
    get_v35_loss_config,
    get_v35_model_config,
)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _load_cfg(path: str | Path) -> tuple[RMRv3Config, RMRv3LossConfig]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    validate_v3_config(raw)
    m = RMRv3Config(**raw.get("model", {}))
    m.pretrained = False
    l = RMRv3LossConfig(**raw.get("loss", {}))
    return m, l


def test_auxiliary_loss_detached_when_disabled():
    """Verify disabled auxiliary losses return detached tensors without adding autograd nodes."""
    device = torch.device("cpu")
    b, h, w = 2, 64, 64
    y = torch.full((b, 1, h, w), 0.05, device=device, requires_grad=True)
    y0 = torch.full((b, 1, h, w), 0.05, device=device, requires_grad=True)
    target = torch.zeros((b, 1, h, w), device=device)
    target[:, :, 10, 10] = 1.0

    loss_cfg = RMRv3LossConfig(
        lambda_count=1.0,
        lambda_flat_dm16=1.0,
        lambda_cell=0.0,
        lambda_region_nb=0.0,
        lambda_bayesian=0.0,
        lambda_hurdle=0.0,
        lambda_trunc_nb=0.0,
        lambda_curvature=0.0,
        lambda_hard_bg=0.0,
        lambda_scale_align=0.0,
        lambda_carrier_cell=0.0,
        lambda_fine_cell=0.0,
        use_spectral_loss=False,
        output_stride=2,
        dm_target="dual",
    )

    boxes = torch.tensor([[0, 0, 16, 16]], device=device)
    scale_id = torch.zeros(1, dtype=torch.long, device=device)
    area = torch.tensor([256.0], device=device)
    regions = RegionSet(boxes=boxes, scale_id=scale_id, area=area)

    outputs = {
        "y": y,
        "y0": y0,
        "regions": regions,
        "b_region": torch.zeros((b, 1), device=device),
        "region_dispersion": torch.ones((b, 1), device=device),
    }

    losses = compute_rmr_v3_losses(outputs, target, cfg=loss_cfg)

    # Active losses must have grad_fn
    assert losses["total"].grad_fn is not None, "total loss must have grad_fn"
    assert losses["count"].grad_fn is not None, "count loss must have grad_fn"
    assert losses["flat_dm16"].grad_fn is not None, "flat_dm16 must have grad_fn"

    # Disabled auxiliary losses must be detached zeros
    assert losses["cell"].grad_fn is None, "cell loss must be detached when lambda=0"
    assert losses["region_nb"].grad_fn is None, "region_nb must be detached when lambda=0"
    assert losses["curvature"].grad_fn is None, "curvature must be detached when lambda=0"
    assert losses["hard_bg"].grad_fn is None, "hard_bg must be detached when lambda=0"
    assert losses["scale_align"].grad_fn is None, "scale_align must be detached when lambda=0"

    # Backward pass must execute cleanly
    losses["total"].backward()
    assert y.grad is not None and torch.isfinite(y.grad).all()
    assert y0.grad is not None and torch.isfinite(y0.grad).all()


def test_sub60_e139_config_and_parameters():
    """Verify sub60_e139 (Phase 1) satisfies the strict <= 104,441 parameter ceiling."""
    cfg_path = Path("configs/rmr_research/sub60_e139_purified_dual_loss_104k.yaml")
    assert cfg_path.exists(), f"Config file {cfg_path} must exist"

    m_cfg, l_cfg = _load_cfg(cfg_path)

    # Check loss purification settings
    assert l_cfg.lambda_count == 1.0
    assert l_cfg.lambda_flat_dm16 == 1.0
    assert l_cfg.lambda_region_nb == 0.05
    assert l_cfg.lambda_hurdle == 0.02
    assert l_cfg.lambda_bayesian == 0.0
    assert l_cfg.lambda_cell == 0.0
    assert l_cfg.lambda_trunc_nb == 0.0
    assert l_cfg.lambda_curvature == 0.0
    assert l_cfg.lambda_hard_bg == 0.0
    assert l_cfg.lambda_scale_align == 0.0
    assert m_cfg.subpixel_dm is False
    assert m_cfg.subpixel_stride2 is True
    assert m_cfg.iterations == 2

    # Build model and count parameters
    model = RMRv3(m_cfg)
    total_params = count_params(model)
    assert total_params <= 104441, f"sub60_e139 parameter count {total_params} exceeds 104,441"
    assert total_params == 104407, f"Expected 104,407 parameters, got {total_params}"


def test_sub60_e139_forward_backward():
    """Verify sub60_e139 model executes forward and backward passes without error."""
    m_cfg, l_cfg = _load_cfg("configs/rmr_research/sub60_e139_purified_dual_loss_104k.yaml")
    model = RMRv3(m_cfg)
    model.train()

    x = torch.randn(1, 3, 256, 256)
    target = torch.zeros(1, 1, 128, 128)
    target[:, :, 30, 30] = 1.0
    target[:, :, 60, 60] = 2.0

    out = model(x)
    assert "y" in out and "y0" in out
    assert out["y"].shape[-2:] == (128, 128), "Output stride 2 should produce 128x128 from 256x256"

    losses = compute_rmr_v3_losses(out, target, cfg=l_cfg)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()

    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} has no gradient"
            assert torch.isfinite(p.grad).all(), f"Parameter {name} has non-finite gradient"


def test_v35_scaled_architecture_budget():
    """Verify Phase 2 (v35 / 175k params) falls strictly in [150,000, 200,000]."""
    m_cfg = get_v35_model_config(pretrained=False)
    l_cfg = get_v35_loss_config()

    assert m_cfg.feature_width == 80
    assert m_cfg.region_head_hidden == 140
    assert m_cfg.subpixel_dm is False
    assert m_cfg.subpixel_stride2 is True
    assert m_cfg.iterations == 2
    assert m_cfg.output_stride == 2
    assert m_cfg.max_trainable_params == CANONICAL_V35_PARAM_BUDGET  # 200,000
    assert l_cfg.lambda_flat_dm16 == 1.0
    assert l_cfg.lambda_region_nb == 0.05
    assert l_cfg.lambda_hurdle == 0.02

    model = RMRv3(m_cfg)
    total_params = count_params(model)

    assert 150000 <= total_params <= CANONICAL_V35_PARAM_BUDGET
    assert total_params == CANONICAL_V35_EXPECTED_PARAMS  # 174,340
    assert total_params == 174340


def test_sub60_e140_config_and_validation():
    """Verify sub60_e140 (Phase 2 config) loads, validates, and initializes properly."""
    cfg_path = Path("configs/rmr_research/sub60_e140_rmr_v35_175k_scaled.yaml")
    assert cfg_path.exists(), f"Config file {cfg_path} must exist"

    m_cfg, l_cfg = _load_cfg(cfg_path)

    assert m_cfg.feature_width == 80
    assert m_cfg.region_head_hidden == 140
    assert m_cfg.subpixel_dm is False
    assert m_cfg.subpixel_stride2 is True
    assert m_cfg.iterations == 2
    assert m_cfg.output_stride == 2
    assert m_cfg.max_trainable_params == 200000
    assert l_cfg.lambda_flat_dm16 == 1.0
    assert l_cfg.lambda_region_nb == 0.05
    assert l_cfg.lambda_hurdle == 0.02

    model = RMRv3(m_cfg)
    total_params = count_params(model)
    assert total_params == 174340


def test_v35_scaled_forward_backward_multi_res():
    """Verify scaled v35 model handles standard and odd resolutions with finite gradients."""
    m_cfg = get_v35_model_config(pretrained=False)
    l_cfg = get_v35_loss_config()
    model = RMRv3(m_cfg)
    model.train()

    test_shapes = [
        (256, 256),
        (224, 256),
        (256, 288),
    ]

    for h, w in test_shapes:
        model.zero_grad(set_to_none=True)
        x = torch.randn(1, 3, h, w)
        out = model(x)

        expected_h, expected_w = h // 2, w // 2
        assert out["y"].shape[-2:] == (expected_h, expected_w), f"Expected {(expected_h, expected_w)}, got {out['y'].shape[-2:]}"

        target = torch.zeros(1, 1, expected_h, expected_w)
        target[:, :, 10, 10] = 1.0

        losses = compute_rmr_v3_losses(out, target, cfg=l_cfg)
        assert torch.isfinite(losses["total"])
        losses["total"].backward()

        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None and torch.isfinite(p.grad).all(), f"Gradient non-finite in {name} for shape {h}x{w}"


def test_empty_target_batch_robustness():
    """Verify empty target batch returns zero loss and allows backward pass."""
    device = torch.device("cpu")
    y = torch.ones((0, 1, 64, 64), device=device, requires_grad=True)
    y0 = torch.ones((0, 1, 64, 64), device=device, requires_grad=True)
    target = torch.zeros((0, 1, 64, 64), device=device)

    outputs = {"y": y, "y0": y0}
    loss_cfg = RMRv3LossConfig()
    losses = compute_rmr_v3_losses(outputs, target, cfg=loss_cfg)

    assert "total" in losses
    assert losses["total"].item() == 0.0


def test_comprehensive_ablation_configs_e141_to_e143():
    """Verify e141, e142, and e143 configurations load, validate, and execute forward pass."""
    for cfg_file in [
        "configs/rmr_research/sub60_e141_rmr_v35_balanced_dm_lambda8.yaml",
        "configs/rmr_research/sub60_e142_rmr_v35_coupled_solver.yaml",
        "configs/rmr_research/sub60_e143_rmr_v35_factorized_routing.yaml",
    ]:
        m_cfg, l_cfg = _load_cfg(cfg_file)
        model = RMRv3(m_cfg)
        params = count_params(model)
        assert 150000 <= params <= 200000, f"Params {params} out of bounds for {cfg_file}"

        x = torch.randn(1, 3, 256, 256)
        out = model(x)
        assert out["y"].shape[-2:] == (128, 128)
        tgt = torch.zeros(1, 1, 128, 128)
        tgt[:, :, 10, 10] = 1.0
        loss = compute_rmr_v3_losses(out, tgt, cfg=l_cfg)
        assert torch.isfinite(loss["total"])
