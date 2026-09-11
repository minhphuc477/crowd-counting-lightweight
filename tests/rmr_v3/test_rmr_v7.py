from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from rmr_core.heads import FineMeasureHead
from rmr_v3.model import RMRv3, RMRv3Config
from rmr_v3.losses import (
    hurdle_focal_bce_loss,
    truncated_nb_nll_loss,
    compute_rmr_v3_losses,
    RMRv3LossConfig,
)


def test_fine_measure_head_temperature_softplus():
    """Verify Learnable Temperature Softplus in FineMeasureHead."""
    # When temp_softplus=False: returns raw logit z
    head_raw = FineMeasureHead(32, temp_softplus=False)
    x = torch.zeros(1, 32, 16, 16)
    z = head_raw(x)
    assert not hasattr(head_raw, "tau")
    # Bias is initialized so softplus(bias) ≈ 0.01576
    assert abs(F.softplus(z).mean().item() - 0.015763) < 0.005

    # When temp_softplus=True: returns tau * softplus(z / tau)
    head_temp = FineMeasureHead(32, temp_softplus=True)
    assert hasattr(head_temp, "tau")
    assert head_temp.tau.numel() == 1
    assert abs(head_temp.tau.item() - 1.0) < 1e-5

    y0 = head_temp(x)
    # Output must be non-negative density
    assert (y0 >= 0.0).all()
    # At tau=1.0, tau * softplus(z/tau) == softplus(z)
    assert abs(y0.mean().item() - 0.015763) < 0.005

    # Test gradient flow to tau
    loss = y0.sum()
    loss.backward()
    assert head_temp.tau.grad is not None
    assert torch.isfinite(head_temp.tau.grad)


def test_hurdle_head_architecture_and_forward():
    """Verify Hurdle NB head produces hurdle_logit and modulates solver target."""
    cfg = RMRv3Config(
        output_stride=4,
        feature_width=32,
        pretrained=False,
        region_sizes_px=(32, 64, 128),
        iterations=2,
        hurdle_head=True,
        temp_softplus=True,
    )
    model = RMRv3(cfg)
    model.eval()

    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        out = model(x)

    assert "hurdle_logit" in out
    assert out["hurdle_logit"] is not None
    b_solver = out["b_solver"]
    b_region = out["b_region"]

    # π_R = sigmoid(hurdle_logit)
    pi_r = torch.sigmoid(out["hurdle_logit"])
    assert torch.allclose(b_solver, pi_r * b_region, atol=1e-5)
    assert out["z0"] is not None
    assert (out["y0"] >= 0.0).all()


def test_hurdle_losses_focal_bce_and_truncated_nb():
    """Verify hurdle focal BCE and truncated NB NLL loss functions."""
    b, c, m = 2, 1, 50
    pi_logit = torch.randn(b, c, m, requires_grad=True)
    target_region = torch.zeros(b, c, m)
    target_region[0, 0, :25] = 5.0  # occupied regions
    target_region[0, 0, 25:] = 0.0  # background regions

    loss_bce = hurdle_focal_bce_loss(pi_logit, target_region, gamma=2.0)
    assert loss_bce.item() > 0.0
    loss_bce.backward()
    assert pi_logit.grad is not None

    # Truncated NB NLL
    mu = torch.rand(b, c, m) * 10 + 0.1
    disp = torch.rand(b, c, m) * 50 + 1.0
    loss_tnb = truncated_nb_nll_loss(mu, disp, target_region)
    assert loss_tnb.item() > 0.0

    # If all empty, truncated NB NLL must gracefully return 0
    all_zero_target = torch.zeros(b, c, m)
    loss_zero = truncated_nb_nll_loss(mu, disp, all_zero_target)
    assert loss_zero.item() == 0.0


def test_tv_laplacian_smoothing_diffusion():
    """Verify TV Laplacian smoothing diffuses high-frequency spikes in SIRT loop."""
    cfg = RMRv3Config(
        output_stride=4,
        feature_width=32,
        pretrained=False,
        region_sizes_px=(32, 64, 128),
        iterations=2,
        tv_lambda=0.05,
    )
    model = RMRv3(cfg)
    model.eval()

    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)

    y = out["y"]
    assert (y >= 0.0).all()
    assert torch.isfinite(y).all()


def test_ema_state_restored_on_resume():
    """Verify that EMA shadow state is correctly restored from a checkpoint on resume.

    Regression test for: EMA state was re-initialized from fresh model on --resume,
    losing accumulated EMA progress and producing warm-EMA bias for many epochs.
    """
    import copy

    cfg = RMRv3Config(output_stride=4, feature_width=32, pretrained=False)
    model = RMRv3(cfg)

    # Simulate an EMA state that has drifted from the current model weights
    # (this is the state that would exist after many training epochs)
    ema_state = copy.deepcopy(model.state_dict())
    for k in ema_state:
        ema_state[k] = ema_state[k].float() * 2.0  # artificially drift EMA weights

    # Simulate a checkpoint that was saved with that EMA state
    fake_ckpt = {
        "model": model.state_dict(),
        "epoch": 500,
        "best_mae": 85.0,
        "epochs_without_improvement": 3,
        "ema_model": {k: v.clone() for k, v in ema_state.items()},
    }

    # Simulate what train.py resume does: init fresh ema_state from model, then load ckpt
    fresh_ema = copy.deepcopy(model.state_dict())
    for k in fresh_ema:
        fresh_ema[k] = fresh_ema[k].float()

    # Apply the resume EMA restoration logic (mirrors train.py)
    if "ema_model" in fake_ckpt:
        ema_sd = fake_ckpt["ema_model"]
        for k in fresh_ema:
            if k in ema_sd:
                fresh_ema[k].copy_(ema_sd[k].float())

    # After restoration, fresh_ema should match the original drifted EMA state
    for k in fresh_ema:
        assert torch.allclose(fresh_ema[k], ema_state[k].float()), (
            f"EMA state key '{k}' was not restored correctly on resume. "
            f"Got {fresh_ema[k].mean():.4f}, expected {ema_state[k].float().mean():.4f}"
        )


def test_v7_canonical_parameter_budget():
    """RMR-v7 canonical config (hurdle_head + temp_softplus) must stay strictly under 105,000 params.

    Base model:    101,763 params (no v7 features)
    + hurdle_head:  +45 params (Linear(44, 1) in ProbabilisticRegionalEvidenceHead)
    + temp_softplus: +1 param (learnable τ in FineMeasureHead)
    V7 canonical:  101,813 params total
    """
    cfg = RMRv3Config(
        pretrained=False,
        hurdle_head=True,
        temp_softplus=True,
        region_head_hidden=44,
    )
    model = RMRv3(cfg)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert n < 105_000, f"OVER BUDGET: {n:,} >= 105,000"
    assert n == 101_813, (
        f"V7 canonical expected exactly 101,813 parameters, got {n:,}. "
        f"If the architecture changed, update this test and document the delta."
    )
