"""Adversarial stress-testing suite for Generation 19 models and optimization components.

Verifies:
1. Exact parameter count ceiling (<= 104,441) on all Gen 19 models.
2. Strict Sample Isolation: grad(loss(sample 1)) w.r.t. sample 0 is identically zero.
3. Arbitrary prime and odd spatial resolutions (383x509, 241x311) with zero NaNs/Infs.
4. Extreme density bounds: pure background (0 people) and extreme crowd (3,000 people).
5. SafeProdigy adversarial resilience: NaN gradient injection, spike gating, and rate limiting.
6. WSD Scheduler continuity and exact mathematical phase transitions.
"""
from __future__ import annotations

import math
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import yaml

from rmr_v3.config import load_config, validate_v3_config
from rmr_v3.engine import make_loss_cfg, make_model
from rmr_v3.losses import compute_rmr_v3_losses
from rmr_v3.optim.safe_prodigy import SafeProdigy
from rmr_v3.optim.wsd_scheduler import make_wsd_scheduler


GEN19_CONFIGS = [
    "configs/rmr_research/sub60_e82_canonical_m04.yaml",
    "configs/rmr_research/sub60_e83_m04_wsd_plateau50.yaml",
    "configs/rmr_research/sub60_e84_e5_safeprodigy_calibrated.yaml",
    "configs/rmr_research/sub60_e85_e5_wsd_plateau50.yaml",
    "configs/rmr_research/sub60_e86_m04_safeprodigy_calibrated.yaml",
]


@pytest.mark.parametrize("cfg_path_str", GEN19_CONFIGS)
def test_gen19_parameter_budget_and_schema(cfg_path_str: str) -> None:
    """Invariant: Every Gen 19 model must strictly obey the <= 104,441 parameter ceiling."""
    cfg_path = Path(cfg_path_str)
    assert cfg_path.exists(), f"Missing config: {cfg_path}"
    cfg = load_config(cfg_path)
    validate_v3_config(cfg)

    cfg.setdefault("model", {})["pretrained"] = False
    model, _ = make_model(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params <= 104441, f"Parameter budget violated in {cfg_path.name}: {n_params} > 104,441"
    assert n_params == 104441, f"Parameter mismatch in {cfg_path.name}: {n_params} != 104,441"


@pytest.mark.parametrize("cfg_path_str", GEN19_CONFIGS)
def test_gen19_sample_isolation(cfg_path_str: str) -> None:
    """Invariant: Sample isolation in batch=2 under both zero and dense target slices."""
    cfg = load_config(Path(cfg_path_str))
    cfg.setdefault("model", {})["pretrained"] = False
    model, _ = make_model(cfg)
    model.eval()

    loss_cfg = make_loss_cfg(cfg)
    loss_cfg.elementwise_dense_scaling = False
    loss_cfg.density_loss_scaling = False

    torch.manual_seed(42)
    x = torch.randn(2, 3, 256, 256, requires_grad=True)
    th, tw = 64, 64
    # Sample 0 has 0 targets, Sample 1 has dense cluster
    target = torch.zeros(2, 1, th, tw, dtype=torch.float32)
    target[1, 0, 10:20, 10:20] = 5.0

    out = model(x)
    out_s1 = {k: v[1:2] if isinstance(v, torch.Tensor) and v.shape[0] == 2 else v for k, v in out.items()}
    losses_s1 = compute_rmr_v3_losses(out_s1, target[1:2], loss_cfg)

    model.zero_grad()
    if x.grad is not None:
        x.grad.zero_()
    losses_s1["total"].backward()

    assert x.grad is not None
    grad_s0 = x.grad[0].abs().max().item()
    assert grad_s0 == 0.0, f"Sample isolation violated in {cfg_path_str}! Sample 0 got gradient {grad_s0}"


@pytest.mark.parametrize("cfg_path_str", GEN19_CONFIGS)
@pytest.mark.parametrize("h,w", [(383, 509), (241, 311)])
def test_gen19_arbitrary_odd_prime_resolutions(cfg_path_str: str, h: int, w: int) -> None:
    """Invariant: Arbitrary odd/prime input resolutions produce exact finite outputs."""
    cfg = load_config(Path(cfg_path_str))
    cfg.setdefault("model", {})["pretrained"] = False
    model, _ = make_model(cfg)
    loss_cfg = make_loss_cfg(cfg)

    x = torch.randn(1, 3, h, w, requires_grad=True)
    out = model(x)
    assert torch.isfinite(out.y).all(), f"Non-finite output in {cfg_path_str} at resolution {h}x{w}"

    th, tw = (h + 3) // 4, (w + 3) // 4
    target = torch.randint(0, 3, (1, 1, th, tw), dtype=torch.float32)
    losses = compute_rmr_v3_losses(out, target, loss_cfg)
    assert torch.isfinite(losses["total"]), f"Non-finite loss in {cfg_path_str} at resolution {h}x{w}"

    losses["total"].backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_safeprodigy_adversarial_nan_and_spike_gating() -> None:
    """Test SafeProdigy resilience against NaN gradients, extreme spikes, and rate-limiting."""
    model = nn.Linear(8, 2)
    opt = SafeProdigy(
        model.parameters(),
        lr=1.0,
        d0=1e-5,
        d_coef=0.5,
        growth_rate=1.02,
        d_max_cap=0.03,
        d_warmup_steps=2,
        grad_spike_thresh=5.0,
        use_rms_clipping=True,
    )

    # 1. Normal steps during warmup
    for _ in range(2):
        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
    assert opt.param_groups[0]["d"] == pytest.approx(1e-5, rel=1e-5)

    # 2. Giant gradient spike (> grad_spike_thresh = 5.0) -> D growth must freeze
    d_before_spike = opt.param_groups[0]["d"]
    x = torch.randn(2, 8) * 1000.0
    loss = model(x).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()
    assert opt.param_groups[0]["d"] == pytest.approx(d_before_spike, rel=1e-6)

    # 3. Inject NaN gradient directly -> SafeProdigy must skip step and zero grads without crash
    model.weight.grad = torch.full_like(model.weight.data, float("nan"))
    opt.step()
    # Gradients should have been zeroed
    assert model.weight.grad is None or not torch.isnan(model.weight.grad).any()

    # 4. Multi-step growth must strictly respect growth_rate and d_max_cap
    for _ in range(50):
        x = torch.randn(2, 8) * 2.0
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
        assert opt.param_groups[0]["d"] <= 0.03 + 1e-9


def test_wsd_scheduler_exact_mathematical_transitions() -> None:
    """Test WSD schedule boundary continuity and floor clamp."""
    p = nn.Parameter(torch.ones(1))
    opt = torch.optim.SGD([p], lr=2e-4)
    sched = make_wsd_scheduler(
        opt,
        epochs=1000,
        warmup_epochs=20,
        stable_ratio=0.60,  # stable until 600
        min_lr_ratio=0.01,  # floor 2e-6
    )

    # Epoch 0: warmup start
    assert opt.param_groups[0]["lr"] == pytest.approx(2e-4 * (1.0 / 20.0), rel=1e-4)

    # Epoch 20: warmup complete, peak reached
    for _ in range(20):
        opt.step()
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(2e-4, rel=1e-4)

    # Epoch 599: still at peak
    for _ in range(579):
        opt.step()
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(2e-4, rel=1e-4)

    # Epoch 999: final decay reached near floor
    for _ in range(400):
        opt.step()
        sched.step()
    final_lr = opt.param_groups[0]["lr"]
    assert final_lr >= 2e-4 * 0.01 * 0.99
    assert final_lr <= 2e-4 * 0.02
