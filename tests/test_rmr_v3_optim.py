"""Unit tests for RMR-v3 optimization suite: SafeLRFinder, SafeProdigy, and WSD Scheduler."""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from rmr_v3.optim.builder import build_v3_optimizer, build_v3_scheduler, maybe_run_safe_lr_finder
from rmr_v3.optim.lr_finder import SafeLRFinder
from rmr_v3.optim.safe_prodigy import SafeProdigy
from rmr_v3.optim.wsd_scheduler import make_wsd_scheduler


class ToyModel(nn.Module):
    """Simple model with backbone-like and head-like parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encoder(x)
        out = self.head(feat)
        return {"y": out, "total": out.sum()}


def test_wsd_scheduler_phases() -> None:
    """Test Warmup-Stable-Decay schedule across all three distinct phases."""
    param = nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=1e-3)
    epochs = 100
    warmup_epochs = 10
    stable_ratio = 0.70  # Stable until 10 + 0.70 * 90 = 73
    min_lr_ratio = 0.05

    sched = make_wsd_scheduler(
        opt,
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        stable_ratio=stable_ratio,
        min_lr_ratio=min_lr_ratio,
    )

    # Epoch 0 (Warmup start: (0 + 1) / 10 = 0.1)
    lr0 = opt.param_groups[0]["lr"]
    assert lr0 == pytest.approx(1e-3 * (1.0 / warmup_epochs), rel=1e-3)

    # Epoch 10 (Warmup end / Start of flat plateau)
    for _ in range(10):
        opt.step()
        sched.step()
    lr10 = opt.param_groups[0]["lr"]
    assert lr10 == pytest.approx(1e-3, rel=1e-3)

    # Epoch 50 (In the middle of flat stable plateau)
    for _ in range(40):
        opt.step()
        sched.step()
    lr50 = opt.param_groups[0]["lr"]
    assert lr50 == pytest.approx(1e-3, rel=1e-3)

    # Epoch 99 (Decay end)
    for _ in range(49):
        opt.step()
        sched.step()
    lr_end = opt.param_groups[0]["lr"]
    assert lr_end <= 1e-3 * 0.15
    assert lr_end >= 1e-3 * min_lr_ratio * 0.95


def test_safe_prodigy_invariants() -> None:
    """Test SafeProdigy distance bounds, rate limiting, and RMS normalization."""
    model = nn.Linear(10, 2)
    opt = SafeProdigy(
        model.parameters(),
        lr=1.0,
        d0=1e-5,
        d_coef=0.5,
        growth_rate=1.02,
        d_max_cap=0.05,
        d_warmup_steps=5,
        grad_spike_thresh=3.0,
        use_rms_clipping=True,
    )

    # 1. Warmup steps: D must remain frozen at d0
    for step in range(5):
        x = torch.randn(4, 10)
        y = model(x).sum()
        y.backward()
        opt.step()
        opt.zero_grad()
        assert opt.param_groups[0]["d"] == pytest.approx(1e-5, rel=1e-4)

    # 2. Steps after warmup: D can grow but strictly <= d * growth_rate and <= d_max_cap
    prev_d = opt.param_groups[0]["d"]
    for step in range(30):
        x = torch.randn(4, 10) * 10.0
        y = model(x).sum()
        y.backward()
        opt.step()
        opt.zero_grad()
        curr_d = opt.param_groups[0]["d"]
        assert curr_d <= prev_d * 1.0201 + 1e-9  # Rate-limit verified
        assert curr_d <= 0.05 + 1e-9  # Ceiling cap verified
        prev_d = curr_d

    # 3. State dict serialization and deserialization
    state = opt.state_dict()
    new_opt = SafeProdigy(model.parameters(), lr=1.0)
    new_opt.load_state_dict(state)
    assert new_opt.param_groups[0]["d"] == pytest.approx(curr_d, rel=1e-5)


def test_safe_lr_finder_execution() -> None:
    """Test SafeLRFinder range test execution and weight restoration."""
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Record initial weights
    w_initial = model[0].weight.clone()

    x = torch.randn(32, 8)
    y = torch.randn(32, 1)
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=8, shuffle=True)

    def toy_loss(out: torch.Tensor, tgt: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"total": nn.functional.mse_loss(out, tgt)}

    finder = SafeLRFinder(
        model=model,
        optimizer=opt,
        loss_fn=lambda out, tgt: toy_loss(out, tgt),
        device=torch.device("cpu"),
    )

    suggested_lr = finder.range_test(
        loader,
        min_lr=1e-6,
        max_lr=1e-1,
        num_iter=20,
        max_safe_lr=5e-3,
    )

    # Verify suggested LR is strictly positive and within safe ceiling
    assert 1e-6 <= suggested_lr <= 5e-3

    # Verify model weights are 100% restored
    w_restored = model[0].weight
    assert torch.allclose(w_initial, w_restored, atol=1e-7)

    # Verify optimizer group lr is restored
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-4, rel=1e-5)


def test_builder_factories() -> None:
    """Test build_v3_optimizer and build_v3_scheduler with various configurations."""
    model = ToyModel()

    # 1. Standard AdamW + Cosine
    cfg_adamw = {
        "train": {
            "optimizer": "adamw",
            "scheduler_type": "cosine",
            "lr": 2e-4,
            "backbone_lr_scale": 0.2,
            "weight_decay": 1e-4,
        }
    }
    opt1 = build_v3_optimizer(model, cfg_adamw, lr_init=2e-4)
    assert isinstance(opt1, torch.optim.AdamW)
    assert len(opt1.param_groups) == 4
    assert opt1.param_groups[0]["lr"] == pytest.approx(4e-5, rel=1e-5)  # 2e-4 * 0.2
    assert opt1.param_groups[2]["lr"] == pytest.approx(2e-4, rel=1e-5)

    sched1 = build_v3_scheduler(opt1, cfg_adamw, epochs=100)
    assert sched1 is not None

    # 2. SafeProdigy + WSD
    cfg_prodigy = {
        "train": {
            "optimizer": "safeprodigy",
            "scheduler_type": "wsd",
            "wsd_stable_ratio": 0.75,
            "prodigy_d_coef": 0.5,
            "prodigy_growth_rate": 1.015,
            "prodigy_d_max_cap": 0.08,
            "prodigy_d_warmup_steps": 200,
        }
    }
    opt2 = build_v3_optimizer(model, cfg_prodigy, lr_init=1.0)
    assert isinstance(opt2, SafeProdigy)
    assert opt2.param_groups[0]["d"] == pytest.approx(1e-5, rel=1e-5)

    sched2 = build_v3_scheduler(opt2, cfg_prodigy, epochs=100)
    assert sched2 is not None

    # 3. maybe_run_safe_lr_finder with auto_lr_finder: false -> returns None
    res = maybe_run_safe_lr_finder(
        model, opt1, None, None, {"train": {"auto_lr_finder": False}}, torch.device("cpu")
    )
    assert res is None
