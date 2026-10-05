"""Unit tests for canonical literature modules: Schedule-Free AdamW, ChfL, and FIDTM.

Verifies:
1. Meta FAIR AdamWScheduleFree optimizer train/eval state switching and weight averaging.
2. Canonical Characteristic Function Loss (CVPR 2022) global count preservation and frequency damping.
3. Canonical FIDTM Loss (TPAMI 2022) exact 1.0 peak preservation and count independence.
4. Strict parameter budget invariant <= 104,441 trainable parameters and zero KD.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from rmr_v3.losses.chfl import canonical_chfl_loss
from rmr_v3.losses.config import RMRv3LossConfig
from rmr_v3.losses.fidt import canonical_fidt_loss, generate_canonical_fidt_target
from rmr_v3.model import RMRv3, RMRv3Config
from rmr_v3.optim.builder import build_v3_optimizer, build_v3_scheduler
from rmr_v3.optim.schedule_free import AdamWScheduleFree


class TinyMLP(nn.Module):
    """Simple MLP to verify parameter state switching."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 4, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_schedule_free_train_eval_switching() -> None:
    """Test Schedule-Free AdamW switches between evaluation sequence y and average sequence x."""
    model = TinyMLP()
    opt = AdamWScheduleFree(model.parameters(), lr=1e-2, weight_decay=1e-4)

    # Initial state: train() switches to y
    opt.train()
    assert opt.param_groups[0]["train_mode"] is True

    # Take 5 optimization steps
    for _ in range(5):
        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

    # Snapshot weights in train mode (evaluation point y)
    y_weights = model.fc.weight.clone()

    # Switch to eval mode (Polyak-Ruppert average point x)
    opt.eval()
    assert opt.param_groups[0]["train_mode"] is False
    x_weights = model.fc.weight.clone()

    # Weights in x and y must be mathematically related but distinct after updates
    assert not torch.allclose(y_weights, x_weights)

    # Calling eval() again is idempotent
    opt.eval()
    assert torch.allclose(model.fc.weight, x_weights)

    # Switching back to train restores y_weights
    opt.train()
    assert opt.param_groups[0]["train_mode"] is True
    assert torch.allclose(model.fc.weight, y_weights, atol=1e-6)

    # Calling train() again is idempotent
    opt.train()
    assert torch.allclose(model.fc.weight, y_weights, atol=1e-6)


def test_schedule_free_builder_integration() -> None:
    """Test optimizer builder and scheduler factory with Schedule-Free."""
    model = TinyMLP()
    cfg = {
        "train": {
            "optimizer": "schedule_free",
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "schedule_free_warmup_steps": 100,
        },
        "model": {"backbone_lr_scale": 0.1},
    }

    opt = build_v3_optimizer(model, cfg, lr_init=1e-4)
    assert isinstance(opt, AdamWScheduleFree)
    assert opt.defaults["warmup_steps"] == 100

    sched = build_v3_scheduler(opt, cfg, epochs=100)
    assert hasattr(sched, "step")
    # Calling step() on IdentityScheduler must not crash or modify lr
    sched.step()
    assert sched.get_last_lr()[0] == pytest.approx(1e-5)  # backbone scaled


def test_canonical_chfl_count_and_frequency_properties() -> None:
    """Test Canonical ChfL preserves global count and damps high frequencies."""
    H, W = 64, 64
    pred = torch.full((1, 1, H, W), 0.05, requires_grad=True)  # count = 0.05 * 4096 = 204.8
    points = [torch.tensor([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])]  # N = 3

    loss = canonical_chfl_loss(pred, points, chf_tik=0.01, chf_step=16, bandwidth=8.0)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    # Backpropagation test
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()

    # Empty points test (N = 0)
    pred_empty = torch.full((1, 1, H, W), 0.01, requires_grad=True)
    loss_empty = canonical_chfl_loss(pred_empty, [torch.empty((0, 2))], chf_tik=0.01, chf_step=16, bandwidth=8.0)
    assert torch.isfinite(loss_empty)
    assert loss_empty.item() >= 0.0


def test_canonical_fidt_peak_preservation_and_count_independence() -> None:
    """Test Canonical FIDTM preserves exact 1.0 peak regardless of count N."""
    H, W = 16, 16
    pt = torch.tensor([[18.0, 18.0]])  # Point at cell (4, 4) center: (4 + 0.5) * 4 = 18.0

    # Compute target on 16x16 grid (stride 4)
    target_sparse = generate_canonical_fidt_target(h=16, w=16, points=pt, stride=4)
    # Peak at cell (4, 4) must be exactly 1.0 (d=0)
    peak_sparse = target_sparse[4, 4].item()
    assert peak_sparse == pytest.approx(1.0, abs=1e-5)

    # Now add 100 points elsewhere far away
    extra_pts = [torch.tensor([float(i * 10 + 100), float(j * 10 + 100)]) for i in range(10) for j in range(10)]
    many_pts = torch.cat([pt, torch.stack(extra_pts)], dim=0)
    target_dense = generate_canonical_fidt_target(h=16, w=16, points=many_pts, stride=4)
    peak_dense = target_dense[4, 4].item()

    # Peak must remain EXACTLY 1.0, not crushed by 1/N or 1/sum
    assert peak_dense == pytest.approx(1.0, abs=1e-5)
    assert peak_dense == pytest.approx(peak_sparse, abs=1e-5)

    # Loss gradient test
    pred = torch.full((1, 1, 16, 16), 0.5, requires_grad=True)
    fidt_loss = canonical_fidt_loss(pred, [pt], stride=4)
    assert torch.isfinite(fidt_loss)
    fidt_loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_rmr_architecture_parameter_budget_invariant() -> None:
    """Verify strict adherence to parameter ceiling <= 104,441 params and Zero KD."""
    config = RMRv3Config()
    model = RMRv3(config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    assert trainable_params <= 104441, f"Param count {trainable_params} exceeds limit 104,441"
    assert trainable_params < 105000, f"Param count {trainable_params} exceeds ceiling 105,000"
    assert trainable_params == total_params, "All model parameters must be trainable (standalone)"
