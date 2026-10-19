from __future__ import annotations

import torch
import pytest

from rmr_core.operators import (
    build_multiscale_regions,
    closed_form_radon_nikodym_transport,
    regional_sum,
)
from rmr_v3.losses.spatial_priors import radon_nikodym_transport_energy_loss
from rmr_v3.losses.config import RMRv3LossConfig
from rmr_v3.losses.orchestration import compute_rmr_v3_losses
from rmr_v3.model.architecture import RMRv3
from rmr_v3.model.config import RMRv3Config


def test_rn_transport_loss_zero_when_consistent():
    """Verify loss is exactly zero when multiplier is 1.0 everywhere."""
    y0 = torch.rand(2, 1, 32, 32)
    s_field = torch.ones(2, 1, 32, 32)
    loss = radon_nikodym_transport_energy_loss(y0, s_field)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_rn_transport_loss_positive_when_inconsistent():
    """Verify loss is strictly positive when carrier deviates from regional evidence."""
    y0 = torch.rand(2, 1, 32, 32) + 0.1
    s_field = torch.full((2, 1, 32, 32), 1.25)  # 25% under-count
    loss = radon_nikodym_transport_energy_loss(y0, s_field)
    assert loss.item() > 0.0
    expected = (1.25 - 1.0) ** 2  # 0.0625
    assert loss.item() == pytest.approx(expected, rel=1e-3)


def test_rn_transport_loss_background_immunity():
    """Verify background cells (y0 == 0) contribute 0 to the loss and gradient."""
    y0 = torch.zeros(2, 1, 32, 32, requires_grad=True)
    s_field = torch.full((2, 1, 32, 32), 5.0)  # Huge discrepancy on empty background
    loss = radon_nikodym_transport_energy_loss(y0, s_field)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)

    loss.backward()
    assert torch.all(y0.grad == 0.0)


def test_rn_transport_loss_end_to_end_closed_form_rn():
    """Verify end-to-end forward, loss computation, and backward with closed_form_rn."""
    cfg = RMRv3Config(
        output_stride=4,
        feature_width=32,
        backbone_name="mobilenetv4_conv_small_050.e3000_r224_in1k",
        pretrained=False,
        solver_mode="closed_form_rn",
        enable_solver=True,
        head_blocks=2,
    )
    model = RMRv3(cfg)
    model.train()

    x = torch.randn(2, 3, 128, 128)
    out = model(x)

    assert "rn_multiplier" in out
    assert out.rn_multiplier is not None
    assert out.rn_multiplier.shape[-2:] == out.y0.shape[-2:]

    loss_cfg = RMRv3LossConfig(
        lambda_count=1.0,
        lambda_flat_dm16=1.0,
        lambda_region_nb=0.20,
        lambda_rn_transport=0.10,
        lambda_cell=0.0,
        lambda_hurdle=0.0,
        lambda_trunc_nb=0.0,
        lambda_scale_align=0.0,
        lambda_curvature=0.0,
        lambda_hard_bg=0.0,
    )

    target_y = torch.zeros(2, 1, 32, 32, dtype=torch.float32)
    target_y[0, 0, 5, 5] = 1.0
    target_y[1, 0, 10, 12] = 2.0
    pts = [torch.tensor([[20.0, 20.0]]), torch.tensor([[40.0, 48.0], [42.0, 50.0]])]

    losses = compute_rmr_v3_losses(out, target_y, loss_cfg, points=pts)

    assert "rn_transport" in losses
    assert losses["rn_transport"].item() >= 0.0
    assert torch.isfinite(losses["total"])

    losses["total"].backward()

    # Verify gradients reach both head and backbone
    has_grad = False
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.all(torch.isfinite(p.grad)), "Non-finite gradient detected!"
            has_grad = True
    assert has_grad, "Zero gradients backpropagated!"
