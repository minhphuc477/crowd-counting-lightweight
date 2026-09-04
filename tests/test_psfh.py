"""Unit tests for PS-FH-CMICF components:
- FractionalPrefixPreconditioner
- balanced_sobolev_smooth_l1
- partition_grid_into_blocks
- PSFHCMICFLoss (signed projected inequality AL)
- MICFLite strict-local FH cumulative head (single-block and multi-block composition)
"""

import math
import pytest
import torch
import torch.nn as nn

from hpc.losses.ps_fh_cmicf import (
    FractionalPrefixPreconditioner,
    balanced_sobolev_smooth_l1,
    partition_grid_into_blocks,
    PSFHCMICFLoss,
)
from hpc.losses.micf import (
    cell_counts_to_cumulative_field,
    discrete_mixed_difference,
)
from hpc.models.micf_lite import MICFLite


def test_fractional_prefix_preconditioner_svd():
    p = FractionalPrefixPreconditioner(k=4, alpha=0.5)
    assert p.k == 4
    assert math.isclose(p.alpha, 0.5)
    assert 29.0 < p.prefix_condition_number < 30.0
    assert math.isclose(p.quadratic_condition_number, p.prefix_condition_number, rel_tol=1e-3)
    assert p.P_alpha.shape == (16, 16)


def test_fractional_prefix_preconditioner_forward():
    p = FractionalPrefixPreconditioner(k=4, alpha=0.5)
    x = torch.randn(8, 1, 4, 4, requires_grad=True)
    out = p(x)
    assert out.shape == (8, 1, 4, 4)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))


def test_balanced_sobolev_smooth_l1():
    pred = torch.randn(4, 1, 4, 4)
    target = torch.zeros(4, 1, 4, 4)
    target[0, 0, 1, 1] = 2.0
    target[2, 0, 3, 2] = 1.0

    loss, stats = balanced_sobolev_smooth_l1(pred, target, beta=1.0)
    assert loss.ndim == 0
    assert "sobolev_pos_loss" in stats
    assert "sobolev_zero_loss" in stats
    assert stats["positive_cell_fraction"] == pytest.approx(2.0 / 64.0)
    assert stats["zero_cell_fraction"] == pytest.approx(62.0 / 64.0)


def test_partition_grid_into_blocks():
    x = torch.arange(1, 17, dtype=torch.float32).view(1, 1, 4, 4)
    blocks, nh, nw = partition_grid_into_blocks(x, k=2)
    assert nh == 2
    assert nw == 2
    assert blocks.shape == (4, 1, 2, 2)


def test_ps_fh_cmicf_signed_al_forward_and_dual():
    loss_fn = PSFHCMICFLoss(
        k=4,
        precondition_alpha=0.5,
        lambda_sobolev=1.0,
        lambda_count=1.0,
        al_rho=1.0,
        al_dual_init=0.0,
        al_dual_max=100.0,
        al_update_mode="dual_ascent",
    )

    B, H, W = 2, 16, 16
    pred_c = torch.rand(B, 1, H, W, requires_grad=True)
    target_y = torch.zeros(B, 1, H, W)
    target_y[:, :, 2, 2] = 1.0
    target_y[:, :, 8, 8] = 3.0
    target_c = cell_counts_to_cumulative_field(target_y, orientation="TL")

    pred_c_blocks, _, _ = partition_grid_into_blocks(pred_c, k=4)
    total_loss, comp = loss_fn(
        pred_c=pred_c,
        target_c=target_c,
        target_y=target_y,
        pred_c_blocks=pred_c_blocks,
        return_components=True,
    )

    assert total_loss.ndim == 0
    assert torch.isfinite(total_loss)
    assert "ps_pc_loss" in comp
    assert "ps_sobolev_loss" in comp
    assert "ps_count_loss" in comp
    assert "ps_constraint" in comp
    assert "ps_dual_lambda" in comp
    assert "ps_dual_lambda_max" in comp
    assert "ps_dual_lambda_terminal" in comp

    # Test 1: When violation occurs (c = -y > 0, i.e. y < 0), lambda INCREASES
    viol_c = torch.zeros(10, 1, 4, 4)
    viol_c[:, :, 3, 3] = 0.5  # violation of 0.5 at terminal phase
    loss_fn.update_dual(viol_c)
    assert loss_fn.al_lambda[0, 0, 3, 3].item() == pytest.approx(0.5)
    assert loss_fn.al_lambda[0, 0, 0, 0].item() == pytest.approx(0.0)

    # Test 2: When constraint is strictly satisfied (c = -y < 0, i.e. y > 0), lambda DECREASES!
    sat_c = torch.zeros(10, 1, 4, 4)
    sat_c[:, :, 3, 3] = -0.3  # satisfied with margin 0.3
    loss_fn.update_dual(sat_c)
    assert loss_fn.al_lambda[0, 0, 3, 3].item() == pytest.approx(0.2)  # 0.5 - 0.3 = 0.2!

    # Test 3: Clamping at 0 when margin exceeds lambda
    sat_large_c = torch.zeros(10, 1, 4, 4)
    sat_large_c[:, :, 3, 3] = -0.5
    loss_fn.update_dual(sat_large_c)
    assert loss_fn.al_lambda[0, 0, 3, 3].item() == pytest.approx(0.0)


def test_ps_fh_model_strict_local_single_block():
    model = MICFLite(
        backbone_name="mobilenetv4_conv_small_050.e3000_r224_in1k",
        pretrained=False,
        output_stride=16,
        finite_horizon=4,
        extent_aware=True,
        fh_strict_local=True,
        fh_local_norm="group",
    )
    x = torch.randn(2, 3, 64, 64)
    c_global, aux = model.forward_field_with_aux(x)
    assert c_global.shape == (2, 1, 4, 4)
    assert "c_blocks" in aux
    assert "y_blocks" in aux
    assert aux["c_blocks"].shape == (2, 1, 4, 4)


def test_ps_fh_model_strict_local_multi_block_256():
    """Multi-block test on 256x256 image with stride 16 and K=4.
    
    16x16 grid contains 4x4 = 16 blocks. For batch 2, total blocks = 32.
    Verifies that block-partitioned features are exactly composed into the global field.
    """
    model = MICFLite(
        backbone_name="mobilenetv4_conv_small_050.e3000_r224_in1k",
        pretrained=False,
        output_stride=16,
        finite_horizon=4,
        extent_aware=True,
        fh_strict_local=True,
        fh_local_norm="group",
    )
    B = 2
    x = torch.randn(B, 3, 256, 256)
    c_global, aux = model.forward_field_with_aux(x)

    assert c_global.shape == (B, 1, 16, 16)
    assert aux["c_blocks"].shape == (B * 16, 1, 4, 4)
    assert aux["y_blocks"].shape == (B * 16, 1, 4, 4)
    assert aux["n_h"] == 4
    assert aux["n_w"] == 4

    # Check that discrete mixed difference of c_global matches reassembled y_blocks
    y_global = discrete_mixed_difference(c_global)
    y_from_blocks = (
        aux["y_blocks"]
        .view(B, 4, 4, 1, 4, 4)
        .permute(0, 3, 1, 4, 2, 5)
        .contiguous()
        .view(B, 1, 16, 16)
    )
    diff = (y_global - y_from_blocks).abs().max().item()
    assert diff < 2e-5, f"Composition mismatch: max diff {diff}"
