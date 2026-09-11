from __future__ import annotations

import copy
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from rmr_core.data import rasterize_points
from rmr_core.evaluation import evaluate_dataset, predict_tiled
from rmr_core.losses import (
    balanced_smooth_l1,
    count_magnitude_loss,
    flat_dm16_loss,
    multiscale_dm_loss,
    negative_binomial_nll_mean_dispersion,
)
from rmr_core.operators import (
    build_multiscale_regions,
    charbonnier_tv_step,
    multiplicative_gated_adjoint,
    regional_adjoint,
    regional_sum,
)
from rmr_core.training import build_checkpoint, compute_file_sha256, get_git_info
from rmr_v3.config import RMRv3Config
from rmr_v3.eval import load_model_from_ckpt
from rmr_v3.losses import (
    RMRv3LossConfig,
    compute_rmr_v3_losses,
    hurdle_focal_bce_loss,
    mass_weighted_cell_loss,
    truncated_nb_nll_loss,
)
from rmr_v3.model import (
    RMRv3,
    reliability_from_nb,
    weighted_coverage,
    weighted_normalized_adjoint_field,
    weighted_regional_energy,
)


def test_autograd_graph_completeness_on_empty_background_batch():
    cfg = RMRv3Config(
        pretrained=False,
        neck_type="aspp_lite",
        hurdle_head=True,
        temp_softplus=True,
        iterations=1,
    )
    model = RMRv3(cfg)
    model.train()

    x = torch.randn(2, 3, 64, 64)
    target_y = torch.zeros(2, 1, 16, 16)

    out = model(x)
    loss_cfg = RMRv3LossConfig(
        lambda_hurdle=1.0,
        lambda_trunc_nb=1.0,
        lambda_region_nb=0.0,
    )
    losses = compute_rmr_v3_losses(out, target_y, loss_cfg)

    assert abs(losses["trunc_nb"].item()) < 1e-7
    losses["total"].backward()

    assert model.region_head.hurdle_head_layer.weight.grad is not None
    assert torch.isfinite(model.region_head.hurdle_head_layer.weight.grad).all()
    assert model.region_head.hurdle_head_layer.bias.grad.item() > 0.0


def test_aspp_gap_toggle_exact_parameter_delta():
    cfg_with_gap = RMRv3Config(
        pretrained=False,
        neck_type="aspp_lite",
        use_aspp_gap=True,
        region_head_hidden=44,
        hurdle_head=True,
        regional_feature_stats="mean_std",
    )
    model_with = RMRv3(cfg_with_gap)
    params_with = sum(p.numel() for p in model_with.parameters() if p.requires_grad)

    cfg_no_gap = RMRv3Config(
        pretrained=False,
        neck_type="aspp_lite",
        use_aspp_gap=False,
        region_head_hidden=44,
        hurdle_head=True,
        regional_feature_stats="mean_std",
    )
    model_no = RMRv3(cfg_no_gap)
    params_no = sum(p.numel() for p in model_no.parameters() if p.requires_grad)

    expected_delta = 32 * 32 + 32
    assert params_with - params_no == expected_delta, f"Expected delta {expected_delta}, got {params_with - params_no}"

    x = torch.randn(1, 3, 64, 64)
    out_with = model_with(x)
    out_no = model_no(x)
    assert out_with["y"].shape == (1, 1, 16, 16)
    assert out_no["y"].shape == (1, 1, 16, 16)


def test_extreme_crowd_density_numerical_stability():
    cfg = RMRv3Config(
        pretrained=False,
        neck_type="aspp_lite",
        regional_feature_stats="mean_std",
        region_head_hidden=44,
        hurdle_head=True,
        temp_softplus=True,
        solver_mode="multiplicative",
        tv_type="charbonnier",
        iterations=2,
    )
    model = RMRv3(cfg)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    torch.manual_seed(42)
    pts = torch.rand(5000, 2) * 255.0
    target_y = rasterize_points(pts, 256, 256, stride=4).unsqueeze(0)
    assert target_y.sum().item() == 5000.0

    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    assert torch.isfinite(out["y"]).all()

    loss_cfg = RMRv3LossConfig(
        cell_loss_mode="mass_weighted",
        lambda_hurdle=0.5,
        lambda_trunc_nb=0.5,
    )
    losses = compute_rmr_v3_losses(out, target_y, loss_cfg)

    for k, v in losses.items():
        assert torch.isfinite(v).all(), f"Loss component {k} is not finite: {v.item()}"

    losses["total"].backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()

    optimizer.step()
    for name, p in model.named_parameters():
        assert torch.isfinite(p).all()


def test_checkpoint_exact_round_trip_determinism():
    cfg_dict = {
        "model": {
            "neck_type": "aspp_lite",
            "use_aspp_gap": True,
            "regional_feature_stats": "mean_std",
            "region_head_hidden": 44,
            "hurdle_head": True,
            "temp_softplus": True,
            "solver_mode": "multiplicative",
            "tv_type": "charbonnier",
            "iterations": 2,
        }
    }
    model_orig = RMRv3(RMRv3Config.from_dict(cfg_dict["model"], pretrained=False))
    model_orig.eval()

    opt = torch.optim.AdamW(model_orig.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    ema_state = {k: v.clone().float() * 0.99 for k, v in model_orig.state_dict().items()}

    ckpt = build_checkpoint(
        epoch=42,
        model=model_orig,
        optimizer=opt,
        scheduler=None,
        scaler=scaler,
        config=cfg_dict,
        config_hash="dummy_hash_1234",
        best_mae=75.42,
        epochs_without_improvement=3,
        solver_strength=1.0,
        ema_state=ema_state,
    )

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = Path(f.name)

    try:
        torch.save(ckpt, tmp_path)

        model_ema, _, _, loaded_ckpt = load_model_from_ckpt(tmp_path, device=torch.device("cpu"), use_ema=True)
        assert loaded_ckpt["epoch"] == 42
        assert loaded_ckpt["best_mae"] == 75.42
        assert loaded_ckpt["config_hash"] == "dummy_hash_1234"

        for k, v in model_ema.state_dict().items():
            if k in ema_state and v.is_floating_point():
                torch.testing.assert_close(v, ema_state[k].to(dtype=v.dtype), atol=1e-5, rtol=1e-5)

        model_live, _, _, _ = load_model_from_ckpt(tmp_path, device=torch.device("cpu"), use_ema=False)
        for k, v in model_live.state_dict().items():
            torch.testing.assert_close(v, model_orig.state_dict()[k], atol=1e-5, rtol=1e-5)

        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out_orig = model_orig(x)
            out_live = model_live(x)
        torch.testing.assert_close(out_orig["y"], out_live["y"], atol=1e-6, rtol=1e-6)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_multiscale_dm_allocation_scale_invariance():
    torch.manual_seed(42)
    pred_map = torch.rand(2, 1, 32, 32) * 2.0 + 0.1
    target_map = torch.randint(0, 5, (2, 1, 32, 32)).float()

    loss_1x = multiscale_dm_loss(pred_map, target_map, block_sizes_px=(16, 32), stride=4)
    loss_10x = multiscale_dm_loss(pred_map * 10.0, target_map, block_sizes_px=(16, 32), stride=4)
    loss_01x = multiscale_dm_loss(pred_map * 0.1, target_map, block_sizes_px=(16, 32), stride=4)

    assert abs(loss_1x.item() - loss_10x.item()) < 1e-5
    assert abs(loss_1x.item() - loss_01x.item()) < 1e-5


def test_multiscale_dm_zero_ground_truth_returns_zero():
    pred_map = torch.rand(2, 1, 32, 32) * 1.5
    target_map = torch.zeros(2, 1, 32, 32)

    loss_zero = multiscale_dm_loss(pred_map, target_map, block_sizes_px=(16, 32), stride=4)
    assert abs(loss_zero.item()) < 1e-7


def test_multiplicative_sirt_monotonic_energy_decay():
    h, w = 32, 32
    regions = build_multiscale_regions(h, w, output_stride=4, region_sizes_px=(16, 32), device=torch.device("cpu"))

    torch.manual_seed(42)
    y = torch.rand(1, 1, h, w) * 0.5 + 0.05
    weight = torch.ones((1, 1, regions.boxes.shape[0]))
    b_solver = regional_sum(y, regions.boxes) * 0.7

    cov_w = weighted_coverage(weight, regions, h, w)

    energies = []
    for step in range(4):
        e = weighted_regional_energy(y, b_solver, weight, regions).item()
        energies.append(e)

        field = weighted_normalized_adjoint_field(
            y, b_solver, weight, regions,
            weighted_cov=cov_w,
            solver_mode="multiplicative",
            density_gate_rho=0.02,
            density_gate_floor=0.02,
        )
        y = torch.clamp_min(y - 0.8 * field, 0.0)

    for i in range(len(energies) - 1):
        assert energies[i + 1] <= energies[i] + 1e-5, f"Energy increased at step {i}: {energies[i]:.6f} -> {energies[i+1]:.6f}"


def test_coordinate_attention_axis_specific_modulation():
    from rmr_core.necks import CoordinateAttention

    ca = CoordinateAttention(channels=32, reduction=4)
    ca.eval()

    torch.manual_seed(42)
    x = torch.zeros(2, 32, 24, 24)
    x[0, :, 10:14, :] = 5.0
    x[1, :, :, 10:14] = 5.0

    with torch.no_grad():
        out = ca(x)

    diff = (out[0] - out[1]).abs().max().item()
    assert diff > 0.1
    assert out.shape == (2, 32, 24, 24)


def test_predict_tiled_arbitrary_non_square_coverage():
    h, w, stride = 131, 219, 4

    class IdentityModel(nn.Module):
        def forward(self, img, **kwargs):
            _, _, ih, iw = img.shape
            gh = math.ceil(ih / stride)
            gw = math.ceil(iw / stride)
            return {"y": torch.full((1, 1, gh, gw), 0.05, dtype=torch.float32)}

    model = IdentityModel()
    image = torch.zeros(3, h, w)

    canvas_tiled = predict_tiled(
        model,
        image,
        output_stride=stride,
        tile_size=64,
        halo=16,
    )

    expected_gh = math.ceil(h / stride)
    expected_gw = math.ceil(w / stride)
    assert canvas_tiled.shape == (1, expected_gh, expected_gw)
    assert (canvas_tiled > 0.0).all()
    assert torch.allclose(canvas_tiled, torch.tensor(0.05), atol=1e-5)


def test_rmr_core_losses_parity():
    torch.manual_seed(42)
    pred = torch.rand(4, 1, 32, 32)
    tgt = torch.rand(4, 1, 32, 32)

    l_smooth = balanced_smooth_l1(pred, tgt)
    assert torch.isfinite(l_smooth)

    l_count = count_magnitude_loss(pred, tgt, mode="nb", dispersion=25.0)
    assert torch.isfinite(l_count)

    l_dm16 = flat_dm16_loss(pred, tgt)
    assert torch.isfinite(l_dm16)
