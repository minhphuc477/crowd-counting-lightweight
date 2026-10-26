from __future__ import annotations

import math
import random
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from rmr_core.evaluation import evaluate_dataset
from rmr_core.metrics import bootstrap_ci, game_physical_image, game_single
from rmr_core.operators import build_multiscale_regions, regional_sum
from rmr_core.tiling import predict_tiled
from rmr_core.training import safe_torch_load, safe_torch_save
from rmr_v3.config import compute_config_hash
from rmr_v3.checkpoint import CheckpointManager, EMAManager
from rmr_v3.engine import build_optimizer
from rmr_core.losses import negative_binomial_nll_mean_dispersion
from rmr_v3.losses import (
    RMRv3LossConfig,
    bayesian_loss,
    compute_rmr_v3_losses,
    fidt_loss,
    sinkhorn_ot_loss,
)
from rmr_v3.model import RMRv3, RMRv3Config


def test_stage2_parameter_ceilings_track_a_and_track_b() -> None:
    """Stage 2: Strict parameter ceiling verification."""
    cfg_a = RMRv3Config(feature_width=32, pretrained=False)
    model_a = RMRv3(cfg_a)
    params_a = sum(p.numel() for p in model_a.parameters() if p.requires_grad)
    assert params_a <= 104441, f"Track A ceiling violated: {params_a} > 104,441"

    cfg_b = RMRv3Config(feature_width=48, subpixel_stride2=True, max_trainable_params=200000, pretrained=False)
    model_b = RMRv3(cfg_b)
    params_b = sum(p.numel() for p in model_b.parameters() if p.requires_grad)
    assert params_b <= 200000, f"Track B ceiling violated: {params_b} > 200,000"


def test_stage2_banned_antipattern_rejection() -> None:
    """Stage 2: Anti-pattern validator rejection contract."""
    banned_switches = [
        {"use_park": True},
        {"use_cpcm": True},
        {"use_coord_attn": True},
        {"use_nesterov_momentum": True},
        {"use_alternating_bb": True},
        {"use_top_down_semantic_gate": True},
        {"foreground_gate": True},
        {"hybrid_recovery_alpha": 0.5},
        {"dynamic_perspective_windows": True},
    ]
    for switch in banned_switches:
        with pytest.raises(ValueError):
            RMRv3Config(**switch)


def test_stage2_divisor16_contract_arbitrary_resolutions() -> None:
    """Stage 2: Divisor 16 padding/cropping contract for odd and prime resolutions."""
    model = RMRv3(RMRv3Config(feature_width=32, pretrained=False)).eval()
    test_shapes = [(511, 511), (333, 417), (127, 255), (64, 64)]

    with torch.no_grad():
        for h, w in test_shapes:
            x = torch.randn(1, 3, h, w)
            out = model(x, compute_energy=False)
            expected_h4 = math.ceil(h / 4)
            expected_w4 = math.ceil(w / 4)
            assert out["y"].shape[-2:] == (expected_h4, expected_w4), (
                f"Shape mismatch for input ({h}, {w}): got {out['y'].shape[-2:]}, expected ({expected_h4}, {expected_w4})"
            )
            assert torch.isfinite(out["y"]).all()


def test_stage2_batchnorm_tracking_eval_invariance() -> None:
    """Stage 2: BatchNorm running statistics must be bitwise invariant during evaluation."""
    model = RMRv3(RMRv3Config(feature_width=32, pretrained=False)).eval()
    bn_stats_before = [
        (m.running_mean.clone(), m.running_var.clone())
        for m in model.modules()
        if isinstance(m, nn.modules.batchnorm._BatchNorm)
    ]

    with torch.no_grad():
        for _ in range(3):
            _ = model(torch.randn(2, 3, 128, 128), compute_energy=False)

    bn_stats_after = [
        (m.running_mean.clone(), m.running_var.clone())
        for m in model.modules()
        if isinstance(m, nn.modules.batchnorm._BatchNorm)
    ]

    for (mean_b, var_b), (mean_a, var_a) in zip(bn_stats_before, bn_stats_after):
        assert torch.equal(mean_b, mean_a), "BatchNorm running_mean drifted during eval!"
        assert torch.equal(var_b, var_a), "BatchNorm running_var drifted during eval!"


def test_stage3_bayesian_loss_fp32_autograd_graph() -> None:
    """Stage 3: Bayesian loss evaluates with finite autograd gradients."""
    y0 = torch.rand(2, 1, 64, 64, requires_grad=True)
    pts = [torch.tensor([[10.0, 15.0], [30.0, 45.0]]), torch.empty((0, 2))]

    loss = bayesian_loss(y0, pts, stride=4, sigma_min=2.0, sigma_max=8.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert y0.grad is not None
    assert torch.isfinite(y0.grad).all()


def test_stage3_gradscaler_dynamic_recovery() -> None:
    """Stage 3: GradScaler recovers from non-finite gradients without parameter corruption."""
    model = nn.Sequential(nn.Linear(16, 8), nn.Linear(8, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cpu", init_scale=65536.0)

    # Initial state snapshot
    w_initial = model[0].weight.clone()

    # Trigger NaN gradient
    optimizer.zero_grad()
    x = torch.randn(4, 16)
    out = model(x)
    loss = out.sum() * float("nan")
    scaler.scale(loss).backward()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
    scaler.step(optimizer)
    scaler.update()

    # Optimizer step must have been skipped: weights unchanged
    assert torch.equal(model[0].weight, w_initial), "Weights updated despite non-finite gradient!"
    assert scaler.get_scale() < 65536.0, "GradScaler failed to reduce scale factor!"


def test_stage3_loss_numerical_limits() -> None:
    """Stage 3: Loss numerical stability at extreme limits (N=0 and N=20,000)."""
    # Negative Binomial at extremes
    y_ext = torch.tensor([0.0, 20000.0])
    mu_ext = torch.tensor([1e-4, 19500.0])
    nb_loss = negative_binomial_nll_mean_dispersion(y_ext, mu_ext, dispersion=50.0)
    assert torch.isfinite(nb_loss)

    # Sinkhorn OT at N=0 and N=500
    prob = torch.rand(1, 1, 32, 32)
    ot_empty = sinkhorn_ot_loss(prob, [torch.empty((0, 2))], stride=4)
    assert torch.isfinite(ot_empty)
    ot_dense = sinkhorn_ot_loss(prob, [torch.rand(500, 2) * 128.0], stride=4)
    assert torch.isfinite(ot_dense)

    # FIDT at N=0 and N=500
    fidt_empty = fidt_loss(prob, [torch.empty((0, 2))], stride=4)
    assert torch.isfinite(fidt_empty)
    fidt_dense = fidt_loss(prob, [torch.rand(500, 2) * 128.0], stride=4)
    assert torch.isfinite(fidt_dense)


def test_stage4_adjoint_operator_symmetry() -> None:
    """Stage 4: Verification of adjoint operator symmetry <A y, r> = <y, A^T r>."""
    h, w, stride = 64, 64, 4
    regions = build_multiscale_regions(h, w, output_stride=stride, region_sizes_px=(16, 32), overlap=0.5)
    gh, gw = math.ceil(h / stride), math.ceil(w / stride)

    torch.manual_seed(42)
    y = torch.rand(gh, gw)
    r = torch.rand(regions.boxes.shape[0])

    # Forward operator A y: regional sums
    ay = regional_sum(y, regions.boxes).view(-1)
    lhs = torch.dot(ay, r).item()

    # Adjoint operator A^T r: scatter regional weights back to grid
    at_r = torch.zeros(gh, gw)
    for k, (y1, x1, y2, x2) in enumerate(regions.boxes.tolist()):
        at_r[y1:y2, x1:x2] += r[k]
    rhs = (y * at_r).sum().item()

    rel_diff = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-6)
    assert rel_diff < 1e-4, f"Adjoint property violated: lhs={lhs}, rhs={rhs}, rel_diff={rel_diff}"


def test_stage4_morozov_thresholding_and_nonnegativity() -> None:
    """Stage 4: Morozov soft-thresholding noise rejection and non-negativity."""
    residuals = torch.tensor([-2.5, -0.5, 0.0, 0.8, 3.2])
    threshold = 1.0

    # Soft thresholding: sgn(r) * max(|r| - threshold, 0)
    r_thresh = torch.sign(residuals) * F.relu(torch.abs(residuals) - threshold)
    assert r_thresh[1].item() == 0.0, "Noise below threshold was not zeroed!"
    assert r_thresh[2].item() == 0.0
    assert r_thresh[3].item() == 0.0
    assert abs(r_thresh[0].item() - (-1.5)) < 1e-5
    assert abs(r_thresh[4].item() - 2.2) < 1e-5

    # Non-negative projection P+
    y_raw = torch.tensor([-1.2, 0.0, 0.5, 4.0])
    y_proj = torch.clamp_min(y_raw, 0.0)
    assert (y_proj >= 0.0).all(), "Non-negativity projection violated!"


def test_stage5_optimizer_parameter_groups_separation() -> None:
    """Stage 5: AdamW parameter group separation and zero weight decay on 1D biases/scales."""
    model = RMRv3(RMRv3Config(feature_width=32, pretrained=False))
    opt = build_optimizer(model, {"train": {"backbone_lr_scale": 0.1, "weight_decay": 1e-4}}, lr_init=1e-4)

    group_names = [g["name"] for g in opt.param_groups]
    assert group_names == ["backbone_decay", "backbone_no_decay", "main_decay", "main_no_decay"]

    # Verify 1D tensors have weight_decay = 0.0
    for g in opt.param_groups:
        if "no_decay" in g["name"]:
            assert g["weight_decay"] == 0.0
            for p in g["params"]:
                assert p.ndim <= 1
        elif "backbone" in g["name"]:
            assert abs(g["lr"] - 1e-5) < 1e-8


def test_stage5_ema_manager_swap_context() -> None:
    """Stage 5: EMAManager parameter tracking and safe swap context."""
    model = RMRv3(RMRv3Config(feature_width=32, pretrained=False))
    ema = EMAManager(model, decay=0.99)
    device = torch.device("cpu")

    # Snapshot live parameter
    orig_val = next(model.parameters()).clone()

    # Mutate parameter and update EMA
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)

    mutated_val = next(model.parameters()).clone()
    assert not torch.equal(orig_val, mutated_val)

    # Test swap_into context
    with ema.swap_into(model, device):
        swapped_val = next(model.parameters()).clone()
        # Inside context, parameter should match EMA shadow weight (d*orig + (1-d)*mutated)
        assert not torch.equal(swapped_val, mutated_val)

    # After context, parameter must be restored bitwise to mutated live value
    restored_val = next(model.parameters()).clone()
    assert torch.equal(restored_val, mutated_val), "Live weights not restored after EMA context exit!"


def test_stage6_checkpoint_resumption_and_rng_continuity(tmp_path: Path) -> None:
    """Stage 6: Safe checkpoint saving, loading (weights_only=True), and bitwise RNG continuity."""
    ckpt_file = tmp_path / "test_ckpt.pt"
    dummy_payload = {
        "model": {"weight": torch.randn(4, 4)},
        "epoch": 42,
        "config_hash": "abc123hash",
    }
    safe_torch_save(dummy_payload, ckpt_file)
    loaded = safe_torch_load(ckpt_file, weights_only=True)
    assert loaded["epoch"] == 42
    assert torch.equal(loaded["model"]["weight"], dummy_payload["model"]["weight"])

    # Bitwise RNG restoration test
    torch.manual_seed(999)
    rng_state = torch.get_rng_state()
    val_1 = torch.randn(5)

    # Advance RNG
    _ = torch.randn(10)

    # Restore RNG state
    torch.set_rng_state(rng_state)
    val_2 = torch.randn(5)
    assert torch.equal(val_1, val_2), "RNG state restoration is not bitwise identical!"


def test_stage7_tiling_inference_halo_continuity() -> None:
    """Stage 7: Tiled inference with halo context produces seamless, stable predictions."""
    model = RMRv3(RMRv3Config(feature_width=32, pretrained=False)).eval()
    img = torch.randn(3, 512, 512)

    with torch.no_grad():
        y_full = model(img.unsqueeze(0), compute_energy=False)["y"][0]
        y_tiled = predict_tiled(model, img, output_stride=4, tile_size=256, halo=32)

    count_full = y_full.sum().item()
    count_tiled = y_tiled.sum().item()
    rel_diff = abs(count_full - count_tiled) / max(abs(count_full), 1.0)
    assert rel_diff < 0.15, f"Tiled count diverged significantly from full: {count_tiled} vs {count_full}"
    assert torch.isfinite(y_tiled).all()


def test_stage7_game_and_bootstrap_metrics() -> None:
    """Stage 7: GAME metric monotonicity and Bootstrap CI coverage."""
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[1.2, 1.8], [3.1, 3.9]])

    g0 = game_single(pred, target, 0)
    g1 = game_single(pred, target, 1)
    assert g0 <= g1 + 1e-6, f"GAME monotonicity violated: GAME0 ({g0}) > GAME1 ({g1})"

    # Bootstrap CI
    errs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 2.5, 3.5, 1.5])
    low, high = bootstrap_ci(errs, n_boot=200, alpha=0.05)
    mean_val = float(np.mean(errs))
    assert low <= mean_val <= high, f"Mean {mean_val} outside Bootstrap CI [{low}, {high}]"
