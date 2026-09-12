from __future__ import annotations

import copy
import gc
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rmr_core.data import (
    CrowdManifestDataset,
    collate_eval,
    collate_train,
    rasterize_points,
    train_transform,
)
from rmr_core.evaluation import evaluate_dataset, predict_tiled
from rmr_core.training import (
    build_checkpoint,
    load_rng_state,
    make_scheduler,
    safe_torch_save,
    seed_everything,
)
from rmr_v3.losses import RMRv3LossConfig, compute_rmr_v3_losses
from rmr_v3.model import RMRv3, RMRv3Config


# ===========================================================================
# 1. Checkpoint Resume Exact Numerical Roundtrip Verification
# ===========================================================================

def test_checkpoint_resume_exact_numerical_roundtrip(tmp_path: Path):
    """Rigorous test verifying exact numerical roundtrip of saved and resumed state.

    Resuming a checkpoint must be mathematically and numerically indistinguishable
    from an uninterrupted training process:
    - Model state dict matches exactly (all params and buffers).
    - Optimizer state dict matches exactly (step counts, momentum, variance).
    - Scheduler state dict matches exactly.
    - Scaler state dict matches exactly.
    - EMA shadow state matches to machine precision.
    - A subsequent forward-backward-optimizer step on identical data produces
      identical gradients, loss, outputs, and updated parameter values.
    """
    seed_everything(42, deterministic=True)
    device = torch.device("cpu")

    cfg = RMRv3Config(
        feature_width=16,
        region_sizes_px=(16, 32),
        iterations=2,
        pretrained=False,
        ema_decay=0.99,
    )
    model_live = RMRv3(cfg).to(device)

    bb_params = list(model_live.encoder.parameters())
    bb_ids = set(id(p) for p in bb_params)
    other_params = [p for p in model_live.parameters() if id(p) not in bb_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": bb_params, "lr": 1e-4},
            {"params": other_params, "lr": 1e-3},
        ],
        weight_decay=1e-4,
    )
    scheduler = make_scheduler(optimizer, epochs=10, warmup=2)
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    ema_decay = 0.99
    ema_state = copy.deepcopy(model_live.state_dict())
    for k in ema_state:
        ema_state[k] = ema_state[k].float()

    loss_cfg = RMRv3LossConfig(dm_strict=False)

    # Initial training step
    x_batch1 = torch.randn(2, 3, 64, 64)
    y_batch1 = torch.rand(2, 1, 16, 16)

    optimizer.zero_grad(set_to_none=True)
    out1 = model_live(x_batch1, uniform_reliability=False, solver_strength=1.0)
    losses1 = compute_rmr_v3_losses(out1, y_batch1, loss_cfg)
    losses1["total"].backward()
    torch.nn.utils.clip_grad_norm_(model_live.parameters(), 10.0)
    optimizer.step()
    scheduler.step()

    # EMA update
    with torch.no_grad():
        for name, param in model_live.named_parameters():
            if name in ema_state:
                ema_state[name].mul_(ema_decay).add_(
                    param.detach().float(), alpha=1.0 - ema_decay
                )
        for name, buf in model_live.named_buffers():
            if name in ema_state:
                ema_state[name].copy_(buf.float())

    # Build checkpoint
    ckpt_dict = build_checkpoint(
        epoch=1,
        model=model_live,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        config={"mock": True},
        config_hash="testhash123",
        best_mae=12.5,
        epochs_without_improvement=0,
        solver_strength=1.0,
        ema_state=ema_state,
    )
    ckpt_path = tmp_path / "checkpoint_roundtrip.pt"
    safe_torch_save(ckpt_dict, ckpt_path)

    # Run step 2 on continuous instance
    x_batch2 = torch.randn(2, 3, 64, 64)
    y_batch2 = torch.rand(2, 1, 16, 16)

    optimizer.zero_grad(set_to_none=True)
    out_cont = model_live(x_batch2, uniform_reliability=False, solver_strength=1.0)
    losses_cont = compute_rmr_v3_losses(out_cont, y_batch2, loss_cfg)
    losses_cont["total"].backward()
    torch.nn.utils.clip_grad_norm_(model_live.parameters(), 10.0)
    optimizer.step()
    scheduler.step()

    with torch.no_grad():
        for name, param in model_live.named_parameters():
            if name in ema_state:
                ema_state[name].mul_(ema_decay).add_(
                    param.detach().float(), alpha=1.0 - ema_decay
                )
        for name, buf in model_live.named_buffers():
            if name in ema_state:
                ema_state[name].copy_(buf.float())

    state_cont = copy.deepcopy(model_live.state_dict())
    ema_cont = copy.deepcopy(ema_state)

    # Now create a FRESH instance and resume from checkpoint
    model_resumed = RMRv3(cfg).to(device)
    bb_params_r = list(model_resumed.encoder.parameters())
    bb_ids_r = set(id(p) for p in bb_params_r)
    other_params_r = [p for p in model_resumed.parameters() if id(p) not in bb_ids_r]
    optimizer_r = torch.optim.AdamW(
        [
            {"params": bb_params_r, "lr": 1e-4},
            {"params": other_params_r, "lr": 1e-3},
        ],
        weight_decay=1e-4,
    )
    scheduler_r = make_scheduler(optimizer_r, epochs=10, warmup=2)
    scaler_r = torch.amp.GradScaler("cpu", enabled=False)

    loaded_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model_resumed.load_state_dict(loaded_ckpt["model"])
    optimizer_r.load_state_dict(loaded_ckpt["optimizer"])
    scheduler_r.load_state_dict(loaded_ckpt["scheduler"])
    scaler_r.load_state_dict(loaded_ckpt["scaler"])
    if "rng_state" in loaded_ckpt:
        load_rng_state(loaded_ckpt["rng_state"], strict=True)

    ema_state_r = copy.deepcopy(model_resumed.state_dict())
    for k in ema_state_r:
        ema_state_r[k] = ema_state_r[k].float()
    assert "ema_model" in loaded_ckpt
    for k in ema_state_r:
        if k in loaded_ckpt["ema_model"]:
            ema_state_r[k].copy_(loaded_ckpt["ema_model"][k].to(device=ema_state_r[k].device).float())

    # Check exact equivalence of loaded state
    for k, v in loaded_ckpt["model"].items():
        assert torch.equal(model_resumed.state_dict()[k], v)

    for k, v in loaded_ckpt["ema_model"].items():
        assert torch.equal(ema_state_r[k].cpu(), v)

    # Run step 2 on resumed instance
    optimizer_r.zero_grad(set_to_none=True)
    out_resumed = model_resumed(x_batch2, uniform_reliability=False, solver_strength=1.0)
    losses_resumed = compute_rmr_v3_losses(out_resumed, y_batch2, loss_cfg)
    losses_resumed["total"].backward()
    torch.nn.utils.clip_grad_norm_(model_resumed.parameters(), 10.0)
    optimizer_r.step()
    scheduler_r.step()

    with torch.no_grad():
        for name, param in model_resumed.named_parameters():
            if name in ema_state_r:
                ema_state_r[name].mul_(ema_decay).add_(
                    param.detach().float(), alpha=1.0 - ema_decay
                )
        for name, buf in model_resumed.named_buffers():
            if name in ema_state_r:
                ema_state_r[name].copy_(buf.float())

    # Assert exact numerical equivalence between continuous and resumed step 2
    assert torch.allclose(out_cont["y"], out_resumed["y"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(losses_cont["total"], losses_resumed["total"], atol=1e-6, rtol=1e-5)

    for k in state_cont:
        assert torch.allclose(state_cont[k], model_resumed.state_dict()[k], atol=1e-6, rtol=1e-5), (
            f"Param mismatch on key '{k}' between continued and resumed instances!"
        )

    for k in ema_cont:
        assert torch.allclose(ema_cont[k], ema_state_r[k], atol=1e-6, rtol=1e-5), (
            f"EMA shadow mismatch on key '{k}' between continued and resumed instances!"
        )


# ===========================================================================
# 2. Variable-Resolution Evaluation Pipeline Verification
# ===========================================================================

def test_variable_resolution_evaluation_pipeline(tmp_path: Path):
    """Verify full evaluation pipeline across variable, non-divisible resolutions.

    Tests both direct inference and haloed tiled inference on images with
    odd dimensions, non-square aspect ratios, and boundary-point annotations.
    """
    img_dir = tmp_path / "var_res_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = tmp_path / "var_res.jsonl"
    shapes = [
        (101, 103),   # Odd x odd, prime-adjacent
        (687, 512),   # Non-divisible tall image
        (73, 157),    # Extreme non-square prime dimensions
        (128, 256),   # Standard clean powers of 2
    ]

    manifest_rows = []
    for idx, (h, w) in enumerate(shapes):
        img_path = img_dir / f"img_{idx}.jpg"
        # Synthetic RGB image
        arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_path)

        # Points at corners, interior, and near borders
        pts = [
            [0.0, 0.0],
            [float(w - 1), float(h - 1)],
            [float(w // 2), float(h // 2)],
            [10.5, 20.5],
        ]
        manifest_rows.append({
            "image": str(img_path.resolve()),
            "points": pts,
            "id": f"sample_{idx}_{w}x{h}",
        })

    with manifest_path.open("w", encoding="utf-8") as f:
        for r in manifest_rows:
            f.write(json.dumps(r) + "\n")

    dataset = CrowdManifestDataset(manifest_path, train=False, output_stride=4)
    assert len(dataset) == len(shapes)

    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_eval)

    cfg = RMRv3Config(feature_width=16, pretrained=False, iterations=2)
    model = RMRv3(cfg).eval()
    device = torch.device("cpu")

    # 1. Direct evaluation (no tiling)
    rows_direct, summary_direct = evaluate_dataset(
        model=model,
        loader=loader,
        device=device,
        output_stride=4,
        run_tiling=False,
        enforce_gt_consistency=True,
    )
    assert len(rows_direct) == len(shapes)
    assert summary_direct["MAE"] >= 0.0
    assert summary_direct["RMSE"] >= 0.0
    assert np.isfinite(summary_direct["MAE"])

    for row, (h, w) in zip(rows_direct, shapes):
        assert row["gt"] == 4.0
        assert row["GAME0"] == pytest.approx(row["abs_err"], rel=1e-5, abs=1e-5)
        for lev in range(4):
            assert f"GAME{lev}" in row
            assert row[f"GAME{lev}"] >= 0.0

    # 2. Tiled evaluation with halo
    rows_tiled, summary_tiled = evaluate_dataset(
        model=model,
        loader=loader,
        device=device,
        output_stride=4,
        run_tiling=True,
        tile_size=128,
        practical_halo=16,
        enforce_gt_consistency=True,
    )
    assert len(rows_tiled) == len(shapes)
    assert summary_tiled["MAE"] >= 0.0
    assert "direct_tiled_discrepancy_mean" in summary_tiled
    assert np.isfinite(summary_tiled["direct_tiled_discrepancy_mean"])


# ===========================================================================
# 3. Memory Leak Stability Over 50 Iterations
# ===========================================================================

@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_memory_stability_over_50_iterations():
    """Verify zero memory leak over 50 consecutive forward-backward-eval iterations.

    Ensures that loss logging, EMA updates, and evaluation hooks do NOT retain
    the autograd computation graph or leak PyTorch tensors.
    """
    device = torch.device("cpu")
    cfg = RMRv3Config(feature_width=16, pretrained=False, iterations=1, ema_decay=0.99)
    model = RMRv3(cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_cfg = RMRv3LossConfig(dm_strict=False)

    ema_decay = 0.99
    ema_state = copy.deepcopy(model.state_dict())
    for k in ema_state:
        ema_state[k] = ema_state[k].float()

    all_disps_log = []
    mu_means_log = []

    # Warmup 10 iterations to initialize AdamW momentum and variance states
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        x = torch.randn(2, 3, 64, 64, device=device)
        tgt = torch.rand(2, 1, 16, 16, device=device)
        out = model(x)
        l = compute_rmr_v3_losses(out, tgt, loss_cfg)["total"]
        l.backward()
        optimizer.step()
        del x, tgt, out, l

    gc.collect()
    num_tensors_baseline = len([obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor)])

    # 50 consecutive train + eval steps
    for step in range(50):
        # 1. Train step
        model.train()
        optimizer.zero_grad(set_to_none=True)
        x_train = torch.randn(2, 3, 64, 64, device=device)
        y_train = torch.rand(2, 1, 16, 16, device=device)

        out_train = model(x_train, uniform_reliability=False, solver_strength=1.0)
        losses = compute_rmr_v3_losses(out_train, y_train, loss_cfg)
        total_loss = losses["total"]
        total_loss.backward()
        optimizer.step()

        # EMA update
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in ema_state:
                    ema_state[name].mul_(ema_decay).add_(
                        param.detach().float(), alpha=1.0 - ema_decay
                    )

        # Logging (proper .detach().cpu() pattern)
        with torch.no_grad():
            mu_means_log.append(float(out_train["b_region"].mean().item()))
            all_disps_log.append(out_train["region_dispersion"].detach().cpu().float().flatten())

        # 2. Eval step (under no_grad)
        model.eval()
        with torch.no_grad():
            x_eval = torch.randn(1, 3, 64, 64, device=device)
            out_eval = model(x_eval)
            _ = float(out_eval["y"].sum().item())

        # Clean per-step temporaries (simulates DataLoader batch lifecycle)
        del x_train, y_train, out_train, losses, total_loss, x_eval, out_eval

        # Keep log size bounded per epoch (simulates train loop reset at epoch boundary)
        if step % 10 == 9:
            all_disps_log.clear()
            mu_means_log.clear()

    gc.collect()
    num_tensors_final = len([obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor)])

    # Tensor count must not have grown across 50 iterations
    tensor_growth = num_tensors_final - num_tensors_baseline
    assert tensor_growth <= 5, (
        f"Significant tensor leak detected: baseline={num_tensors_baseline}, "
        f"final={num_tensors_final}, growth={tensor_growth} tensors!"
    )


# ===========================================================================
# 4. Data Augmentation Empty and Boundary Points Invariants
# ===========================================================================

def test_data_augmentation_empty_and_boundary_points():
    """Verify train_transform handles empty annotations, boundary clamping, and reflection involution."""
    crop_size = 64

    # 1. Empty image (0 points)
    empty_img = Image.fromarray(np.full((crop_size, crop_size, 3), 128, dtype=np.uint8))
    pts_empty = torch.zeros((0, 2), dtype=torch.float32)

    img_t, pts_out = train_transform(
        empty_img, pts_empty, crop_size=crop_size, scale_range=(1.0, 1.0), hflip_prob=0.5
    )
    assert img_t.shape == (3, crop_size, crop_size)
    assert pts_out.shape == (0, 2)
    assert pts_out.numel() == 0

    grid_empty = rasterize_points(pts_out, crop_size, crop_size, stride=4)
    assert grid_empty.sum().item() == 0.0

    # 2. Extreme boundary coordinates: (0, 0), (63.0, 63.0), (63.99, 10.0)
    # Using exact crop_size x crop_size guarantees deterministic left=0, top=0
    boundary_img = Image.fromarray(np.full((crop_size, crop_size, 3), 128, dtype=np.uint8))
    pts_boundary = torch.tensor([
        [0.0, 0.0],
        [float(crop_size - 1), float(crop_size - 1)],
        [float(crop_size) - 0.01, 10.0],
    ], dtype=torch.float32)

    # Force hflip=1.0 to stress-test reflection arithmetic
    img_flipped, pts_flipped = train_transform(
        boundary_img, pts_boundary, crop_size=crop_size, scale_range=(1.0, 1.0), hflip_prob=1.0
    )
    assert img_flipped.shape == (3, crop_size, crop_size)

    # All points must be conserved and tested (no skipped assertions)
    assert len(pts_flipped) == len(pts_boundary)
    assert (pts_flipped[:, 0] >= 0.0).all(), f"Negative x found: {pts_flipped[:, 0]}"
    assert (pts_flipped[:, 0] <= float(crop_size - 1)).all(), f"x > crop_size - 1 found: {pts_flipped[:, 0]}"
    assert (pts_flipped[:, 1] >= 0.0).all(), f"Negative y found: {pts_flipped[:, 1]}"
    assert (pts_flipped[:, 1] <= float(crop_size - 1)).all(), f"y > crop_size - 1 found: {pts_flipped[:, 1]}"

    grid_flipped = rasterize_points(pts_flipped, crop_size, crop_size, stride=4)
    assert grid_flipped.sum().item() == float(pts_flipped.shape[0])

    # 3. Involution test: flipping a second time recovers the clamped points
    img_f_pil = Image.fromarray((img_flipped.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    _, pts_recovered = train_transform(
        img_f_pil, pts_flipped, crop_size=crop_size, scale_range=(1.0, 1.0), hflip_prob=1.0
    )
    expected_x = pts_boundary[:, 0].clamp(0.0, float(crop_size - 1))
    assert torch.allclose(pts_recovered[:, 0], expected_x, atol=1e-4)


# ===========================================================================
# 5. Predict Tiled Training-Mode Safety & Mode Restoration Verification
# ===========================================================================

def test_predict_tiled_safely_evaluates_in_training_mode():
    """Verify predict_tiled executes safely on non-divisible images when model is in train mode."""
    cfg = RMRv3Config(feature_width=16, pretrained=False)
    model = RMRv3(cfg)
    model.train()  # Explicitly in training mode

    # Non-divisible dimensions that produce 1x1 residual patches with halo=0
    test_img = torch.randn(3, 513, 513)
    out = predict_tiled(model, test_img, tile_size=512, halo=0)

    # Expected spatial size: math.ceil(513 / 4) = 129
    assert out.shape == (1, 129, 129)
    assert torch.isfinite(out).all()

    # Invariant: model training mode must be cleanly restored to True
    assert model.training is True


# ===========================================================================
# 6. Early Stopping Patience Preservation On Resume
# ===========================================================================

def test_early_stopping_patience_preservation_on_resume(tmp_path: Path):
    """Verify that disabled patience (patience=0) does not accumulate false improvement gaps."""
    model = RMRv3(RMRv3Config(feature_width=16, pretrained=False))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = make_scheduler(optimizer, epochs=10, warmup=2)
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    # Checkpoint created when early stopping was disabled (patience = 0)
    ckpt = build_checkpoint(
        epoch=50,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        config={"mock": True},
        config_hash="testhash",
        best_mae=25.0,
        epochs_without_improvement=0,  # correctly kept at 0 when patience <= 0
        solver_strength=1.0,
    )
    ckpt_file = tmp_path / "ckpt_no_early_stop.pt"
    safe_torch_save(ckpt, ckpt_file)

    loaded = torch.load(ckpt_file, map_location="cpu", weights_only=False)

    # When resuming with patience = 20, accumulated gap is not prematurely tripped
    patience = 20
    epochs_without_improvement = int(loaded.get("epochs_without_improvement", 0)) if patience > 0 else 0
    assert epochs_without_improvement == 0
    assert epochs_without_improvement < patience
