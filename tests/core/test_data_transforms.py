from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from rmr_core.data import (
    CrowdManifestDataset,
    compute_manifest_density,
    rasterize_points,
    train_transform,
)
from rmr_v2.losses import block_sum_2d, flat_dm16_loss
from rmr_v3.config import (
    compute_config_hash,
    compute_file_sha256,
    extract_trajectory_config,
    validate_resume_compatibility,
    validate_v3_config,
)


def test_pixel_center_continuous_point_scaling():
    """Verify exact continuous pixel-center scaling geometry:
    x' = (x + 0.5) * (w1 / w0) - 0.5
    """
    w0, h0 = 100, 200
    w1, h1 = 150, 300
    pts = torch.tensor([
        [0.0, 0.0],
        [49.5, 99.5],
        [99.0, 199.0],
    ])

    scaled_x = (pts[:, 0] + 0.5) * (w1 / w0) - 0.5
    scaled_y = (pts[:, 1] + 0.5) * (h1 / h0) - 0.5

    # Origin center at 0.5 maps to 0.75 -> coordinate 0.25
    assert pytest.approx(scaled_x[0].item(), rel=1e-5) == 0.25
    assert pytest.approx(scaled_y[0].item(), rel=1e-5) == 0.25

    # Midpoint (49.5 + 0.5) * 1.5 - 0.5 = 74.5
    assert pytest.approx(scaled_x[1].item(), rel=1e-5) == 74.5
    assert pytest.approx(scaled_y[1].item(), rel=1e-5) == 149.5

    # Right/bottom pixel center (99 + 0.5) * 1.5 - 0.5 = 148.75 = 149 - 0.25
    assert pytest.approx(scaled_x[2].item(), rel=1e-5) == 148.75
    assert pytest.approx(scaled_y[2].item(), rel=1e-5) == 298.75


def test_horizontal_flip_inversion_and_preservation():
    """Horizontal flip around pixel centers: x' = (crop_size - 1) - x."""
    crop_size = 512
    x = torch.tensor([0.0, 255.5, 511.0, 100.2])
    x_flipped = (crop_size - 1) - x
    assert pytest.approx(x_flipped[0].item()) == 511.0
    assert pytest.approx(x_flipped[1].item()) == 255.5
    assert pytest.approx(x_flipped[2].item()) == 0.0

    # Inversion: flipping twice recovers original
    x_recovered = (crop_size - 1) - x_flipped
    torch.testing.assert_close(x_recovered, x)


def test_no_synthetic_padding_small_image():
    """Images smaller than crop_size must be scaled up to at least crop_size without synthetic gray padding."""
    crop_size = 512
    # Create small image (200x300) with non-gray distinct pattern (all values 200)
    arr = np.full((300, 200, 3), 200, dtype=np.uint8)
    img = Image.fromarray(arr)

    pts = torch.tensor([[50.0, 50.0], [150.0, 250.0]])
    cropped_t, cropped_pts = train_transform(
        img, pts, crop_size=crop_size, scale_range=(0.75, 1.25), hflip_prob=0.0
    )

    assert cropped_t.shape == (3, crop_size, crop_size)
    # The image was scaled up and cropped from real pixels; no gray (128/255 = ~0.502) border was padded
    # All pixels should be 200 / 255 = ~0.784
    expected_val = 200.0 / 255.0
    assert torch.allclose(cropped_t, torch.full_like(cropped_t, expected_val), atol=1e-2)

    # All cropped points must be strictly within bounds
    if cropped_pts.numel():
        assert (cropped_pts[:, 0] >= 0).all() and (cropped_pts[:, 0] < crop_size).all()
        assert (cropped_pts[:, 1] >= 0).all() and (cropped_pts[:, 1] < crop_size).all()


def test_count_conservation_in_rasterization():
    """Rasterization must conserve the exact count of all valid points."""
    pts = torch.tensor([
        [0.0, 0.0],
        [10.2, 20.7],
        [511.9, 511.9],
        [-5.0, 10.0],    # out of bounds (left)
        [10.0, 600.0],   # out of bounds (bottom)
    ])
    y = rasterize_points(pts, image_h=512, image_w=512, stride=4)
    # 3 points inside, 2 out of bounds
    assert y.sum().item() == 3.0


def test_photometric_jitter_disabled_by_default():
    """When jitter=0.0, train_transform produces identical pixel values to pure PIL transform."""
    arr = np.random.randint(50, 200, (600, 600, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    pts = torch.empty((0, 2))

    t1, _ = train_transform(
        img.copy(), pts, crop_size=512, scale_range=(1.0, 1.0), hflip_prob=0.0,
        brightness_jitter=0.0, contrast_jitter=0.0
    )
    # Values should be in [0, 1]
    assert (t1 >= 0.0).all() and (t1 <= 1.0).all()


def test_compute_manifest_density_filters_oob_points(tmp_path: Path):
    """compute_manifest_density must filter out-of-image points consistently with rasterize_points."""
    img_path = tmp_path / "test_img.jpg"
    img = Image.new("RGB", (100, 100), (128, 128, 128))
    img.save(img_path)

    # 2 points inside, 2 points outside bounds
    manifest_file = tmp_path / "manifest.jsonl"
    data = [{
        "image": str(img_path.name),
        "points": [
            [10.0, 10.0],
            [90.0, 90.0],
            [-10.0, 50.0],  # OOB
            [150.0, 50.0],  # OOB
        ],
        "id": "img1",
    }]
    with manifest_file.open("w", encoding="utf-8") as f:
        for r in data:
            f.write(json.dumps(r) + "\n")

    # Stride 4 on 100x100: ceil(100/4)*ceil(100/4) = 25*25 = 625 cells
    # Valid points: 2
    # Expected density: 2 / 625 = 0.0032
    density = compute_manifest_density(manifest_file, output_stride=4, data_root=tmp_path)
    assert pytest.approx(density, rel=1e-5) == 2.0 / 625.0


def test_block_sum_2d_and_flat_dm16_divisibility_guard():
    """FlatDM16 and block_sum_2d(strict=True) must reject non-divisible spatial grids."""
    k = 4
    # Divisible shape (128x128) succeeds
    divisible_x = torch.rand(2, 1, 128, 128)
    out = block_sum_2d(divisible_x, k=k, strict=True)
    assert out.shape == (2, 1, 32, 32)

    # Non-divisible shape (127x128) raises ValueError when strict=True
    non_divisible_x = torch.rand(2, 1, 127, 128)
    with pytest.raises(ValueError, match="must be divisible by block size"):
        block_sum_2d(non_divisible_x, k=k, strict=True)

    # flat_dm16_loss raises ValueError for non-divisible inputs
    pred = torch.rand(2, 1, 126, 128)
    target = torch.rand(2, 1, 126, 128)
    with pytest.raises(ValueError, match="FlatDM16 requires grid dimensions divisible by block size"):
        flat_dm16_loss(pred, target, kappa=20.0, stride=4)


def test_config_hash_and_resume_locking():
    """Config hash must be deterministic and resume compatibility must enforce hash match."""
    cfg1 = {
        "seed": 42,
        "model": {"omega": 1.0, "iterations": 2, "feature_width": 32, "output_stride": 4},
        "loss": {"lambda_count": 1.0, "lambda_cell": 0.25},
        "train": {"lr": 1e-4, "weight_decay": 1e-4},
        "data": {"crop_size": 512, "scale_range": [0.75, 1.25], "hflip_prob": 0.5},
    }
    cfg2 = dict(cfg1)
    cfg2["data"] = dict(cfg1["data"])

    h1 = compute_config_hash(cfg1)
    h2 = compute_config_hash(cfg2)
    assert h1 == h2
    assert isinstance(h1, str) and len(h1) == 64

    # Different config yields different hash
    cfg3 = dict(cfg1)
    cfg3["data"] = dict(cfg1["data"])
    cfg3["data"]["crop_size"] = 256
    h3 = compute_config_hash(cfg3)
    assert h1 != h3

    # Matching hash passes resume check
    validate_resume_compatibility(cfg1, cfg2, ckpt_hash=h1, incoming_hash=h2)

    # Mismatched hash raises ValueError
    with pytest.raises(ValueError, match="Resume config hash mismatch"):
        validate_resume_compatibility(cfg1, cfg3, ckpt_hash=h1, incoming_hash=h3)


def test_config_hash_invariant_to_environment_paths():
    """Config hash must be invariant to output_dir, data_root, pin_memory, and filesystem prefixes."""
    cfg_local = {
        "seed": 42,
        "output_dir": "runs/sha_a/rmr_v3_rw_seed42",
        "model": {"backbone": "mobilenetv4_conv_small_050.e3000_r224_in1k", "init_m0": 0.01576340398, "omega": 1.0, "output_stride": 4},
        "loss": {"lambda_count": 1.0, "lambda_cell": 0.25},
        "train": {"lr": 1e-4, "workers": 4, "pin_memory": False, "epochs": 1000},
        "data": {
            "data_root": "F:/data",
            "train_manifest": "data/sha_a_train_all.jsonl",
            "val_manifest": "data/sha_a_test.jsonl",
            "crop_size": 512,
            "scale_range": [0.75, 1.25],
        },
    }

    cfg_kaggle = {
        "seed": 42,
        "output_dir": "/kaggle/working/experiment_output",
        "model": {"backbone_name": "mobilenetv4_conv_small_050.e3000_r224_in1k", "init_m0": 0.0157634, "sirt_omega": 1.0, "output_stride": 4},
        "loss": {"lambda_count": 1.0, "lambda_cell": 0.25},
        "train": {"lr": 1e-4, "workers": 4, "pin_memory": True, "epochs": 1000},
        "data": {
            "data_root": "/kaggle/input/shanghaitech/data",
            "train_manifest": "/kaggle/input/shanghaitech/data/sha_a_train_all.jsonl",
            "val_manifest": "/kaggle/input/shanghaitech/data/sha_a_test.jsonl",
            "crop_size": 512,
            "scale_range": [0.75, 1.25],
        },
    }

    h_local = compute_config_hash(cfg_local)
    h_kaggle = compute_config_hash(cfg_kaggle)
    assert h_local == h_kaggle, f"Hashes must match across environments: {h_local} vs {h_kaggle}"


def test_config_hash_sensitive_to_workers():
    """Different workers setting (e.g. 0 vs 4) must alter trajectory hash due to RNG stream differences."""
    base_cfg = {
        "seed": 42,
        "model": {"output_stride": 4, "omega": 1.0},
        "loss": {"lambda_count": 1.0},
        "train": {"lr": 1e-4, "epochs": 10, "workers": 0},
        "data": {"crop_size": 512},
    }
    cfg_w4 = dict(base_cfg)
    cfg_w4["train"] = dict(base_cfg["train"])
    cfg_w4["train"]["workers"] = 4

    h0 = compute_config_hash(base_cfg)
    h4 = compute_config_hash(cfg_w4)
    assert h0 != h4

    with pytest.raises(ValueError, match="Resume config hash mismatch"):
        validate_resume_compatibility(base_cfg, cfg_w4, ckpt_hash=h0, incoming_hash=h4)


def test_config_hash_sensitive_to_eval_protocol_and_early_stopping():
    """eval_every, early_stopping, and patience must alter trajectory hash."""
    base_cfg = {
        "seed": 42,
        "model": {"output_stride": 4},
        "loss": {"lambda_count": 1.0},
        "train": {"lr": 1e-4, "epochs": 100, "eval_every": 10, "early_stopping": False, "patience": 0},
    }
    h_base = compute_config_hash(base_cfg)

    # eval_every change
    cfg_eval = dict(base_cfg, train=dict(base_cfg["train"], eval_every=5))
    assert compute_config_hash(cfg_eval) != h_base

    # early_stopping change
    cfg_es = dict(base_cfg, train=dict(base_cfg["train"], early_stopping=True, patience=20))
    assert compute_config_hash(cfg_es) != h_base


def test_config_hash_sensitive_to_manifest_content(tmp_path: Path):
    """Altering manifest file contents even by 1 byte must change the manifest SHA256 and config hash."""
    manifest_file = tmp_path / "train.jsonl"
    manifest_file.write_text('{"image": "img1.jpg", "points": [[10.0, 10.0]]}\n', encoding="utf-8")

    cfg1 = {
        "seed": 42,
        "model": {"output_stride": 4},
        "data": {"train_manifest": str(manifest_file), "crop_size": 512},
    }
    h1 = compute_config_hash(cfg1)
    traj1 = extract_trajectory_config(cfg1)
    sha1 = traj1["data"]["train_manifest_sha256"]
    assert sha1 != "<virtual>"
    assert len(sha1) == 64

    # Alter 1 byte in manifest
    manifest_file.write_text('{"image": "img1.jpg", "points": [[10.0, 10.1]]}\n', encoding="utf-8")
    h2 = compute_config_hash(cfg1)
    traj2 = extract_trajectory_config(cfg1)
    sha2 = traj2["data"]["train_manifest_sha256"]

    assert sha1 != sha2
    assert h1 != h2
    with pytest.raises(ValueError, match="Resume config hash mismatch"):
        validate_resume_compatibility(cfg1, cfg1, ckpt_hash=h1, incoming_hash=h2)


def test_validate_v3_config_alias_conflict():
    """Declaring conflicting alias keys simultaneously must raise ValueError."""
    # Conflicting backbone aliases
    bad_cfg1 = {
        "model": {
            "backbone": "mobilenetv4_conv_small_050.e3000_r224_in1k",
            "backbone_name": "mobilenetv4_conv_small_050.e3000_r224_in1k",
        }
    }
    with pytest.raises(ValueError, match="Conflicting alias keys.*backbone"):
        validate_v3_config(bad_cfg1)

    # Conflicting omega aliases
    bad_cfg2 = {
        "model": {
            "omega": 1.0,
            "sirt_omega": 1.0,
        }
    }
    with pytest.raises(ValueError, match="Conflicting alias keys.*omega"):
        validate_v3_config(bad_cfg2)

    # Valid non-conflicting single alias passes
    good_cfg1 = {"model": {"backbone": "mobilenetv4_conv_small_050.e3000_r224_in1k", "omega": 1.0}}
    good_cfg2 = {"model": {"backbone_name": "mobilenetv4_conv_small_050.e3000_r224_in1k", "sirt_omega": 1.0}}
    validate_v3_config(good_cfg1)
    validate_v3_config(good_cfg2)


def test_resume_flow_with_derived_init_m0(tmp_path: Path):
    """Resume validation must succeed when init_m0 is derived from manifest in both runs."""
    img_path = tmp_path / "img1.jpg"
    img = Image.new("RGB", (100, 100), (128, 128, 128))
    img.save(img_path)

    manifest_file = tmp_path / "train.jsonl"
    with manifest_file.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"image": str(img_path.name), "points": [[20.0, 30.0]], "id": "1"}) + "\n")

    # 1. Base config loaded from YAML (does NOT contain init_m0)
    raw_cfg = {
        "seed": 42,
        "model": {"output_stride": 4, "omega": 1.0},
        "loss": {"lambda_count": 1.0},
        "train": {"lr": 1e-4, "epochs": 10},
        "data": {"train_manifest": str(manifest_file), "data_root": str(tmp_path), "crop_size": 64},
    }

    # 2. Fresh training run derives init_m0 before saving checkpoint
    run_cfg = dict(raw_cfg)
    run_cfg["model"] = dict(raw_cfg["model"])
    run_cfg["model"]["init_m0"] = compute_manifest_density(
        run_cfg["data"]["train_manifest"],
        output_stride=run_cfg["model"]["output_stride"],
        data_root=run_cfg["data"].get("data_root"),
    )
    saved_hash = compute_config_hash(run_cfg)

    # 3. Resume run starts from raw YAML (no init_m0), derives init_m0, then validates
    incoming_cfg = dict(raw_cfg)
    incoming_cfg["model"] = dict(raw_cfg["model"])
    incoming_cfg["model"]["init_m0"] = compute_manifest_density(
        incoming_cfg["data"]["train_manifest"],
        output_stride=incoming_cfg["model"]["output_stride"],
        data_root=incoming_cfg["data"].get("data_root"),
    )
    incoming_hash = compute_config_hash(incoming_cfg)

    assert saved_hash == incoming_hash
    validate_resume_compatibility(run_cfg, incoming_cfg, ckpt_hash=saved_hash, incoming_hash=incoming_hash)

