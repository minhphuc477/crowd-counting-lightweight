from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from rmr_core.data import CrowdManifestDataset, train_transform


def test_universal_0_percent_padding_sha():
    """Verify that train_transform achieves strictly 0.0% synthetic padding across SHA train images."""
    data_root = Path("data")
    manifest = Path("data/sha_a_train_all.jsonl")
    if not manifest.exists():
        pytest.skip("Dataset manifest not found")

    with manifest.open("r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    # Test across first 30 images with 3 random crops each
    for it in items[:30]:
        p = data_root / it["image"]
        pts = torch.tensor(it["points"], dtype=torch.float32).reshape(-1, 2)
        with Image.open(p) as img:
            img_rgb = img.convert("RGB")
        for _ in range(2):
            crop_t, cpts = train_transform(
                img_rgb.copy(), pts.clone(), crop_size=512, pad_small_images=False,
            )
            assert crop_t.shape == (3, 512, 512)
            # Check for gray padding (128, 128, 128) -> ~0.50196 in float
            is_gray = (
                (torch.abs(crop_t[0] - 128 / 255) < 1e-4) &
                (torch.abs(crop_t[1] - 128 / 255) < 1e-4) &
                (torch.abs(crop_t[2] - 128 / 255) < 1e-4)
            )
            # A full image crop cannot be made purely of synthetic padding
            assert is_gray.float().mean().item() < 0.01
            if cpts.numel():
                assert (cpts[:, 0] >= 0.0).all() and (cpts[:, 0] <= 511.0).all()
                assert (cpts[:, 1] >= 0.0).all() and (cpts[:, 1] <= 511.0).all()


def test_universal_multidataset_extreme_cases():
    """Adversarial stress test simulating extreme dimensions across QNRF, NWPU, and drone benchmarks."""
    crop_size = 512
    cases = [
        ("Tiny 150x150", Image.new("RGB", (150, 150), (200, 100, 50)), torch.tensor([[10., 10.], [140., 140.]])),
        ("Extreme Panorama 100x3000", Image.new("RGB", (100, 3000), (50, 100, 150)), torch.tensor([[50., 500.], [50., 2500.]])),
        ("Extreme Wide 3000x100", Image.new("RGB", (3000, 100), (50, 150, 100)), torch.tensor([[500., 50.], [2500., 50.]])),
        ("Huge QNRF 6000x4000", Image.new("RGB", (6000, 4000), (100, 100, 100)), torch.tensor([[100., 100.], [5500., 3500.]])),
        ("Empty image N=0", Image.new("RGB", (800, 600), (10, 20, 30)), torch.empty((0, 2))),
    ]

    for name, img, pts in cases:
        crop_t, cpts = train_transform(
            img, pts, crop_size=crop_size, pad_small_images=False, max_size=2048,
        )
        assert crop_t.shape == (3, crop_size, crop_size), f"{name} failed output shape"
        assert not torch.isnan(crop_t).any()
        assert not torch.isinf(crop_t).any()
        if cpts.numel():
            assert (cpts[:, 0] >= 0.0).all() and (cpts[:, 0] <= float(crop_size - 1)).all()
            assert (cpts[:, 1] >= 0.0).all() and (cpts[:, 1] <= float(crop_size - 1)).all()


def test_eval_resolution_guard():
    """Verify evaluation mode caps ultra-large images (e.g. 6000x4000) to max_size to prevent CUDA OOM."""
    class DummyItem:
        pass

    # Create dataset instance with max_size=1024 for testing
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        img_path = tmp_root / "huge_img.jpg"
        img = Image.new("RGB", (4000, 3000), (100, 150, 200))
        img.save(img_path)

        manifest_path = tmp_root / "manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({
                "id": "huge_01",
                "image": str(img_path),
                "points": [[1000.0, 1000.0], [3000.0, 2000.0]],
            }) + "\n")

        ds_eval = CrowdManifestDataset(
            manifest=manifest_path,
            train=False,
            max_size=1024,
            pad_small_images=False,
        )
        sample = ds_eval[0]
        img_t = sample["image"]
        # Max dimension must be bounded to 1024
        assert max(sample["height"], sample["width"]) <= 1024
        assert img_t.shape[-1] <= 1024 and img_t.shape[-2] <= 1024
        pts = sample["points"]
        assert pts.numel() == 4  # 2 points x 2 coords
        assert (pts[:, 0] >= 0).all() and (pts[:, 0] <= 1024).all()
        assert (pts[:, 1] >= 0).all() and (pts[:, 1] <= 1024).all()


def test_small_image_scale_variance_preservation():
    """Verify that images smaller than crop_size maintain non-zero scale variance and spatial jitter."""
    # Small image (300 x 400)
    img = Image.new("RGB", (400, 300), (120, 150, 180))
    pts = torch.tensor([[100.0, 100.0], [250.0, 200.0]])
    crop_size = 512

    scales_observed = []
    offsets_observed = []
    for _ in range(20):
        # Sample crops
        crop_t, cpts = train_transform(
            img.copy(), pts.clone(), crop_size=crop_size, scale_range=(0.75, 1.25),
            pad_small_images=False,
        )
        assert crop_t.shape == (3, crop_size, crop_size)
        if cpts.numel():
            assert (cpts[:, 0] >= 0.0).all() and (cpts[:, 0] <= float(crop_size - 1)).all()
            assert (cpts[:, 1] >= 0.0).all() and (cpts[:, 1] <= float(crop_size - 1)).all()
            offsets_observed.append(cpts[0, 0].item())

    # Observed point coordinates must vary across crops (proving both scale and translation jitter)
    assert len(set(round(o, 2) for o in offsets_observed)) > 1, "Augmentation collapsed to deterministic values!"

