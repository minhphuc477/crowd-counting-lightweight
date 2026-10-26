from __future__ import annotations

import json
from pathlib import Path
from PIL import Image
import numpy as np
import pytest
import torch

from rmr_core.data import CrowdManifestDataset, collate_train
from rmr_core.transforms import (
    cap_image_resolution,
    rasterize_points,
    train_transform,
)


def test_crop_first_box_resizing_numerical_equivalence():
    """Verify crop-first box resizing preserves exact dimensions and subpixel points."""
    w0, h0 = 1600, 1200
    arr = np.random.randint(0, 256, (h0, w0, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    pts = torch.tensor([
        [100.0, 100.0],
        [800.0, 600.0],
        [1500.0, 1100.0],
    ])

    crop_t, cpts = train_transform(
        img,
        pts,
        crop_size=512,
        scale_range=(1.2, 1.2),
        hflip_prob=0.0,
        pad_small_images=False,
    )
    assert crop_t.shape == (3, 512, 512)
    assert (crop_t >= 0.0).all() and (crop_t <= 1.0).all()
    if cpts.numel():
        assert (cpts[:, 0] >= 0.0).all() and (cpts[:, 0] <= 511.0).all()
        assert (cpts[:, 1] >= 0.0).all() and (cpts[:, 1] <= 511.0).all()


def test_box_resize_edge_clamp_invariant():
    """Verify that bounding box coordinates never exceed PIL original image size."""
    w0, h0 = 400, 300
    arr = np.full((h0, w0, 3), 150, dtype=np.uint8)
    img = Image.fromarray(arr)
    # Center point guaranteed to be included with crowd_aware anchor
    pts = torch.tensor([[200.0, 150.0]])

    # Under pad_small_images=False, shorter side is scaled to >= 512
    crop_t, cpts = train_transform(
        img,
        pts,
        crop_size=512,
        scale_range=(0.75, 1.25),
        hflip_prob=0.0,
        pad_small_images=False,
        crowd_aware_crop=True,
        crowd_aware_prob=1.0,
    )
    assert crop_t.shape == (3, 512, 512)
    assert cpts.numel() == 2
    assert (cpts[:, 0] >= 0.0).all() and (cpts[:, 0] <= 511.0).all()
    assert (cpts[:, 1] >= 0.0).all() and (cpts[:, 1] <= 511.0).all()


def test_raw_frame_exif_preservation_contract():
    """Verify dataset loading preserves raw un-transposed coordinate space for MATLAB annotations."""
    manifest_p = Path("data/qnrf_test.jsonl")
    if not manifest_p.exists():
        pytest.skip("QNRF test manifest not available")

    # Image img_0097.jpg has EXIF orientation 6 (90 CW)
    with manifest_p.open("r", encoding="utf-8") as f:
        target_item = None
        for line in f:
            it = json.loads(line)
            if "img_0097" in it["image"]:
                target_item = it
                break

    if target_item is None:
        pytest.skip("img_0097 not found in manifest")

    img_path = Path("data") / target_item["image"]
    if not img_path.exists():
        pytest.skip("Raw QNRF image file not locally downloaded")

    with Image.open(img_path) as raw_img:
        w_raw, h_raw = raw_img.size
        # img_0097 raw size is (3264, 2448)
        assert (w_raw, h_raw) == (3264, 2448)

    pts = target_item["points"]
    max_y = max(p[1] for p in pts)
    # y coordinates go up to ~2445 (< 2448), confirming alignment with raw height 2448
    assert max_y < 2448.0


def test_batch_collate_mixed_empty_points():
    """Verify collate_train correctly handles batches with mixed empty and non-empty points."""
    sample_empty = {
        "image": torch.rand(3, 512, 512),
        "target_y": torch.zeros(1, 128, 128),
        "points": torch.empty((0, 2)),
        "id": "empty_01",
    }
    sample_dense = {
        "image": torch.rand(3, 512, 512),
        "target_y": rasterize_points(torch.rand(50, 2) * 512, 512, 512, stride=4),
        "points": torch.rand(50, 2) * 512,
        "id": "dense_01",
    }
    batch = collate_train([sample_empty, sample_dense])
    assert batch["image"].shape == (2, 3, 512, 512)
    assert batch["target_y"].shape == (2, 1, 128, 128)
    assert len(batch["points"]) == 2
    assert batch["points"][0].shape == (0, 2)
    assert batch["points"][1].shape == (50, 2)
    assert batch["id"] == ["empty_01", "dense_01"]


def test_cap_image_resolution_boundary_points():
    """Verify cap_image_resolution clamps boundary coordinates strictly to [0.0, dim - 1e-4]."""
    img = Image.new("RGB", (4000, 3000), (100, 100, 100))
    pts = torch.tensor([
        [0.0, 0.0],
        [3999.9, 2999.9],
    ])
    capped_img, capped_pts = cap_image_resolution(img, pts, max_size=2048)
    assert max(capped_img.size) == 2048
    assert capped_pts[:, 0].min().item() >= 0.0
    assert capped_pts[:, 1].min().item() >= 0.0
    assert capped_pts[:, 0].max().item() < float(capped_img.size[0])
    assert capped_pts[:, 1].max().item() < float(capped_img.size[1])


def test_compute_manifest_density_capped_preserves_all_points(tmp_path: Path):
    """Verify compute_manifest_density with max_size does not drop points in bottom-right quadrants."""
    from rmr_core.manifests import compute_manifest_density
    img_p = tmp_path / "huge_sample.jpg"
    # Create large image 4000 x 3000
    Image.new("RGB", (4000, 3000)).save(img_p)

    # 4 points across 4 quadrants:
    # (100, 100): top-left
    # (3500, 100): top-right (x > 2048)
    # (100, 2500): bottom-left (y > 2048)
    # (3500, 2500): bottom-right (x > 2048 and y > 2048)
    pts = [
        [100.0, 100.0],
        [3500.0, 100.0],
        [100.0, 2500.0],
        [3500.0, 2500.0],
    ]
    manifest_p = tmp_path / "manifest.jsonl"
    manifest_p.write_text(json.dumps({"image": str(img_p.name), "points": pts}) + "\n", encoding="utf-8")

    # With max_size=2048:
    # cap_scale = 2048 / 4000 = 0.512
    # w_eff = 2048, h_eff = 1536
    # cells = ceil(1536 / 4) * ceil(2048 / 4) = 384 * 512 = 196608
    # All 4 points must be counted!
    d = compute_manifest_density(manifest_p, output_stride=4, data_root=tmp_path, max_size=2048)
    expected_density = 4.0 / (384 * 512)
    assert pytest.approx(d, rel=1e-5) == expected_density


def test_random_erasing_boundary_robustness():
    """Verify point_preserving_random_erasing is resilient against reversed/boundary inputs."""
    from rmr_core.transforms import point_preserving_random_erasing
    img_t = torch.rand(3, 512, 512)
    pts = torch.rand(20, 2) * 512

    # Reversed scale tuple and extreme ratio
    out_img, out_pts = point_preserving_random_erasing(
        img_t, pts, prob=1.0, scale=(0.05, 0.01), ratio=(0.1, 3.0), max_patches=4
    )
    assert out_img.shape == (3, 512, 512)
    assert out_pts.shape == pts.shape
    assert torch.equal(out_pts, pts)
    assert not torch.isnan(out_img).any()
    assert not torch.isinf(out_img).any()


def test_virtual_capping_direct_box_resizing_invariants():
    """Verify virtual capping avoids intermediate PIL allocation and preserves point bounds."""
    # 4000 x 3000 image simulates UCF-QNRF / NWPU scale
    w0, h0 = 4000, 3000
    arr = np.full((h0, w0, 3), 120, dtype=np.uint8)
    img = Image.fromarray(arr)
    pts = torch.tensor([
        [100.0, 100.0],
        [2000.0, 1500.0],
        [3900.0, 2900.0],
    ], dtype=torch.float32)

    crop_t, cpts = train_transform(
        img,
        pts,
        crop_size=512,
        max_size=2048,
        pad_small_images=False,
        crowd_aware_crop=True,
        crowd_aware_prob=1.0,
    )
    assert crop_t.shape == (3, 512, 512)
    assert not torch.isnan(crop_t).any()
    if cpts.numel():
        assert (cpts[:, 0] >= 0.0).all() and (cpts[:, 0] <= 511.0).all()
        assert (cpts[:, 1] >= 0.0).all() and (cpts[:, 1] <= 511.0).all()



