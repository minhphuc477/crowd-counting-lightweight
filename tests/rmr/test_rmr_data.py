import torch

from rmr_count.data import rasterize_points


def test_rasterize_points_conserves_count():
    pts = torch.tensor([[0.0, 0.0], [3.6, 4.0], [7.4, 7.4], [15.0, 15.0]])
    y = rasterize_points(pts, 16, 16, stride=4)
    assert y.sum().item() == 4


def test_oob_points_are_ignored_not_clipped():
    pts = torch.tensor([[-1.0, 2.0], [2.0, 2.0], [20.0, 3.0]])
    y = rasterize_points(pts, 16, 16, stride=4)
    assert y.sum().item() == 1


def test_rasterize_points_conserves_edge_points():
    # Boundary conservation: point at x=15.9, y=15.9 in 16x16 image with stride 4.
    # floor((15.9+0.5)/4) = 4 >= gw=4.
    # The point is valid (inside image) and must land in the last cell (j=3, i=3),
    # not be silently dropped.
    pts = torch.tensor([[15.9, 15.9], [0.0, 0.0], [15.1, 0.5]])
    y = rasterize_points(pts, 16, 16, stride=4)
    assert y.sum().item() == 3
    assert y[0, 3, 3].item() == 1
    assert y[0, 0, 0].item() == 1
    assert y[0, 0, 3].item() == 1


def test_dataset_ram_caching_train_and_eval():
    from rmr_core.data import CrowdManifestDataset
    # Test Train Caching via compressed raw bytes store
    train_ds = CrowdManifestDataset("data/sha_a_train_all.jsonl", train=True, cache_images=True)
    assert len(train_ds._raw_bytes_cache) == 0
    s0 = train_ds[0]
    assert len(train_ds._raw_bytes_cache) == 1
    assert 0 in train_ds._raw_bytes_cache
    s0_again = train_ds[0]
    assert len(train_ds._raw_bytes_cache) == 1

    # Test Eval Caching via compressed raw bytes store
    eval_ds = CrowdManifestDataset("data/sha_a_test.jsonl", train=False, cache_images=True)
    assert len(eval_ds._raw_bytes_cache) == 0
    e0 = eval_ds[0]
    assert len(eval_ds._raw_bytes_cache) == 1
    e0_again = eval_ds[0]
    assert torch.equal(e0["image"], e0_again["image"])
    assert torch.equal(e0["target_y"], e0_again["target_y"])

    # Verify returned points clone protects cached points tensor
    e0_again["points"][0, 0] = 9999.0
    e0_third = eval_ds[0]
    assert e0_third["points"][0, 0].item() != 9999.0

    # Test Preload
    preload_ds = CrowdManifestDataset("data/sha_a_test.jsonl", train=False, cache_images=True, preload=True)
    assert len(preload_ds._raw_bytes_cache) == 182


