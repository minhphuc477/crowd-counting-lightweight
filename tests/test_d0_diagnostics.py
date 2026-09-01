"""Unit tests for D0 Diagnostic Suite (D-R, D-K, D-L, D-M)."""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from hpc.diagnostics.effective_rank import (
    compute_spectral_rank_metrics,
    evaluate_effective_rank_single_image,
)
from hpc.diagnostics.gradient_allocation import evaluate_gradient_allocation_single_batch
from hpc.diagnostics.phase_shift import (
    crop_valid_center,
    evaluate_phase_shift_single_image,
    inverse_align_feature,
)
from hpc.diagnostics.separability import (
    evaluate_separability_single_image,
    sample_feature_at_image_coord,
)
from hpc.losses.ntpc import NTPCConfig, NTPCLoss
from hpc.models.hpc_lite import HPCLite


def test_phase_shift_roll_and_center_recovery():
    x = torch.zeros(1, 1, 64, 64)
    x[..., 31, 31] = 1.0

    # Shift using torch.roll
    shifted = torch.roll(x, shifts=(-1, 2), dims=(-2, -1))
    aligned = inverse_align_feature(shifted, dx_img=2, dy_img=-1, stride=1.0, device=torch.device("cpu"))
    central = crop_valid_center(aligned, margin_px=16, stride=1)
    central_x = crop_valid_center(x, margin_px=16, stride=1)
    assert torch.allclose(central, central_x, atol=1e-5)


def test_inter_person_dissimilarity_decreases_when_features_merge():
    f1 = torch.tensor([[1.0, 0.0]])
    f2_distinct = torch.tensor([[0.0, 1.0]])
    f2_merged = torch.tensor([[1.0, 0.0]])

    distinct = 1.0 - F.cosine_similarity(f1, f2_distinct, dim=-1)
    merged = 1.0 - F.cosine_similarity(f1, f2_merged, dim=-1)
    assert distinct.item() > merged.item()


def test_spectral_rank_metrics_sample_matched():
    # Rank-1 matrix with variance (outer product u @ v)
    u = torch.randn(256, 1)
    v = torch.randn(1, 32)
    rank1 = u @ v
    metrics_rank1 = compute_spectral_rank_metrics(rank1)
    assert metrics_rank1["normalized_participation_ratio"] <= 0.10
    assert metrics_rank1["top1_energy_ratio"] >= 0.95
    
    # Random orthogonal matrix of 32 channels
    q, _ = torch.linalg.qr(torch.randn(256, 32))
    metrics_q = compute_spectral_rank_metrics(q)
    assert metrics_q["normalized_participation_ratio"] >= 0.85


def test_spectral_rank_scale_invariant():
    x = torch.randn(128, 32)
    a = compute_spectral_rank_metrics(x)
    b = compute_spectral_rank_metrics(x * 1e-5)
    assert np.isclose(
        a["normalized_participation_ratio"],
        b["normalized_participation_ratio"],
        rtol=1e-4,
        atol=1e-6,
    )


def test_gradient_allocation_preserves_frozen_model_state():
    model = HPCLite(
        backbone_name="mobilenetv4_conv_small_050",
        pretrained=False,
        neck_width=16,
        feature_reductions=(4, 8, 16),
    )
    model.eval()
    
    # Snapshot buffer states (BatchNorm running mean/var if any)
    orig_buffers = {k: v.clone() for k, v in model.named_buffers()}
    
    criterion = NTPCLoss(NTPCConfig(mode="r2_flat_dm", root_loss="nb"))
    img = torch.randn(1, 3, 64, 64)
    targets = {4: torch.zeros(1, 16, 16), 16: torch.zeros(1, 4, 4), "N": torch.zeros(1)}
    
    res = evaluate_gradient_allocation_single_batch(model, criterion, img, targets)
    assert "C16_fg_energy_fraction" in res
    assert "C16_gradient_enrichment" in res
    
    # Check that model remains in eval mode and buffers are unchanged
    assert not model.training
    for k, v in model.named_buffers():
        assert torch.equal(v, orig_buffers[k])


def test_sample_feature_at_image_coord_impulse():
    # Feature map of shape (1, 1, 16, 16) representing stride 4 on 64x64 image
    feat = torch.zeros(1, 1, 16, 16)
    # Feature cell (4, 4) center in image space is at: (4 + 0.5)*4 - 0.5 = 17.5
    feat[0, 0, 4, 4] = 10.0
    
    # Query at exact image coordinate of feature cell (4, 4) center: (17.5, 17.5)
    query_exact = torch.tensor([[17.5, 17.5]], dtype=torch.float32)
    val_exact = sample_feature_at_image_coord(feat, query_exact, reduction=4)
    assert np.isclose(val_exact.item(), 10.0, atol=1e-2)
    
    # Query far away at (40.0, 40.0)
    query_far = torch.tensor([[40.0, 40.0]], dtype=torch.float32)
    val_far = sample_feature_at_image_coord(feat, query_far, reduction=4)
    assert np.isclose(val_far.item(), 0.0, atol=1e-3)


def test_feature_sampling_odd_image_extent():
    reduction = 16
    # Equivalent to a feature map from an odd-sized input:
    # image ~ 317 x 411 -> feature ~ 20 x 26.
    feat = torch.zeros(1, 1, 20, 26)
    fy = 7
    fx = 10
    feat[0, 0, fy, fx] = 9.0

    # Aligned image-space center represented by that feature location
    x = (fx + 0.5) * reduction - 0.5
    y = (fy + 0.5) * reduction - 0.5
    xy = torch.tensor([[x, y]], dtype=torch.float32)
    value = sample_feature_at_image_coord(feat, xy, reduction=reduction)
    assert torch.allclose(value, torch.tensor([[9.0]]), atol=1e-5)


def test_dk_kdtree_matches_dense_knn_small_case():
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, size=(25, 2)).astype(np.float32)

    # Brute-force pairwise distance matrix
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    brute_nn_idx = np.argmin(dists, axis=-1)
    brute_nn_dist = np.min(dists, axis=-1)

    # cKDTree query
    tree = cKDTree(pts)
    nn_dist, nn_idx = tree.query(pts, k=2)

    np.testing.assert_array_equal(nn_idx[:, 1], brute_nn_idx)
    np.testing.assert_allclose(nn_dist[:, 1], brute_nn_dist, rtol=1e-5)


def test_factorial_configs_are_exact_2x2():
    import yaml

    cfgs = {}
    for letter, fname in [
        ("A", "configs/factorial_a_crop256_c16.yaml"),
        ("B", "configs/factorial_b_crop256_c32.yaml"),
        ("C", "configs/factorial_c_crop448_c16.yaml"),
        ("D", "configs/factorial_d_crop448_c32.yaml"),
    ]:
        with open(fname, "r", encoding="utf-8") as f:
            cfgs[letter] = yaml.safe_load(f)

    # All must use mode: r2_flat_dm with kappa_flat16: 20
    for letter, cfg in cfgs.items():
        assert cfg["loss"]["mode"] == "r2_flat_dm"
        assert cfg["loss"]["kappa_flat16"] == 20.0
        assert cfg["optimizer"]["lr"] == 1e-4
        assert cfg["optimizer"]["grad_clip"] == 500.0

    # Check 2x2 factor isolation
    assert cfgs["A"]["dataset"]["crop_size"] == 256
    assert cfgs["A"]["model"]["features"] == ["C4", "C8", "C16"]

    assert cfgs["B"]["dataset"]["crop_size"] == 256
    assert cfgs["B"]["model"]["features"] == ["C4", "C8", "C16", "C32"]

    assert cfgs["C"]["dataset"]["crop_size"] == 448
    assert cfgs["C"]["model"]["features"] == ["C4", "C8", "C16"]

    assert cfgs["D"]["dataset"]["crop_size"] == 448
    assert cfgs["D"]["model"]["features"] == ["C4", "C8", "C16", "C32"]


def test_dm_diagnostic_uses_unpadded_natural_crop():
    from tools.run_d0_diagnostics import make_natural_dm_crop

    img = torch.randn(1, 3, 300, 400)
    pts = np.array([[10.0, 20.0], [200.0, 150.0], [390.0, 290.0]], dtype=np.float32)

    crop_sample = make_natural_dm_crop(img, pts, max_crop=256)
    assert crop_sample is not None
    crop_img, crop_pts = crop_sample

    # Dimensions must be divisible by 64 and <= max_crop
    _, _, ch, cw = crop_img.shape
    assert ch % 64 == 0
    assert cw % 64 == 0
    assert ch <= 256
    assert cw <= 256

    # All cropped points must lie within the crop bounds
    if len(crop_pts) > 0:
        assert np.all(crop_pts[:, 0] >= -0.5)
        assert np.all(crop_pts[:, 0] <= cw - 0.5)
        assert np.all(crop_pts[:, 1] >= -0.5)
        assert np.all(crop_pts[:, 1] <= ch - 0.5)


def test_tool_imports():
    import tools.architecture_table
    import tools.eval_localization
    import tools.run_all_ablations
    import tools.run_d0_diagnostics
    import tools.run_factorial_abcd
    import tools.summary_runs


def test_python_sources_compile():
    import compileall
    for target in ["hpc", "tools", "legacy", "tests", "train_ntpc.py"]:
        if target.endswith(".py"):
            assert compileall.compile_file(target, quiet=1, force=True)
        else:
            assert compileall.compile_dir(target, quiet=1, force=True)



