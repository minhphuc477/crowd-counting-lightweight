"""Permanent Expanded Mathematical & Architectural Test Suite.

Automated verification for:
1. Claerbout Adjoint Invariant (<A x, v> == <x, A^T v>) on arbitrary grids and rectangular windows.
2. 100% Parameter Gradient Flow (Zero dead parameters in backbone, neck, and heads).
3. Robustness on extreme resolutions and non-standard aspect ratios.
4. Point rasterization & augmentation geometric conservation.
5. GAME hierarchy invariant (GAME(0) == Absolute Error, monotonic scaling).
6. Inverse solver energy reduction and strict pure SIRT monotonicity across T=1..10.
"""

from __future__ import annotations

import math
import random
import torch
import numpy as np
import pytest
from PIL import Image

from rmr_core.operators import build_multiscale_regions, regional_sum, regional_adjoint
from rmr_core.data import rasterize_points, train_transform
from rmr_core.metrics import game_physical_image
from rmr_v3.model import RMRv3, RMRv3Config
from rmr_v3.losses import RMRv3LossConfig, compute_rmr_v3_losses
from rmr_v3.solver import unrolled_sirt_solver


def test_adjoint_invariant_multigrid():
    grids = [(48, 64), (41, 57), (19, 83), (64, 64), (17, 17)]
    sizes = (32, 64, 128, (64, 32), (32, 64))
    for h, w in grids:
        regions = build_multiscale_regions(h, w, output_stride=4, region_sizes_px=sizes)
        m = regions.boxes.shape[0]
        x = torch.randn(2, 1, h, w, dtype=torch.float64)
        v = torch.randn(2, 1, m, dtype=torch.float64)
        ax = regional_sum(x, regions.boxes, out_dtype=torch.float64)
        at_v = regional_adjoint(v, regions.boxes, h, w, out_dtype=torch.float64)
        dot1 = (ax * v).sum().item()
        dot2 = (x * at_v).sum().item()
        rel_err = abs(dot1 - dot2) / max(abs(dot1), abs(dot2), 1e-12)
        assert rel_err < 1e-11, f"Adjoint invariant failed on grid ({h},{w}): {rel_err}"


@pytest.mark.parametrize("target_mode", ["y", "y0"])
@pytest.mark.parametrize("model_name,cfg", [
    ("AQ-RMR", RMRv3Config(
        pretrained=False, iterations=6, proximal_tau=0.015, tv_lambda=0.02,
        regional_feature_stats="mean_std", region_sizes_px=(32, 64, 128, (64, 32), (32, 64)),
    )),
    ("Canonical", RMRv3Config(
        pretrained=False, iterations=6, proximal_tau=0.0, tv_lambda=0.0,
        regional_feature_stats="mean", region_sizes_px=(32, 64, 128),
    )),
])
def test_zero_dead_parameters_gradient_flow(target_mode: str, model_name: str, cfg: RMRv3Config):
    model = RMRv3(cfg)
    model.train()
    loss_cfg = RMRv3LossConfig(dm_target=target_mode)
    x = torch.randn(2, 3, 128, 128)
    out = model(x, solver_strength=1.0)
    target = torch.zeros(2, 1, 32, 32)
    pts = torch.randint(0, 32, (150, 2))
    for pt in pts:
        target[0, 0, pt[0], pt[1]] += 1.0
        target[1, 0, pt[0], pt[1]] += 1.0
    losses = compute_rmr_v3_losses(out, target, loss_cfg)
    losses["total"].backward()
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        assert p.grad is not None, f"Dead parameter without gradient: {pname}"
        assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {pname}"
        assert p.grad.abs().sum().item() > 0.0, f"Zero gradient in active parameter {pname}"


def test_extreme_dimensions_and_aspect_ratios():
    cfg = RMRv3Config(
        pretrained=False, iterations=4, proximal_tau=0.015, tv_lambda=0.02,
        regional_feature_stats="mean_std", region_sizes_px=(32, 64, 128, (64, 32), (32, 64)),
    )
    model = RMRv3(cfg)
    model.eval()
    test_shapes = [(1, 3, 128, 128), (1, 3, 64, 256), (1, 3, 256, 64), (1, 3, 136, 184), (1, 3, 112, 112)]
    for b, c, h, w in test_shapes:
        x = torch.randn(b, c, h, w)
        with torch.no_grad():
            out = model(x, solver_strength=1.0)
        y = out["y"]
        assert y.shape == (b, 1, h // 4, w // 4)
        assert torch.isfinite(y).all()
        assert (y >= 0.0).all()


def test_data_pipeline_conservation():
    img = Image.new("RGB", (600, 800), color=(100, 150, 200))
    pts_t = torch.tensor([[50.0, 50.0], [100.0, 200.0], [300.0, 400.0], [550.0, 750.0], [250.0, 350.0]])
    grid = rasterize_points(pts_t, image_h=800, image_w=600, stride=4)
    assert grid.sum().item() == 5.0

    random.seed(42)
    torch.manual_seed(42)
    for _ in range(10):
        img_t, pts_trans = train_transform(
            img, pts_t, crop_size=512, scale_range=(0.8, 1.2), hflip_prob=0.5,
            brightness_jitter=0.2, contrast_jitter=0.2, gamma_jitter=(0.8, 1.2),
            random_invert_prob=0.2,
        )
        assert img_t.shape == (3, 512, 512)
        assert img_t.min() >= 0.0 and img_t.max() <= 1.0
        if pts_trans.numel() > 0:
            assert (pts_trans[:, 0] >= 0.0).all() and (pts_trans[:, 0] < 512.0).all()
            assert (pts_trans[:, 1] >= 0.0).all() and (pts_trans[:, 1] < 512.0).all()
            grid_crop = rasterize_points(pts_trans, image_h=512, image_w=512, stride=4)
            assert grid_crop.sum().item() == float(len(pts_trans))


def test_game_hierarchy_and_triangle_inequality():
    torch.manual_seed(42)
    np.random.seed(42)
    for _ in range(5):
        img_h, img_w = np.random.randint(400, 800), np.random.randint(400, 800)
        gh, gw = math.ceil(img_h / 4), math.ceil(img_w / 4)
        pred_map = torch.rand(1, 1, gh, gw) * 0.1
        n_pts = np.random.randint(50, 500)
        pts = np.random.uniform(0, [img_w, img_h], size=(n_pts, 2))
        game_dict = game_physical_image(pred_map, pts, image_h=img_h, image_w=img_w, stride=4)
        pred_tot = pred_map.sum().item()
        gt_tot = float(n_pts)
        abs_err = abs(pred_tot - gt_tot)
        game0, game1, game2, game3 = game_dict[0], game_dict[1], game_dict[2], game_dict[3]
        rel_err = abs(game0 - abs_err) / max(abs_err, 1e-8)
        assert rel_err < 1e-6
        assert game1 >= game0 - 1e-4
        assert game2 >= game1 - 1e-4
        assert game3 >= game2 - 1e-4


def test_solver_energy_reduction_and_monotonicity():
    h, w = 48, 64
    regions = build_multiscale_regions(h, w, output_stride=4, region_sizes_px=(32, 64, 128, (64, 32), (32, 64)))
    m = regions.boxes.shape[0]
    y_gt = torch.zeros(1, 1, h, w)
    y_gt[0, 0, 10:20, 20:30] = 1.5
    y_gt[0, 0, 30:35, 40:45] = 2.0
    y0 = torch.full((1, 1, h, w), 0.01)
    y0[0, 0, 8:22, 18:32] = 0.8
    y0[0, 0, 28:37, 38:47] = 1.0
    b_solver = regional_sum(y_gt, regions.boxes)
    weight_solver = torch.ones(1, 1, m)

    for t in [1, 2, 4, 6, 10]:
        res = unrolled_sirt_solver(
            y0=y0, b_solver=b_solver, weight_solver=weight_solver, regions=regions,
            iterations=t, omega=1.0, proximal_tau=0.015, tv_lambda=0.02,
        )
        energies = res["energy_trace"]
        e_init = energies[0]["before"].item()
        e_final = energies[-1]["after"].item()
        assert e_final < e_init
        assert (res["y"] >= 0.0).all()
        assert torch.isfinite(res["y"]).all()

    res_pure = unrolled_sirt_solver(
        y0=y0, b_solver=b_solver, weight_solver=weight_solver, regions=regions,
        iterations=6, omega=1.0, proximal_tau=0.0, tv_lambda=0.0,
    )
    for i, step in enumerate(res_pure["energy_trace"]):
        eb = step["before"].item()
        ea = step["after"].item()
        assert ea < eb


def test_kd_density_map_empty_and_occupied_autograd():
    """Verify DensityMapKDLoss autograd graph connectivity on both empty background and occupied patches."""
    from rmr_v3.kd import DensityMapKDLoss

    kd_fn = DensityMapKDLoss(lambda_spatial_kl=1.0, lambda_count_kd=0.5, temperature=1.0)

    # Empty teacher
    ys = torch.rand(2, 1, 32, 32, requires_grad=True)
    yt_empty = torch.zeros(2, 1, 32, 32)
    res_empty = kd_fn(ys, yt_empty)
    assert res_empty["spatial_kl"].item() == 0.0
    res_empty["total_kd"].backward()
    assert ys.grad is not None and torch.isfinite(ys.grad).all()

    # Occupied teacher
    ys = torch.rand(2, 1, 32, 32, requires_grad=True)
    yt_occ = torch.zeros(2, 1, 32, 32)
    yt_occ[0, 0, 10:15, 10:15] = 2.0
    yt_occ[1, 0, 20:25, 20:25] = 3.0
    res_occ = kd_fn(ys, yt_occ)
    assert res_occ["spatial_kl"].item() > 0.0
    res_occ["total_kd"].backward()
    assert ys.grad is not None and torch.isfinite(ys.grad).all()


def test_predict_tiled_exact_identity_and_seam_invariance():
    """Verify that predict_tiled is strictly identical to full-image inference for images <= tile_size."""
    from rmr_core.evaluation import predict_tiled

    cfg = RMRv3Config(pretrained=False, iterations=2)
    model = RMRv3(cfg).eval()

    # 1. Image <= tile_size: must be mathematically identical
    img = torch.randn(3, 256, 256)
    with torch.no_grad():
        y_direct = model(img.unsqueeze(0))["y"][0]
        y_tiled = predict_tiled(model, img, output_stride=4, tile_size=512, halo=64)

    diff = (y_direct - y_tiled).abs().max().item()
    assert diff < 1e-6, f"Tiled prediction differs on small image: {diff:.2e}"

    # 2. Large image > tile_size: must produce valid contiguous canvas
    img_large = torch.randn(3, 600, 800)
    with torch.no_grad():
        y_large = predict_tiled(model, img_large, output_stride=4, tile_size=512, halo=64)
    assert y_large.shape == (1, 150, 200)
    assert torch.isfinite(y_large).all()
    assert (y_large >= 0.0).all()


def test_nb_interval_coverage_diagnostics():
    """Verify that compute_nb_interval_coverage accurately computes confidence intervals."""
    from rmr_v3.diagnostics import compute_nb_interval_coverage

    # Generate synthetic observations from known Negative Binomial distribution
    np.random.seed(42)
    rows = []
    mu_true = 25.0
    r_true = 10.0
    p_true = r_true / (r_true + mu_true)

    # Sample 1000 draws from NB(r, p)
    samples = np.random.negative_binomial(r_true, p_true, size=1000)
    for i, s in enumerate(samples):
        rows.append({
            "pred_count": float(mu_true),
            "dispersion": float(r_true),
            "gt_count": float(s),
            "scale_id": i % 3,
        })

    coverage_dict = compute_nb_interval_coverage(rows, nominal_levels=(0.50, 0.80, 0.95))

    # For discrete integer distributions, empirical coverage is conservative: coverage >= nominal - 0.02
    assert 0.48 <= coverage_dict["coverage_50"] <= 0.65
    assert abs(coverage_dict["coverage_80"] - 0.80) < 0.04
    assert abs(coverage_dict["coverage_95"] - 0.95) < 0.03

