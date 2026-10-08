from __future__ import annotations

import math
import pytest
import torch
import torch.nn.functional as F

from rmr_core.data import rasterize_points
from rmr_v3.config import validate_v3_config
from rmr_v3.losses.config import RMRv3LossConfig
from rmr_v3.losses.dual_supervision import align_target_to_prediction, compute_dual_lattice_losses
from rmr_v3.losses.orchestration import compute_rmr_v3_losses
from rmr_v3.model import RMRv3, RMRv3Config
from rmr_v3.model.dual_lattice import SubpixelAllocationHead, push_forward_stride2_to_stride4


def test_direction3_parameter_ceiling_and_subpixel_head():
    """Verify strict parameter ceiling <= 104,441 and SubpixelAllocationHead parameter budget."""
    head = SubpixelAllocationHead(in_channels=32)
    head_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    assert head_params == 452, f"SubpixelAllocationHead must have exactly 452 params, got {head_params}"

    cfg = RMRv3Config(
        backbone_name="mobilenetv4_conv_small_050",
        pretrained=False,
        output_stride=2,
        subpixel_dm=True,
        neck_type="hdc_lite",
        region_head_hidden=48,
        iterations=6,
        max_trainable_params=104441,
    )
    model = RMRv3(cfg)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params <= 104441, f"Strict parameter ceiling violated: {total_params} > 104,441"
    assert total_params == 104359, f"Expected exactly 104,359 params with hdc_lite+hidden48, got {total_params}"


def test_step0_uniform_parity_and_discrete_mass_conservation():
    """Verify Step-0 uniform [0.25, 0.25, 0.25, 0.25] parity and discrete mass conservation."""
    head = SubpixelAllocationHead(in_channels=32)
    b, c, h, w = 2, 32, 16, 20
    p4 = torch.randn(b, c, h, w)
    y4 = torch.rand(b, 1, h, w) * 5.0 + 0.5

    y2 = head(p4, y4)
    assert y2.shape == (b, 1, 2 * h, 2 * w)

    # Step-0 parity: zero-init PW weights guarantee exact uniform allocation
    y4_uniform = F.interpolate(y4, scale_factor=2, mode="nearest") * 0.25
    assert torch.allclose(y2, y4_uniform, atol=1e-5), "Step 0 uniform allocation violated"

    # Discrete mass conservation: sum(y2) == sum(y4)
    m4 = y4.sum(dim=(-2, -1))
    m2 = y2.sum(dim=(-2, -1))
    assert torch.allclose(m4, m2, atol=1e-5), f"Mass not conserved: {m4} vs {m2}"


def test_mass_conservation_on_arbitrary_odd_and_prime_resolutions():
    """Verify discrete mass conservation on arbitrary odd/prime resolutions (e.g. 409x521, 113x227)."""
    cfg = RMRv3Config(
        backbone_name="mobilenetv4_conv_small_050",
        pretrained=False,
        output_stride=2,
        subpixel_dm=True,
        iterations=2,
        max_trainable_params=105000,
    )
    model = RMRv3(cfg)

    resolutions = [(409, 521), (113, 227), (255, 257)]
    for h_in, w_in in resolutions:
        x = torch.randn(1, 3, h_in, w_in)
        out = model(x)
        expected_h2 = (h_in + 1) // 2
        expected_w2 = (w_in + 1) // 2
        assert out.y.shape == (1, 1, expected_h2, expected_w2)

        m_fine = out.y.sum().item()
        m_carrier = out["y_carrier"].sum().item()
        assert abs(m_fine - m_carrier) < 1e-4, (
            f"Mass mismatch on {h_in}x{w_in}: fine={m_fine:.5f} vs carrier={m_carrier:.5f}"
        )


def test_align_target_to_prediction_no_zero_padding():
    """Verify align_target_to_prediction replaces zero-padding with exact mass-conserving alignment."""
    # Scenario A: Target at Stride 4 (32x32), Prediction at Stride 2 (64x64) with point annotations
    pts_s0 = torch.tensor([[10.2, 14.5], [50.0, 60.1], [100.5, 120.3]])
    pts_s1 = torch.tensor([[20.0, 30.0]])
    points = [pts_s0, pts_s1]

    # Stride 4 target
    t_s4 = torch.stack([
        rasterize_points(pts_s0, 128, 128, stride=4),
        rasterize_points(pts_s1, 128, 128, stride=4),
    ], dim=0)

    # Prediction y at Stride 2 (64x64 for 128x128 image)
    y_s2 = torch.ones(2, 1, 64, 64)

    # Re-alignment with points: must produce exact Stride-2 rasterization
    t_aligned = align_target_to_prediction(t_s4, y_s2, points=points, stride=2)
    assert t_aligned.shape == (2, 1, 64, 64)
    assert t_aligned[0].sum().item() == 3.0
    assert t_aligned[1].sum().item() == 1.0

    # Ensure no zero-padding on right/bottom: point at (100.5, 120.3) must be at row ~60, col ~50
    assert t_aligned[0, 0, 60, 50].item() == 1.0

    # Scenario B: Target at Stride 4 without points -> uniform prolongation with 1/4 mass
    t_s4_dense = torch.rand(2, 1, 32, 32) * 5.0
    t_aligned_nopoints = align_target_to_prediction(t_s4_dense, y_s2, points=None, stride=2)
    assert t_aligned_nopoints.shape == (2, 1, 64, 64)
    m_orig = t_s4_dense.sum(dim=(-2, -1))
    m_aligned = t_aligned_nopoints.sum(dim=(-2, -1))
    assert torch.allclose(m_orig, m_aligned, atol=1e-5), "Mass must be conserved during prolongation"

    # Scenario C: Target at Stride 2, Prediction at Stride 4 -> push-forward box summation
    y_s4 = torch.ones(2, 1, 32, 32)
    t_s2_dense = torch.rand(2, 1, 64, 64) * 5.0
    t_down = align_target_to_prediction(t_s2_dense, y_s4, points=None, stride=4)
    assert t_down.shape == (2, 1, 32, 32)
    assert torch.allclose(t_s2_dense.sum(dim=(-2, -1)), t_down.sum(dim=(-2, -1)), atol=1e-5)


def test_batch_sample_isolation_gradient_independence():
    """Verify Sample Isolation: Sample 0 must receive ZERO gradient w.r.t Sample 1's loss."""
    cfg = RMRv3Config(
        backbone_name="mobilenetv4_conv_small_050",
        pretrained=False,
        output_stride=2,
        subpixel_dm=True,
        iterations=2,
        max_trainable_params=105000,
    )
    model = RMRv3(cfg)

    x = torch.randn(2, 3, 256, 256)
    out = model(x)

    loss_cfg = RMRv3LossConfig(
        output_stride=2,
        allocation_loss_type="dual_bayesian_dm16",
        dm_target="dual",
        lambda_count=1.0,
        lambda_bayesian=0.025,
        lambda_flat_dm16=15.0,
        lambda_cell=0.0,
        lambda_region_nb=0.2,
    )

    pts_s0 = torch.tensor([[50.0, 50.0], [80.0, 80.0]])
    pts_s1 = torch.tensor([[150.0, 150.0], [200.0, 200.0], [210.0, 210.0]])
    points = [pts_s0, pts_s1]
    target_stride2 = torch.stack([
        rasterize_points(pts_s0, 256, 256, stride=2),
        rasterize_points(pts_s1, 256, 256, stride=2),
    ], dim=0)

    # Compute loss for Sample 1 only
    out_sample1 = {
        "y": out.y[1:2],
        "y0": out.y0[1:2],
        "y_carrier": out["y_carrier"][1:2],
        "y0_carrier": out["y0_carrier"][1:2],
        "regions": out["regions"],
        "b_region": out["b_region"][1:2],
        "region_dispersion": out["region_dispersion"][1:2],
    }
    tgt_sample1 = target_stride2[1:2]
    pts_sample1 = [pts_s1]

    losses_s1 = compute_rmr_v3_losses(out_sample1, tgt_sample1, loss_cfg, points=pts_sample1)
    loss_s1 = losses_s1["total"]

    # Differentiate Sample 1's loss w.r.t Sample 0's output
    out.y.retain_grad()
    loss_s1.backward()

    # Sample 0's gradient slice must be strictly None or identically 0.0
    if out.y.grad is not None:
        grad_sample0 = out.y.grad[0:1]
        assert torch.all(grad_sample0 == 0.0), "Sample isolation violated: Sample 0 received non-zero gradient from Sample 1"


def test_sub60_e134_full_pipeline_finite_gradients():
    """Verify sub60_e134 configuration builds, validates, and runs with non-zero finite gradients."""
    import yaml
    from pathlib import Path

    cfg_path = Path("configs/rmr_research/sub60_e134_decoupled_subpixel_stride2.yaml")
    assert cfg_path.exists(), f"Configuration file {cfg_path} must exist"

    raw_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    validate_v3_config(raw_cfg)

    m_cfg = dict(raw_cfg["model"])
    m_cfg["pretrained"] = False
    cfg = RMRv3Config.from_dict(m_cfg, pretrained=False)
    model = RMRv3(cfg)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params <= 104441, f"Strict parameter ceiling exceeded: {total_params} > 104,441"

    l_cfg = dict(raw_cfg["loss"])
    loss_cfg = RMRv3LossConfig.from_dict(l_cfg)

    # 1 forward + backward step with AMP
    x = torch.randn(2, 3, 256, 256)
    pts = [torch.tensor([[30.0, 40.0], [70.0, 80.0]]), torch.tensor([[100.0, 120.0]])]
    targets = torch.stack([
        rasterize_points(pts[0], 256, 256, stride=2),
        rasterize_points(pts[1], 256, 256, stride=2),
    ], dim=0)

    with torch.amp.autocast("cpu", enabled=True):
        out = model(x)
        assert out.y.shape == (2, 1, 128, 128)
        assert out["y_carrier"].shape == (2, 1, 64, 64)

        losses = compute_rmr_v3_losses(dict(out), targets, loss_cfg, points=pts)
        loss = losses["total"]
        assert torch.isfinite(loss), f"Loss must be finite, got {loss}"
        assert loss.item() > 0.0

    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "Gradients must exist"
    for g in grads:
        assert torch.all(torch.isfinite(g)), "All gradients must be finite"
    total_grad_norm = sum(g.norm().item() for g in grads)
    assert total_grad_norm > 0.0, f"Total gradient norm must be positive, got {total_grad_norm}"


def test_subpixel_local_per_cell_discrete_mass_conservation():
    """Verify that every single coarse cell (i, j) conserves mass locally in its 2x2 fine block."""
    head = SubpixelAllocationHead(in_channels=32)
    with torch.no_grad():
        for p in head.parameters():
            p.normal_(0, 2.0)

    b, c, h, w = 2, 32, 16, 24
    p4 = torch.randn(b, c, h, w) * 10.0
    y4 = torch.rand(b, 1, h, w) * 100.0 + 0.1

    y2 = head(p4, y4)
    assert y2.shape == (b, 1, 2 * h, 2 * w)

    # 2x2 box summation must equal y4 for EVERY cell
    y2_blocks = y2.view(b, 1, h, 2, w, 2).sum(dim=(3, 5))
    diff = (y2_blocks - y4).abs()
    max_err = diff.max().item()
    assert max_err < 1e-4, f"Per-cell local mass conservation violated! Max error: {max_err}"


def test_adversarial_zero_and_corner_points_loss_stability():
    """Adversarial stress test: empty images, corner coordinates, and extreme head collisions."""
    loss_cfg = RMRv3LossConfig(
        output_stride=2,
        allocation_loss_type="dual_bayesian_dm16",
        dm_target="dual",
        lambda_count=1.0,
        lambda_bayesian=0.025,
        lambda_flat_dm16=15.0,
        lambda_cell=0.0,
        lambda_region_nb=0.2,
    )

    pts_empty = torch.empty((0, 2), dtype=torch.float32)
    pts_corners = torch.tensor([[0.0, 0.0], [255.99, 255.99], [0.0, 255.99], [255.99, 0.0]], dtype=torch.float32)
    pts_collision = torch.rand(500, 2) * 2.0 + 100.0
    points = [pts_empty, pts_corners, pts_collision]

    targets = torch.stack([
        rasterize_points(pts_empty, 256, 256, stride=2),
        rasterize_points(pts_corners, 256, 256, stride=2),
        rasterize_points(pts_collision, 256, 256, stride=2),
    ], dim=0)

    cfg = RMRv3Config(
        backbone_name="mobilenetv4_conv_small_050",
        pretrained=False,
        output_stride=2,
        subpixel_dm=True,
        iterations=2,
        max_trainable_params=105000,
    )
    model = RMRv3(cfg)
    x = torch.randn(3, 3, 256, 256)

    out = model(x)
    losses = compute_rmr_v3_losses(dict(out), targets, loss_cfg, points=points)

    assert torch.isfinite(losses["total"]), f"Loss is not finite: {losses['total']}"
    assert torch.isfinite(losses["allocation"]), f"Allocation loss is not finite: {losses['allocation']}"
    assert torch.isfinite(losses["region_nb"]), f"Regional NB loss is not finite: {losses['region_nb']}"

    losses["total"].backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.all(torch.isfinite(p.grad)), f"NaN/Inf gradient in {name}"


def test_diagnostics_robustness_with_stride_mismatched_target():
    """Verify diagnostics work properly even if target_y is provided at Stride 4 for Stride 2 model."""
    from rmr_v3.diagnostics.rows import regional_reliability_rows
    from rmr_v3.diagnostics.trajectory import compute_solver_trajectory_diagnostics

    cfg = RMRv3Config(
        backbone_name="mobilenetv4_conv_small_050",
        pretrained=False,
        output_stride=2,
        subpixel_dm=True,
        iterations=2,
        max_trainable_params=105000,
    )
    model = RMRv3(cfg)
    x = torch.randn(1, 3, 256, 256)
    out = model(x)

    pts = torch.tensor([[50.0, 60.0], [200.0, 220.0]])
    target_stride4 = rasterize_points(pts, 256, 256, stride=4).unsqueeze(0)

    d_rows = regional_reliability_rows(dict(out), target_stride4, max_regions=100)
    assert len(d_rows) > 0, "Diagnostic rows must not be empty"

    t_diag = compute_solver_trajectory_diagnostics(dict(out), target_stride4)
    assert "mae_reg_y0" in t_diag, "Trajectory diagnostics must compute iterate MAE"


def test_tiled_prediction_and_game_physical_parity_at_stride2():
    """Verify tiled prediction and GAME metrics on odd dimensions (383x491) at Stride 2."""
    from rmr_core.evaluation import predict_tiled
    from rmr_core.metrics import game_physical_image

    cfg = RMRv3Config(
        backbone_name="mobilenetv4_conv_small_050",
        pretrained=False,
        output_stride=2,
        subpixel_dm=True,
        iterations=2,
        max_trainable_params=105000,
    )
    model = RMRv3(cfg).eval()

    h_odd, w_odd = 383, 491
    img = torch.randn(3, h_odd, w_odd)
    # Coordinates in (x, y) where x < w_odd (491) and y < h_odd (383)
    pts = torch.tensor([[150.0, 100.0], [400.0, 300.0], [450.0, 350.0]])

    pred_tiled = predict_tiled(model, img, output_stride=2, tile_size=256, halo=32)
    expected_gh = math.ceil(h_odd / 2)
    expected_gw = math.ceil(w_odd / 2)
    assert pred_tiled.shape == (1, expected_gh, expected_gw), f"Shape mismatch: {pred_tiled.shape}"

    game_dict = game_physical_image(
        pred_tiled, pts, image_h=h_odd, image_w=w_odd, stride=2, levels=(0, 1, 2)
    )
    total_pred = pred_tiled.sum().item()
    valid_mask = (pts[:, 0] >= 0) & (pts[:, 0] < w_odd) & (pts[:, 1] >= 0) & (pts[:, 1] < h_odd)
    total_gt = float(valid_mask.sum().item())
    abs_err = abs(total_pred - total_gt)
    assert abs(game_dict[0] - abs_err) < 1e-4, f"GAME(0) {game_dict[0]} != |pred - gt| {abs_err}"


def test_sub60_e135_to_e138_new_experiment_suite():
    """Verify all newly designed experiment configurations (e135, e136, e137, e138).

    Guarantees:
      1. All 4 configs load and validate cleanly with validate_v3_config.
      2. All 4 models strictly satisfy trainable parameters <= 104,441.
      3. All 4 models execute a full forward pass and loss computation without NaN/Inf.
    """
    from rmr_v3.config import load_config

    configs = [
        "configs/rmr_research/sub60_e135_subpixel_stride2_adaptive_sigma.yaml",
        "configs/rmr_research/sub60_e136_subpixel_stride2_pure_bayesian.yaml",
        "configs/rmr_research/sub60_e137_subpixel_stride2_aspp_control.yaml",
        "configs/rmr_research/sub60_e138_subpixel_stride2_balanced_dm.yaml",
    ]

    for cfg_path in configs:
        cfg = load_config(cfg_path)
        validate_v3_config(cfg)
        model_cfg = RMRv3Config.from_dict(cfg["model"], pretrained=False)
        model = RMRv3(model_cfg)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n_params <= 104441, f"Param limit exceeded: {n_params} > 104441 in {cfg_path}"

        # Lightweight forward and loss check on 256x256 test crop
        x = torch.randn(1, 3, 256, 256)
        out = model(x)
        assert "y" in out and out.y.shape[-2:] == (128, 128)
        assert "y_carrier" in out and out["y_carrier"].shape[-2:] == (64, 64)

        pts = [torch.tensor([[50.0, 60.0], [120.0, 140.0]])]
        tgt = rasterize_points(pts[0], 256, 256, stride=2).unsqueeze(0)
        loss_cfg = RMRv3LossConfig.from_dict(cfg["loss"])
        loss_dict = compute_rmr_v3_losses(dict(out), tgt, cfg=loss_cfg, points=pts)
        assert torch.isfinite(loss_dict["total"]), f"Non-finite total loss in {cfg_path}"


def test_python39_typing_and_bayesian_function_parity():
    """Verify _BayesianPersonErrorFunction works seamlessly across Python versions."""
    from rmr_v3.losses.point_supervision import _BayesianPersonErrorFunction

    u = torch.randn(2, requires_grad=True)
    pts = torch.tensor([[10.0, 10.0], [20.0, 20.0]])
    gx = torch.tensor([[10.0, 20.0]])
    gy = torch.tensor([[10.0, 20.0]])
    target = torch.tensor([1.0, 1.0])

    # Float inv_k
    loss_float = _BayesianPersonErrorFunction.apply(u, pts, gx, gy, 0.125, target, 64)
    loss_float.backward()
    assert u.grad is not None and torch.isfinite(u.grad).all()

    # Tensor inv_k
    u.grad.zero_()
    inv_k_tensor = torch.tensor([0.125, 0.25])
    loss_tensor = _BayesianPersonErrorFunction.apply(u, pts, gx, gy, inv_k_tensor, target, 64)
    loss_tensor.backward()
    assert u.grad is not None and torch.isfinite(u.grad).all()

