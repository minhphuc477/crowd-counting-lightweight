from __future__ import annotations

import math
import pytest
import torch
import torch.nn.functional as F
import yaml

from rmr_core.heads import FineMeasureHead
from rmr_core.scale_routing import FactorizedRoutingHead
from rmr_v3.model import RMRv3, RMRv3Config
from rmr_v3.losses import RMRv3LossConfig, compute_rmr_v3_losses, physical_scale_alignment_loss


def test_rmr_v19_parameter_budget():
    """Verify RMR-v19 complies strictly with the <= 105,000 parameter budget."""
    with open("configs/rmr_v19/rmr_v19_factorized_aspect.yaml") as f:
        cfg = yaml.safe_load(f)

    model_cfg = RMRv3Config(**cfg["model"])
    model = RMRv3(model_cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert n_params == 104509, f"Expected 104509 params, got {n_params}"
    assert n_params <= 105000, f"Exceeded budget: {n_params} > 105000"
    assert 105000 - n_params == 491, f"Headroom should be exactly 491 params, got {105000 - n_params}"


def test_factorized_routing_parameters():
    """Verify exact parameter count of FactorizedRoutingHead (551 parameters)."""
    router = FactorizedRoutingHead(
        in_channels=32,
        num_scales=3,
        num_aspect_ratios=2,
        perspective_bias=True,
    )
    router_params = sum(p.numel() for p in router.parameters() if p.requires_grad)
    # DW: 32*9 + 32 = 320
    # GN: 32*2 = 64
    # PW scale: 32*3 + 3 = 99
    # PW aspect: 32*2 + 2 = 66
    # Persp bias: 2
    # Total = 320 + 64 + 99 + 66 + 2 = 551
    assert router_params == 551, f"Expected 551 parameters for FactorizedRoutingHead, got {router_params}"


def test_factorized_routing_forward_properties():
    """Verify partition of unity, marginal scale conservation, and output shapes."""
    router = FactorizedRoutingHead(
        in_channels=32,
        num_scales=3,
        num_aspect_ratios=2,
        perspective_bias=True,
    )
    x = torch.randn(2, 32, 64, 64)
    joint_pi, pi_scale, pi_aspect = router(x)

    assert joint_pi.shape == (2, 4, 64, 64), f"Unexpected joint_pi shape: {joint_pi.shape}"
    assert pi_scale.shape == (2, 3, 64, 64), f"Unexpected pi_scale shape: {pi_scale.shape}"
    assert pi_aspect.shape == (2, 2, 64, 64), f"Unexpected pi_aspect shape: {pi_aspect.shape}"

    # 1. Partition of unity for joint distribution
    sum_joint = joint_pi.sum(dim=1)
    assert torch.allclose(sum_joint, torch.ones_like(sum_joint), atol=1e-6), (
        "Partition of unity violated on joint distribution!"
    )

    # 2. Partition of unity for marginal scale and aspect distributions
    sum_scale = pi_scale.sum(dim=1)
    sum_aspect = pi_aspect.sum(dim=1)
    assert torch.allclose(sum_scale, torch.ones_like(sum_scale), atol=1e-6), (
        "Partition of unity violated on marginal scale!"
    )
    assert torch.allclose(sum_aspect, torch.ones_like(sum_aspect), atol=1e-6), (
        "Partition of unity violated on aspect ratio!"
    )

    # 3. Marginal scale conservation: pi_1 + pi_2 == pi_scale[:, 1]
    pi_scale_1_reconstructed = joint_pi[:, 1] + joint_pi[:, 2]
    assert torch.allclose(pi_scale_1_reconstructed, pi_scale[:, 1], atol=1e-6), (
        "Marginal scale conservation violated! Sum of 64sq and 64rect does not match marginal pi_scale_64."
    )

    # 4. Pure scale conservation: pi_0 == pi_scale[:, 0] and pi_3 == pi_scale[:, 2]
    assert torch.allclose(joint_pi[:, 0], pi_scale[:, 0], atol=1e-6), "Scale 0 (32px) mismatch!"
    assert torch.allclose(joint_pi[:, 3], pi_scale[:, 2], atol=1e-6), "Scale 2 (128px) mismatch!"


def test_factorized_routing_step0_identity():
    """At step 0 (persp_weight_aspect == 0), aspect logits are unperturbed by perspective."""
    router = FactorizedRoutingHead(
        in_channels=32,
        num_scales=3,
        num_aspect_ratios=2,
        perspective_bias=True,
    )
    # With zero weights initialized in pw_scale and pw_aspect
    x = torch.zeros(1, 32, 32, 32)
    _, pi_scale, pi_aspect = router(x)

    # Output probabilities should be uniform at init
    expected_scale = torch.full_like(pi_scale, 1.0 / 3.0)
    expected_aspect = torch.full_like(pi_aspect, 0.5)
    assert torch.allclose(pi_scale, expected_scale, atol=1e-5), "Step 0 uniform scale prior violated"
    assert torch.allclose(pi_aspect, expected_aspect, atol=1e-5), "Step 0 uniform aspect prior violated"


def test_factorized_routing_perspective_modulation():
    """Verify that vertical perspective bias shifts aspect ratio probability smoothly from horizon to foreground."""
    router = FactorizedRoutingHead(
        in_channels=32,
        num_scales=3,
        num_aspect_ratios=2,
        perspective_bias=True,
    )
    # Set positive perspective bias on aspect 1 (vertical rectangle: 2:1)
    # and negative on aspect 0 (square: 1:1)
    with torch.no_grad():
        router.persp_weight_aspect.copy_(torch.tensor([-2.0, 2.0]))

    x = torch.zeros(1, 32, 64, 64)
    joint_pi, pi_scale, pi_aspect = router(x)

    # Top rows (v ≈ -0.5, horizon): square (aspect 0) should dominate
    top_aspect_0 = pi_aspect[0, 0, 0, :].mean().item()
    top_aspect_1 = pi_aspect[0, 1, 0, :].mean().item()
    assert top_aspect_0 > top_aspect_1, f"At horizon, expected square > vertical, got {top_aspect_0} vs {top_aspect_1}"

    # Bottom rows (v ≈ +0.5, foreground): vertical rectangle (aspect 1) should dominate
    bottom_aspect_0 = pi_aspect[0, 0, -1, :].mean().item()
    bottom_aspect_1 = pi_aspect[0, 1, -1, :].mean().item()
    assert bottom_aspect_1 > bottom_aspect_0, f"At foreground, expected vertical > square, got {bottom_aspect_1} vs {bottom_aspect_0}"


def test_factorized_routing_gradient_flow():
    """Verify gradients propagate back into both scale and aspect branches and perspective bias."""
    router = FactorizedRoutingHead(
        in_channels=32,
        num_scales=3,
        num_aspect_ratios=2,
        perspective_bias=True,
    )
    x = torch.randn(2, 32, 32, 32, requires_grad=True)
    joint_pi, pi_scale, pi_aspect = router(x)

    loss = joint_pi[:, 1].sum() + pi_aspect[:, 1].sum()
    loss.backward()

    assert router.pw_scale.weight.grad is not None
    assert router.pw_aspect.weight.grad is not None
    assert router.persp_weight_aspect.grad is not None
    assert router.dw.weight.grad is not None
    assert x.grad is not None

    assert torch.isfinite(router.pw_scale.weight.grad).all()
    assert torch.isfinite(router.pw_aspect.weight.grad).all()
    assert torch.isfinite(router.persp_weight_aspect.grad).all()
    assert (router.persp_weight_aspect.grad != 0).any(), "Gradient to persp_weight_aspect is identically zero"


def test_physical_scale_alignment_loss_monotonicity():
    """Verify that scale alignment loss operates monotonically on 3-scale marginal pi_scale."""
    # Scale 0: 32px (dense)
    # Scale 1: 64px (moderate)
    # Scale 2: 128px (sparse)
    b, h, w = 1, 32, 32

    # High density map (all heads clustered)
    target_dense = torch.full((b, 1, h, w), 0.25)
    # Sparse density map
    target_sparse = torch.full((b, 1, h, w), 0.04)

    # Case 1: Router predicts dense scale (scale 0)
    pred_dense = torch.zeros(b, 3, h, w)
    pred_dense[:, 0] = 1.0  # all probability on 32px

    # Case 2: Router predicts sparse scale (scale 2)
    pred_sparse = torch.zeros(b, 3, h, w)
    pred_sparse[:, 2] = 1.0  # all probability on 128px

    loss_dense_correct = physical_scale_alignment_loss(
        scale_weights=pred_dense,
        target_y=target_dense,
        tau_dense=0.12,
        tau_sparse=0.03,
        mask_background=False,
    )
    loss_dense_wrong = physical_scale_alignment_loss(
        scale_weights=pred_sparse,
        target_y=target_dense,
        tau_dense=0.12,
        tau_sparse=0.03,
        mask_background=False,
    )
    assert loss_dense_correct < loss_dense_wrong, (
        f"Dense target should favor scale 0: correct={loss_dense_correct:.4f}, wrong={loss_dense_wrong:.4f}"
    )

    loss_sparse_correct = physical_scale_alignment_loss(
        scale_weights=pred_sparse,
        target_y=target_sparse,
        tau_dense=0.12,
        tau_sparse=0.03,
        mask_background=False,
    )
    loss_sparse_wrong = physical_scale_alignment_loss(
        scale_weights=pred_dense,
        target_y=target_sparse,
        tau_dense=0.12,
        tau_sparse=0.03,
        mask_background=False,
    )
    assert loss_sparse_correct < loss_sparse_wrong, (
        f"Sparse target should favor scale 2: correct={loss_sparse_correct:.4f}, wrong={loss_sparse_wrong:.4f}"
    )


def test_gated_density_curvature_mechanics():
    """Verify that density gating eliminates curvature squaring on pavement/facades while preserving it on dense crowds."""
    head = FineMeasureHead(
        width=32,
        density_curvature=True,
        gated_density_curvature=True,
        curvature_dense_threshold=0.15,
        curvature_gate_beta=0.03,
        curvature_pool_kernel=8,
    )
    # Set curvature alpha = log(e^0.5 - 1) so softplus(alpha) = 0.5
    with torch.no_grad():
        head.curvature_alpha.fill_(0.0)  # softplus(0) = log(2) ≈ 0.693

    # Scenario A: Pavement / textured background with low density (y_base ≈ 0.05)
    # Without gating, y_base^2 = 0.0025 with multiplier 0.693 gives +0.0017 (+3.4% error)
    # With gating (0.05 - 0.15) / 0.03 = -3.33, sigmoid(-3.33) ≈ 0.034 -> curvature suppressed by 97%!
    z_low = torch.full((1, 1, 32, 32), -4.0)  # low density
    y_low = head.activate(z_low)
    y_base_low = F.softplus(z_low)
    curv_excess_low = (y_low - y_base_low).max().item()

    # Scenario B: High density crowd cluster (y_base ≈ 1.0)
    # Local density = 1.0 >> 0.15, sigmoid((1.0 - 0.15)/0.03) ≈ 1.0 -> full curvature expansion
    z_high = torch.full((1, 1, 32, 32), 1.0)  # high density
    y_high = head.activate(z_high)
    y_base_high = F.softplus(z_high)
    curv_excess_high = (y_high - y_base_high).mean().item()
    expected_expansion = math.log(2.0) * (y_base_high.mean().item() ** 2)

    # Curvature expansion on high density should be >= 95% of theoretical full expansion
    assert curv_excess_high > 0.95 * expected_expansion, (
        f"Expected high-density expansion ~{expected_expansion:.4f}, got {curv_excess_high:.4f}"
    )

    # Curvature on low density should be virtually silenced compared to high density
    ratio = curv_excess_low / max(curv_excess_high, 1e-6)
    assert ratio < 0.01, f"Low-density pavement curvature was not silenced! Ratio: {ratio:.6f}"


def test_gated_density_curvature_gradient_flow():
    """Verify gradients propagate to curvature_alpha through the density gate."""
    head = FineMeasureHead(
        width=32,
        density_curvature=True,
        gated_density_curvature=True,
        curvature_dense_threshold=0.15,
        curvature_gate_beta=0.03,
        curvature_pool_kernel=8,
    )
    z = torch.full((1, 1, 32, 32), 1.0, requires_grad=True)
    y = head.activate(z)
    loss = y.sum()
    loss.backward()

    assert head.curvature_alpha.grad is not None
    assert torch.isfinite(head.curvature_alpha.grad).all()
    assert head.curvature_alpha.grad.item() != 0.0, "Curvature alpha received zero gradient!"


def test_rmr_v19_full_pipeline_forward_and_backward():
    """Verify end-to-end forward and backward pass of RMR-v19 model and loss composite."""
    with open("configs/rmr_v19/rmr_v19_factorized_aspect.yaml") as f:
        cfg = yaml.safe_load(f)

    model_cfg = RMRv3Config(**cfg["model"])
    model = RMRv3(model_cfg)
    loss_cfg = RMRv3LossConfig(**cfg["loss"])

    x = torch.randn(2, 3, 256, 256)
    target_y = torch.zeros(2, 1, 64, 64)
    # Put a few simulated head peaks
    target_y[0, 0, 16:20, 16:20] = 0.5
    target_y[1, 0, 32:40, 32:40] = 1.2

    outputs = model(x, solver_strength=1.0)

    # Verify output dictionary structure
    assert "y" in outputs
    assert "y0" in outputs
    assert "scale_weights" in outputs
    assert "pi_scale" in outputs
    assert "pi_aspect" in outputs
    assert outputs["scale_weights"].shape == (2, 4, 64, 64)
    assert outputs["pi_scale"].shape == (2, 3, 64, 64)
    assert outputs["pi_aspect"].shape == (2, 2, 64, 64)

    # Compute loss composite
    losses = compute_rmr_v3_losses(outputs, target_y, loss_cfg)
    assert "total" in losses
    assert "scale_align" in losses
    assert torch.isfinite(losses["total"]).all()

    # Backward pass
    losses["total"].backward()

    # Check that gradients exist for backbone, neck, routing, and heads
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.encoder.parameters()), "Encoder received no gradients"
    assert model.fine_head.curvature_alpha.grad is not None
    assert model.scale_router.pw_scale.weight.grad is not None
    assert model.scale_router.pw_aspect.weight.grad is not None
    assert model.scale_router.persp_weight_aspect.grad is not None


def test_rmr_v19_numerical_stability_edge_cases():
    """Verify stability on empty background (0 count), extreme count, and non-power-of-2 shapes."""
    with open("configs/rmr_v19/rmr_v19_factorized_aspect.yaml") as f:
        cfg = yaml.safe_load(f)

    model_cfg = RMRv3Config(**cfg["model"])
    model = RMRv3(model_cfg)
    loss_cfg = RMRv3LossConfig(**cfg["loss"])

    # Edge Case 1: Empty background image (all zero ground truth)
    x_empty = torch.zeros(1, 3, 192, 192)
    t_empty = torch.zeros(1, 1, 48, 48)
    out_empty = model(x_empty)
    losses_empty = compute_rmr_v3_losses(out_empty, t_empty, loss_cfg)
    assert torch.isfinite(losses_empty["total"]).all()
    assert not torch.isnan(losses_empty["total"]).any()

    # Edge Case 2: Extreme density (> 2500 heads)
    x_dense = torch.randn(1, 3, 256, 256)
    t_dense = torch.full((1, 1, 64, 64), 0.75)  # sum = 64*64*0.75 = 3072 heads
    out_dense = model(x_dense)
    losses_dense = compute_rmr_v3_losses(out_dense, t_dense, loss_cfg)
    assert torch.isfinite(losses_dense["total"]).all()
    assert not torch.isnan(losses_dense["total"]).any()

    # Edge Case 3: Arbitrary rectangular dimension (not divisible by 32, stride 4 aligned: 268x332)
    x_rect = torch.randn(1, 3, 268, 332)
    out_rect = model(x_rect)
    assert out_rect["y"].shape == (1, 1, 67, 83)
    assert out_rect["scale_weights"].shape == (1, 4, 67, 83)
    sum_rect = out_rect["scale_weights"].sum(dim=1)
    assert torch.allclose(sum_rect, torch.ones_like(sum_rect), atol=1e-5)


def test_rmr_v19_all_suite_configs_load_and_parameter_check():
    """Verify all 5 configs in the RMR-v19 experimental suite load cleanly and satisfy budget."""
    expected_budgets = {
        "configs/rmr_v19/rmr_v19_factorized_aspect.yaml": 104509,
        "configs/rmr_v19/rmr_v19_canonical_isotropic.yaml": 104441,
        "configs/rmr_v19/rmr_v19_ablation_no_gated_curv.yaml": 104509,
        "configs/rmr_v19/rmr_v19_ablation_no_curvature.yaml": 104508,
        "configs/rmr_v19/rmr_v19_control_no_solver.yaml": 104509,
    }

    for cfg_path, exp_params in expected_budgets.items():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        model_cfg = RMRv3Config(**cfg["model"])
        model = RMRv3(model_cfg)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert n_params == exp_params, f"{cfg_path}: expected {exp_params} params, got {n_params}"
        assert n_params <= 105000, f"{cfg_path}: exceeded 105,000 budget: {n_params}"


def test_rmr_v19_ablation_behavior_isolation():
    """Verify causal isolation between the 5 configurations in the RMR-v19 suite."""
    # 1. Canonical Isotropic: 3 scale weights, no aspect ratio branch
    with open("configs/rmr_v19/rmr_v19_canonical_isotropic.yaml") as f:
        cfg_iso = yaml.safe_load(f)
    model_iso = RMRv3(RMRv3Config(**cfg_iso["model"]))
    x = torch.randn(1, 3, 128, 128)
    out_iso = model_iso(x)
    assert out_iso["scale_weights"].shape[1] == 3
    assert "pi_aspect" not in out_iso

    # 2. Control No Solver: y must be identically y0
    with open("configs/rmr_v19/rmr_v19_control_no_solver.yaml") as f:
        cfg_nosolver = yaml.safe_load(f)
    model_nosolver = RMRv3(RMRv3Config(**cfg_nosolver["model"]))
    out_nosolver = model_nosolver(x)
    assert torch.equal(out_nosolver["y"], out_nosolver["y0"]), "Control no solver did not output identical y == y0"

    # 3. Ablation No Curvature: fine_head.density_curvature must be False
    with open("configs/rmr_v19/rmr_v19_ablation_no_curvature.yaml") as f:
        cfg_nocurv = yaml.safe_load(f)
    model_nocurv = RMRv3(RMRv3Config(**cfg_nocurv["model"]))
    assert not getattr(model_nocurv.fine_head, "density_curvature", False)
    assert not hasattr(model_nocurv.fine_head, "curvature_alpha")

    # 4. Ablation No Gated Curvature: density_curvature True, gated_density_curvature False
    with open("configs/rmr_v19/rmr_v19_ablation_no_gated_curv.yaml") as f:
        cfg_nogated = yaml.safe_load(f)
    model_nogated = RMRv3(RMRv3Config(**cfg_nogated["model"]))
    assert getattr(model_nogated.fine_head, "density_curvature", False)
    assert not getattr(model_nogated.fine_head, "gated_density_curvature", False)

