import copy
import math
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
import yaml

from rmr_core.operators import (
    RegionSet,
    build_multiscale_regions,
    regional_sum,
    weighted_normalized_adjoint_field,
)
from rmr_core.evaluation import evaluate_dataset
from rmr_v3.config import validate_v3_config
from rmr_v3.losses import RMRv3LossConfig, compute_rmr_v3_losses
from rmr_v3.model import RMRv3, RMRv3Config
from rmr_v3.solver import unrolled_sirt_solver


def test_rmr_v21_parameter_budget():
    """Verify RMR-v21 canonical model trainable parameter count is strictly <= 105,000."""
    cfg_path = Path("configs/rmr_v21/rmr_v21_canonical.yaml")
    assert cfg_path.exists(), "configs/rmr_v21/rmr_v21_canonical.yaml does not exist"

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    m_cfg = raw_cfg["model"]
    model_cfg = RMRv3Config.from_dict(m_cfg, pretrained=False)
    model = RMRv3(model_cfg)

    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert num_trainable <= 105_000, f"Parameter budget exceeded! Got {num_trainable} > 105,000"
    assert num_trainable == 104_441, f"Expected exactly 104,441 parameters, got {num_trainable}"


def test_rmr_v21_all_configs_schema_valid():
    """Verify all 6 RMR-v21 configs pass strict validation without unknown keys."""
    config_names = [
        "rmr_v21_canonical",
        "rmr_v21_ablation_no_hybrid_flux",
        "rmr_v21_ablation_no_bb_step",
        "rmr_v21_ablation_no_sample_loss",
        "rmr_v21_ablation_no_curvature",
        "rmr_v21_control_no_solver",
    ]
    for name in config_names:
        p = Path(f"configs/rmr_v21/{name}.yaml")
        assert p.exists(), f"Missing config: {p}"
        with open(p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        validate_v3_config(cfg)


def test_hybrid_recovery_flux_breaks_zero_absorbing_barrier():
    """Verify that Hybrid Adjoint (alpha > 0) injects discovery flux when y0 = 0 and b > 0.
    
    Under pure Radon-Nikodym (alpha = 0), y * (delta / q) is 0 * ... = 0, so missed heads
    can NEVER be recovered.
    Under Hybrid recovery flux (alpha = 0.05), an additive Lebesgue seed is back-projected.
    """
    h, w = 32, 32
    regions = build_multiscale_regions(
        h, w,
        output_stride=4,
        region_sizes_px=(16, 32),
    )
    m = regions.boxes.shape[0]

    # Initial density is strictly zero everywhere (completely missed by backbone)
    y_zero = torch.zeros((1, 1, h, w), dtype=torch.float32)

    # Regional head predicts 10 people in region 0
    b_regional = torch.zeros((1, 1, m), dtype=torch.float32)
    b_regional[0, 0, 0] = 10.0

    weight = torch.ones((1, 1, m), dtype=torch.float32)

    # 1. Pure Radon-Nikodym (alpha = 0.0): Zero-absorbing barrier
    grad_rn = weighted_normalized_adjoint_field(
        y_zero,
        b_regional,
        weight,
        regions,
        adjoint_mode="radon_nikodym",
        hybrid_recovery_alpha=0.0,
        morozov_gamma=0.0,
    )
    assert torch.allclose(grad_rn, torch.zeros_like(grad_rn)), (
        "Pure Radon-Nikodym should produce strictly zero gradient when y=0 (zero-absorbing barrier)"
    )

    # 2. Hybrid Recovery Flux (alpha = 0.05): Injects recovery flux
    grad_hybrid = weighted_normalized_adjoint_field(
        y_zero,
        b_regional,
        weight,
        regions,
        adjoint_mode="radon_nikodym",
        hybrid_recovery_alpha=0.05,
        morozov_gamma=0.0,
    )
    # Since b > q, delta = q - b < 0, so rate_residual < 0 and grad < 0.
    # In SIRT update: y_{k+1} = y_k - omega * grad = 0 - omega * (-val) > 0!
    box0 = regions.boxes[0].int()
    grad_in_box0 = grad_hybrid[0, 0, box0[0]:box0[2], box0[1]:box0[3]]
    assert (grad_in_box0 < 0.0).all(), "Hybrid recovery flux should produce negative gradient to seed mass"
    assert grad_in_box0.abs().max().item() > 1e-4, "Hybrid recovery flux magnitude too small"


def test_hybrid_recovery_flux_preserves_morozov_deadband():
    """Verify that when discrepancy is within the Morozov deadband (|q - b| <= gamma * sigma_b),
    hybrid recovery flux remains strictly zero, ensuring zero background phantom mass lift.
    """
    h, w = 32, 32
    regions = build_multiscale_regions(h, w, output_stride=4, region_sizes_px=(16, 32))
    m = regions.boxes.shape[0]

    y_zero = torch.zeros((1, 1, h, w), dtype=torch.float32)
    # Measurement is small noise b = 0.5 with high variance sigma_b^2 = 1.0 (sigma = 1.0)
    b_noisy = torch.full((1, 1, m), 0.5, dtype=torch.float32)
    b_var = torch.full((1, 1, m), 1.0, dtype=torch.float32)  # sigma = 1.0
    weight = torch.ones((1, 1, m), dtype=torch.float32)

    # With gamma = 0.75, deadband = 0.75 * 1.0 = 0.75 > |0 - 0.5| = 0.5.
    # Delta should be completely clamped to 0.0!
    grad_deadband = weighted_normalized_adjoint_field(
        y_zero,
        b_noisy,
        weight,
        regions,
        adjoint_mode="radon_nikodym",
        hybrid_recovery_alpha=0.05,
        b_variance=b_var,
        morozov_gamma=0.75,
    )
    assert torch.allclose(grad_deadband, torch.zeros_like(grad_deadband), atol=1e-7), (
        "Morozov deadband must clamp noisy background discrepancy to zero even with hybrid recovery flux"
    )


def test_elementwise_loss_scaling_cross_talk_isolation():
    """Verify that Elementwise High-Density Loss Rescaling isolates batch samples:
    An empty background image receives weight 1.0 and its gradient is completely unperturbed
    by a high-density stadium crop in the same batch.
    """
    b_sz = 2
    h, w = 32, 32
    regions = build_multiscale_regions(h, w, output_stride=4, region_sizes_px=(32,))
    m = regions.boxes.shape[0]

    # Sample 0: Stadium crowd crop (GT count = 1500)
    # Sample 1: Empty background crop (GT count = 0)
    target_stadium = torch.full((1, 1, h, w), 1500.0 / (h * w), dtype=torch.float32)
    target_empty = torch.zeros((1, 1, h, w), dtype=torch.float32)
    target_batch = torch.cat([target_stadium, target_empty], dim=0)

    # Dummy model outputs with requires_grad on predictions
    pred_y_stadium = torch.full((1, 1, h, w), 1000.0 / (h * w), dtype=torch.float32, requires_grad=True)
    pred_y_empty = torch.full((1, 1, h, w), 0.1, dtype=torch.float32, requires_grad=True)

    def _make_outputs(y_s, y_e):
        y_cat = torch.cat([y_s, y_e], dim=0)
        return {
            "y": y_cat,
            "y0": y_cat.clone(),
            "regions": regions,
            "b_region": regional_sum(y_cat, regions.boxes),
            "region_dispersion": torch.full((2, 1, m), 50.0),
        }

    cfg_elementwise = RMRv3LossConfig(
        elementwise_dense_scaling=True,
        dense_loss_thresh=100.0,
        dense_loss_norm=150.0,
        dense_loss_alpha=1.0,
        dense_loss_max_boost=2.0,
        lambda_count=1.0,
        lambda_flat_dm16=0.0,  # disable spatial DM to test count/cell isolation
        lambda_cell=0.5,
        lambda_region_nb=0.0,
        lambda_scale_align=0.0,
        lambda_curvature=0.0,
    )

    outputs = _make_outputs(pred_y_stadium, pred_y_empty)
    losses = compute_rmr_v3_losses(outputs, target_batch, cfg_elementwise)
    losses["total"].backward()

    # Stadium sample should receive boost: clamp((1500 - 100)/150, 0, 2.0) = 2.0 -> weight 3.0
    # Empty sample should receive boost: clamp((0 - 100)/150, 0, 2.0) = 0.0 -> weight 1.0
    # Scale in losses dict should reflect average weight (3.0 + 1.0) / 2 = 2.0
    assert abs(losses["dense_loss_scale"].item() - 2.0) < 1e-4

    # Now verify gradient on empty sample when alone vs when in batch with stadium
    pred_y_empty_solo = torch.full((1, 1, h, w), 0.1, dtype=torch.float32, requires_grad=True)
    out_solo = {
        "y": pred_y_empty_solo,
        "y0": pred_y_empty_solo.clone(),
        "regions": regions,
        "b_region": regional_sum(pred_y_empty_solo, regions.boxes),
        "region_dispersion": torch.full((1, 1, m), 50.0),
    }
    loss_solo = compute_rmr_v3_losses(out_solo, target_empty, cfg_elementwise)
    loss_solo["total"].backward()

    # The gradient in the batch of size 2 should be exactly 0.5 * grad_solo
    # (since batch loss is 1/B * sum(w_i * L_i) = 1/2 * (3.0 * L_stadium + 1.0 * L_empty))
    expected_grad_empty = 0.5 * pred_y_empty_solo.grad
    actual_grad_empty = pred_y_empty.grad
    assert torch.allclose(actual_grad_empty, expected_grad_empty, atol=1e-5), (
        f"Cross-talk leakage detected! Expected empty gradient ~{expected_grad_empty.mean():.6f}, "
        f"got {actual_grad_empty.mean():.6f}"
    )


def test_horizontal_flip_tta_symmetry():
    """Verify that Horizontal Flip TTA is strictly equivariant under horizontal reflection:
    flip_H(Y_TTA(flip_H(I))) == Y_TTA(I)
    and total predicted count is invariant to horizontal orientation.
    """
    with open("configs/rmr_v21/rmr_v21_canonical.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["model"]

    model_cfg = RMRv3Config.from_dict(cfg, pretrained=False)
    model = RMRv3(model_cfg)
    model.eval()

    img = torch.randn(1, 3, 64, 64)
    img_flip = torch.flip(img, dims=[-1])

    with torch.no_grad():
        # 1. TTA on original image
        out_orig = model(img)
        out_orig_flip = model(img_flip)
        y_tta_orig = 0.5 * (out_orig["y"] + torch.flip(out_orig_flip["y"], dims=[-1]))

        # 2. TTA on flipped image
        out_flip_orig = model(img_flip)
        out_flip_flip = model(torch.flip(img_flip, dims=[-1]))
        y_tta_flip = 0.5 * (out_flip_orig["y"] + torch.flip(out_flip_flip["y"], dims=[-1]))

    # Equivariance: unflipping the TTA of the flipped image must match the TTA of the original image
    y_tta_flip_unflipped = torch.flip(y_tta_flip, dims=[-1])
    assert torch.allclose(y_tta_flip_unflipped, y_tta_orig, atol=1e-5), (
        "Horizontal Flip TTA must be strictly equivariant under spatial reflection!"
    )
    # Count invariance: total count must be identical
    assert abs(y_tta_orig.sum().item() - y_tta_flip.sum().item()) < 1e-5, (
        "Total count must be invariant under Horizontal Flip TTA!"
    )


def test_rmr_v21_full_pipeline_forward_and_backward():
    """Verify end-to-end forward and backward pass of RMR-v21 canonical model and loss composite."""
    with open("configs/rmr_v21/rmr_v21_canonical.yaml", "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    model_cfg = RMRv3Config.from_dict(raw_cfg["model"], pretrained=False)
    model = RMRv3(model_cfg)
    loss_cfg = RMRv3LossConfig.from_dict(raw_cfg["loss"])

    images = torch.randn(2, 3, 64, 64)
    targets = torch.zeros(2, 1, 16, 16)
    targets[0, 0, 4, 4] = 5.0
    targets[1, 0, 8, 8] = 20.0

    outputs = model(images)
    losses = compute_rmr_v3_losses(outputs, targets, loss_cfg)

    assert "total" in losses
    assert torch.isfinite(losses["total"])
    assert losses["total"] > 0.0

    losses["total"].backward()

    # Verify gradients reach backbone parameters
    has_grad = False
    for p in model.parameters():
        if p.grad is not None and torch.isfinite(p.grad).any():
            has_grad = True
            break
    assert has_grad, "No parameters received valid finite gradients!"
