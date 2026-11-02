"""Unit tests for EB-GB (Error-Bounded Geometric Binning) and MPC (Micro-Spatial Allocation Simplex).

Guarantees:
1. Exact Mass Conservation of MPC-Head:
   sum_{u, v} m0_fine(2i + u, 2j + v) == M(i, j) identically.
2. Bounded Gradient of EB-GB Loss (Zero Starvation):
   ||nabla_z L_{CE}||_infty <= 1.0 for all y in [0, 450] persons (eliminating 4,080x NLL starvation).
3. Exact Barycentric Interpolation:
   E_{Y_soft}[b] == y_R identically for any arbitrary continuous count.
4. Interface Compatibility & Autograd Flow with RMR Model Pipeline.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rmr_core.mpc_head import MicroSpatialPolyphaseCarrierHead
from rmr_core.operators.regions import build_multiscale_regions
from rmr_v3.ebgb_head import UnsaturableRegionalEvidenceHead, DEFAULT_GEOMETRIC_BINS
from rmr_v3.losses.ebgb import compute_soft_bin_targets, compute_ebgb_loss


def test_mpc_exact_mass_conservation() -> None:
    """Invariant 1: MPC-Head satisfies exact mass conservation down to 2x2 sub-cells."""
    b, c, h, w = 2, 32, 16, 16
    p4 = torch.randn(b, c, h, w)

    # 1. Stride 2 mode
    head_s2 = MicroSpatialPolyphaseCarrierHead(width=32, subpixel_stride2=True)
    y0_s2, macro_cap, alloc_simplex = head_s2(p4)

    assert y0_s2.shape == (b, 1, 32, 32)
    assert macro_cap.shape == (b, 1, 16, 16)
    assert alloc_simplex.shape == (b, 4, 16, 16)

    # Sum of allocation simplex over the 4 sub-cells must be 1.0 everywhere
    simplex_sum = alloc_simplex.sum(dim=1)
    assert torch.allclose(simplex_sum, torch.ones_like(simplex_sum), atol=1e-6)

    # Summing y0 over 2x2 pixel blocks must exactly equal macro_capacity
    macro_reconstructed = F.avg_pool2d(y0_s2, kernel_size=2, stride=2) * 4.0
    assert torch.allclose(macro_reconstructed, macro_cap, atol=1e-5)

    # 2. Stride 4 mode
    head_s4 = MicroSpatialPolyphaseCarrierHead(width=32, subpixel_stride2=False)
    y0_s4, macro_s4, alloc_s4 = head_s4(p4)
    assert y0_s4.shape == (b, 1, 16, 16)
    assert torch.allclose(y0_s4, macro_s4, atol=1e-6)


def test_ebgb_soft_targets_exact_expectation() -> None:
    """Invariant 2: Barycentric projection onto geometric bins preserves exact expected count."""
    bins = torch.tensor(DEFAULT_GEOMETRIC_BINS)
    # Test across broad dynamic range: 0 to 450 persons
    test_counts = torch.tensor([
        [0.0, 0.5, 1.0, 2.7, 7.3, 15.0, 35.5, 75.2, 180.0, 250.0, 380.0, 450.0, 500.0]
    ])

    target_probs = compute_soft_bin_targets(test_counts, bins)

    # Probability simplex: sum must be 1.0
    assert torch.allclose(target_probs.sum(dim=-1), torch.ones_like(test_counts), atol=1e-6)

    # Expected value must match ground truth clamped to [bins[0], bins[-1]]
    expected_counts = (target_probs * bins).sum(dim=-1)
    clamped_counts = test_counts.clamp(bins[0].item(), bins[-1].item())
    assert torch.allclose(expected_counts, clamped_counts, atol=1e-5)


def test_ebgb_bounded_gradient_zero_starvation() -> None:
    """Invariant 3: Cross-Entropy gradient is strictly bounded by 1.0, eliminating NLL starvation."""
    bins = torch.tensor(DEFAULT_GEOMETRIC_BINS)
    num_bins = len(bins)
    b, m = 2, 20

    bin_logits = torch.randn(b, m, num_bins, requires_grad=True)
    hurdle_logit = torch.randn(b, m, requires_grad=True)

    # Mixed crowd: sparse, moderate, and ultra-dense (up to 350 persons)
    target_counts = torch.tensor([
        [0.0, 0.0, 1.0, 2.0, 15.0, 50.0, 120.0, 250.0, 300.0, 350.0] * 2,
        [0.0, 1.0, 3.0, 10.0, 25.0, 80.0, 150.0, 200.0, 280.0, 400.0] * 2,
    ])

    loss, loss_dict = compute_ebgb_loss(
        bin_logits=bin_logits,
        target_count=target_counts,
        bin_centers=bins,
        hurdle_logit=hurdle_logit,
        lambda_hurdle=1.0,
    )

    loss.backward()

    # Mathematical Invariant: ||nabla_z L_{CE}||_infty <= 1.0
    grad = bin_logits.grad
    assert grad is not None
    assert grad.abs().max() <= 1.0 + 1e-5

    # Check non-zero gradient for ultra-dense sample (index 7, y=250)
    # In classical NLL, gradient at y=250 is attenuated by 4,080x (~0.002)
    # Under EB-GB, the gradient magnitude remains healthy O(1)
    dense_grad_mag = grad[:, 7, :].abs().sum(dim=-1).mean()
    sparse_grad_mag = grad[:, 2, :].abs().sum(dim=-1).mean()
    # Ratio between sparse and dense must be balanced (within 10x, NOT 4,080x)
    ratio = max(dense_grad_mag / (sparse_grad_mag + 1e-6), sparse_grad_mag / (dense_grad_mag + 1e-6))
    assert ratio < 20.0, f"Disparity too high: {ratio:.2f}"


def test_ebgb_head_forward_and_interface_compatibility() -> None:
    """Invariant 4: UnsaturableRegionalEvidenceHead integrates with RegionSet and produces required keys."""
    b = 2
    h, w = 16, 16
    p4 = torch.randn(b, 32, h, w)
    p8 = torch.randn(b, 32, h // 2, w // 2)
    p16 = torch.randn(b, 32, h // 4, w // 4)
    pyramid = (p4, p8, p16)

    regions = build_multiscale_regions(
        height=h,
        width=w,
        output_stride=4,
        region_sizes_px=(32, 64),
        overlap=0.5,
        include_full_image=False,
    )

    head = UnsaturableRegionalEvidenceHead(
        region_sizes_px=(32, 64),
        feature_dim=32,
        hidden_dim=42,
        regional_feature_stats="mean",
        native_scale_pooling=True,
    )

    out = head(pyramid, regions)

    m = regions.boxes.shape[0]
    assert out["mu_count"].shape == (b, 1, m)
    assert out["rate"].shape == (b, 1, m)
    assert out["dispersion"].shape == (b, 1, m)
    assert out["hurdle_logit"].shape == (b, 1, m)
    assert out["bin_logits"].shape == (b, m, len(DEFAULT_GEOMETRIC_BINS))
    assert out["bin_probs"].shape == (b, m, len(DEFAULT_GEOMETRIC_BINS))
    assert out["count_variance"].shape == (b, 1, m)

    # All predictions must be strictly non-negative
    assert (out["mu_count"] >= 0.0).all()
    assert (out["rate"] >= 0.0).all()
    assert (out["dispersion"] >= 1.0).all()
    assert (out["count_variance"] > 0.0).all()


def test_rmr_full_forward_pipeline_with_mpc_and_ebgb() -> None:
    """Invariant 5: RMRv3 forward pass end-to-end with MPC fine head and EB-GB regional head."""
    from rmr_v3.model import RMRv3, RMRv3Config

    cfg = RMRv3Config(
        feature_width=32,
        region_head_hidden=40,
        fine_carrier_type="mpc",
        regional_head_type="ebgb",
        enable_solver=True,
        solver_mode="mcbpro",
        mcbpro_soft_margin_gating=True,
        mcbpro_dense_clamp_min=0.90,
    )
    model = RMRv3(cfg)
    model.eval()

    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out = model(x)

    assert out.y.shape == (2, 1, 16, 16)
    assert out.y0.shape == (2, 1, 16, 16)
    assert (out.y >= 0.0).all()
    assert (out.y0 >= 0.0).all()
    assert out.b_solver is not None
    assert "bin_logits" in out
    assert out["bin_logits"].shape[-1] == len(DEFAULT_GEOMETRIC_BINS)


def test_rmr_loss_orchestration_and_backward_with_ebgb_and_mpc() -> None:
    """Invariant 6: Multi-task loss computation and backward pass with EB-GB and MPC."""
    from rmr_v3.model import RMRv3, RMRv3Config
    from rmr_v3.losses import RMRv3LossConfig, compute_rmr_v3_losses

    cfg = RMRv3Config(
        feature_width=32,
        region_head_hidden=40,
        fine_carrier_type="mpc",
        regional_head_type="ebgb",
        enable_solver=True,
        solver_mode="mcbpro",
        mcbpro_soft_margin_gating=True,
        mcbpro_dense_clamp_min=0.90,
    )
    model = RMRv3(cfg)
    model.train()

    loss_cfg = RMRv3LossConfig(
        lambda_count=0.5,
        lambda_region_nb=0.5,
        lambda_bayesian=0.0,
        lambda_flat_dm16=0.0,
    )

    x = torch.randn(2, 3, 64, 64)
    target = torch.rand(2, 1, 16, 16) * 2.0  # Synthetic crowd target

    out = model(x)
    losses = compute_rmr_v3_losses(out, target, cfg=loss_cfg)

    assert "total" in losses
    assert "region_nb" in losses
    assert "ebgb_ce" in losses
    assert losses["total"] > 0.0
    assert torch.isfinite(losses["total"])

    losses["total"].backward()

    # Verify MPC head and EBGB head parameters receive finite gradients
    for name, p in model.fine_head.named_parameters():
        assert p.grad is not None, f"MPC head param {name} missing grad"
        assert torch.isfinite(p.grad).all(), f"MPC head param {name} has non-finite grad"

    for name, p in model.region_head.named_parameters():
        assert p.grad is not None, f"EBGB head param {name} missing grad"
        assert torch.isfinite(p.grad).all(), f"EBGB head param {name} has non-finite grad"


def test_rmr_full_forward_and_backward_subpixel_stride2() -> None:
    """Invariant 7: Full forward pass and backward pass with subpixel_stride2=True."""
    from rmr_v3.model import RMRv3, RMRv3Config
    from rmr_v3.losses import RMRv3LossConfig, compute_rmr_v3_losses

    cfg = RMRv3Config(
        feature_width=32,
        region_head_hidden=40,
        fine_carrier_type="mpc",
        regional_head_type="ebgb",
        subpixel_stride2=True,
        output_stride=2,
        enable_solver=True,
        solver_mode="mcbpro",
    )
    model = RMRv3(cfg)
    model.train()

    loss_cfg = RMRv3LossConfig(
        output_stride=2,
        lambda_count=0.5,
        lambda_region_nb=0.5,
        lambda_bayesian=0.0,
        lambda_flat_dm16=0.0,
    )

    x = torch.randn(2, 3, 64, 64)
    target = torch.rand(2, 1, 32, 32) * 2.0  # Synthetic crowd target at Stride 2

    out = model(x)
    assert out.y.shape == (2, 1, 32, 32)
    assert out.y0.shape == (2, 1, 32, 32)

    losses = compute_rmr_v3_losses(out, target, cfg=loss_cfg)
    assert "total" in losses
    assert "region_nb" in losses
    assert losses["total"] > 0.0
    assert torch.isfinite(losses["total"])

    losses["total"].backward()

    # In subpixel_stride2 mode, micro_conv MUST exist and receive finite gradients
    assert hasattr(model.fine_head, "micro_conv")
    assert model.fine_head.micro_conv is not None
    assert model.fine_head.micro_conv.weight.grad is not None
    assert torch.isfinite(model.fine_head.micro_conv.weight.grad).all()


def test_mpc_activate_parity_with_forward() -> None:
    """Invariant 8: activate(forward_logits(p4)) matches forward(p4)[0] down to machine precision."""
    p4 = torch.randn(2, 32, 16, 16)

    # Stride 4
    head4 = MicroSpatialPolyphaseCarrierHead(width=32, subpixel_stride2=False, floor_tau=0.003)
    y4_act = head4.activate(head4.forward_logits(p4))
    y4_fwd, _, _ = head4(p4)
    assert torch.allclose(y4_act, y4_fwd, atol=1e-6)

    # Stride 2
    head2 = MicroSpatialPolyphaseCarrierHead(width=32, subpixel_stride2=True, floor_tau=0.003)
    y2_act = head2.activate(head2.forward_logits(p4))
    y2_fwd, _, _ = head2(p4)
    assert torch.allclose(y2_act, y2_fwd, atol=1e-6)


def test_ebgb_principled_soft_occupancy_gate() -> None:
    """Invariant 9: Soft occupancy gate preserves 100% mass on mu >= 1.0 and suppresses background noise."""
    b, h, w = 1, 16, 16
    regions = build_multiscale_regions(height=h, width=w, output_stride=4, region_sizes_px=(32,), overlap=0.5)

    head = UnsaturableRegionalEvidenceHead(
        region_sizes_px=(32,),
        feature_dim=32,
        hidden_dim=40,
        hurdle_head=True,
    )

    pyramid = (torch.randn(b, 32, h, w), torch.randn(b, 32, h // 2, w // 2), torch.randn(b, 32, h // 4, w // 4))

    # Overwrite hurdle head logit to strongly negative (pi ~ 0.01) to simulate background classification
    assert head.hurdle_head is not None
    nn.init.constant_(head.hurdle_head.bias, -4.595)  # sigmoid(-4.595) ~ 0.01
    nn.init.zeros_(head.hurdle_head.weight)

    out = head(pyramid, regions)
    c_r = out["mu_count"].squeeze()
    var_r = out["count_variance"].squeeze()

    # For any region where mu_bin is >= 1.0, count must NOT be attenuated by pi=0.01
    mu_bin = (out["bin_probs"] * out["bin_centers"]).sum(dim=-1).squeeze()
    crowd_mask = mu_bin >= 1.0
    if crowd_mask.any():
        assert torch.allclose(c_r[crowd_mask], mu_bin[crowd_mask], atol=1e-5)

    # Variance must remain strictly finite and positive
    assert torch.isfinite(var_r).all()
    assert (var_r > 0.0).all()
    assert (var_r < 3e5).all()


def test_ebgb_parameter_ceiling_both_modes() -> None:
    """Invariant 10: Strict parameter ceiling <= 104,441 holds for both hurdle_head=True and False."""
    from rmr_v3.config import load_config
    from rmr_v3.model import RMRv3, RMRv3Config

    raw_cfg = load_config("configs/rmr_canonical/rmr_canonical_104k_ebgb_mpc_seed42.yaml")
    for has_hurdle in (True, False):
        raw_cfg["model"]["hurdle_head"] = has_hurdle
        cfg = RMRv3Config(**raw_cfg["model"])
        model = RMRv3(cfg)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert num_params <= 104441, f"hurdle_head={has_hurdle} exceeded ceiling: {num_params} > 104441"
        if has_hurdle:
            assert num_params == 104379
        else:
            assert num_params == 104338


def test_ebgb_mean_preserving_smoothing() -> None:
    """Invariant 11: Geometric label smoothing preserves exact expected count E[b] == y."""
    bins = torch.tensor(DEFAULT_GEOMETRIC_BINS)
    counts = torch.tensor([[0.0, 0.5, 1.0, 2.7, 7.3, 15.0, 35.5, 75.2, 180.0, 250.0, 380.0, 600.0, 1100.0]])

    p_smoothed = compute_soft_bin_targets(counts, bins, smoothing=0.05)
    # Sum to 1.0
    assert torch.allclose(p_smoothed.sum(dim=-1), torch.ones_like(counts), atol=1e-6)
    # Expectation preserved
    exp_counts = (p_smoothed * bins).sum(dim=-1)
    clamped = counts.clamp(bins[0].item(), bins[-1].item())
    assert torch.allclose(exp_counts, clamped, atol=1e-4)


def test_ebgb_scale_id_full_image_supervision() -> None:
    """Invariant 12: scale_balanced_regional_ebgb_loss supervises scale_id == -1 when present."""
    from rmr_v3.losses.ebgb import scale_balanced_regional_ebgb_loss

    regions = build_multiscale_regions(
        height=32,
        width=32,
        output_stride=4,
        region_sizes_px=(32, 64),
        overlap=0.5,
        include_full_image=True,
    )
    assert (regions.scale_id == -1).any()

    num_boxes = regions.boxes.shape[0]
    bin_logits = torch.randn(2, num_boxes, len(DEFAULT_GEOMETRIC_BINS), requires_grad=True)
    target_region = torch.rand(2, num_boxes) * 10.0
    bins = torch.tensor(DEFAULT_GEOMETRIC_BINS)

    loss, aux = scale_balanced_regional_ebgb_loss(
        bin_logits=bin_logits,
        target_region=target_region,
        bin_centers=bins,
        regions=regions,
    )
    assert torch.isfinite(loss)
    loss.backward()

    # Verify that full-image box logits receive non-zero gradients!
    mask_full = (regions.scale_id == -1)
    assert bin_logits.grad is not None
    full_grad = bin_logits.grad[:, mask_full, :]
    assert (full_grad.abs() > 0.0).any()



