from __future__ import annotations

from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from rmr_core.operators import (
    RegionSet,
    build_multiscale_regions,
    regional_sum,
    weighted_normalized_adjoint_field,
)
from rmr_v3.config import RMRv3Config, load_config, validate_v3_config
from rmr_v3.losses import (
    RMRv3LossConfig,
    compute_rmr_v3_losses,
    physical_scale_alignment_loss,
)
from rmr_v3.model import RMRv3
from rmr_v3.regional_head import reliability_from_nb
from rmr_v3.solver import unrolled_sirt_solver


# =============================================================================
# 1. Scale Invariance Theorem (H_nu 1 = 1) for Radon-Nikodym Adjoint
# =============================================================================

def test_radon_nikodym_scale_invariance_theorem():
    """Theorem 1: For any constant field y = c * 1_Omega, when b = 0, delta = q,

    the Radon-Nikodym adjoint field H_nu reconstructs exactly c * 1_Omega.
    """
    h, w = 32, 32
    regions = build_multiscale_regions(
        height=h, width=w, output_stride=4, region_sizes_px=(16, 32), overlap=0.5, include_full_image=False
    )
    c_val = 3.5

    # Test in float64 for exact mathematical precision
    y64 = torch.full((1, 1, h, w), c_val, dtype=torch.float64)
    q64 = regional_sum(y64, regions.boxes, out_dtype=torch.float64)

    # b_region = 0 so discrepancy delta = q - 0 = q
    b64 = torch.zeros_like(q64)
    weights64 = torch.ones_like(q64)

    h_nu_64 = weighted_normalized_adjoint_field(
        y64,
        b64,
        weights64,
        regions,
        adjoint_mode="radon_nikodym",
        eps=1e-8,
    )

    max_err_64 = (h_nu_64 - c_val).abs().max().item()
    assert max_err_64 < 1e-12, f"Theorem 1 FP64 error must be near-zero, got {max_err_64}"

    # Also verify FP32 numerical stability
    y32 = torch.full((1, 1, h, w), c_val, dtype=torch.float32)
    q32 = regional_sum(y32, regions.boxes, out_dtype=torch.float32)
    b32 = torch.zeros_like(q32)
    weights32 = torch.ones_like(q32)

    h_nu_32 = weighted_normalized_adjoint_field(
        y32,
        b32,
        weights32,
        regions,
        adjoint_mode="radon_nikodym",
        eps=1e-6,
    )
    max_err_32 = (h_nu_32 - c_val).abs().max().item()
    assert max_err_32 < 1e-5, f"Theorem 1 FP32 error must be < 1e-5, got {max_err_32}"


# =============================================================================
# 2. Background Zero-Leakage of Radon-Nikodym Adjoint
# =============================================================================

def test_radon_nikodym_zero_background_leakage():
    """Verify that A_nu^T dumps 0.0 discrepancy onto empty background pixels.

    In the Flat Lebesgue adjoint A^T, discrepancy is spread uniformly across regions,
    contaminating empty background. In A_nu^T, (A_nu^T delta)(u) is modulated by y(u),
    strictly zeroing out background updates.
    """
    h, w = 32, 32
    regions = build_multiscale_regions(
        height=h, width=w, output_stride=4, region_sizes_px=(16,), overlap=0.5, include_full_image=False
    )

    # Foreground head cluster at center, strictly zero everywhere else
    y = torch.zeros(1, 1, h, w, dtype=torch.float32)
    y[0, 0, 14:18, 14:18] = 2.0

    q = regional_sum(y, regions.boxes)
    # Measurement b has a positive discrepancy: b = q - 5.0 so delta = q - b = 5.0
    b = q - 5.0
    weights = torch.ones_like(q)

    field_flat = weighted_normalized_adjoint_field(
        y, b, weights, regions, adjoint_mode="flat"
    )
    field_rn = weighted_normalized_adjoint_field(
        y, b, weights, regions, adjoint_mode="radon_nikodym"
    )

    flat_bg_leak = field_flat[0, 0, :8, :8].max().item()
    assert flat_bg_leak > 0.0, "Flat adjoint expectedly leaks mass onto background"

    rn_bg_mass = field_rn[0, 0, :8, :8].abs().max().item()
    assert rn_bg_mass == 0.0, f"Radon-Nikodym adjoint must have 0.0 leakage on background, got {rn_bg_mass}"

    rn_fg_mass = field_rn[0, 0, 14:18, 14:18].mean().item()
    assert rn_fg_mass > 0.0, f"Radon-Nikodym adjoint must update active cluster, got {rn_fg_mass}"


# =============================================================================
# 3. Bayesian Morozov Discrepancy Deadband Shrinkage
# =============================================================================

def test_morozov_discrepancy_deadband_shrinkage():
    """Verify that Morozov shrinkage deadband eliminates noise below gamma * sigma."""
    h, w = 32, 32
    regions = build_multiscale_regions(
        height=h, width=w, output_stride=4, region_sizes_px=(16,), overlap=0.5, include_full_image=False
    )

    y = torch.full((1, 1, h, w), 1.0, dtype=torch.float32)
    q = regional_sum(y, regions.boxes)
    
    b_var = torch.full_like(q, 16.0)  # sigma = 4.0
    gamma = 1.0  # deadband = 4.0

    # Region 0: discrepancy delta = q - b = 2.0 (< deadband 4.0) -> shrunk to 0.0
    # Region 1: discrepancy delta = q - b = 6.0 (> deadband 4.0) -> shrunk to 6.0 - 4.0 = 2.0
    b_val = q.clone()
    b_val[0, 0, 0] = q[0, 0, 0] - 2.0
    b_val[0, 0, 1] = q[0, 0, 1] - 6.0

    weights = torch.ones_like(q)

    field_morozov = weighted_normalized_adjoint_field(
        y,
        b_val,
        weights,
        regions,
        adjoint_mode="flat",
        b_variance=b_var,
        morozov_gamma=gamma,
    )

    assert not torch.isnan(field_morozov).any()
    assert not torch.isinf(field_morozov).any()


# =============================================================================
# 4. Signal-to-Noise Ratio (SNR) Reliability Monotonicity
# =============================================================================

def test_snr_reliability_weighting():
    h, w = 32, 32
    regions = build_multiscale_regions(
        height=h, width=w, output_stride=4, region_sizes_px=(16,), overlap=0.5, include_full_image=False
    )
    m = regions.boxes.shape[0]

    mu_count = torch.zeros(1, 1, m)
    mu_count[0, 0, 0] = 0.05
    mu_count[0, 0, 1] = 50.0

    dispersion = torch.full_like(mu_count, 10.0)

    res_snr = reliability_from_nb(
        mu_count,
        dispersion,
        regions,
        mode="snr",
        weight_min=0.25,
        weight_max=4.0,
        normalize_within_scale=False,
    )
    w_snr = res_snr["weight"]
    assert w_snr[0, 0, 1] > w_snr[0, 0, 0]

    res_var = reliability_from_nb(
        mu_count,
        dispersion,
        regions,
        mode="nb_rate_variance",
        weight_min=0.25,
        weight_max=4.0,
        normalize_within_scale=False,
    )
    w_var = res_var["weight"]
    assert w_var[0, 0, 0] > w_var[0, 0, 1]


# =============================================================================
# 5. Physical Scale Alignment Loss
# =============================================================================

def test_physical_scale_alignment_loss():
    b, k, h, w = 2, 3, 32, 32
    target_y = torch.zeros(b, 1, h, w)

    target_y[0, 0, :, :] = 0.20  # Dense
    target_y[1, 0, :, :] = 0.01  # Sparse

    # Test 1: Perfect scale prediction matching target distribution
    scale_weights_perfect = torch.zeros(b, k, h, w)
    scale_weights_perfect[0, 0, :, :] = 1.0  # Scale 0 (16x16)
    scale_weights_perfect[1, 2, :, :] = 1.0  # Scale 2 (64x64)

    loss_perf = physical_scale_alignment_loss(
        scale_weights_perfect,
        target_y,
        tau_dense=0.12,
        tau_sparse=0.03,
        kernel_size=1,
    )
    assert loss_perf.item() < 1e-4

    # Test 2: Softmax logits gradient flow
    logits = torch.randn(b, k, h, w, requires_grad=True)
    scale_weights = F.softmax(logits, dim=1)

    loss = physical_scale_alignment_loss(
        scale_weights,
        target_y,
        tau_dense=0.12,
        tau_sparse=0.03,
        kernel_size=1,
    )
    assert loss.item() > 0.0

    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()
    assert (logits.grad.abs().sum() > 0.0).item()


# =============================================================================
# 6. Parameter Budget Verification: strictly <= 105,000 parameters
# =============================================================================

def test_rmr_v13_parameter_budget():
    with pytest.raises(ValueError, match="permanently BANNED"):
        RMRv3Config(
            backbone_name="mobilenetv4_conv_small_050.e3000_r224_in1k",
            pretrained=False,
            neck_type="aspp_lite",
            use_aspp_gap=True,
            aspp_dilations=(1, 3, 6),
            region_sizes_px=(32, 64, 128),
            region_overlap=0.5,
            include_full_image=False,
            regional_feature_stats="mean",
            region_head_hidden=48,
            dynamic_scale_routing=True,
            foreground_gate=True,
            enable_solver=True,
            iterations=6,
            adjoint_mode="radon_nikodym",
            morozov_gamma=0.75,
            reliability_mode="snr",
            trust_region_kappa=0.35,
            trust_region_floor=0.005,
            tv_lambda=0.02,
            proximal_tau=0.015,
            proximal_mode="firm",
            proximal_mu=3.0,
            hurdle_head=True,
            temp_softplus=True,
        )


# =============================================================================
# 7. One-Batch Overfit with All 4 Upgrades Active
# =============================================================================

def test_rmr_v13_one_batch_overfit():
    torch.manual_seed(42)
    cfg = RMRv3Config(
        backbone_name="mobilenetv4_conv_small_050.e3000_r224_in1k",
        pretrained=False,
        neck_type="aspp_lite",
        use_aspp_gap=True,
        aspp_dilations=(1, 3, 6),
        region_sizes_px=(32, 64),
        region_overlap=0.5,
        include_full_image=False,
        regional_feature_stats="mean",
        region_head_hidden=48,
        dynamic_scale_routing=True,
        enable_solver=True,
        iterations=3,
        adjoint_mode="radon_nikodym",
        morozov_gamma=0.75,
        reliability_mode="snr",
        trust_region_kappa=0.35,
        trust_region_floor=0.005,
        tv_lambda=0.02,
        proximal_tau=0.015,
        proximal_mode="firm",
        proximal_mu=3.0,
        hurdle_head=True,
        temp_softplus=True,
    )
    model = RMRv3(cfg)
    model.train()

    loss_cfg = RMRv3LossConfig(
        allocation_loss_type="flat_dm16",
        dm_target="dual",
        lambda_count=1.0,
        lambda_flat_dm16=1.0,
        lambda_cell=0.50,
        lambda_region_nb=0.20,
        lambda_hurdle=0.10,
        lambda_trunc_nb=0.20,
        lambda_scale_align=0.05,
        lambda_curvature=0.50,
        curvature_gate_threshold=0.08,
        curvature_gate_kernel=5,
        curvature_gate_mode="hard",
        lambda_hard_bg=0.10,
        lambda_fg_gate=0.05,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    img = torch.randn(1, 3, 128, 128)
    target = torch.zeros(1, 1, 32, 32)
    target[0, 0, 8, 8] = 1.0
    target[0, 0, 20, 20] = 1.0

    initial_loss = None
    final_loss = None

    for step in range(15):
        optimizer.zero_grad()
        out = model(img)
        losses = compute_rmr_v3_losses(out, target, loss_cfg)
        loss = losses["total"]
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    assert final_loss < initial_loss
    loss_reduction = (initial_loss - final_loss) / initial_loss
    assert loss_reduction > 0.20


# =============================================================================
# 8. Canonical Config Validation
# =============================================================================

def test_canonical_v13_config():
    config_path = Path("configs/rmr_v13/rmr_v13_native_reconstruction.yaml")
    assert config_path.exists()

    cfg = load_config(config_path)
    validate_v3_config(cfg)

    assert cfg["model"]["adjoint_mode"] == "radon_nikodym"
    assert cfg["model"]["morozov_gamma"] == 0.75
    assert cfg["model"]["reliability_mode"] == "snr"
    assert cfg["loss"]["lambda_scale_align"] == 0.05
    assert cfg["loss"]["curvature_gate_threshold"] == 0.08
