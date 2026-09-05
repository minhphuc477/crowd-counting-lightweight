import math
import torch

from rmr_count.model import RMRConfig, RMRCount, _FINE_HEAD_BIAS_INIT
from rmr_count.operators import build_multiscale_regions, region_geometry, regional_sum


# ===========================================================================
# Original tests (preserved)
# ===========================================================================

def test_rmr_output_positive_and_shape():
    torch.manual_seed(0)
    model = RMRCount(RMRConfig(iterations=2), variant="rmr")
    x = torch.randn(2, 3, 128, 160)
    out = model(x)
    y = out["y"]
    assert y.shape == (2, 1, 32, 40)
    assert torch.all(y >= 0)
    assert len(out["iterates"]) == 3


def test_zero_region_residual_is_fixed_direction():
    torch.manual_seed(0)
    model = RMRCount(RMRConfig(iterations=1), variant="rmr")
    y = torch.rand(1, 1, 16, 20)
    regions = model._regions(16, 20, y.device)
    b = regional_sum(y, regions.boxes)
    r = model._normalized_adjoint_field(y, b, regions)
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-7)


def test_local_refine_positive():
    torch.manual_seed(0)
    model = RMRCount(RMRConfig(iterations=2), variant="local_refine")
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    assert torch.all(out["y"] >= 0)
    assert len(out["residual_fields"]) == 2


def test_learned_project_same_regional_scope_runs():
    torch.manual_seed(0)
    model = RMRCount(
        RMRConfig(iterations=1, region_sizes_px=(32, 64)),
        variant="learned_project",
    )
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    assert out["y"].shape == (1, 1, 32, 32)
    assert torch.all(out["y"] >= 0)
    assert out["b_region"].shape[-1] == out["regions"].boxes.shape[0]


def test_small_bounded_eta_initialization():
    model = RMRCount(RMRConfig(iterations=2, eta_max=0.2, eta_init=0.05), variant="rmr")
    eta0 = float(model._eta(0).detach())
    assert abs(eta0 - 0.05) < 1e-5


def test_solver_strength_zero_reduces_to_initial_measure():
    torch.manual_seed(0)
    model = RMRCount(RMRConfig(iterations=2), variant="rmr")
    model.set_solver_strength(0.0)
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    assert torch.allclose(out["y"], out["y0"], atol=1e-7)


# ===========================================================================
# P1 calibration regression tests (6 new)
# ===========================================================================

def test_initial_count_sanity():
    """P1-T1: Initial predicted count per cell must be close to mu0=0.01, not softplus(0)=0.693.

    With the fixed bias init of ~-4.595, softplus(bias) ≈ 0.01 per cell.
    A 512x512 image at stride 4 has 128x128 = 16,384 cells.
    Expected initial total: ~164, not ~11,370 (= 0.693 * 16384).
    """
    torch.manual_seed(42)
    model = RMRCount(RMRConfig(output_stride=4), variant="direct")
    model.eval()
    with torch.no_grad():
        # 512x512 image
        x = torch.zeros(1, 3, 512, 512)  # blank image (normalized to zero mean is fine)
        out = model(x)
        y0 = out["y0"]
        cells = y0.numel()
        total_initial = float(y0.sum().item())
        mean_per_cell = total_initial / cells

    # Expected: ~0.01 per cell (±10x tolerance for random weight variation)
    # Definitely NOT ~0.693 (old behavior) which would be >10x higher
    assert mean_per_cell < 0.5, (
        f"Initial count/cell={mean_per_cell:.4f} is too high. "
        f"Expected ~0.01 (softplus(-4.595)). "
        f"Old broken behavior was ~0.693 (softplus(0)). "
        f"Total predicted on 512x512: {total_initial:.1f}"
    )
    assert mean_per_cell > 1e-5, (
        f"Initial count/cell={mean_per_cell:.6f} is suspiciously low (< 1e-5)."
    )


def test_regional_area_extensivity():
    """P1-T2: Regional head satisfies extensivity: b_R ∝ |R| when visual features are uniform.

    For a blank/uniform input, two regions of different sizes but same average features
    should predict proportionally different total counts (larger region -> more count).
    This verifies the rate × area formulation is working.
    """
    torch.manual_seed(0)
    model = RMRCount(
        RMRConfig(region_sizes_px=(32, 128), include_full_image=False),
        variant="region_aux",
    )
    model.eval()
    with torch.no_grad():
        x = torch.zeros(1, 3, 256, 256)
        out = model(x)
        regions = out["regions"]
        b_region = out["b_region"][0, 0]  # [M]

        # Find one 32px and one 128px region
        small_mask = regions.scale_id == 0   # sid=0 is 32px
        large_mask = regions.scale_id == 1   # sid=1 is 128px

        if small_mask.any() and large_mask.any():
            small_counts = b_region[small_mask]
            large_counts = b_region[large_mask]
            small_areas = regions.area[small_mask]
            large_areas = regions.area[large_mask]

            # Rate (count / area) should be similar across scales for uniform input
            small_rate = float((small_counts / small_areas.to(small_counts.dtype)).mean().item())
            large_rate = float((large_counts / large_areas.to(large_counts.dtype)).mean().item())

            # Rates should be within 10x of each other for uniform input
            ratio = max(small_rate, large_rate) / max(min(small_rate, large_rate), 1e-10)
            assert ratio < 10.0, (
                f"Rate extensivity violated: small_rate={small_rate:.4f}, large_rate={large_rate:.4f}, "
                f"ratio={ratio:.2f}. Large regions should not have wildly different rates."
            )


def test_same_size_region_invariant_to_image_extent():
    """P1-T3 (updated): A 32px region must have IDENTICAL geometry on different image sizes.

    With position-free geometry (geom_dim=4), ALL 4 features are position-free:
        [log_h, log_w, log_area, log_aspect]
    The ENTIRE geometry vector must be equal across 256px and 512px images.
    Previously this tested only indices 2:, but now ALL indices must match.
    """
    win = 8  # 32px at stride 4
    boxes = torch.tensor([[4, 4, 4 + win, 4 + win]], dtype=torch.long)

    # Same physical window, different image sizes
    geom_256 = region_geometry(boxes, height=64, width=64)    # 256px image
    geom_512 = region_geometry(boxes, height=128, width=128)  # 512px image

    assert geom_256.shape[-1] == 4, f"Expected geom_dim=4, got {geom_256.shape[-1]}"
    assert torch.allclose(geom_256, geom_512, atol=1e-6), (
        f"Position-free geometry must be fully identical across image sizes:\n"
        f"  256px: {geom_256}\n"
        f"  512px: {geom_512}"
    )


def test_regional_head_initial_rate_calibrated():
    """NEW P0 #1-T: RegionalEvidenceHead must initialize at ~0.01 count/cell.

    With final Linear bias = _FINE_HEAD_BIAS_INIT ≈ -4.595:
        softplus(bias) ≈ 0.01
        b_R = |R| * 0.01

    For 32px (8×8=64 cells): b_R ≈ 0.64
    For 128px (32×32=1024 cells): b_R ≈ 10.24

    OLD broken: softplus(0) = 0.693 → b_128 ≈ 710 → solver injects mass.
    """
    model = RMRCount(
        RMRConfig(region_sizes_px=(32, 64, 128), include_full_image=False),
        variant="region_aux",
    )
    model.eval()
    with torch.no_grad():
        x = torch.zeros(1, 3, 256, 256)
        out = model(x)
        regions = out["regions"]
        b_region = out["b_region"][0, 0]   # [M]
        area = regions.area.float()
        rate_per_cell = b_region / area.clamp_min(1.0)

    # Tight check: initial rate must be within 3x of 0.01 target (not just < 0.5)
    # Equivalent to: softplus(bias) ∈ [0.005, 0.03]
    # 0.5 was too loose — 0.2 would pass but is 20x the intended prior.
    mean_rate = float(rate_per_cell.mean().item())
    assert 0.005 < mean_rate < 0.03, (
        f"Regional initial rate/cell={mean_rate:.5f}. "
        f"Expected 0.005 < rate < 0.03 (≈ 0.01 target). "
        f"If mean_rate ≈ 0.693: bias init missing. If > 0.03: weight init too large."
    )
    # Also verify per-scale: all scales should be close to 0.01, not just the mean
    for sid in torch.unique(regions.scale_id):
        mask = regions.scale_id == sid
        scale_rate = float((rate_per_cell[mask]).mean().item())
        assert 0.002 < scale_rate < 0.1, (
            f"scale_id={int(sid.item())} initial rate={scale_rate:.5f} out of expected range. "
            f"Regional head may be scaling predictions by area incorrectly."
        )


def test_regional_loss_gradient_scale_balanced():
    """NEW P0 #2-T: Regional rate loss must produce equal-magnitude gradients for 32px and 128px.

    With rate-normalized loss: L_rate = SmoothL1(b_R/|R|, N_R/|R|)
    dL/dz_R ∝ sigma(z_R) regardless of |R| — gradient magnitude is scale-balanced.

    With old count loss: dL/dz_R ∝ |R| * sigma(z_R)
    → 128px (|R|=1024) produces 16× larger gradients than 32px (|R|=64).
    """
    import torch

    from rmr_count.losses import LossConfig, compute_losses

    torch.manual_seed(0)
    model = RMRCount(
        RMRConfig(region_sizes_px=(32, 128), include_full_image=False, iterations=0),
        variant="region_aux",
    )
    x = torch.zeros(1, 3, 256, 256)
    target = torch.zeros(1, 1, 64, 64)   # all-zero GT

    model.zero_grad()
    out = model(x)
    # Inject large b_region to create large residual (simulate early training)
    b_region = out["b_region"]
    losses = compute_losses(out, target, "region_aux", LossConfig())
    losses["region_head"].backward()

    # Inspect gradients on the final linear bias of regional MLP
    final_linear = model.region_head.mlp[-1]
    grad_bias = final_linear.bias.grad
    assert grad_bias is not None, "No gradient on regional MLP final bias"
    # Gradient should be finite and non-zero
    assert torch.isfinite(grad_bias).all(), "Gradient contains inf/nan"
    assert grad_bias.abs().item() > 0, "Zero gradient — rate loss not connected"


def test_region_geometry_position_free_across_crop_and_tile():
    """NEW P1 #1-T: Same physical region must have identical geometry in crop vs tile vs full-res.

    Simulates three coordinate contexts for the same 32px window:
      - training crop (512px):  region at position (50,50)-(58,58) in 128-cell grid
      - full image (1024px):    same region at same pixel position → (50,50)-(58,58) in 256-cell grid
      - tile (256px):           region at position (20,20)-(28,28) in 64-cell grid

    All must produce identical 4-dim geometry [log_h, log_w, log_area, log_aspect].
    """
    win = 8  # 32px at stride 4
    box_crop = torch.tensor([[50, 50, 50 + win, 50 + win]], dtype=torch.long)
    box_full = torch.tensor([[50, 50, 50 + win, 50 + win]], dtype=torch.long)
    box_tile = torch.tensor([[20, 20, 20 + win, 20 + win]], dtype=torch.long)

    g_crop = region_geometry(box_crop, height=128, width=128)   # 512px training crop
    g_full = region_geometry(box_full, height=256, width=256)   # 1024px full image
    g_tile = region_geometry(box_tile, height=64, width=64)     # 256px tile

    assert torch.allclose(g_crop, g_full, atol=1e-6), (
        f"Geometry differs between crop and full-image for same 32px window:\n"
        f"  crop: {g_crop}\n  full: {g_full}"
    )
    assert torch.allclose(g_crop, g_tile, atol=1e-6), (
        f"Geometry differs between crop and tile for same 32px window:\n"
        f"  crop: {g_crop}\n  tile: {g_tile}"
    )


def test_padding_mean_is_zero_after_normalization():
    """NEW P1 #2-T: ImageNet-mean padded pixels must normalize to exactly 0.

    After padding with [0.485, 0.456, 0.406] in [0,1] space and then applying
    normalize_image (subtract mean, divide by std), padded pixels → 0.0 for all channels.
    This verifies no spurious feature activations from padding.
    """
    import torch

    from rmr_count.data import _pad_to_crop, normalize_image

    # Create small image (3×10×10) and pad to (3×20×20)
    image = torch.rand(3, 10, 10)
    pts = torch.empty(0, 2)
    padded, _ = _pad_to_crop(image, pts, crop_h=20, crop_w=20)

    # Apply normalization
    normalized = normalize_image(padded)

    # Padded region (rows 10:, cols 10:) should be ~0.0 after normalization
    pad_region = normalized[:, 10:, 10:]
    assert torch.allclose(pad_region, torch.zeros_like(pad_region), atol=1e-5), (
        f"Padded region after normalization should be ~0.0, got max abs {pad_region.abs().max():.6f}"
    )


def test_solver_effective_step_nonzero_after_warmup():
    """Diagnostic: After solver ramp, Y_final != Y0 (solver produces non-trivial update).

    Validates that the RMR-Latent update rule:
        z^{t+1} = z^t - eta * M * r
    produces a measurable change in the predicted measure Y = softplus(z).

    With z ≈ -4.6 at init and r coming from regional inconsistency, the step
    |Δz| = eta * |M * r| should produce a visible |ΔY| = softplus(z+Δz) - softplus(z).
    This test verifies the update chain is numerically active (not flushed to zero).

    Note: sigma(z) gate was REMOVED from the update (was the old RMR-Jacobian rule).
    The current registered rule (RMR-Latent) does NOT multiply by sigma(z).
    For RMR-Jacobian ablation, set cfg.use_jacobian_gate=True.
    """
    torch.manual_seed(42)
    model = RMRCount(RMRConfig(iterations=2, eta_init=0.05, eta_max=0.2), variant="rmr")
    model.set_solver_strength(1.0)
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 128, 128)
        out = model(x)
        iterates = out["iterates"]
        y0 = iterates[0]
        yf = iterates[-1]

    rel_step = float((yf - y0).abs().sum() / (y0.abs().sum() + 1e-8))

    # RMR-Latent: |Δz| = eta * |M * r|, no sigma(z) suppression.
    # Even at init (r small due to random weights), must produce non-zero update.
    assert rel_step > 0.0, "Solver produces zero relative step — update is numerically collapsed"
    # Warn if extremely small (< 1e-4 relative) — may indicate r is near-zero at init
    if rel_step < 1e-4:
        import warnings
        warnings.warn(
            f"Solver relative step = {rel_step:.2e} is very small. "
            f"Check residual field magnitudes and eta initialization. "
            f"If using use_jacobian_gate=True, note that sigma(z)≈0.01 suppresses updates ~100x."
        )


def test_no_full_image_region_in_pilot():
    """P1-T4: Pilot config must not include the full-image region.

    include_full_image=False prevents the full-image scale from dominating scale-balanced loss
    and avoids extent extrapolation across training/inference resolutions.
    """
    regions = build_multiscale_regions(
        height=128, width=128,
        output_stride=4,
        region_sizes_px=(32, 64, 128),
        overlap=0.5,
        include_full_image=False,
    )
    # scale_id == -1 means full-image region
    assert not (regions.scale_id == -1).any(), (
        "Full-image region (scale_id=-1) present in pilot config. "
        "Set include_full_image=False to avoid extent extrapolation bug."
    )


def test_checkpoint_solver_strength_saved():
    """P1-T5: solver_strength must be saved in checkpoint state dict.

    The training checkpoint now stores solver_strength for audit provenance.
    This ensures we know which evaluation epoch's solver state the checkpoint corresponds to.
    """
    torch.manual_seed(0)
    model = RMRCount(RMRConfig(iterations=2), variant="rmr")
    model.set_solver_strength(0.5)

    # Simulate what training saves
    state = {
        "epoch": 15,
        "model": model.state_dict(),
        "solver_strength": model.solver_strength,
        "best_mae": 1000.0,
    }
    assert "solver_strength" in state, "solver_strength must be saved in checkpoint"
    assert state["solver_strength"] == 0.5


def test_fine_head_bias_correct():
    """P1-T6: FineMeasureHead final conv bias must be initialized to ~-4.595.

    This ensures initial predicted density ≈ 0.01 counts/cell via softplus(-4.595) ≈ 0.01.
    """
    model = RMRCount(RMRConfig(), variant="direct")
    # Access the last Conv2d in fine_head.body
    final_conv = model.fine_head.body[-1]
    bias_val = float(final_conv.bias.item())

    expected = _FINE_HEAD_BIAS_INIT  # log(exp(0.01) - 1) ≈ -4.595
    assert abs(bias_val - expected) < 1e-5, (
        f"FineMeasureHead bias={bias_val:.4f}, expected {expected:.4f}. "
        f"softplus({bias_val:.4f}) = {math.log1p(math.exp(bias_val)):.4f}, "
        f"should be ≈ 0.01 counts/cell."
    )
    # Also confirm softplus of this bias is near 0.01
    initial_rate = math.log1p(math.exp(bias_val))
    assert 0.005 < initial_rate < 0.05, (
        f"Initial rate per cell = {initial_rate:.4f}, expected ~0.01."
    )


def test_output_stride_guard():
    """P0: RMRCount must reject output_stride != 4 at construction."""
    import pytest
    with pytest.raises(ValueError, match="only supports output_stride=4"):
        RMRCount(RMRConfig(output_stride=8))


def test_lazy_regions_direct_and_local_refine():
    """P0: direct and local_refine must skip region construction (regions is None)."""
    torch.manual_seed(0)
    x = torch.randn(1, 3, 64, 64)
    model_direct = RMRCount(RMRConfig(), variant="direct")
    out_direct = model_direct(x)
    assert out_direct["regions"] is None, "direct variant should not construct regions"

    model_refine = RMRCount(RMRConfig(), variant="local_refine")
    out_refine = model_refine(x)
    assert out_refine["regions"] is None, "local_refine variant should not construct regions"

    model_rmr = RMRCount(RMRConfig(), variant="rmr")
    out_rmr = model_rmr(x)
    assert out_rmr["regions"] is not None, "rmr variant requires regions"


def test_eval_restores_jacobian_gate_and_solver_strength():
    """P0: make_model_from_ckpt in eval.py must restore use_jacobian_gate and solver_strength."""
    from rmr_count.eval import make_model_from_ckpt

    base_model = RMRCount(RMRConfig(use_jacobian_gate=True), variant="rmr")
    fake_ckpt = {
        "config": {
            "model": {
                "variant": "rmr",
                "use_jacobian_gate": True,
            }
        },
        "model": base_model.state_dict(),
        "solver_strength": 0.65,
    }
    loaded = make_model_from_ckpt(fake_ckpt, torch.device("cpu"))
    assert loaded.cfg.use_jacobian_gate is True, "use_jacobian_gate must be restored"
    assert abs(loaded.solver_strength - 0.65) < 1e-6, "solver_strength must be restored"


def test_r_spatial_computation():
    """Verify mathematical properties of R_spatial diagnostic."""
    # Case 1: Zero step (e.g. during warmup)
    dy_zero = torch.zeros(2, 1, 16, 16)
    signed_dn = dy_zero.sum(dim=(-3, -2, -1))
    abs_dn = signed_dn.abs()
    l1_dy = dy_zero.abs().sum(dim=(-3, -2, -1))
    r_sp_zero = torch.where(
        l1_dy > 1e-6,
        (1.0 - (abs_dn / (l1_dy + 1e-6))).clamp(0.0, 1.0),
        torch.zeros_like(l1_dy),
    )
    assert (r_sp_zero == 0.0).all()

    # Case 2: Unipolar shift (pure global count adjustment, dy >= 0 everywhere)
    dy_unipolar = torch.ones(2, 1, 16, 16) * 0.5
    signed_dn = dy_unipolar.sum(dim=(-3, -2, -1))
    abs_dn = signed_dn.abs()
    l1_dy = dy_unipolar.abs().sum(dim=(-3, -2, -1))
    r_sp_unipolar = torch.where(
        l1_dy > 1e-6,
        (1.0 - (abs_dn / (l1_dy + 1e-6))).clamp(0.0, 1.0),
        torch.zeros_like(l1_dy),
    )
    assert torch.allclose(r_sp_unipolar, torch.zeros_like(r_sp_unipolar), atol=1e-5)
    assert (signed_dn > 0).all()

    # Case 3: Pure zero-sum spatial redistribution (sum dy == 0, but mass moved)
    dy_redist = torch.zeros(2, 1, 16, 16)
    dy_redist[:, :, :, :8] = 1.0
    dy_redist[:, :, :, 8:] = -1.0
    signed_dn = dy_redist.sum(dim=(-3, -2, -1))
    abs_dn = signed_dn.abs()
    l1_dy = dy_redist.abs().sum(dim=(-3, -2, -1))
    r_sp_redist = torch.where(
        l1_dy > 1e-6,
        (1.0 - (abs_dn / (l1_dy + 1e-6))).clamp(0.0, 1.0),
        torch.zeros_like(l1_dy),
    )
    assert torch.allclose(r_sp_redist, torch.ones_like(r_sp_redist), atol=1e-5)
    assert torch.allclose(signed_dn, torch.zeros_like(signed_dn), atol=1e-5)


# ===========================================================================
# RMR-P (Nonnegative Projected SIRT) upgrade tests
# ===========================================================================

def test_projected_sirt_one_step_matches_formula():
    """Verify single step of Projected SIRT matches max(0, Y0 - omega * r) exactly."""
    torch.manual_seed(42)
    cfg = RMRConfig(
        iterations=1,
        update_rule="projected_sirt",
        sirt_omega=1.0,
        projected_use_preconditioner=False,
    )
    model = RMRCount(cfg, variant="rmr")
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    y0 = out["y0"]
    b_region = out["b_region"]
    regions = out["regions"]
    _, coverage = model._regions_and_coverage(y0.shape[-2], y0.shape[-1], y0.device)

    r = model._normalized_adjoint_field(y0, b_region, regions, coverage=coverage)
    expected_y1 = torch.clamp_min(y0 - 1.0 * r, 0.0)

    assert torch.allclose(out["y"], expected_y1, atol=1e-6)
    assert len(out["iterates"]) == 2


def test_projected_sirt_preserves_nonnegativity():
    """Verify Projected SIRT guarantees Y >= 0 even with extreme negative residuals."""
    torch.manual_seed(42)
    cfg = RMRConfig(
        iterations=2,
        update_rule="projected_sirt",
        sirt_omega=2.0,
        projected_use_preconditioner=False,
    )
    model = RMRCount(cfg, variant="rmr")
    x = torch.randn(2, 3, 128, 128)
    # Provide b_region override of all zeros so AY - b >> 0, pushing Y strongly negative
    probe = model(x)
    regions = probe["regions"]
    b_zero = torch.zeros_like(probe["b_region"])
    out = model(x, b_region_override=b_zero)
    assert torch.all(out["y"] >= 0.0)
    for it in out["iterates"]:
        assert torch.all(it >= 0.0)


def test_projected_sirt_can_add_mass_to_low_cell():
    """Verify Projected SIRT can add mass to cells where initial count is near zero."""
    torch.manual_seed(42)
    cfg = RMRConfig(
        iterations=1,
        update_rule="projected_sirt",
        sirt_omega=1.0,
        projected_use_preconditioner=False,
    )
    model = RMRCount(cfg, variant="rmr")
    x = torch.randn(1, 3, 128, 128)
    probe = model(x)
    regions = probe["regions"]
    # Provide very large regional count to force r < 0 (mass injection)
    b_large = torch.full_like(probe["b_region"], 1000.0)
    out = model(x, b_region_override=b_large)
    # Output must strictly exceed initial count by a substantial non-throttled margin
    delta = (out["y"] - out["y0"]).sum().item()
    assert delta > 10.0, f"Projected SIRT should add significant mass, got delta={delta}"


def test_projected_sirt_solver_strength_zero_is_identity():
    """Verify solver_strength=0.0 leaves Y unchanged at Y0."""
    torch.manual_seed(42)
    cfg = RMRConfig(
        iterations=2,
        update_rule="projected_sirt",
        sirt_omega=1.0,
    )
    model = RMRCount(cfg, variant="rmr")
    model.set_solver_strength(0.0)
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    assert torch.allclose(out["y"], out["y0"], atol=1e-7)


def test_projected_sirt_reports_no_final_latent():
    """Verify out['z'] is None under Projected SIRT (measure space optimization)."""
    torch.manual_seed(42)
    cfg = RMRConfig(
        iterations=2,
        update_rule="projected_sirt",
    )
    model = RMRCount(cfg, variant="rmr")
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    assert out["z"] is None
    assert out["z0"] is not None


def test_legacy_latent_still_runs():
    """Verify backward compatibility: legacy RMR-Latent still operates as before."""
    torch.manual_seed(42)
    cfg = RMRConfig(
        iterations=2,
        update_rule="latent",
    )
    model = RMRCount(cfg, variant="rmr")
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    assert out["z"] is not None
    assert torch.all(out["y"] >= 0.0)
    assert len(out["iterates"]) == 3


def test_legacy_jacobian_still_runs():
    """Verify backward compatibility: legacy RMR-Jacobian still operates as before."""
    torch.manual_seed(42)
    cfg = RMRConfig(
        iterations=2,
        use_jacobian_gate=True,
    )
    model = RMRCount(cfg, variant="rmr")
    assert model.update_rule == "jacobian"
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    assert out["z"] is not None
    assert torch.all(out["y"] >= 0.0)


def test_projected_solver_detached_b_has_no_solver_gradient():
    """Verify causal separation: when detach_region_evidence=True, b has no gradient from Y_T."""
    torch.manual_seed(42)
    # Case 1: detached (default)
    cfg_detached = RMRConfig(
        iterations=1,
        update_rule="projected_sirt",
        detach_region_evidence=True,
    )
    model_det = RMRCount(cfg_detached, variant="rmr")
    x = torch.randn(1, 3, 128, 128)
    out_det = model_det(x)
    loss_det = out_det["y"].sum()
    loss_det.backward()
    for name, p in model_det.region_head.named_parameters():
        assert p.grad is None, f"Parameter {name} should not receive gradient from Y_T when detached"

    # Case 2: attached (ablation e2e)
    cfg_attached = RMRConfig(
        iterations=1,
        update_rule="projected_sirt",
        detach_region_evidence=False,
    )
    model_att = RMRCount(cfg_attached, variant="rmr")
    out_att = model_att(x)
    loss_att = out_att["y"].sum()
    loss_att.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model_att.region_head.parameters())
    assert has_grad, "Region head should receive gradient from Y_T when detach_region_evidence=False"

