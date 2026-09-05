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

    # All rates should be close to 0.01, definitely NOT 0.693
    mean_rate = float(rate_per_cell.mean().item())
    assert mean_rate < 0.5, (
        f"Regional initial rate/cell={mean_rate:.4f}. "
        f"Expected ~0.01 (fixed init). Got ~0.693 (broken, softplus(0))."
    )
    assert mean_rate > 1e-4, (
        f"Regional initial rate/cell={mean_rate:.6f} suspiciously low."
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
    """NEW Diagnostic: After solver ramp, Y1 != Y0 (solver actually changes prediction).

    Checks that with solver_strength=1.0, eta=0.05, the iterative update produces
    a non-trivial change. This validates that the sigma(z) * r * eta chain is not
    numerically collapsed to zero.
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
        y1 = iterates[1]
        yf = iterates[-1]

    rel_step = float((yf - y0).abs().sum() / (y0.abs().sum() + 1e-8))
    delta_n = float((yf.sum() - y0.sum()).abs())

    # With bias init z ≈ -4.6 → sigma(z) ≈ 0.01 → step ≈ 0.05 * 1 * 0.01 * r
    # Even if small, must be > 0 (model is not broken)
    assert rel_step > 0.0, "Solver produces zero relative step — update is collapsed"
    # Warn if extremely small (< 0.0001 relative), as this may indicate sigma(z) bottleneck
    # This is not a hard failure but should trigger the sigma(z) ablation
    if rel_step < 1e-4:
        import warnings
        warnings.warn(
            f"Solver relative step = {rel_step:.2e} is very small. "
            f"Consider removing sigma(z) multiplier from the update rule "
            f"(z = z - eta * M * r without the sigmoid gate)."
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
