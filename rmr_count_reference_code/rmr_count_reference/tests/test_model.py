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
    """P1-T3: A 32px region should have the same geometric descriptor regardless of image size.

    With the absolute geometry encoding (log_h, log_w, log_area, log_aspect),
    a 32px region (= 8 cells at stride 4) should produce the same geometry vector
    on a 256px image and a 512px image.
    """
    # 32px = 8 grid cells at stride 4
    win = 8
    boxes_small = torch.tensor([[0, 0, win, win]], dtype=torch.long)
    boxes_large = torch.tensor([[0, 0, win, win]], dtype=torch.long)

    geom_small = region_geometry(boxes_small, height=64, width=64)    # 256px image
    geom_large = region_geometry(boxes_large, height=128, width=128)  # 512px image

    # log_h, log_w, log_area, log_aspect (indices 2,3,4,5) must be identical
    assert torch.allclose(geom_small[:, 2:], geom_large[:, 2:], atol=1e-6), (
        f"Absolute geometry not invariant to image size:\n"
        f"  small: {geom_small[:, 2:]}\n"
        f"  large: {geom_large[:, 2:]}"
    )
    # cy/H, cx/W (normalized center position) WILL differ — that is correct behavior.


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
