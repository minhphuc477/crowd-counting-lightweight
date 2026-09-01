import pytest
import torch

from hpc.metrics.otm import OTMConfig, _epsilon_scaling_transport_plan, otm_localize


def test_official_otm_is_deterministic_for_one_seed():
    mass = torch.zeros(32, 32)
    mass[4, 5] = 0.8
    mass[20, 22] = 1.2
    first = otm_localize(mass, seed=9, max_source_points=None)
    second = otm_localize(mass, seed=9, max_source_points=None)
    assert torch.equal(first, second)
    assert len(first) == round(float(mass.sum()))


def test_otm_cell_center_coordinate_offset():
    """Stride-4 mass cell at (row=5, col=5) should place source mass at pixel center (21.5, 21.5)."""
    mass = torch.zeros(16, 16)
    mass[5, 5] = 1.0  # Exactly 1 point
    points = otm_localize(mass, output_stride=4, outer_iterations=1, seed=42, max_source_points=None)
    assert len(points) == 1
    # Pixel center coordinate of stride-4 cell 5 is 5 * 4 + 1.5 = 21.5
    assert points[0, 0].item() == pytest.approx(21.5, abs=1e-3)
    assert points[0, 1].item() == pytest.approx(21.5, abs=1e-3)


def test_otm_simultaneous_dual_update_numerical_invariants():
    """Verify simultaneous dual update produces valid non-negative transport plan matching costs."""
    torch.manual_seed(42)
    # S=2 source points, T=2 target points with diagonal cost 0 and off-diagonal cost 16.0
    source_weight = torch.tensor([[[0.5], [0.5]]], dtype=torch.float32)  # [1, 2, 1]
    target_weight = torch.tensor([[[0.5], [0.5]]], dtype=torch.float32)  # [1, 2, 1]
    cost = torch.tensor([[[0.0, 16.0], [16.0, 0.0]]], dtype=torch.float32)  # [1, S=2, T=2]

    plan = _epsilon_scaling_transport_plan(
        source_weight=source_weight,
        target_weight=target_weight,
        cost=cost,
        blur=0.05,
        scaling=0.75,
    )
    assert plan.shape == (1, 2, 2)
    assert torch.isfinite(plan).all()
    assert (plan >= 0).all()
    # Diagonal matches (cost=0) must receive virtually all transport mass over off-diagonal (cost=16)
    assert plan[0, 0, 0] > plan[0, 0, 1] * 50
    assert plan[0, 1, 1] > plan[0, 1, 0] * 50
    # Marginals must sum to source and target weights
    torch.testing.assert_close(plan.sum(dim=2), source_weight.squeeze(-1), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(plan.sum(dim=1), target_weight.squeeze(-1), atol=1e-3, rtol=1e-3)


def test_otm_rounding_and_consistency_diagnostics():
    mass = torch.zeros(16, 16)
    mass[2, 3] = 1.2
    mass[8, 9] = 1.3
    points, diagnostics = otm_localize(
        mass, seed=42, max_source_points=None, return_diagnostics=True
    )
    assert len(points) == 3
    assert diagnostics["localized_count"] == 3
    assert diagnostics["cardinality_gap"] == pytest.approx(0.5)
    assert diagnostics["source_retained_mass_ratio"] == pytest.approx(1.0)


def test_softplus_source_sparsification_is_explicit_and_bounded():
    mass = torch.full((32, 32), 1e-8)
    mass[5, 5] = 1.0
    mass[20, 20] = 1.0
    _, diagnostics = otm_localize(
        mass,
        seed=42,
        max_source_points=64,
        return_diagnostics=True,
    )
    assert diagnostics["source_points"] <= 64
    assert diagnostics["source_retained_mass_ratio"] > 0.999


def test_otm_rejects_nan_and_inf():
    nan_mass = torch.full((8, 8), float("nan"))
    with pytest.raises(ValueError, match="NaN or Inf"):
        otm_localize(nan_mass)


def test_deprecated_fixed_epsilon_api_fails_instead_of_mislabeling_otm():
    with pytest.raises(ValueError, match="old approximation"):
        otm_localize(torch.ones(4, 4), epsilon=0.02)


def test_otm_config_rejects_invalid_scaling():
    with pytest.raises(ValueError):
        OTMConfig(ot_scaling=1.0)


def test_otm_memory_guard_raises_on_huge_transport():
    """OT-M should raise MemoryError when estimated transport matrix exceeds limit."""
    mass = torch.ones(10, 10)  # 100 target points; with max_transport_elements=50, even 1 source atom requires 100 elements
    with pytest.raises(MemoryError, match="OT-M transport matrix requires"):
        otm_localize(mass, max_transport_elements=50, max_source_points=None)


def test_otm_partial_edge_center():
    """OT-M source distribution on partial border cells must compute coordinate centers bounded by image_hw."""
    from hpc.metrics.otm import _source_distribution, OTMConfig

    mass = torch.zeros((75, 103))
    mass[74, 102] = 1.0
    config = OTMConfig(output_stride=4)

    weight, coord_yx, _ = _source_distribution(mass, config, image_hw=(298, 410))
    # Row 74 center: 0.5 * (296 + 297) = 296.5
    # Col 102 center: 0.5 * (408 + 409) = 408.5
    assert abs(float(coord_yx[0, 0]) - 296.5) < 1e-4
    assert abs(float(coord_yx[0, 1]) - 408.5) < 1e-4


def test_otm_effective_source_cap_prevents_memory_error():
    """OT-M should adapt effective source cap to respect max_transport_elements under high predicted count."""
    mass = torch.ones(64, 64) * 0.5  # sum = 2048
    # With 2048 target points and default max_transport_elements=5_000_000,
    # effective source cap adapts to ~2441 points, successfully avoiding MemoryError.
    pts = otm_localize(mass, outer_iterations=2, max_transport_elements=5_000_000)
    assert len(pts) == 2048


def test_otm_oracle_cardinality():
    """OT-M with target_point_count must decode exactly the specified oracle cardinality."""
    mass = torch.ones(20, 20) * 0.1  # sum = 40.0
    pts = otm_localize(mass, target_point_count=50, outer_iterations=2)
    assert len(pts) == 50


def test_otm_fullres_initialization_guard_and_explicit_stride_grid_mode():
    mass = torch.zeros(4, 4)
    mass[1, 1] = 1.0
    with pytest.raises(MemoryError, match="full-resolution initialization"):
        otm_localize(mass, image_hw=(100, 100), max_initialization_pixels=1_000)
    points, diagnostics = otm_localize(
        mass,
        image_hw=(100, 100),
        initialization_mode="stride_grid",
        max_initialization_pixels=1_000,
        return_diagnostics=True,
    )
    assert len(points) == 1
    assert diagnostics["config_initialization_mode"] == "stride_grid"


def test_otm_tall_map_tiny_source_cap():
    from hpc.metrics.otm import OTMConfig, _source_distribution

    mass = torch.ones(100, 1, dtype=torch.float32)
    cfg = OTMConfig(max_source_points=1)
    weights, coords, diag = _source_distribution(mass, cfg, image_hw=(400, 4))
    assert len(weights) == 1
    assert len(coords) == 1
    assert diag["source_coarse_height"] >= 1
    assert diag["source_coarse_width"] >= 1


