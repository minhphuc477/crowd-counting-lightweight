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
    mass = torch.ones(10, 10)  # 100 source cells, 100 target points -> 10,000 elements
    with pytest.raises(MemoryError, match="OT-M transport matrix requires"):
        otm_localize(mass, max_transport_elements=100, max_source_points=None)
