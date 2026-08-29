import pytest
import torch

from hpc.metrics.otm import OTMConfig, otm_localize


def test_official_otm_is_deterministic_for_one_seed():
    mass = torch.zeros(32, 32)
    mass[4, 5] = 0.8
    mass[20, 22] = 1.2
    first = otm_localize(mass, seed=9, max_source_points=None)
    second = otm_localize(mass, seed=9, max_source_points=None)
    assert torch.equal(first, second)
    assert len(first) == round(float(mass.sum()))


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


def test_deprecated_fixed_epsilon_api_fails_instead_of_mislabeling_otm():
    with pytest.raises(ValueError, match="old approximation"):
        otm_localize(torch.ones(4, 4), epsilon=0.02)


def test_otm_config_rejects_invalid_scaling():
    with pytest.raises(ValueError):
        OTMConfig(ot_scaling=1.0)
