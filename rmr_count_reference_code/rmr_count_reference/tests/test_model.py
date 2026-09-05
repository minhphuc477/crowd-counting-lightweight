import torch

from rmr_count.model import RMRConfig, RMRCount
from rmr_count.operators import regional_sum


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
    r = model._rmr_field(y, b, regions)
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
