import torch

from hpc.models.backbone import MobileNetV4Backbone
from hpc.models.hpc_lite import HPCLite
from hpc.models.neck import AdditiveFPNNeck


def test_model_forward_and_positivity():
    """HPCLite forward output must have shape (B, 1, H/4, W/4) and be strictly positive."""
    model = HPCLite(pretrained=False, neck_width=32)
    model.eval()

    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        d_map = model(x)

    assert d_map.shape == (2, 1, 64, 64)
    assert (d_map >= 1e-8).all()
    assert torch.isfinite(d_map).all()


def test_parameter_budget():
    """Total deployed parameters must be under 1.5M (target ~0.35M)."""
    model = HPCLite(pretrained=False, neck_width=32)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 1_500_000, f"Parameter count {n_params} exceeds 1.5M budget"
    assert n_params < 500_000, f"Parameter count {n_params} exceeds 0.5M lightweight budget"


def test_arbitrary_padded_inference():
    """predict() must handle arbitrary resolutions not divisible by 32 or 4."""
    model = HPCLite(pretrained=False, neck_width=32)
    model.eval()

    # Image dimensions not divisible by 32 or 4
    x = torch.randn(1, 3, 317, 411)
    with torch.no_grad():
        count, d_valid = model.predict(x, pad_multiple=32)

    assert count.ndim == 1 and count.shape[0] == 1
    assert torch.isfinite(count).all()
    assert d_valid.shape == (1, 1, 80, 103)
