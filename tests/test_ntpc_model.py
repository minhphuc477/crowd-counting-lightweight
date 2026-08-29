import torch

from hpc.models.backbone import MobileNetV4Backbone
from hpc.models.factory import build_model_from_config
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
    """Total deployed parameters must match the ~0.35M Carrier budget."""
    model = HPCLite(pretrained=False, neck_width=32)
    n_params = sum(p.numel() for p in model.parameters())
    assert 349_000 <= n_params <= 352_000, f"Parameter count {n_params} drifted outside [349k, 352k]"


def test_factory_build_model_equivalence():
    """build_model_from_config must build model with matching architecture."""
    cfg = {
        "model": {
            "backbone": "mobilenetv4_conv_small_050",
            "pretrained": False,
            "neck_width": 32,
            "context_dilations": [1, 2, 3],
            "use_p8_context": False,
            "use_repblock": False,
            "eps_d": 1e-8,
        }
    }
    model = build_model_from_config(cfg)
    assert isinstance(model, HPCLite)
    n_params = sum(p.numel() for p in model.parameters())
    assert 349_000 <= n_params <= 352_000


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


def test_padding_multiple_invariance_diagnostic():
    """Diagnostic test: measure count variance across pad_multiple=16, 32, 64 for non-divisible image."""
    torch.manual_seed(42)
    model = HPCLite(pretrained=False, neck_width=32)
    model.eval()

    # Arbitrary non-divisible image (317, 411)
    x = torch.randn(1, 3, 317, 411)
    with torch.no_grad():
        count_16, d_16 = model.predict(x, pad_multiple=16)
        count_32, d_32 = model.predict(x, pad_multiple=32)
        count_64, d_64 = model.predict(x, pad_multiple=64)

    # Valid output shapes must be identical (80, 103) regardless of pad_multiple
    assert d_16.shape == d_32.shape == d_64.shape == (1, 1, 80, 103)
    diff_32_16 = abs(count_32.item() - count_16.item())
    diff_64_32 = abs(count_64.item() - count_32.item())
    assert diff_32_16 < 0.5 and diff_64_32 < 0.5, (
        f"Significant count drift across padding policies: 16={count_16.item():.3f}, "
        f"32={count_32.item():.3f}, 64={count_64.item():.3f}"
    )
