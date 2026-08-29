import pytest
import torch

from hpc.models.backbone import MobileNetV4Backbone
from hpc.models.factory import assert_checkpoint_compatible, build_model_from_config
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


def test_direct_arbitrary_resolution():
    """HPCLite forward must process arbitrary resolutions directly without zero-padding distortion."""
    model = HPCLite(pretrained=False, neck_width=32).eval()
    x = torch.randn(1, 3, 317, 411)
    with torch.no_grad():
        d = model(x)
    assert d.shape == (1, 1, 80, 103)
    assert torch.isfinite(d).all()


def test_arbitrary_direct_and_padded_inference():
    """predict() must handle arbitrary resolutions directly (pad_multiple=None) or with padding."""
    model = HPCLite(pretrained=False, neck_width=32)
    model.eval()

    # Image dimensions not divisible by 32 or 4
    x = torch.randn(1, 3, 317, 411)
    with torch.no_grad():
        count_direct, d_direct = model.predict(x, pad_multiple=None)
        count_padded, d_padded = model.predict(x, pad_multiple=32)

    assert count_direct.ndim == 1 and count_direct.shape[0] == 1
    assert torch.isfinite(count_direct).all()
    assert d_direct.shape == (1, 1, 80, 103)

    assert count_padded.ndim == 1 and count_padded.shape[0] == 1
    assert torch.isfinite(count_padded).all()
    assert d_padded.shape == (1, 1, 80, 103)


def test_checkpoint_compatibility_assertion():
    """assert_checkpoint_compatible must accept matching configs and reject mismatches."""
    cfg = {
        "model": {"backbone": "mobilenetv4_conv_small_050", "neck_width": 32},
        "dataset": {"name": "sha", "part": "part_A", "coordinate_base": 1},
    }

    # Matching checkpoint
    assert_checkpoint_compatible({"config": cfg}, cfg)

    # Model architecture mismatch
    bad_model_ckpt = {
        "config": {
            "model": {"backbone": "mobilenetv4_conv_small_050", "neck_width": 64},
            "dataset": {"name": "sha", "part": "part_A", "coordinate_base": 1},
        }
    }
    with pytest.raises(ValueError, match="Model config mismatch"):
        assert_checkpoint_compatible(bad_model_ckpt, cfg)

    # Dataset preprocessing mismatch
    bad_ds_ckpt = {
        "config": {
            "model": {"backbone": "mobilenetv4_conv_small_050", "neck_width": 32},
            "dataset": {"name": "sha", "part": "part_B", "coordinate_base": 1},
        }
    }
    with pytest.raises(ValueError, match="Dataset/preprocessing config mismatch"):
        assert_checkpoint_compatible(bad_ds_ckpt, cfg)

