import pytest
import torch

from rmr_v2.losses import LossConfig, compute_losses
from rmr_v2.model import RMRConfig, RMRCount, count_parameters


def test_rmr_v2_model_and_losses():
    cfg = RMRConfig(output_stride=4, feature_width=32, backbone_name="tiny", iterations=2)
    model = RMRCount(cfg, variant="rmr")
    model.eval()

    x = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
    out = model(x)

    assert "y" in out
    assert "y0" in out
    assert "regions" in out
    assert out["y"].shape == (1, 1, 16, 16)
    assert (out["y"] >= 0).all()

    target_y = torch.zeros((1, 1, 16, 16), dtype=torch.float32)
    target_y[0, 0, 4, 4] = 5.0

    losses = compute_losses(out, target_y, variant="rmr", cfg=LossConfig())
    assert "total" in losses
    assert "count" in losses
    assert "flat_dm16" in losses
    assert "cell" in losses
    assert "region_head" in losses
    assert torch.isfinite(losses["total"])


def test_rmr_v2_backward_compatibility():
    import rmr_count
    import rmr_v2
    # Ensure RMRConfig and RMRCount in rmr_v2 and rmr_count match signatures
    cfg1 = rmr_count.model.RMRConfig()
    cfg2 = rmr_v2.RMRConfig()
    assert cfg1.output_stride == cfg2.output_stride
    assert cfg1.feature_width == cfg2.feature_width
