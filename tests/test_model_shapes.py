import pytest
import torch
from hpc.models.hpc_lite import HPCLite
from tools.profile_model import count_parameters


def test_t1_model_shapes_and_reductions():
    """T1: Check backbone reductions and head output stride 4."""
    model = HPCLite(pretrained=False, neck_width=32, truncate_backbone=True)
    model.eval()
    
    # Test 448x448 input
    x448 = torch.randn(2, 3, 448, 448)
    c4, c8, c16 = model.backbone(x448)
    assert c4.shape[-2:] == (112, 112), f"C4 shape expected (112, 112), got {c4.shape[-2:]}"
    assert c8.shape[-2:] == (56, 56), f"C8 shape expected (56, 56), got {c8.shape[-2:]}"
    assert c16.shape[-2:] == (28, 28), f"C16 shape expected (28, 28), got {c16.shape[-2:]}"
    
    d448 = model(x448)
    assert d448.shape == (2, 1, 112, 112), f"D shape expected (2, 1, 112, 112), got {d448.shape}"
    
    # Test 672x672 input
    x672 = torch.randn(1, 3, 672, 672)
    d672 = model(x672)
    assert d672.shape == (1, 1, 168, 168), f"D shape expected (1, 1, 168, 168), got {d672.shape}"


def test_t1_arbitrary_padded_inference():
    """T1 & Section 20: Arbitrary input resolution with reflect padding.

    The model pads to the next multiple of pad_multiple, then crops back to
    exactly ceil(H/4) x ceil(W/4) output cells. Floor division would silently
    drop border content on odd-height/width images (spec §4.6).
    """
    import math
    model = HPCLite(pretrained=False, neck_width=32, truncate_backbone=True)
    model.eval()

    # Odd resolution 513 x 387
    x_odd = torch.randn(1, 3, 513, 387)
    count, d_valid = model.predict(x_odd, pad_multiple=16)

    expected_h = math.ceil(513 / 4)  # 129
    expected_w = math.ceil(387 / 4)  # 97
    assert d_valid.shape == (1, 1, expected_h, expected_w), \
        f"Expected (1, 1, {expected_h}, {expected_w}), got {tuple(d_valid.shape)}"
    assert count.shape == (1,)
    assert torch.isfinite(count)
    assert count.item() >= 0.0



def test_t12_parameter_budget():
    """T12: Verify deployed parameter budget < 1.5M with physical backbone truncation."""
    model = HPCLite(pretrained=False, neck_width=32, truncate_backbone=True)
    total_params = count_parameters(model)
    print(f"Total model parameters: {total_params:,}")
    assert total_params < 1_500_000, f"Model parameter count {total_params} exceeds 1.5M budget!"
