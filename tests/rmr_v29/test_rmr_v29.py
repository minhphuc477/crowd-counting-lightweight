import math
from pathlib import Path
import pytest
import torch
import yaml

from rmr_v3.config import validate_v3_config
from rmr_v3.model import RMRv3, RMRv3Config
from rmr_v3.losses import compute_rmr_v3_losses, RMRv3LossConfig


CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "rmr_v29"


def test_v29_configs_exist_and_validate():
    """Verify that all RMR-v29 configuration files exist and pass validation."""
    configs = [
        "rmr_v29_step0_v19_anchor.yaml",
        "rmr_v29_h1_depth8.yaml",
        "rmr_v29_h2_subpixel2.yaml",
    ]
    for cfg_name in configs:
        p = CONFIG_DIR / cfg_name
        assert p.is_file(), f"Missing config file: {p}"
        raw = yaml.safe_load(p.read_text())
        validate_v3_config(raw)


def test_v29_parameter_budget():
    """Verify strict parameter ceilings (<= 105,000) across all v29 variants."""
    configs = {
        "rmr_v29_step0_v19_anchor.yaml": 104441,
        "rmr_v29_h1_depth8.yaml": 104441,
        "rmr_v29_h2_subpixel2.yaml": 104540,
    }
    for cfg_name, expected_params in configs.items():
        p = CONFIG_DIR / cfg_name
        raw = yaml.safe_load(p.read_text())
        m_cfg = RMRv3Config.from_dict(raw.get("model", {}))
        model = RMRv3(m_cfg)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable == expected_params, f"{cfg_name}: expected {expected_params} params, got {trainable}"
        assert trainable <= 105000, f"{cfg_name}: exceeds 105,000 parameter budget: {trainable}"


def test_v29_single_variable_isolation():
    """Verify single-variable hypothesis isolation across the Two-Loop ladder."""
    p_step0 = CONFIG_DIR / "rmr_v29_step0_v19_anchor.yaml"
    p_h1 = CONFIG_DIR / "rmr_v29_h1_depth8.yaml"
    p_h2 = CONFIG_DIR / "rmr_v29_h2_subpixel2.yaml"

    c_step0 = yaml.safe_load(p_step0.read_text())
    c_h1 = yaml.safe_load(p_h1.read_text())
    c_h2 = yaml.safe_load(p_h2.read_text())

    # Step 0 vs H1 differs ONLY in iterations
    assert c_step0["model"]["iterations"] == 6
    assert c_h1["model"]["iterations"] == 8
    c_s0_copy = dict(c_step0["model"])
    c_h1_copy = dict(c_h1["model"])
    c_s0_copy.pop("iterations")
    c_h1_copy.pop("iterations")
    assert c_s0_copy == c_h1_copy, "H1 introduces uncontrolled variables against Step 0!"

    # Step 0 vs H2 differs ONLY in subpixel_stride2 and output_stride
    assert c_h2["model"]["subpixel_stride2"] is True
    assert c_h2["model"]["output_stride"] == 2
    c_s0_m = dict(c_step0["model"])
    c_h2_m = dict(c_h2["model"])
    c_s0_m.pop("output_stride")
    c_h2_m.pop("output_stride")
    c_h2_m.pop("subpixel_stride2")
    assert c_s0_m == c_h2_m, "H2 introduces uncontrolled model variables against Step 0!"
    assert c_step0["loss"] == c_h2["loss"], "H2 loss differs from Step 0!"
    assert c_step0["train"] == c_h2["train"], "H2 train differs from Step 0!"
    assert c_step0["eval"] == c_h2["eval"], "H2 eval differs from Step 0!"


def test_v29_step0_forward_backward_gradient_flow():
    """Verify forward-backward gradient flow through the unrolled SIRT solver for Step 0."""
    p = CONFIG_DIR / "rmr_v29_step0_v19_anchor.yaml"
    raw = yaml.safe_load(p.read_text())
    m_cfg = RMRv3Config.from_dict(raw.get("model", {}))
    l_cfg = RMRv3LossConfig.from_dict(raw.get("loss", {}))

    model = RMRv3(m_cfg)
    model.train()

    # Synthetic batch: 2 images of 256x256
    x = torch.randn(2, 3, 256, 256, requires_grad=False)
    target_y = torch.zeros(2, 1, 64, 64)
    target_y[0, 0, 10, 10] = 1.0
    target_y[1, 0, 20, 20] = 1.0

    out = model(x)
    assert out.y.shape == (2, 1, 64, 64), f"Expected shape (2, 1, 64, 64), got {out.y.shape}"
    assert len(out.iterates) == 7  # y0 + 6 iterations

    losses = compute_rmr_v3_losses(out, target_y, l_cfg)
    assert torch.isfinite(losses["total"]), "Total loss is non-finite!"

    losses["total"].backward()

    # Check gradient flow into backbone and heads
    has_grad = False
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN/Inf gradient in parameter {name}"
            has_grad = True
    assert has_grad, "No parameters received gradients!"


def test_v29_h2_subpixel2_forward_backward():
    """Verify forward-backward gradient flow and Stride 2 output resolution for H2."""
    p = CONFIG_DIR / "rmr_v29_h2_subpixel2.yaml"
    raw = yaml.safe_load(p.read_text())
    m_cfg = RMRv3Config.from_dict(raw.get("model", {}))
    l_cfg = RMRv3LossConfig.from_dict(raw.get("loss", {}))

    model = RMRv3(m_cfg)
    model.train()

    # Synthetic batch: 2 images of 256x256
    x = torch.randn(2, 3, 256, 256, requires_grad=False)
    # Stride 2 target: 128x128
    target_y = torch.zeros(2, 1, 128, 128)
    target_y[0, 0, 20, 20] = 1.0
    target_y[1, 0, 40, 40] = 1.0

    out = model(x)
    assert out.y.shape == (2, 1, 128, 128), f"Expected shape (2, 1, 128, 128), got {out.y.shape}"

    losses = compute_rmr_v3_losses(out, target_y, l_cfg)
    assert torch.isfinite(losses["total"]), "H2 Total loss is non-finite!"

    losses["total"].backward()

    has_grad = False
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN/Inf gradient in parameter {name}"
            has_grad = True
    assert has_grad, "H2: No parameters received gradients!"
