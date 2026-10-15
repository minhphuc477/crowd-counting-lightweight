from __future__ import annotations

import copy
from pathlib import Path
import pytest
import torch
import yaml
from PIL import Image

from rmr_core.data import CrowdManifestDataset
from rmr_core.transforms import train_transform
from rmr_v3.config import validate_v3_config
from rmr_v3.engine import make_loss_cfg, make_model
from rmr_v3.losses import compute_rmr_v3_losses


V42_CONFIGS = [
    "configs/rmr_v42/sub60_v42_m04_scale_wide.yaml",
    "configs/rmr_v42/sub60_v42_m04_color_invariance.yaml",
    "configs/rmr_v42/sub60_v42_m04_reg_wd.yaml",
]


@pytest.mark.parametrize("cfg_path", V42_CONFIGS)
def test_v42_configs_schema_and_parameter_ceiling(cfg_path: str):
    p = Path(cfg_path)
    assert p.is_file(), f"Missing config file: {cfg_path}"
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. Schema and invariants validation
    validate_v3_config(cfg)

    # 2. Model parameter ceiling <= 104,441
    model, _ = make_model(cfg)
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    assert trainable_params <= 104441, f"Params exceeded budget: {trainable_params} > 104,441"
    assert trainable_params == 104441, f"Expected exactly 104,441 params, got {trainable_params}"


def test_v42_single_variable_isolation():
    """Verify that each v42 config differs from m04 by exactly its isolated variable."""
    m04_path = Path("runs/sha_a/matrix/m04_iso_fractional_loss/resolved_config.yaml")
    assert m04_path.is_file()
    with m04_path.open("r", encoding="utf-8") as f:
        m04_cfg = yaml.safe_load(f)

    # Config 1: Wide Scale Range
    with open("configs/rmr_v42/sub60_v42_m04_scale_wide.yaml", "r", encoding="utf-8") as f:
        c1 = yaml.safe_load(f)
    assert c1["data"]["scale_range"] == [0.5, 1.5]
    c1_norm = copy.deepcopy(c1)
    c1_norm["data"]["scale_range"] = m04_cfg["data"]["scale_range"]
    c1_norm["output_dir"] = m04_cfg["output_dir"]
    c1_norm["train"]["workers"] = m04_cfg["train"].get("workers", 4)
    c1_norm["train"].pop("num_threads", None)
    assert c1_norm["model"] == m04_cfg["model"]
    assert c1_norm["loss"] == m04_cfg["loss"]

    # Config 2: Color Invariance
    with open("configs/rmr_v42/sub60_v42_m04_color_invariance.yaml", "r", encoding="utf-8") as f:
        c2 = yaml.safe_load(f)
    assert c2["data"]["saturation_jitter"] == 0.3
    assert c2["data"]["grayscale_prob"] == 0.2
    c2_norm = copy.deepcopy(c2)
    c2_norm["data"].pop("saturation_jitter", None)
    c2_norm["data"].pop("grayscale_prob", None)
    c2_norm["output_dir"] = m04_cfg["output_dir"]
    c2_norm["train"]["workers"] = m04_cfg["train"].get("workers", 4)
    c2_norm["train"].pop("num_threads", None)
    assert c2_norm["data"] == m04_cfg["data"]
    assert c2_norm["model"] == m04_cfg["model"]
    assert c2_norm["loss"] == m04_cfg["loss"]

    # Config 3: Weight Regularization
    with open("configs/rmr_v42/sub60_v42_m04_reg_wd.yaml", "r", encoding="utf-8") as f:
        c3 = yaml.safe_load(f)
    assert c3["train"]["weight_decay"] == 0.001
    c3_norm = copy.deepcopy(c3)
    c3_norm["train"]["weight_decay"] = m04_cfg["train"]["weight_decay"]
    c3_norm["output_dir"] = m04_cfg["output_dir"]
    c3_norm["train"]["workers"] = m04_cfg["train"].get("workers", 4)
    c3_norm["train"].pop("num_threads", None)
    assert c3_norm["data"] == m04_cfg["data"]
    assert c3_norm["model"] == m04_cfg["model"]
    assert c3_norm["loss"] == m04_cfg["loss"]


def test_v42_photometric_transforms():
    """Verify saturation_jitter and grayscale_prob transforms function correctly."""
    img = Image.new("RGB", (600, 600), color=(100, 150, 200))
    pts = torch.tensor([[100.0, 100.0], [300.0, 300.0]])

    # Saturation jitter preserves shape and range
    t_sat, p_sat = train_transform(img, pts, saturation_jitter=0.5, crop_size=512)
    assert t_sat.shape == (3, 512, 512)
    assert (t_sat >= 0.0).all() and (t_sat <= 1.0).all()

    # Grayscale prob = 1.0 makes all 3 channels identical
    t_gray, p_gray = train_transform(img, pts, grayscale_prob=1.0, crop_size=512)
    assert t_gray.shape == (3, 512, 512)
    assert torch.allclose(t_gray[0], t_gray[1]) and torch.allclose(t_gray[1], t_gray[2])


@pytest.mark.parametrize("cfg_path", V42_CONFIGS)
def test_v42_forward_and_backward_pass(cfg_path: str):
    """Verify complete forward pass, loss computation, and backpropagation."""
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model, _ = make_model(cfg)
    model.train()
    x = torch.rand(2, 3, 512, 512)
    out = model(x)
    assert "y" in out and "y0" in out and "b_solver" in out

    loss_cfg = make_loss_cfg(cfg)
    target_y = torch.zeros(2, 1, 128, 128)
    target_y[0, 0, 50, 50] = 1.0
    target_y[1, 0, 80, 80] = 2.0
    points = [torch.tensor([[200.0, 200.0]]), torch.tensor([[320.0, 320.0], [321.0, 321.0]])]

    losses = compute_rmr_v3_losses(out, target_y, loss_cfg, points=points)
    assert "total" in losses
    assert torch.isfinite(losses["total"])

    losses["total"].backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Missing gradient for {name}"
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {name}"
