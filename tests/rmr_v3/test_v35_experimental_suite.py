"""Validation tests for the 4-Run Controlled Experimental Suite configs.

Verifies:
1. Configurations load and validate cleanly via validate_v3_config.
2. Param counts strictly respect the assigned boundaries (104,407 <= 104,441 or 174,340 in [150k, 200k]).
3. Forward and backward passes execute cleanly with non-zero gradients on CPU.
"""

from __future__ import annotations

from pathlib import Path
import pytest
import torch
import yaml

from rmr_v3.config import validate_v3_config
from rmr_v3.losses import RMRv3LossConfig, compute_rmr_v3_losses
from rmr_v3.model import RMRv3, RMRv3Config


EXPERIMENT_CONFIGS = [
    ("configs/rmr_research/rmr_v35_purified_104k.yaml", 104407, 104441),
    ("configs/rmr_research/rmr_v35_scaled_165k.yaml", 174340, 200000),
    ("configs/rmr_research/rmr_v35_wsd_104k.yaml", 104407, 104441),
    ("configs/rmr_research/rmr_v35_bayesian_104k.yaml", 104407, 104441),
]


@pytest.mark.parametrize("cfg_path,expected_params,max_budget", EXPERIMENT_CONFIGS)
def test_experimental_configs_integrity_and_forward(cfg_path: str, expected_params: int, max_budget: int):
    full_path = Path(cfg_path)
    assert full_path.exists(), f"Config file not found: {cfg_path}"

    raw = yaml.safe_load(full_path.read_text(encoding="utf-8"))
    validate_v3_config(raw)

    m_cfg = RMRv3Config(**raw.get("model", {}))
    m_cfg.pretrained = False
    model = RMRv3(m_cfg)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_params == expected_params, f"Param count mismatch: {trainable_params} != {expected_params}"
    assert trainable_params <= max_budget, f"Param budget violated: {trainable_params} > {max_budget}"

    # Verify lightweight forward & backward
    loss_cfg = RMRv3LossConfig(**raw.get("loss", {}))
    model.train()

    x = torch.randn(1, 3, 256, 256)
    target = torch.zeros(1, 1, 128, 128)
    target[:, :, 30, 30] = 1.0
    target[:, :, 60, 60] = 2.0

    out = model(x)
    assert "y" in out and "y0" in out
    assert out["y"].shape[-2:] == (128, 128), "Output stride 2 should produce 128x128 from 256x256"

    losses = compute_rmr_v3_losses(out, target, cfg=loss_cfg)
    assert torch.isfinite(losses["total"]), f"Total loss is non-finite: {losses['total']}"
    losses["total"].backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)
    assert has_grad, "No parameters received non-zero gradients"
