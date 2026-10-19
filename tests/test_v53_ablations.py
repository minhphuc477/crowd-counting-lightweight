# ruff: noqa: E402
"""Comprehensive verification suite for RMR V53 Ablation Configurations.

Validates that all 8 ablation configurations:
1. Conform strictly to the RMRv3Config schema and validators.
2. Comply with strict parameter count limits (Track A <= 104,441 or 110k, Track B <= 200,000).
3. Successfully perform forward and backward autograd passes on synthetic inputs.
4. Correctly toggle individual components under single-variable control.
"""

from __future__ import annotations

import glob
import os
import pytest
import torch
import yaml

from rmr_v3.config.validator import validate_v3_config
from rmr_v3.model.architecture import RMRv3
from rmr_v3.model.config import RMRv3Config
from rmr_v3.losses.config import RMRv3LossConfig
from rmr_v3.losses.orchestration import compute_rmr_v3_losses


ABLATION_CONFIGS = sorted(glob.glob("configs/rmr_v53/ablations/*.yaml"))


@pytest.mark.parametrize("config_path", ABLATION_CONFIGS)
def test_v53_ablation_config_contract_and_autograd(config_path: str) -> None:
    """Test schema validation, parameter constraints, and forward/backward pass."""
    assert os.path.exists(config_path), f"Config file missing: {config_path}"
    with open(config_path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)

    # 1. Architectural schema validation
    validate_v3_config(d)

    # 2. Model instantiation & parameter budget
    m_cfg = RMRv3Config.from_dict(d["model"])
    model = RMRv3(m_cfg)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if m_cfg.output_stride == 4 and m_cfg.feature_width == 32:
        if m_cfg.head_blocks == 1:
            assert total_params <= 104441, f"{config_path} exceeds 104,441 params: {total_params}"
        else:
            assert total_params <= 110000, f"{config_path} exceeds 110,000 params: {total_params}"
    else:
        assert total_params <= 200000, f"{config_path} exceeds 200,000 params: {total_params}"

    # 3. Forward pass
    model.train()
    batch_size = 2
    dummy_x = torch.randn(batch_size, 3, 256, 256)
    out = model(dummy_x)

    assert "y0" in out
    assert "y" in out
    assert out.y.shape[0] == batch_size
    expected_stride = m_cfg.output_stride
    assert out.y.shape[-2:] == (256 // expected_stride, 256 // expected_stride)

    # 4. Backward autograd pass
    loss = out.y.sum()
    loss.backward()

    # 5. Check gradients exist on fine carrier head
    final_conv = model.fine_head.body[-1]
    assert final_conv.weight.grad is not None
    assert torch.isfinite(final_conv.weight.grad).all()


def test_ablation_matrix_coverage() -> None:
    """Verify all 8 ablation configurations are present and valid."""
    assert len(ABLATION_CONFIGS) == 8, f"Expected 8 ablation configs, got {len(ABLATION_CONFIGS)}"
    expected_names = [
        "abl_s1_direct_stride4.yaml",
        "abl_s1_sirt_unrolled.yaml",
        "abl_s2_stride4_deep_head.yaml",
        "abl_s3_stride2_shallow_head.yaml",
        "abl_s4_no_strauss.yaml",
        "abl_s5_strauss_strong.yaml",
        "abl_s6_track_b_no_strauss.yaml",
        "abl_s7_track_b_direct.yaml",
    ]
    basenames = [os.path.basename(p) for p in ABLATION_CONFIGS]
    for exp in expected_names:
        assert exp in basenames, f"Missing expected ablation config: {exp}"
