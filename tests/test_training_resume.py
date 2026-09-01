"""Tests for training resume reproducibility and RNG checkpointing."""

import random
import numpy as np
import torch

from train_ntpc import get_rng_state


def test_rng_checkpoint_roundtrip():
    state = get_rng_state()
    assert "torch" in state
    assert "numpy" in state
    assert "python" in state
    assert isinstance(state["torch"], torch.Tensor)
    assert isinstance(state["numpy"], tuple)
    assert isinstance(state["python"], tuple)


def test_assert_resume_compatible_detects_crop_size_mismatch():
    import pytest
    from hpc.models.factory import assert_resume_compatible

    cfg_a = {
        "model": {"backbone": "mobilenetv4_conv_small_050", "neck_width": 32},
        "dataset": {"name": "sha", "crop_size": 256},
        "loss": {"mode": "r2_flat_dm"},
    }
    cfg_c = {
        "model": {"backbone": "mobilenetv4_conv_small_050", "neck_width": 32},
        "dataset": {"name": "sha", "crop_size": 448},
        "loss": {"mode": "r2_flat_dm"},
    }

    ckpt_a = {"config": cfg_a}
    # Matching config passes
    assert_resume_compatible(ckpt_a, cfg_a)

    # Mismatched crop_size raises ValueError
    with pytest.raises(ValueError, match="Resume protocol mismatch in: dataset"):
        assert_resume_compatible(ckpt_a, cfg_c)

