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


def test_resume_rejects_statistics_drift():
    import pytest
    from hpc.models.factory import assert_resume_compatible

    old_cfg = {
        "dataset": {"name": "sha", "crop_size": 256},
        "loss": {"mode": "r5_full_ntpc", "dense_threshold_16": "auto"},
        "statistics": {"seed": 12345, "crops_per_image": 3},
    }
    new_cfg = {
        **old_cfg,
        "statistics": {"seed": 999, "crops_per_image": 5},
    }

    checkpoint = {"config": old_cfg}
    with pytest.raises(ValueError, match="Resume protocol mismatch in: statistics"):
        assert_resume_compatible(checkpoint, new_cfg)


def test_resume_rejects_evaluation_cadence_drift():
    import pytest
    from hpc.models.factory import assert_resume_compatible

    old_cfg = {
        "dataset": {"name": "sha", "crop_size": 256},
        "loss": {"mode": "r2_flat_dm"},
        "training": {"evaluate_every": 5},
    }
    new_cfg = {
        **old_cfg,
        "training": {"evaluate_every": 20},
    }

    with pytest.raises(ValueError, match="Resume protocol mismatch in: training"):
        assert_resume_compatible({"config": old_cfg}, new_cfg)


def test_dataset_resolver_matches_loader_defaults():
    from hpc.models.factory import resolve_dataset_config

    assert resolve_dataset_config({"dataset": {"name": "sha"}})["coordinate_base"] == 0
    assert resolve_dataset_config({"dataset": {"name": "qnrf"}})["coordinate_base"] == 1
    assert resolve_dataset_config({"dataset": {"name": "nwpu"}})["coordinate_base"] == 0

    shb = resolve_dataset_config({"dataset": {"name": "shanghaitech_b"}})
    assert shb["part"] == "part_B"
    assert shb["coordinate_base"] == 0


def test_resume_rejects_persistent_workers_drift():
    import pytest
    from hpc.models.factory import assert_resume_compatible

    old_cfg = {
        "dataset": {"name": "sha"},
        "training": {"num_workers": 2, "persistent_workers": False},
    }
    new_cfg = {
        "dataset": {"name": "sha"},
        "training": {"num_workers": 2, "persistent_workers": True},
    }

    with pytest.raises(ValueError, match="Resume protocol mismatch in: training"):
        assert_resume_compatible({"config": old_cfg}, new_cfg)




