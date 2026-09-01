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
