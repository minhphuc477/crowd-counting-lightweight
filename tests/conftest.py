import pytest
import torch


@pytest.fixture(autouse=True)
def reset_torch_deterministic_state():
    """Ensure that deterministic algorithm state does not leak between tests."""
    yield
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass
