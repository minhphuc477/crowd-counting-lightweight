import yaml
import torch
import pytest

from rmr_v3.config.validator import validate_v3_config
from rmr_v3.model.architecture import RMRv3
from rmr_v3.model.config import RMRv3Config
from rmr_v3.losses.config import RMRv3LossConfig
from rmr_v3.losses.orchestration import compute_rmr_v3_losses


@pytest.mark.parametrize("cfg_path", [
    "configs/rmr_v53/v53_track_a_closed_form_rn.yaml",
    "configs/rmr_v53/v53_track_b_closed_form_rn.yaml",
])
def test_v53_config_contract_and_forward_backward(cfg_path: str):
    """Verify v53 configurations pass full architectural validation, parameter bounds, and autograd."""
    with open(cfg_path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)

    # 1. Schema & architectural validator
    validate_v3_config(d)

    # 2. Model instantiation
    m_cfg = RMRv3Config.from_dict(d["model"])
    model = RMRv3(m_cfg)
    model.train()

    # 3. Parameter budget invariants
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    max_params = m_cfg.max_trainable_params
    assert total_params <= max_params, f"Parameter budget exceeded: {total_params} > {max_params}"

    # 4. Forward execution
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    assert "y" in out and "y0" in out
    assert out.y.shape == out.y0.shape
    assert len(out.iterates) == 2

    # 5. Loss execution & backward pass
    l_cfg = RMRv3LossConfig(**d["loss"])
    target_y = torch.zeros_like(out.y)
    target_y[0, 0, 10, 10] = 1.0
    losses = compute_rmr_v3_losses(out, target_y, l_cfg)
    assert torch.isfinite(losses["total"])
    assert "strauss" in losses

    losses["total"].backward()

    # Check finite gradients
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.all(torch.isfinite(p.grad)), f"Non-finite gradient in {name}"
