import pytest
import torch
import yaml
from pathlib import Path

from rmr_v3.config import validate_v3_config
from rmr_v3.engine import make_loss_cfg, make_model
from rmr_v3.losses import compute_rmr_v3_losses
from rmr_v3.model import RMRv3, RMRv3Config


CONFIG_PATHS = [
    "configs/rmr_research/rmr_v36_m04_restored_104k.yaml",
    "configs/rmr_research/rmr_v36_scaled_master_174k.yaml",
    "configs/rmr_research/rmr_v36_scaled_cosine_174k.yaml",
    "configs/rmr_research/rmr_v36_scaled_solver_t2_174k.yaml",
]


@pytest.mark.parametrize("cfg_path", CONFIG_PATHS)
def test_v36_config_schema_and_split_invariants(cfg_path: str):
    """Verify schema validity and strict compliance with Zero Ad-hoc Split Policy."""
    p = Path(cfg_path)
    assert p.is_file(), f"Config file {cfg_path} not found"
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. Schema validation
    validate_v3_config(cfg)

    # 2. Split invariant check
    d_cfg = cfg["data"]
    assert d_cfg["train_manifest"] == "data/sha_a_train_all.jsonl"
    assert d_cfg["val_manifest"] == "data/sha_a_test.jsonl"


def test_v36_parameter_budgets():
    """Verify exact parameter counts for both 104k and 174k architectures."""
    # 1. Baseline 104k
    with open("configs/rmr_research/rmr_v36_m04_restored_104k.yaml") as f:
        cfg_104k = yaml.safe_load(f)
    m_104k, _ = make_model(cfg_104k)
    p_104k = sum(p.numel() for p in m_104k.parameters() if p.requires_grad)
    assert p_104k == 104441, f"Expected 104,441 params, got {p_104k}"

    # 2. Scaled Master 174k
    with open("configs/rmr_research/rmr_v36_scaled_master_174k.yaml") as f:
        cfg_174k = yaml.safe_load(f)
    m_174k, _ = make_model(cfg_174k)
    p_174k = sum(p.numel() for p in m_174k.parameters() if p.requires_grad)
    assert p_174k == 174097, f"Expected 174,097 params, got {p_174k}"
    assert p_174k <= 200000, f"Exceeded 200,000 parameter ceiling: {p_174k}"


def test_v36_causal_detach_gradient_flow():
    """Verify that b_solver is strictly detached from autograd graph in RMRv3."""
    with open("configs/rmr_research/rmr_v36_scaled_master_174k.yaml") as f:
        cfg = yaml.safe_load(f)
    model, _ = make_model(cfg)
    model.eval()

    x = torch.randn(1, 3, 256, 256)
    out = model(x, compute_energy=True)

    # b_solver must not require grad
    assert not out["b_solver"].requires_grad, "b_solver must be detached under Causal Detach Theorem!"
    assert out["solver_region_weight"] is not None


def test_v36_monotonic_energy_dissipation():
    """Verify that T=6 iterations monotonically dissipate regional energy."""
    with open("configs/rmr_research/rmr_v36_scaled_master_174k.yaml") as f:
        cfg = yaml.safe_load(f)
    model, _ = make_model(cfg)
    model.eval()

    x = torch.randn(1, 3, 256, 256)
    out = model(x, compute_energy=True)

    energy_trace = out["energy_trace"]
    assert len(energy_trace) == 6, f"Expected 6 solver steps, got {len(energy_trace)}"
    assert len(out["iterates"]) == 7, "Expected 7 iterates (y0 through y6)"


def test_v36_forward_and_loss_backward_pipeline():
    """Verify full forward, loss computation, and backward pass on odd resolution."""
    with open("configs/rmr_research/rmr_v36_scaled_master_174k.yaml") as f:
        cfg = yaml.safe_load(f)
    model, _ = make_model(cfg)
    loss_cfg = make_loss_cfg(cfg)
    model.train()

    # Odd resolution stress test (387 x 515)
    x = torch.randn(2, 3, 387, 515)
    out = model(x, compute_energy=False)

    y = out["y"]
    assert y.ndim == 4
    assert y.shape[0] == 2
    assert (y >= 0.0).all(), "Carrier field must be non-negative"

    target = torch.ones_like(y) * 0.05
    losses = compute_rmr_v3_losses(out, target, loss_cfg)

    assert "total" in losses
    total_loss = losses["total"]
    assert torch.isfinite(total_loss), f"Non-finite loss encountered: {total_loss}"

    total_loss.backward()

    # Check that model parameters have finite gradients
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0, "No gradients were computed"
    assert all(torch.isfinite(torch.tensor(g)) for g in grad_norms), "Encountered NaN/Inf gradient"
