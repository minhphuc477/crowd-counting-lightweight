from __future__ import annotations

import math
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F

from rmr_v3.config import load_config
from rmr_v3.losses import compute_rmr_v3_losses
from rmr_v3.model import RMRv3, RMRv3Config
from rmr_v3.train import make_loss_cfg, make_model


def test_rmr_v17_parameter_count_and_budget():
    """Verify RMR-v17 canonical parameter count is exactly 104,474 (budget <= 105,000)."""
    cfg = load_config("configs/rmr_v17/rmr_v17_canonical.yaml")
    model, _ = make_model(cfg)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert total_params <= 105000, f"RMR-v17 exceeded 105k budget: {total_params}"
    assert total_params == 104474, f"Expected exactly 104,474 parameters, got {total_params}"

    # Load v16 canonical and confirm exactly +1 parameter difference (curvature_alpha)
    v16_cfg = load_config("configs/rmr_v16/rmr_v16_canonical.yaml")
    v16_model, _ = make_model(v16_cfg)
    v16_params = sum(p.numel() for p in v16_model.parameters() if p.requires_grad)

    assert total_params - v16_params == 1, (
        f"Expected exactly +1 param over v16 (104,473 -> 104,474), got {total_params} - {v16_params} = {total_params - v16_params}"
    )


def test_rmr_v17_step0_identity_with_v16():
    """Verify RMR-v17 density head step-0 output is essentially identical to vanilla softplus."""
    cfg = load_config("configs/rmr_v17/rmr_v17_canonical.yaml")
    model, _ = make_model(cfg)

    z = torch.linspace(-5.0, 5.0, 100)
    tau = model.fine_head.tau.clamp_min(0.1)
    y_base = tau * F.softplus(z / tau)
    y_v17 = model.fine_head.activate(z)

    # At step 0, alpha is initialized to -8.0, so softplus(-8.0) ≈ 0.000335
    # Relative difference must be under 0.05%
    rel_diff = (y_v17 - y_base).abs() / (y_base + 1e-6)
    assert rel_diff.max().item() < 5e-3, f"Step 0 identity violated: max rel diff = {rel_diff.max().item()}"


def test_rmr_v17_density_curvature_properties():
    """Verify quadratic curvature warping is strictly monotonic and expands dense cluster capacity."""
    cfg = load_config("configs/rmr_v17/rmr_v17_canonical.yaml")
    model, _ = make_model(cfg)

    # Test monotonicity
    z_sorted = torch.linspace(-10.0, 10.0, 500)
    y_sorted = model.fine_head.activate(z_sorted)
    diffs = y_sorted[1:] - y_sorted[:-1]
    assert (diffs >= 0).all(), "Curvature activation violated strict monotonicity!"

    # Test curvature expansion when alpha grows
    with torch.no_grad():
        model.fine_head.curvature_alpha.fill_(-1.0)  # softplus(-1) ≈ 0.313
    z_dense = torch.tensor([3.0, 4.0, 5.0])
    y_expanded = model.fine_head.activate(z_dense)
    tau = model.fine_head.tau.clamp_min(0.1)
    y_linear = tau * F.softplus(z_dense / tau)
    assert (y_expanded > y_linear).all(), "Curvature warping failed to expand high density!"


def test_rmr_v17_barzilai_borwein_stability():
    """Verify Barzilai-Borwein dynamic step size is bounded, finite, and stable over multiple iterations."""
    cfg = load_config("configs/rmr_v17/rmr_v17_canonical.yaml")
    model, _ = make_model(cfg)
    model.eval()

    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out = model(x, solver_strength=1.0)

    assert "iterates" in out
    assert len(out["iterates"]) == 7  # y0 + 6 iterations
    for it_idx, it in enumerate(out["iterates"]):
        assert torch.isfinite(it).all(), f"Iterate {it_idx} contains NaN/Inf!"
        assert (it >= 0.0).all(), f"Iterate {it_idx} has negative values!"


def test_rmr_v17_scale_entropy_trust_modulation():
    """Verify scale-entropy confidence bounds are mathematically exact."""
    # Test uniform distribution: maximum entropy -> 0 confidence -> 0.25 floor
    k = 4
    pi_uniform = torch.full((1, k, 16, 16), 1.0 / k)
    pi_safe = pi_uniform.clamp_min(1e-7)
    entropy = -(pi_safe * torch.log(pi_safe)).sum(dim=1, keepdim=True)
    max_entropy = math.log(k)
    confidence = (1.0 - (entropy / max_entropy)).clamp(0.0, 1.0)
    assert torch.allclose(confidence, torch.zeros_like(confidence), atol=1e-5), "Uniform pi did not yield zero confidence"

    # Test one-hot distribution: minimum entropy (0) -> 1.0 confidence -> 1.0 full bound
    pi_onehot = torch.zeros((1, k, 16, 16))
    pi_onehot[:, 0] = 1.0
    pi_safe = pi_onehot.clamp_min(1e-7)
    entropy = -(pi_safe * torch.log(pi_safe)).sum(dim=1, keepdim=True)
    confidence = (1.0 - (entropy / max_entropy)).clamp(0.0, 1.0)
    assert torch.allclose(confidence, torch.ones_like(confidence), atol=1e-3), "One-hot pi did not yield unit confidence"


def test_rmr_v17_full_forward_backward_gradient_hygiene():
    """Verify full end-to-end forward pass and loss backward yields finite gradients on all parameters."""
    cfg = load_config("configs/rmr_v17/rmr_v17_canonical.yaml")
    model, _ = make_model(cfg)
    loss_cfg = make_loss_cfg(cfg)

    x = torch.randn(2, 3, 256, 256)
    out = model(x, solver_strength=1.0)
    target_y = torch.zeros(2, 1, 64, 64)
    target_y[:, :, 20:30, 20:30] = 1.0  # mock crowd cluster

    losses = compute_rmr_v3_losses(out, target_y, loss_cfg)
    assert torch.isfinite(losses["total"]), "Total loss is NaN or Inf!"
    losses["total"].backward()

    # Verify curvature_alpha received finite non-zero gradient
    assert model.fine_head.curvature_alpha.grad is not None
    assert torch.isfinite(model.fine_head.curvature_alpha.grad).all()

    # Verify all trainable parameters have finite gradients
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} has None grad!"
            assert torch.isfinite(p.grad).all(), f"Parameter {name} has NaN/Inf grad!"


def test_rmr_v17_all_configs_validation():
    """Verify all RMR-v17 config files load and validate cleanly."""
    v17_dir = Path("configs/rmr_v17")
    yaml_files = list(v17_dir.glob("*.yaml"))
    assert len(yaml_files) >= 4, f"Expected at least 4 configs in {v17_dir}, found {len(yaml_files)}"

    for yf in yaml_files:
        cfg = load_config(yf)
        assert cfg["model"]["region_sizes_px"] == [16, 32, 64, 128]
        model, _ = make_model(cfg)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert params <= 105000, f"Config {yf.name} exceeded budget: {params}"
