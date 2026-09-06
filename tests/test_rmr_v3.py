"""tests/test_rmr_v3.py — Unit tests for RMR-v3.

All 8 tests must pass before any training run.

Tests:
  T1. Import smoke test
  T2. Forward pass shape correctness
  T3. Parameter budget < 105k
  T4. Nonnegative output (no negative cells)
  T5. Uniform reliability V3-A: w_R = 1.0 everywhere
  T6. Scale-neutral normalization: mean per scale ~1
  T7. FP32 solver correctness (no NaN, no Inf in adjoint field)
  T8. Detach policy: b_region grad does NOT flow through solver step
"""
import pytest
import torch
import math
import sys
import os

# Ensure repo root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rmr_v3.model import RMRv3, RMRv3Config, _scale_neutral_normalize, _weighted_adjoint
from rmr_v3.losses import compute_losses_v3, LossConfigV3
from rmr_count.operators import build_multiscale_regions


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_model(uniform: bool = False) -> RMRv3:
    cfg = RMRv3Config(pretrained=False, uniform_reliability=uniform)
    return RMRv3(cfg).to(DEVICE).eval()


def _make_input(B: int = 2, H: int = 384, W: int = 512) -> torch.Tensor:
    return torch.rand(B, 3, H, W, device=DEVICE)


# ---------------------------------------------------------------------------
# T1. Import smoke test
# ---------------------------------------------------------------------------
def test_t1_import():
    """RMRv3 and RMRv3Config must be importable from rmr_v3."""
    from rmr_v3 import RMRv3, RMRv3Config  # noqa: F401
    assert RMRv3 is not None
    assert RMRv3Config is not None


# ---------------------------------------------------------------------------
# T2. Forward pass shape correctness
# ---------------------------------------------------------------------------
def test_t2_forward_shapes():
    """Forward pass must return correct tensor shapes for all output keys."""
    model = _make_model()
    x = _make_input(B=2, H=384, W=512)
    with torch.no_grad():
        out = model(x)

    B = 2
    # Stride-4 grid: H4 = 384//4 = 96, W4 = 512//4 = 128
    H4, W4 = 96, 128
    M = out["regions"].boxes.shape[0]

    assert out["y"].shape == (B, 1, H4, W4), f"y shape: {out['y'].shape}"
    assert out["y0"].shape == (B, 1, H4, W4), f"y0 shape: {out['y0'].shape}"
    assert out["b_region"].shape == (B, 1, M), f"b_region shape: {out['b_region'].shape}"
    assert out["w_region"].shape == (B, 1, M), f"w_region shape: {out['w_region'].shape}"
    assert out["rate"].shape == (B, 1, M), f"rate shape: {out['rate'].shape}"
    assert len(out["iterates"]) == 3, f"iterates len: {len(out['iterates'])}"  # Y0, Y1, Y2


# ---------------------------------------------------------------------------
# T3. Parameter budget < 105k
# ---------------------------------------------------------------------------
def test_t3_parameter_budget():
    """Total trainable parameters must be < 105,000."""
    model = _make_model()
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable params: {n:,}")
    assert n < 105_000, f"OVER BUDGET: {n:,} >= 105,000"


# ---------------------------------------------------------------------------
# T4. Nonnegative output
# ---------------------------------------------------------------------------
def test_t4_nonnegative_output():
    """All cells in final Y must be >= 0 (nonneg projection guarantee)."""
    model = _make_model()
    x = _make_input(B=4, H=256, W=256)
    with torch.no_grad():
        out = model(x)
    y = out["y"]
    n_neg = (y < 0).sum().item()
    assert n_neg == 0, f"Found {n_neg} negative cells in output Y"


# ---------------------------------------------------------------------------
# T5. Uniform reliability (V3-A): w_R = 1.0 everywhere
# ---------------------------------------------------------------------------
def test_t5_uniform_reliability():
    """When uniform_reliability=True (V3-A), all w_R must equal 1.0."""
    model = _make_model(uniform=True)
    x = _make_input(B=2, H=256, W=256)
    with torch.no_grad():
        out = model(x)
    w = out["w_region"]
    assert torch.allclose(w, torch.ones_like(w), atol=1e-6), \
        f"V3-A weights not uniform: mean={w.mean():.4f}, std={w.std():.4f}"


# ---------------------------------------------------------------------------
# T6. Scale-neutral normalization: mean per scale ~1
# ---------------------------------------------------------------------------
def test_t6_scale_neutral_normalization():
    """After scale-neutral normalization, mean weight per scale must be ~1.0."""
    model = _make_model(uniform=False)
    x = _make_input(B=2, H=256, W=256)
    with torch.no_grad():
        out = model(x)

    w = out["w_region"]  # [B, 1, M] after normalization
    regions = out["regions"]

    for sid in torch.unique(regions.scale_id).tolist():
        mask = regions.scale_id == int(sid)
        w_s = w[:, :, mask]  # [B, 1, M_s]
        mean_s = w_s.mean(dim=-1)  # [B, 1]
        assert torch.allclose(mean_s, torch.ones_like(mean_s), atol=0.05), \
            f"Scale {sid}: mean weight = {mean_s.mean():.4f}, expected ~1.0"


# ---------------------------------------------------------------------------
# T7. FP32 solver: no NaN or Inf
# ---------------------------------------------------------------------------
def test_t7_solver_fp32_no_nan_inf():
    """Solver adjoint field must contain no NaN or Inf values."""
    model = _make_model()
    x = _make_input(B=2, H=256, W=256)
    with torch.no_grad():
        out = model(x)

    for t, y in enumerate(out["iterates"]):
        assert torch.isfinite(y).all(), f"NaN/Inf in iterate Y{t}"

    y_final = out["y"]
    assert torch.isfinite(y_final).all(), "NaN/Inf in final Y"


# ---------------------------------------------------------------------------
# T8. Detach policy: b_region grad does not flow through solver
# ---------------------------------------------------------------------------
def test_t8_detach_policy():
    """With detach_region_mean_in_solver=True, solver step has no gradient to region_head."""
    model = _make_model()
    model.train()

    x = _make_input(B=2, H=256, W=256)
    target_y = torch.rand(2, 1, 64, 64, device=DEVICE) * 0.1  # sparse GT

    out = model(x)

    # Create a loss that ONLY flows through the solver output (y), not region_head loss
    loss = out["y"].mean()  # Deliberately no region_head loss term
    loss.backward()

    # Check that region_head.rate_head parameters have NO gradient
    # (because b_region was detached before solver)
    rate_weight_grad = model.region_head.rate_head.weight.grad
    if rate_weight_grad is not None:
        # rate_head.weight grad should be zero (detached from solver path)
        # Note: grad CAN be non-zero if reliability_head loss is active,
        # but here we only backprop through y.mean(), so it must be zero.
        assert rate_weight_grad.abs().max().item() < 1e-10, \
            f"rate_head.weight has non-zero grad via solver: {rate_weight_grad.abs().max():.2e}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
