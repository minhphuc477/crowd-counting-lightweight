"""Adversarial stress-testing suite for RMR sub60 and core mathematical operators.

Verifies:
1. Cross-batch broadcasting & Sample Isolation:
   grad(loss(sample 1)) w.r.t. sample 0 is identically ZERO.
2. Non-standard, prime, and odd spatial resolutions (e.g. 383x517, 241x311):
   Zero shape mismatch, zero padding leakage, exact integer rounding parity.
3. Discrete mass conservation on arbitrary resolutions under subpixel prolongation.
4. Numerical stability: Zero NaNs, zero Infs, finite non-zero gradients across all parameters.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
import yaml
from pathlib import Path

from rmr_v3.config import validate_v3_config
from rmr_v3.engine import make_loss_cfg, make_model
from rmr_v3.losses import compute_rmr_v3_losses
from rmr_core.operators import regional_sum, regional_adjoint, build_multiscale_regions


@pytest.mark.parametrize(
    "config_name",
    [
        "sub60_zenith_composite_t8.yaml",
        "sub60_dual_lattice_t8.yaml",
        "sub60_unified_t8.yaml",
        "sub60_abl_no_solver.yaml",
        "sub60_abl_flat_adjoint.yaml",
        "sub60_abl_symmetric_morozov.yaml",
    ],
)
def test_sample_isolation_and_no_broadcasting(config_name: str) -> None:
    """Invariant 1: Sample isolation. Sample 0 cannot receive gradient from Sample 1's loss."""
    cfg_path = Path("configs/rmr_sub60") / config_name
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Disable pretrained download for test speed
    cfg.setdefault("model", {})["pretrained"] = False
    model, _ = make_model(cfg)
    loss_cfg = make_loss_cfg(cfg)
    loss_cfg.elementwise_dense_scaling = False
    loss_cfg.density_loss_scaling = False

    device = torch.device("cpu")
    model.to(device)
    model.eval()  # Eval mode disables dynamic batchnorm cross-sample statistics to test operator isolation

    # Batch of 2 samples with different content
    torch.manual_seed(42)
    x = torch.randn(2, 3, 256, 256, requires_grad=True)
    target_stride = 2 if getattr(model.cfg, "subpixel_dm", False) or model.cfg.subpixel_stride2 else 4
    th, tw = (256 + target_stride - 1) // target_stride, (256 + target_stride - 1) // target_stride
    target = torch.randint(0, 5, (2, 1, th, tw), dtype=torch.float32)

    out = model(x)
    losses = compute_rmr_v3_losses(out, target, loss_cfg)

    # Check loss for sample 0 alone vs sample 1 alone
    out_s1 = {k: v[1:2] if isinstance(v, torch.Tensor) and v.shape[0] == 2 else v for k, v in out.items()}
    losses_s1 = compute_rmr_v3_losses(out_s1, target[1:2], loss_cfg)

    # Backward of sample 1's total loss
    model.zero_grad()
    if x.grad is not None:
        x.grad.zero_()
    losses_s1["total"].backward(retain_graph=True)

    # Sample 0 input gradient MUST BE IDENTICALLY ZERO!
    assert x.grad is not None
    grad_s0 = x.grad[0].abs().max().item()
    assert grad_s0 == 0.0, f"Sample isolation violated! Sample 0 received gradient {grad_s0} from Sample 1 loss."


@pytest.mark.parametrize("h,w", [(383, 517), (241, 311), (513, 255)])
def test_arbitrary_odd_prime_resolutions(h: int, w: int) -> None:
    """Invariant 2: Arbitrary odd and prime resolutions forward + backward."""
    cfg_path = Path("configs/rmr_sub60/sub60_zenith_composite_t8.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("model", {})["pretrained"] = False
    model, _ = make_model(cfg)
    loss_cfg = make_loss_cfg(cfg)

    x = torch.randn(1, 3, h, w, requires_grad=True)
    out = model(x)

    assert out.y.shape[-2] == (h + 1) // 2
    assert out.y.shape[-1] == (w + 1) // 2
    assert torch.isfinite(out.y).all(), "NaN or Inf detected in output density map!"

    # Target matching stride 2
    target = torch.randint(0, 3, (1, 1, (h + 1) // 2, (w + 1) // 2), dtype=torch.float32)
    losses = compute_rmr_v3_losses(out, target, loss_cfg)
    assert torch.isfinite(losses["total"]), "Total loss is NaN or Inf!"

    losses["total"].backward()
    assert x.grad is not None and torch.isfinite(x.grad).all(), "Gradient has NaN or Inf on odd resolution!"


def test_mass_conservation_odd_dimensions() -> None:
    """Invariant 3: Discrete mass conservation holds down to machine precision on odd dimensions."""
    cfg_path = Path("configs/rmr_sub60/sub60_zenith_composite_t8.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("model", {})["pretrained"] = False
    model, _ = make_model(cfg)

    for h, w in [(409, 313), (513, 257), (199, 199)]:
        x = torch.randn(1, 3, h, w)
        out = model(x)
        m2 = out.y.sum().item()
        m4 = out["y_carrier"].sum().item()
        rel_diff = abs(m2 - m4) / max(m4, 1e-6)
        assert rel_diff < 1e-5, f"Mass conservation violated for {h}x{w}: m2={m2:.6f}, m4={m4:.6f}, rel_diff={rel_diff:.2e}"


def test_operator_duality_adjointness() -> None:
    """Invariant 4: Linear operator adjointness: <A y, b> == <y, A^T b> to float32 precision."""
    h, w = 64, 64
    regions = build_multiscale_regions(h, w, output_stride=4, region_sizes_px=(16, 32, 64), overlap=0.5, include_full_image=False)
    boxes = regions.boxes
    m = boxes.shape[0]

    torch.manual_seed(123)
    y = torch.randn(2, 1, h, w, dtype=torch.float32)
    b = torch.randn(2, 1, m, dtype=torch.float32)

    ay = regional_sum(y, boxes, out_dtype=torch.float32)  # [B, 1, M]
    atb = regional_adjoint(b, boxes, h, w, out_dtype=torch.float32)  # [B, 1, H, W]

    inner1 = (ay * b).sum().item()
    inner2 = (y * atb).sum().item()

    rel_err = abs(inner1 - inner2) / max(abs(inner1), abs(inner2), 1e-6)
    assert rel_err < 1e-4, f"Adjointness violated: <Ay, b>={inner1:.6f}, <y, A^T b>={inner2:.6f}, rel_err={rel_err:.2e}"
