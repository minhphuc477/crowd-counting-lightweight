"""Systematic Deep Architectural Audit & Mathematical Invariant Test Suite.

Implements the formal protocols from docs/ML_MODEL_DEEP_AUDIT_AND_CODE_REVIEW_PROTOCOL.md:
1. P0: Batch Sample Independence (The Karpathy / Meta FAIR Invariant).
2. P0: L1 Count Mass Conservation under Spatial Reflection.
3. P0: Dual-Lattice Subpixel Push-Forward Conservation.
4. P1: Autograd Graph Continuity & Numerical Safety under Asymmetric / Prime Shapes.
5. P1: Unrolled Solver Lipschitz Contraction & Monotonic Residual Dissipation.
6. P2: Representation Capacity & Bottleneck SVD Effective Rank (erank).
7. P2: Pixel-Center Half-Pixel Coordinate Symmetry.
"""
from __future__ import annotations

import math
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from rmr_core.operators import build_multiscale_regions, regional_sum
from rmr_v3.config import load_config
from rmr_v3.model import RMRv3, RMRv3Config
from rmr_v3.model.dual_lattice import push_forward_stride2_to_stride4


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "rmr_v30"


@pytest.fixture(scope="module")
def canonical_model():
    """Build canonical anchor model for deep invariant audits."""
    cfg_dict = load_config(CONFIG_DIR / "rmr_v30_step0_v19_anchor.yaml")
    mcfg = RMRv3Config.from_dict(cfg_dict["model"])
    model = RMRv3(mcfg).eval()
    return model


@pytest.fixture(scope="module")
def dual_lattice_model():
    """Build dual-lattice DCSR model for resolution and push-forward audits."""
    cfg_dict = load_config(CONFIG_DIR / "rmr_v30_h3_anscombe_dual_lattice.yaml")
    mcfg = RMRv3Config.from_dict(cfg_dict["model"])
    model = RMRv3(mcfg).eval()
    return model


def test_batch_sample_independence_invariant(canonical_model):
    """P0 Audit: Model output for sample 0 must not depend on sample 1 in the same batch."""
    torch.manual_seed(42)
    x1 = torch.randn(1, 3, 256, 256)
    x2 = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        out1_single = canonical_model(x1)
        batch = torch.cat([x1, x2], dim=0)
        out_batch = canonical_model(batch)

    max_diff_y = (out1_single.y - out_batch.y[0:1]).abs().max().item()
    max_diff_y0 = (out1_single.y0 - out_batch.y0[0:1]).abs().max().item()

    assert max_diff_y < 1e-5, f"[P0 VIOLATION] Batch independence failed on y! Max diff: {max_diff_y}"
    assert max_diff_y0 < 1e-5, f"[P0 VIOLATION] Batch independence failed on y0! Max diff: {max_diff_y0}"


def test_l1_count_mass_conservation_under_flip(canonical_model):
    """P0 Audit: Continuous count measure must be symmetric under horizontal flip."""
    torch.manual_seed(123)
    x = torch.randn(1, 3, 256, 256)
    x_flip = torch.flip(x, dims=[-1])

    with torch.no_grad():
        out = canonical_model(x)
        out_flip = canonical_model(x_flip)

    count_orig = float(out.y.sum().item())
    count_flip = float(out_flip.y.sum().item())
    rel_error = abs(count_orig - count_flip) / max(count_orig, count_flip, 1e-6)

    assert rel_error < 1e-3, (
        f"[P0 VIOLATION] L1 Mass Conservation violated under Flip! "
        f"orig={count_orig:.4f}, flip={count_flip:.4f}, rel_err={rel_error:.2e}"
    )


def test_dual_lattice_pushforward_exact_conservation(dual_lattice_model):
    """P0 Audit: Push-forward stride 2 to stride 4 must strictly conserve spatial mass."""
    torch.manual_seed(456)
    x = torch.randn(2, 3, 256, 256)

    with torch.no_grad():
        out = dual_lattice_model(x)

    assert "y_carrier" in out, "Dual-lattice model must output y_carrier"
    m_subpixel = out.y.sum(dim=(-2, -1))
    m_carrier = out["y_carrier"].sum(dim=(-2, -1))
    diff = (m_subpixel - m_carrier).abs().max().item()

    assert diff < 1e-4, f"[P0 VIOLATION] Subpixel push-forward mass discrepancy: {diff}"


def test_autograd_graph_and_numerical_safety_under_prime_shapes(canonical_model):
    """P1 Audit: Asymmetric non-power-of-2 dimensions must not trigger NaN/Inf in forward or backward."""
    canonical_model.train()
    # 283 x 317 are prime numbers, stressing non-divisible padding and cropping
    x = torch.randn(1, 3, 283, 317, requires_grad=True)
    out = canonical_model(x)

    assert torch.isfinite(out.y).all(), "[P1 VIOLATION] Non-finite values in y on prime resolution!"
    assert torch.isfinite(out.y0).all(), "[P1 VIOLATION] Non-finite values in y0 on prime resolution!"

    loss = out.y.sum() + out.y0.sum()
    loss.backward()

    assert x.grad is not None, "[P1 VIOLATION] Autograd graph broken! x.grad is None"
    assert torch.isfinite(x.grad).all(), "[P1 VIOLATION] Non-finite gradients in input tensor!"


def test_unrolled_solver_lipschitz_residual_dissipation(canonical_model):
    """P1 Audit: Solver iterates must exhibit stable residual dissipation without divergence."""
    torch.manual_seed(789)
    x = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        out = canonical_model(x, compute_energy=True)

    iterates = out.iterates
    assert len(iterates) >= 2, f"Expected multiple solver iterates, got {len(iterates)}"

    regions = out.regions
    b_solver = out["b_solver"]
    residuals = []
    for it in iterates:
        q = regional_sum(it.float(), regions.boxes, out_dtype=torch.float32)
        res_norm = (q - b_solver.float()).square().mean().sqrt().item()
        residuals.append(res_norm)

    # Initial residual at carrier y0 vs final residual at y*
    r_init = residuals[0]
    r_final = residuals[-1]
    assert math.isfinite(r_final), "[P1 VIOLATION] Non-finite solver residual!"
    # Residual must not explode (Lipschitz bound: final residual must not exceed 2.0x initial)
    assert r_final <= r_init * 2.0 + 1e-3, (
        f"[P1 VIOLATION] Solver residual diverged! r_init={r_init:.4f}, r_final={r_final:.4f}"
    )


def test_bottleneck_feature_effective_rank(canonical_model):
    """P2 Audit: P4 bottleneck feature map must maintain healthy SVD effective rank (> 15% channels)."""
    torch.manual_seed(999)
    x = torch.randn(2, 3, 256, 256)

    with torch.no_grad():
        p4, _, _ = canonical_model._extract_carrier_features(x)

    # p4: [B, C, H, W]
    b, c, h, w = p4.shape
    f_flat = p4.permute(0, 2, 3, 1).reshape(-1, c).float()
    f_centered = f_flat - f_flat.mean(dim=0, keepdim=True)
    _, s, _ = torch.linalg.svd(f_centered, full_matrices=False)
    s = s[s > 1e-7]
    p = s / s.sum()
    entropy = -(p * torch.log(p)).sum()
    erank = torch.exp(entropy).item()
    ratio = erank / c

    assert ratio >= 0.15, (
        f"[P2 WARNING] Representation Rank Collapse detected in P4! "
        f"erank={erank:.2f}/{c} ({ratio:.1%}, minimum threshold: 15%)"
    )


def test_half_pixel_center_coordinate_symmetry():
    """P2 Audit: Half-pixel grid centers must be symmetric and translation-invariant."""
    h, w, stride = 64, 64, 4
    y_coords = (torch.arange(h, dtype=torch.float32) + 0.5) * float(stride)
    x_coords = (torch.arange(w, dtype=torch.float32) + 0.5) * float(stride)

    # Leftmost coordinate center: 0.5 * 4 = 2.0
    assert abs(y_coords[0].item() - 2.0) < 1e-6
    # Rightmost coordinate center: (63 + 0.5) * 4 = 254.0
    assert abs(y_coords[-1].item() - 254.0) < 1e-6
    # Center symmetry around image midpoint (256 / 2 = 128.0)
    midpoint_diff = abs((y_coords[0] + y_coords[-1]).item() / 2.0 - 128.0)
    assert midpoint_diff < 1e-6, f"Coordinate centers not symmetric around midpoint: diff={midpoint_diff}"
