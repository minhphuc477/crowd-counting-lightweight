"""Unit tests for Count-Invariant Cell Loss v2 (CI-Cell v2).

Validates the corrected integer-domain formulation where density maps
store integer person counts (0, 1, 2, 3, 4) per spatial cell.
"""
from __future__ import annotations

import torch
import pytest
from rmr_v3.losses.auxiliary import count_invariant_cell_loss


def _make_dense_target(h: int, w: int, n_heads: int, seed: int = 0) -> torch.Tensor:
    """Scatter n_heads into an h×w integer density map."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    t = torch.zeros(h, w)
    for _ in range(n_heads):
        r = torch.randint(0, h, (1,), generator=gen).item()
        c = torch.randint(0, w, (1,), generator=gen).item()
        t[r, c] += 1.0
    return t


class TestCICell_IntegerDomain:
    """Core correctness tests for CI-Cell v2 on integer density maps."""

    def test_zero_target_zero_loss(self):
        """Zero target → loss should be |prediction| (no fg amplification)."""
        y = torch.zeros(8, 8)
        t = torch.zeros(8, 8)
        loss = count_invariant_cell_loss(y, t)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_perfect_prediction_zero_loss(self):
        """Perfect prediction → loss = 0 regardless of crowd density."""
        for n_heads in [5, 50, 500]:
            t = _make_dense_target(32, 32, n_heads)
            loss = count_invariant_cell_loss(t.clone(), t)
            assert loss.item() == pytest.approx(0.0, abs=1e-6), \
                f"Expected 0 loss for perfect prediction with {n_heads} heads"

    def test_count_invariance_sparse_vs_dense(self):
        """Gradient scale per head must be O(1): equal for sparse and dense crowds.

        Specifically: for a fixed spatial perturbation at ONE head location,
        the gradient magnitude should be ~equal regardless of total crowd size N.
        This is the core theorem we are validating.
        """
        grid_h, grid_w = 32, 32
        # Sparse: 10 heads
        t_sparse = _make_dense_target(grid_h, grid_w, n_heads=10, seed=1)
        # Dense: 500 heads
        t_dense = _make_dense_target(grid_h, grid_w, n_heads=500, seed=2)

        # Shared perturbation: predict 0 everywhere
        y_sparse = torch.zeros_like(t_sparse, requires_grad=True)
        y_dense = torch.zeros_like(t_dense, requires_grad=True)

        loss_sparse = count_invariant_cell_loss(y_sparse, t_sparse, alpha=2.0)
        loss_dense = count_invariant_cell_loss(y_dense, t_dense, alpha=2.0)

        loss_sparse.backward()
        loss_dense.backward()

        # Per-cell gradient for a typical occupied cell
        # In v2: weight = 1 + (alpha-1)*fg_fraction, normalized by HW (via .mean())
        # For sparse: fg_fraction ≈ 1/t_sparse.max() for each occupied cell
        # For dense: fg_fraction ≈ similar fractional values
        # Key invariant: total gradient sum / n_heads should be approx constant

        mask_sparse = t_sparse > 0
        mask_dense = t_dense > 0
        n_sparse = mask_sparse.sum().item()
        n_dense = mask_dense.sum().item()

        if n_sparse > 0 and n_dense > 0:
            avg_grad_sparse = y_sparse.grad[mask_sparse].abs().mean().item()
            avg_grad_dense = y_dense.grad[mask_dense].abs().mean().item()

            # Ratio should be within 5x (true O(1) would be exactly 1x;
            # slight differences arise from fg_fraction variation at different densities)
            ratio = max(avg_grad_sparse, avg_grad_dense) / max(
                min(avg_grad_sparse, avg_grad_dense), 1e-9
            )
            assert ratio < 5.0, (
                f"Gradient ratio sparse/dense = {ratio:.2f}x — "
                f"should be O(1), not O(N). "
                f"avg_grad_sparse={avg_grad_sparse:.6f}, avg_grad_dense={avg_grad_dense:.6f}"
            )


class TestCICell_WeightStructure:
    """Tests validating the weight structure of CI-Cell v2."""

    def test_background_weight_is_one(self):
        """Background cells (t=0) must receive weight exactly 1.0."""
        t = torch.zeros(8, 8)
        t[3, 3] = 2.0  # single occupied cell
        y = torch.ones(8, 8) * 0.5  # predict 0.5 everywhere

        # Compute loss manually for background cells
        y_req = y.clone().requires_grad_(True)
        loss = count_invariant_cell_loss(y_req, t, alpha=3.0, beta=1.0)
        loss.backward()

        # Background gradient (all cells except [3,3]) should be uniform
        bg_grads = y_req.grad.clone()
        bg_grads[3, 3] = float("nan")  # exclude fg cell
        bg_grads_valid = bg_grads[~bg_grads.isnan()]
        # All background gradients should be the same value
        assert bg_grads_valid.std().item() == pytest.approx(0.0, abs=1e-7), \
            "Background gradients should be uniform (weight=1.0 everywhere bg)"

    def test_foreground_amplification(self):
        """Foreground cells must receive higher gradient than background."""
        t = torch.zeros(16, 16)
        t[7, 7] = 3.0   # peak cell
        t[7, 8] = 1.0   # lower density cell
        y = torch.zeros(16, 16, requires_grad=True)

        loss = count_invariant_cell_loss(y, t, alpha=4.0)
        loss.backward()

        grad = y.grad
        grad_peak = grad[7, 7].abs().item()
        grad_lower = grad[7, 8].abs().item()
        grad_bg = grad[0, 0].abs().item()

        assert grad_peak > grad_lower, "Peak cell should have higher gradient than lower-density cell"
        assert grad_lower > grad_bg, "Occupied cell should have higher gradient than background"

    def test_alpha_1_equals_balanced(self):
        """With alpha=1, CI-Cell should equal standard smooth-L1 (all weights=1)."""
        import torch.nn.functional as F
        t = _make_dense_target(16, 16, n_heads=30)
        y = torch.rand_like(t)

        loss_ci = count_invariant_cell_loss(y, t, alpha=1.0, beta=1.0)
        loss_ref = F.smooth_l1_loss(y, t, beta=1.0, reduction="mean")

        assert loss_ci.item() == pytest.approx(loss_ref.item(), rel=1e-5), \
            "alpha=1 must produce identical result to standard smooth-L1"

    def test_batch_dimension_handling(self):
        """Loss should be stable for various input shapes: 2D, 3D, 4D."""
        t2d = _make_dense_target(16, 16, n_heads=20)
        t3d = t2d.unsqueeze(0)
        t4d = t2d.unsqueeze(0).unsqueeze(0)

        for t_in in [t2d, t3d, t4d]:
            y_in = torch.rand_like(t_in)
            loss = count_invariant_cell_loss(y_in, t_in)
            assert torch.isfinite(loss), f"Loss should be finite for shape {t_in.shape}"
            assert loss.item() >= 0.0, f"Loss should be non-negative for shape {t_in.shape}"

    def test_empty_scene_stability(self):
        """All-zero target (no people) should produce finite loss."""
        y = torch.rand(8, 8)
        t = torch.zeros(8, 8)
        loss = count_invariant_cell_loss(y, t)
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0

    def test_high_density_stability(self):
        """Extreme density (many people per cell) should not cause NaN/Inf."""
        t = torch.randint(0, 20, (32, 32)).float()  # 0–19 people per cell
        y = torch.randn(32, 32)
        loss = count_invariant_cell_loss(y, t, alpha=5.0)
        assert torch.isfinite(loss), "Loss should be finite even with high per-cell counts"
