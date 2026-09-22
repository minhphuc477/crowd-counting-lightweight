import torch
import pytest
from rmr_v3.losses.auxiliary import count_invariant_cell_loss


def test_count_invariant_cell_loss_basic():
    """Verify basic forward and autograd gradient flow."""
    y = torch.ones((2, 1, 64, 64), requires_grad=True)
    target = torch.ones((2, 1, 64, 64))
    loss = count_invariant_cell_loss(y, target, beta=1.0, alpha=2.0, tau_head=0.08)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
    loss.backward()
    assert y.grad is not None
    assert torch.isfinite(y.grad).all()


def test_count_invariant_cell_loss_background_suppression():
    """Verify background suppression stream behaves correctly."""
    # Perfectly matching background: zero loss
    y_bg = torch.zeros((1, 1, 32, 32), requires_grad=True)
    target_bg = torch.zeros((1, 1, 32, 32))
    loss_clean = count_invariant_cell_loss(y_bg, target_bg)
    assert loss_clean.item() == 0.0

    # False positive on background: positive loss
    y_halluc = torch.full((1, 1, 32, 32), 0.5, requires_grad=True)
    loss_halluc = count_invariant_cell_loss(y_halluc, target_bg)
    assert loss_halluc.item() > 0.0


def test_count_invariant_cell_loss_gradient_invariance():
    """Verify that dense crowd heads do NOT suffer from O(1/N) gradient decay."""
    # Case 1: Sparse image (1 head at center)
    y_sparse = torch.zeros((1, 1, 64, 64), requires_grad=True)
    t_sparse = torch.zeros((1, 1, 64, 64))
    t_sparse[0, 0, 32, 32] = 0.08  # 1 head

    loss_sparse = count_invariant_cell_loss(y_sparse, t_sparse, tau_head=0.08, alpha=2.0)
    loss_sparse.backward()
    grad_sparse_head = y_sparse.grad[0, 0, 32, 32].abs().item()

    # Case 2: Dense image (100 heads)
    y_dense = torch.zeros((1, 1, 64, 64), requires_grad=True)
    t_dense = torch.zeros((1, 1, 64, 64))
    t_dense[0, 0, 10:20, 10:20] = 0.08  # 100 heads

    loss_dense = count_invariant_cell_loss(y_dense, t_dense, tau_head=0.08, alpha=2.0)
    loss_dense.backward()
    grad_dense_head = y_dense.grad[0, 0, 15, 15].abs().item()

    # In legacy mass_weighted, grad_dense_head was 100x smaller!
    # In count_invariant_cell_loss, the ratio must be bounded and close to 1.0 (not 100x smaller).
    ratio = grad_dense_head / max(grad_sparse_head, 1e-8)
    assert 0.2 < ratio < 5.0, f"Expected comparable per-head gradient, got ratio {ratio}"
