"""Unit tests for HPC-Lite Probabilistic Hierarchy Learning Formulation."""

import math
import torch
import torch.nn.functional as F

from hpc.losses.probabilistic_hierarchy import (
    RootNegativeBinomialLoss,
    HierarchicalDirichletMultinomialLoss,
    HardZeroMiningLoss,
    MassCalibrationLoss,
    HPCLiteUnifiedCriterion,
    group_quadtree_children,
)


def test_quadtree_grouping():
    """Verify that group_quadtree_children accurately groups 2x2 child cells."""
    B, H, W = 2, 8, 8
    child_grid = torch.arange(B * 1 * (2*H) * (2*W), dtype=torch.float32).view(B, 1, 2*H, 2*W)
    grouped = group_quadtree_children(child_grid)

    assert grouped.shape == (B, H, W, 4), f"Expected shape {(B, H, W, 4)}, got {grouped.shape}"

    # Check child indices for parent (0, 0)
    for b in range(B):
        for h in range(H):
            for w in range(W):
                c0 = child_grid[b, 0, 2*h, 2*w].item()
                c1 = child_grid[b, 0, 2*h, 2*w+1].item()
                c2 = child_grid[b, 0, 2*h+1, 2*w].item()
                c3 = child_grid[b, 0, 2*h+1, 2*w+1].item()
                g = grouped[b, h, w].tolist()
                assert g == [c0, c1, c2, c3], f"Mismatch at ({b}, {h}, {w}): expected {[c0, c1, c2, c3]}, got {g}"

    print("✓ test_quadtree_grouping passed!")


def test_root_nb_loss():
    """Verify Root Negative-Binomial loss gradient and numerical stability."""
    crit = RootNegativeBinomialLoss(dispersion=10.0)
    mu = torch.tensor([10.0, 50.0, 200.0], requires_grad=True)
    gt = torch.tensor([12.0, 48.0, 210.0])

    loss = crit(mu, gt)
    assert not torch.isnan(loss), "NaN in Root NB loss"
    loss.backward()
    assert not torch.isnan(mu.grad).any(), "NaN in Root NB gradients"
    assert (mu.grad is not None) and (mu.grad.shape == mu.shape)

    print("✓ test_root_nb_loss passed!")


def test_hierarchical_dm_loss():
    """Verify Dirichlet-Multinomial quadtree loss and gradients."""
    crit = HierarchicalDirichletMultinomialLoss(concentration_alpha=10.0)

    B = 2
    # Grid 64: 7x7 (stride 64 on 448x448)
    # Grid 32: 14x14
    # Grid 16: 28x28
    mu_blocks = {
        64: torch.rand(B, 1, 7, 7, requires_grad=True),
        32: torch.rand(B, 1, 14, 14, requires_grad=True),
        16: torch.rand(B, 1, 28, 28, requires_grad=True),
    }

    gt_blocks = {
        64: torch.randint(0, 50, (B, 1, 7, 7), dtype=torch.float32),
        32: torch.randint(0, 20, (B, 1, 14, 14), dtype=torch.float32),
        16: torch.randint(0, 10, (B, 1, 28, 28), dtype=torch.float32),
    }

    loss, details = crit(mu_blocks, gt_blocks)
    assert not torch.isnan(loss), "NaN in Hierarchical DM loss"
    loss.backward()

    for b in (32, 16):
        mu = mu_blocks[b]
        assert mu.grad is not None and not torch.isnan(mu.grad).any(), f"NaN in mu_{b} gradients"

    print("✓ test_hierarchical_dm_loss passed!")


def test_hard_zero_mining():
    """Verify hard zero background suppression."""
    crit = HardZeroMiningLoss(topk_fraction=0.10)
    mu16 = torch.tensor([0.1, 0.5, 0.9, 0.05, 0.01, 10.0, 0.0], requires_grad=True)
    y16 = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0])

    loss = crit(mu16, y16)
    assert loss.item() > 0.0, "Hard zero loss should be positive"
    loss.backward()
    assert mu16.grad[5].item() == 0.0, "Non-zero ground truth block should have zero loss gradient in hard-zero mining"
    assert mu16.grad[2].item() > 0.0, "Top false positive block should receive positive gradient to suppress predicted mass"

    print("✓ test_hard_zero_mining passed!")


def test_unified_criterion_end_to_end():
    """Verify HPCLiteUnifiedCriterion full forward/backward pass with a simulated model output."""
    criterion = HPCLiteUnifiedCriterion(
        dispersion_r=10.0,
        dm_concentration_alpha=10.0,
        hz_topk_fraction=0.10,
        lambda_tree=1.0,
        lambda_hard_zero=0.25,
        lambda_mass=1.0,
    )

    B = 2
    raw_logits = torch.randn(B, 1, 112, 112, requires_grad=True)
    density_map = F.softplus(raw_logits)

    gt_blocks = {
        64: torch.randint(0, 50, (B, 1, 7, 7), dtype=torch.float32),
        32: torch.randint(0, 20, (B, 1, 14, 14), dtype=torch.float32),
        16: torch.randint(0, 10, (B, 1, 28, 28), dtype=torch.float32),
    }
    gt_count = torch.tensor([150.0, 230.0], dtype=torch.float32)

    total_loss, details = criterion(density_map, gt_blocks, gt_count)
    assert not torch.isnan(total_loss), "NaN in total unified loss"

    total_loss.backward()
    assert raw_logits.grad is not None and not torch.isnan(raw_logits.grad).any(), "NaN in raw_logits gradients"

    print("\n--- Unified Loss Component Diagnostics ---")
    for k, v in details.items():
        print(f"  {k:20s}: {v.item():.4f}")
    print("✓ test_unified_criterion_end_to_end passed 100% perfectly!")


if __name__ == "__main__":
    test_quadtree_grouping()
    test_root_nb_loss()
    test_hierarchical_dm_loss()
    test_hard_zero_mining()
    test_unified_criterion_end_to_end()
