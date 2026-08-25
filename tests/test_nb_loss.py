import pytest
import torch
from hpc.losses.negative_binomial import nb_nll, HierarchicalNBLoss


def test_t7_nb_nll_numerical_stability():
    """T7: Test Negative-Binomial loss numerical stability across extreme counts."""
    # Test cases: (y, mu, r)
    cases = [
        (0.0, 1e-6, 1.0),
        (0.0, 100.0, 5.0),
        (1.0, 1.0, 2.0),
        (100.0, 100.0, 10.0),
        (1000.0, 950.0, 25.0),
        (20000.0, 19800.0, 50.0),  # Extreme dense crowd point
        (20000.0, 1e-5, 0.5),     # Severe undercount
        (0.0, 20000.0, 0.5),      # Severe overcount
    ]
    
    for y_val, mu_val, r_val in cases:
        y = torch.tensor([y_val], dtype=torch.float32)
        mu = torch.tensor([mu_val], dtype=torch.float32, requires_grad=True)
        r = torch.tensor([r_val], dtype=torch.float32, requires_grad=True)
        
        loss = nb_nll(y, mu, r)
        assert torch.isfinite(loss).all(), f"NB loss not finite for y={y_val}, mu={mu_val}, r={r_val}: got {loss}"
        
        # Test backward
        loss.backward()
        assert torch.isfinite(mu.grad).all(), f"mu grad not finite for y={y_val}, mu={mu_val}"
        assert torch.isfinite(r.grad).all(), f"r grad not finite for y={y_val}, r={r_val}"


def test_t7_hierarchical_nb_module():
    """T7: Test HierarchicalNBLoss module forward & backward."""
    block_sizes = [16, 32, 64]
    quantiles = {
        16: (1.0, 5.0),
        32: (4.0, 20.0),
        64: (16.0, 80.0),
    }
    loss_fn = HierarchicalNBLoss(block_sizes=block_sizes, quantiles=quantiles, use_stratified=True)
    
    d_map = torch.rand(2, 1, 112, 112, requires_grad=True)
    gt_blocks = {
        16: torch.randint(0, 10, (2, 28, 28)).float(),
        32: torch.randint(0, 30, (2, 14, 14)).float(),
        64: torch.randint(0, 100, (2, 7, 7)).float(),
    }
    
    total_loss, details = loss_fn(d_map, gt_blocks)
    assert torch.isfinite(total_loss)
    assert total_loss.item() > 0.0
    
    total_loss.backward()
    assert torch.isfinite(d_map.grad).all()
