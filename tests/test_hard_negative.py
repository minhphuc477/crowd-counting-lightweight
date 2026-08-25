import pytest
import torch
from hpc.losses.hard_negative import HardNegativeMassLoss, WholeImageEmptyLoss
from hpc.losses.allocation import LocalAllocationLoss
from hpc.losses.criterion import HPCLossCriterion


def test_t8_all_empty_batch():
    """T8: Test all-empty batch where count=0 for all samples."""
    criterion = HPCLossCriterion(
        block_sizes=[16, 32, 64],
        allocation_block=16,
        lambda_hnb=1.0,
        lambda_alloc=0.5,
        lambda_hn=0.25,
        lambda_empty=0.5,
        lambda_global=1.0,
    )
    
    b, c, h, w = 2, 1, 112, 112
    d_map = torch.full((b, c, h, w), 0.05, requires_grad=True)
    
    gt_blocks = {
        16: torch.zeros((b, 28, 28)),
        32: torch.zeros((b, 14, 14)),
        64: torch.zeros((b, 7, 7)),
    }
    gt_z_alloc = torch.zeros((b, h, w))
    gt_counts = torch.zeros((b,))
    
    loss, loss_dict = criterion(d_map, gt_blocks, gt_z_alloc, gt_counts, progress=1.0)
    
    assert torch.isfinite(loss)
    assert not torch.isnan(loss)
    assert loss_dict["loss_alloc"].item() == 0.0, "Allocation loss on empty batch must be 0"
    assert loss_dict["loss_empty"].item() > 0.0, "Empty loss on empty batch must penalize predicted mass"
    assert loss_dict["loss_hn"].item() > 0.0, "Hard negative loss on empty batch must penalize zero blocks"
    
    # Backward pass
    loss.backward()
    assert torch.isfinite(d_map.grad).all()


def test_t9_no_zero_block_batch():
    """T9: Test batch with zero empty blocks (all blocks positive)."""
    hn_loss_fn = HardNegativeMassLoss(top_fraction=0.10, block_size=16)
    
    b, c, h, w = 2, 1, 112, 112
    d_map = torch.rand((b, c, h, w), requires_grad=True)
    
    # All 16x16 blocks have count >= 1
    gt_y16 = torch.ones((b, 28, 28)) * 5.0
    
    loss = hn_loss_fn(d_map, gt_y16)
    assert loss.item() == 0.0, "HN loss with no zero blocks must return exactly 0"
