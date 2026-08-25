import pytest
import torch
from hpc.losses.negative_binomial import sum_pool


def test_t6_sum_pooling_exact_conservation():
    """T6: Test sum-pooling exact conservation for all hierarchical block scales."""
    h_out, w_out = 112, 112  # Corresponding to 448x448 crop at stride 4
    d_map = torch.rand(4, 1, h_out, w_out)
    
    exact_sum_d = d_map.sum(dim=(-1, -2, -3))  # (B,)
    
    for input_block_size in [16, 32, 64]:
        mu_b = sum_pool(d_map, input_block_size=input_block_size, output_stride=4)
        sum_mu = mu_b.sum(dim=(-1, -2, -3))  # (B,)
        
        # Account for float32 summation associativity discrepancy over 12,544 elements
        assert torch.allclose(sum_mu, exact_sum_d, rtol=1e-4, atol=1e-3), (
            f"Scale {input_block_size}: Sum pool discrepancy {torch.abs(sum_mu - exact_sum_d)}"
        )
        
    # Test for 672x672 crop (stride 4 -> 168x168) with 16, 32, 96 block sizes
    d_672 = torch.rand(2, 1, 168, 168)
    exact_sum_672 = d_672.sum(dim=(-1, -2, -3))
    
    for input_block_size in [16, 32, 96]:
        mu_b = sum_pool(d_672, input_block_size=input_block_size, output_stride=4)
        sum_mu = mu_b.sum(dim=(-1, -2, -3))
        
        assert torch.allclose(sum_mu, exact_sum_672, rtol=1e-4, atol=1e-3), (
            f"672 crop scale {input_block_size}: Discrepancy {torch.abs(sum_mu - exact_sum_672)}"
        )
