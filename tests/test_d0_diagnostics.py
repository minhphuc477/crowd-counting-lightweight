"""Unit tests for D0 Diagnostic Suite (D-R, D-K, D-L, D-M)."""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from hpc.diagnostics.effective_rank import (
    compute_spectral_rank_metrics,
    evaluate_effective_rank_single_image,
)
from hpc.diagnostics.gradient_allocation import evaluate_gradient_allocation_single_batch
from hpc.diagnostics.phase_shift import (
    evaluate_phase_shift_single_image,
    inverse_align_feature,
    shift_tensor,
)
from hpc.diagnostics.separability import (
    compute_head_scale_proxies,
    evaluate_separability_single_image,
    sample_feature_at_image_coord,
)
from hpc.losses.ntpc import NTPCConfig, NTPCLoss
from hpc.models.hpc_lite import HPCLite


def test_shift_tensor_and_inverse_align_recovery():
    # Synthetic impulse image: single non-zero pixel in the center
    x = torch.zeros(1, 1, 32, 32)
    x[0, 0, 16, 16] = 10.0
    
    # Shift right 2, down 1
    dx, dy = 2, 1
    shifted = shift_tensor(x, dx, dy, mode="replicate")
    assert shifted[0, 0, 17, 18] == 10.0
    
    # Inverse align back with stride 1.0
    inv = inverse_align_feature(shifted, dx, dy, stride=1.0, device=torch.device("cpu"))
    
    # Check that maximum peak is recovered at (16, 16)
    peak_y, peak_x = torch.where(inv[0, 0] == inv[0, 0].max())
    assert peak_y[0].item() == 16
    assert peak_x[0].item() == 16


def test_head_scale_proxies():
    # 4 collinear points separated by 10 px
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]], dtype=np.float32)
    s_heads = compute_head_scale_proxies(pts, k=2)
    assert len(s_heads) == 4
    # Middle points (10, 0) and (20, 0) have 2 nearest neighbors at distance 10 px
    assert np.isclose(s_heads[1], 10.0, atol=1e-3)
    assert np.isclose(s_heads[2], 10.0, atol=1e-3)


def test_spectral_rank_metrics_sample_matched():
    # Rank-1 matrix with variance (outer product u @ v)
    u = torch.randn(256, 1)
    v = torch.randn(1, 32)
    rank1 = u @ v
    metrics_rank1 = compute_spectral_rank_metrics(rank1)
    assert metrics_rank1["normalized_participation_ratio"] <= 0.10
    assert metrics_rank1["top1_energy_ratio"] >= 0.95
    
    # Random orthogonal matrix of 32 channels
    q, _ = torch.linalg.qr(torch.randn(256, 32))
    metrics_q = compute_spectral_rank_metrics(q)
    assert metrics_q["normalized_participation_ratio"] >= 0.85


def test_gradient_allocation_preserves_frozen_model_state():
    model = HPCLite(
        backbone_name="mobilenetv4_conv_small_050",
        pretrained=False,
        neck_width=16,
        feature_reductions=(4, 8, 16),
    )
    model.eval()
    
    # Snapshot buffer states (BatchNorm running mean/var if any)
    orig_buffers = {k: v.clone() for k, v in model.named_buffers()}
    
    criterion = NTPCLoss(NTPCConfig(mode="r2_flat_dm", root_loss="nb"))
    img = torch.randn(1, 3, 64, 64)
    targets = {4: torch.zeros(1, 16, 16), 16: torch.zeros(1, 4, 4), "N": torch.zeros(1)}
    
    res = evaluate_gradient_allocation_single_batch(model, criterion, img, targets)
    assert "C16_fg_energy_fraction" in res
    assert "C16_gradient_enrichment" in res
    
    # Check that model remains in eval mode and buffers are unchanged
    assert not model.training
    for k, v in model.named_buffers():
        assert torch.equal(v, orig_buffers[k])
