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
from hpc.diagnostics.phase_shift import (
    evaluate_phase_shift_single_image,
    shift_tensor,
)
from hpc.diagnostics.separability import (
    compute_knn_spacing,
    evaluate_separability_single_image,
    sample_feature_at_coord,
)
from hpc.models.hpc_lite import HPCLite


def test_shift_tensor_identity_and_shifts():
    x = torch.randn(2, 3, 32, 32)
    shifted_zero = shift_tensor(x, 0, 0)
    assert torch.allclose(x, shifted_zero)
    
    shifted_1_1 = shift_tensor(x, 1, 1)
    assert shifted_1_1.shape == x.shape
    # Check that pixel at (0, 0) shifted into (1, 1)
    assert torch.allclose(shifted_1_1[:, :, 1:, 1:], x[:, :, :-1, :-1])


def test_compute_knn_spacing():
    # 3 points: (0,0), (3,4) dist=5, (0,10) dist=6 from (0,4)
    pts = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 10.0]], dtype=np.float32)
    knn = compute_knn_spacing(pts)
    assert knn.shape == (3,)
    assert np.isclose(knn[0], 5.0, atol=1e-4)
    assert np.isclose(knn[1], 5.0, atol=1e-4)
    assert np.isclose(knn[2], 6.7082, atol=1e-3)


def test_spectral_rank_metrics_bounds():
    # Identity matrix (maximal rank C, centered is C-1)
    eye = torch.eye(16)
    metrics_eye = compute_spectral_rank_metrics(eye)
    assert metrics_eye["normalized_participation_ratio"] >= 0.90
    assert metrics_eye["spectral_entropy_rank"] >= 0.90
    
    # Rank-1 matrix with variance (outer product u @ v)
    u = torch.randn(32, 1)
    v = torch.randn(1, 16)
    rank1 = u @ v
    metrics_rank1 = compute_spectral_rank_metrics(rank1)
    assert metrics_rank1["normalized_participation_ratio"] <= 0.15
    assert metrics_rank1["top1_energy_ratio"] >= 0.95


def test_phase_shift_and_separability_with_model():
    model = HPCLite(
        backbone_name="mobilenetv4_conv_small_050",
        pretrained=False,
        neck_width=16,
        feature_reductions=(4, 8, 16),
    )
    model.eval()
    
    image = torch.randn(1, 3, 64, 64)
    pts = np.array([[16.0, 16.0], [20.0, 20.0], [48.0, 48.0]], dtype=np.float32)
    
    # Test D-R
    dr_res = evaluate_phase_shift_single_image(model, image, shifts=((0, 0), (1, 0), (0, 1)))
    assert "count_relative_std" in dr_res
    assert "interior_mass_mae_mean" in dr_res
    assert "feature_c4_cos_sim" in dr_res
    assert np.isfinite(dr_res["count_relative_std"])
    
    # Test D-K
    dk_res = evaluate_separability_single_image(model, image, pts)
    assert "num_points" in dk_res
    assert dk_res["num_points"] == 3
    assert "bins" in dk_res
    
    # Test D-L
    dl_res = evaluate_effective_rank_single_image(model, image, pts)
    assert "stages" in dl_res
    assert "C4" in dl_res["stages"]
    assert "C16" in dl_res["stages"]
    assert np.isfinite(dl_res["depth_decay_c16_to_c4"])
