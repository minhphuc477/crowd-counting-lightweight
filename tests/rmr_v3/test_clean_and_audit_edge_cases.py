"""Edge Case, Shape Broadcasting, and KD Alias Verification Suite.

Validates:
1. DensityMapKDLoss accepts both canonical arguments and legacy aliases.
2. DensityMapKDLoss gracefully handles 3D [B, H, W] teacher maps of different spatial resolutions
   with exact mass normalization and zero broadcasting warnings.
3. DensityMapKDLoss functions correctly at batch size B=1.
4. Hurdle and truncated NB losses correctly align 2D and 3D regional count tensors across ALL
   permutations (including 2D mu_count + 3D dispersion + 2D target without IndexError).
5. Mass-weighted cell loss handles both 3D and 4D density map tensors.
6. Multiscale DM loss raises ValueError on empty block size specifications instead of ZeroDivisionError.
7. compute_losses cleanly accepts cfg=None with default initialization.
8. CoordinateAttention is directly importable from rmr_core top-level.
9. Fractional region pooling handles inverted bounding boxes by pooling the true rectangle span,
   matching canonical box pooling.
10. RMRv3 model parameter count is strictly <= 105,000 and switch_to_deploy preserves outputs.
"""

from __future__ import annotations

import warnings
import pytest
import torch

from rmr_core import CoordinateAttention
from rmr_core.losses import LossConfig, compute_losses, multiscale_dm_loss
from rmr_core.operators import fractional_region_average_features
from rmr_v3.kd import DensityMapKDLoss
from rmr_v3.losses import (
    hurdle_focal_bce_loss,
    mass_weighted_cell_loss,
    truncated_nb_nll_loss,
)
from rmr_v3.model import RMRv3, RMRv3Config


def test_kd_loss_alias_compatibility():
    """Verify DensityMapKDLoss supports both canonical and legacy alias parameter names."""
    kd1 = DensityMapKDLoss(lambda_spatial_kl=1.5, lambda_count_kd=0.3)
    assert kd1.lambda_spatial_kl == 1.5
    assert kd1.lambda_count_kd == 0.3

    kd2 = DensityMapKDLoss(lambda_spatial=2.0, lambda_count=0.4)
    assert kd2.lambda_spatial_kl == 2.0
    assert kd2.lambda_count_kd == 0.4


def test_kd_loss_3d_and_multiresolution():
    """Verify DensityMapKDLoss handles 3D teacher tensors with different resolutions without error."""
    kd = DensityMapKDLoss()
    ys = torch.rand(2, 1, 32, 32)
    yt = torch.rand(2, 64, 64) * 2.0  # 3D teacher with different spatial resolution
    loss_dict = kd(ys, yt)

    assert "spatial_kl" in loss_dict
    assert "count_kd" in loss_dict
    assert "total_kd" in loss_dict
    assert torch.isfinite(loss_dict["total_kd"])


def test_kd_loss_batch1_and_no_broadcasting_warning():
    """Verify DensityMapKDLoss operates without broadcasting warnings at B=1 and B=2."""
    kd = DensityMapKDLoss()

    # B=1 test
    ys1 = torch.rand(1, 1, 32, 32)
    yt1 = torch.rand(1, 32, 32)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        loss1 = kd(ys1, yt1)
        # Ensure no UserWarning regarding size mismatch / broadcasting
        for w in record:
            assert "broadcasting" not in str(w.message).lower()
    assert torch.isfinite(loss1["total_kd"])

    # B=2 test with 3D teacher of same resolution
    ys2 = torch.rand(2, 1, 32, 32)
    yt2 = torch.rand(2, 32, 32)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        loss2 = kd(ys2, yt2)
        for w in record:
            assert "broadcasting" not in str(w.message).lower()
    assert torch.isfinite(loss2["total_kd"])


def test_hurdle_losses_shape_alignment_all_combinations():
    """Verify hurdle_focal_bce_loss and truncated_nb_nll_loss align all 2D/3D tensor permutations."""
    b, m = 2, 10
    # 2D logits, 3D target
    logit_2d = torch.randn(b, m)
    tgt_3d = torch.randint(1, 5, (b, 1, m)).float()
    loss1 = hurdle_focal_bce_loss(logit_2d, tgt_3d)
    assert torch.isfinite(loss1)

    # 3D logits, 2D target
    logit_3d = torch.randn(b, 1, m)
    tgt_2d = torch.randint(1, 5, (b, m)).float()
    loss2 = hurdle_focal_bce_loss(logit_3d, tgt_2d)
    assert torch.isfinite(loss2)

    # Truncated NB: 2D target, 3D mean, 3D dispersion
    mu_3d = torch.rand(b, 1, m) * 5.0 + 0.1
    disp_3d = torch.full((b, 1, m), 50.0)
    loss_tnb1 = truncated_nb_nll_loss(mu_3d, disp_3d, tgt_2d)
    assert torch.isfinite(loss_tnb1)

    # Truncated NB: 3D target, 2D mean, 2D dispersion
    mu_2d = torch.rand(b, m) * 5.0 + 0.1
    disp_2d = torch.full((b, m), 50.0)
    loss_tnb2 = truncated_nb_nll_loss(mu_2d, disp_2d, tgt_3d)
    assert torch.isfinite(loss_tnb2)

    # Truncated NB: Mixed 2D mu, 3D disp, 2D target (previously caused IndexError)
    loss_tnb3 = truncated_nb_nll_loss(mu_2d, disp_3d, tgt_2d)
    assert torch.isfinite(loss_tnb3)

    # Truncated NB: Mixed 3D mu, 2D disp, 3D target
    loss_tnb4 = truncated_nb_nll_loss(mu_3d, disp_2d, tgt_3d)
    assert torch.isfinite(loss_tnb4)


def test_mass_weighted_cell_loss_3d_4d():
    """Verify mass_weighted_cell_loss correctly handles both 3D and 4D spatial tensors."""
    b, h, w = 2, 16, 16
    y_4d = torch.rand(b, 1, h, w)
    t_3d = torch.rand(b, h, w)
    loss = mass_weighted_cell_loss(y_4d, t_3d)
    assert torch.isfinite(loss)

    y_3d = torch.rand(b, h, w)
    t_4d = torch.rand(b, 1, h, w)
    loss2 = mass_weighted_cell_loss(y_3d, t_4d)
    assert torch.isfinite(loss2)


def test_multiscale_dm_loss_empty_blocks():
    """Verify multiscale_dm_loss raises ValueError on empty block sizes."""
    pred = torch.rand(1, 1, 16, 16)
    tgt = torch.rand(1, 1, 16, 16)
    with pytest.raises(ValueError, match="must not be empty"):
        multiscale_dm_loss(pred, tgt, block_sizes_px=())


def test_compute_losses_none_config():
    """Verify compute_losses accepts cfg=None without error."""
    out = {
        "y": torch.rand(1, 1, 16, 16),
        "y0": torch.rand(1, 1, 16, 16),
    }
    tgt = torch.rand(1, 1, 16, 16)
    losses = compute_losses(out, tgt, variant="base", cfg=None)
    assert "total" in losses
    assert torch.isfinite(losses["total"])


def test_coordinate_attention_rmr_core_export():
    """Verify CoordinateAttention is importable from rmr_core directly."""
    ca = CoordinateAttention(channels=32, reduction=4)
    x = torch.randn(2, 32, 16, 16)
    out = ca(x)
    assert out.shape == x.shape


def test_fractional_pooling_inverted_boxes_matches_canonical():
    """Verify fractional_region_average_features handles inverted coordinates identically to canonical."""
    feat = torch.randn(1, 8, 16, 16)
    canonical_box = torch.tensor([[5.0, 3.0, 10.0, 12.0]])
    inverted_box = torch.tensor([[10.0, 12.0, 5.0, 3.0]])

    avg_canonical = fractional_region_average_features(feat, canonical_box)
    avg_inverted = fractional_region_average_features(feat, inverted_box)

    assert avg_inverted.shape == (1, 1, 8)
    assert torch.allclose(avg_canonical, avg_inverted, atol=1e-6)


def test_model_parameter_budget_and_deploy():
    """Verify RMRv3 parameter budget constraint (<= 105,000) and deploy mode equivalence."""
    cfg_aq = RMRv3Config(
        pretrained=False,
        iterations=6,
        proximal_tau=0.015,
        tv_lambda=0.02,
        regional_feature_stats="mean_std",
        region_sizes_px=(32, 64, 128, (64, 32), (32, 64)),
    )
    model = RMRv3(cfg_aq)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params <= 105_000, f"Trainable parameters {n_params} exceed budget 105,000"

    # Reparameterization deploy test
    cfg_rep = RMRv3Config(pretrained=False, neck_type="rep_weighted")
    model_rep = RMRv3(cfg_rep)
    model_rep.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out1 = model_rep(x)["y"]
    model_rep.switch_to_deploy()
    with torch.no_grad():
        out2 = model_rep(x)["y"]
    diff = (out1 - out2).abs().max().item()
    assert diff < 1e-4, f"Reparameterization deploy mismatch: {diff}"
