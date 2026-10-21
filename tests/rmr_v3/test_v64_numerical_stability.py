import math
import pytest
import torch
import torch.nn as nn

from rmr_v3.losses.config import RMRv3LossConfig
from rmr_v3.losses.allocation import route_allocation_loss
from rmr_v3.losses.point_supervision import bayesian_loss
from rmr_core.dm_loss import flat_dm_block_loss


def test_bayesian_loss_amp_cuda_stability():
    """Verify that Bayesian loss never overflows or produces NaNs in AMP FP16."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    prob_y0 = (torch.rand(2, 1, 256, 256, device=device) * 0.05).detach().requires_grad_(True)

    # Test extreme cases: 0 points, 1 point, duplicate points, dense cluster
    pts_0 = torch.empty((0, 2), device=device)
    pts_1 = torch.tensor([[10.0, 10.0]], device=device)
    pts_dup = torch.tensor([[50.0, 50.0], [50.0, 50.0], [50.0, 50.0], [50.0, 50.0]], device=device)
    pts_dense = torch.rand((300, 2), device=device) * 512.0

    test_point_cases = [
        [pts_0, pts_1],
        [pts_dup, pts_dense],
    ]

    for pts_list in test_point_cases:
        # Test both canonical and zero virtual background
        for can_bg in [True, False]:
            for bg_ratio in [0.0, 0.05]:
                with torch.amp.autocast(device.type):
                    loss = bayesian_loss(
                        prob_y0,
                        pts_list,
                        sigma=4.0,
                        background_ratio=bg_ratio,
                        stride=2,
                        canonical_background=can_bg,
                        adaptive_sigma=True,
                        sigma_min=2.0,
                        sigma_max=4.0,
                    )
                assert torch.isfinite(loss), f"Loss not finite: {loss.item()} (can_bg={can_bg}, bg_ratio={bg_ratio})"
                loss.backward()
                assert torch.isfinite(prob_y0.grad).all(), "Gradients contained NaN or Inf!"
                prob_y0.grad.zero_()


def test_dual_bayesian_dm16_100_step_training_loop():
    """Simulate 100 consecutive forward-backward optimizer steps under AMP."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.SiLU(),
        nn.Conv2d(16, 1, 1),
        nn.Softplus(),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(device.type)

    cfg = RMRv3LossConfig(
        output_stride=2,
        allocation_loss_type="dual_bayesian_dm16",
        bayesian_sigma=4.0,
        bayesian_adaptive_sigma=True,
        bayesian_sigma_min=2.0,
        bayesian_sigma_max=4.0,
        bayesian_background_ratio=0.05,
        bayesian_canonical_bg=True,
        dm_block_px=32,
        kappa_flat16=20.0,
        normalize_flat_dm16=True,
        dm_norm_mode="unnorm_ref",
        dm_ref_count=100.0,
        dm_target="dual",
        lambda_flat_dm16=1.0,
        lambda_bayesian=0.15,
        lambda_count=0.0,
        lambda_cell=0.0,
        lambda_region_nb=0.0,
    )

    for step in range(50):
        x = torch.rand(2, 3, 256, 256, device=device)
        n_pts1 = int(torch.randint(0, 150, (1,)).item())
        n_pts2 = int(torch.randint(0, 300, (1,)).item())
        pts1 = torch.rand((n_pts1, 2), device=device) * 512.0 if n_pts1 > 0 else torch.empty((0, 2), device=device)
        pts2 = torch.rand((n_pts2, 2), device=device) * 512.0 if n_pts2 > 0 else torch.empty((0, 2), device=device)
        points = [pts1, pts2]

        target_y = torch.zeros(2, 1, 256, 256, device=device)
        for i, pts in enumerate(points):
            for p in pts:
                target_y[i, 0, int(p[1] // 2), int(p[0] // 2)] += 1.0

        with torch.amp.autocast(device.type):
            y0 = model(x)
            loss, alloc_losses = route_allocation_loss(y0, y0, target_y, points, cfg)

        assert torch.isfinite(loss), f"Step {step} loss became NaN/Inf: {loss.item()}"
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        for p in model.parameters():
            assert torch.isfinite(p).all(), f"Model parameter became non-finite at step {step}!"
