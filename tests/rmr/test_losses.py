import pytest
import torch
import torch.nn.functional as F

from rmr_count.losses import (
    LossConfig,
    balanced_smooth_l1,
    count_magnitude_loss,
    negative_binomial_nll_mean_dispersion,
    flat_dm16_loss,
    scale_balanced_region_rate_loss,
    compute_losses,
)
from rmr_count.model import RMRConfig, RMRCount
from rmr_count.operators import build_multiscale_regions, regional_sum


def test_negative_binomial_dispersion_validation():
    y = torch.tensor([10.0])
    mu = torch.tensor([10.0])
    with pytest.raises(ValueError):
        negative_binomial_nll_mean_dispersion(y, mu, dispersion=0.0)
    with pytest.raises(ValueError):
        negative_binomial_nll_mean_dispersion(y, mu, dispersion=-1.0)
    with pytest.raises(ValueError):
        negative_binomial_nll_mean_dispersion(y, mu, dispersion=20000.0)
    with pytest.raises(ValueError):
        negative_binomial_nll_mean_dispersion(torch.tensor([-1.0]), mu, dispersion=50.0)


def test_count_magnitude_loss_modes():
    pred = torch.full((2, 1, 32, 32), 0.05, requires_grad=True)
    target = torch.full((2, 1, 32, 32), 0.04)

    for mode in ('nb', 'log1p', 'l1'):
        loss = count_magnitude_loss(pred, target, mode=mode, dispersion=50.0)
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0
        loss.backward(retain_graph=True)
        assert pred.grad is not None


def test_flat_dm16_empty_image_contributes_zero():
    pred = (torch.rand(2, 1, 32, 32) + 0.1).requires_grad_(True)
    target_empty = torch.zeros(2, 1, 32, 32)

    loss = flat_dm16_loss(pred, target_empty, kappa=20.0, stride=4)
    assert loss.item() == 0.0


def test_flat_dm16_scale_invariance():
    pred = torch.rand(2, 1, 32, 32) + 0.1
    target = torch.randint(0, 5, (2, 1, 32, 32)).float()

    loss1 = flat_dm16_loss(pred, target, kappa=20.0, stride=4)
    loss2 = flat_dm16_loss(pred * 5.0, target, kappa=20.0, stride=4)
    torch.testing.assert_close(loss1, loss2, atol=1e-5, rtol=1e-5)


def test_flat_dm16_backward_pass():
    pred = (torch.rand(2, 1, 32, 32) + 0.1).requires_grad_(True)
    target = torch.randint(0, 5, (2, 1, 32, 32)).float()

    loss = flat_dm16_loss(pred, target, kappa=20.0, stride=4)
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


def test_compute_losses_all_variants():
    x = torch.zeros(1, 3, 128, 128)
    target = torch.randint(0, 3, (1, 1, 32, 32)).float()
    variants = ('direct', 'region_loss', 'region_aux', 'local_refine', 'learned_project', 'rmr')
    cfg = LossConfig()

    for v in variants:
        m = RMRCount(RMRConfig(iterations=2), variant=v)
        out = m(x)
        losses = compute_losses(out, target, v, cfg)
        assert 'total' in losses
        assert 'cell' in losses
        assert 'count' in losses
        assert 'flat_dm16' in losses
        assert torch.isfinite(losses['total'])

        losses['total'].backward()
        # Verify gradient flow to encoder stem
        grad_found = False
        for p in m.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                grad_found = True
                break
        assert grad_found, f'Variant {v} produced no parameter gradients'
