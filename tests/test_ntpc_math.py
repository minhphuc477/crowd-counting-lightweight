import pytest
import torch
import torch.nn.functional as F

from hpc.losses.negative_binomial import negative_binomial_nll_mean_dispersion
from hpc.losses.ntpc import (
    dm_from_mass,
    dm_nll_none,
    group_2x2_flat,
    multinomial_nll_none,
    probs_from_positive_mass,
    sum_pool_mass_pyramid,
)


def test_multinomial_normalization_equivalent_to_normalized():
    """multinomial_nll_none with unnormalized pi must match normalized pi."""
    pi_raw = torch.tensor([[10.0, 10.0, 10.0, 10.0]])
    pi_norm = torch.full_like(pi_raw, 0.25)
    y = torch.tensor([[2.0, 1.0, 1.0, 0.0]])
    torch.testing.assert_close(
        multinomial_nll_none(y, pi_raw),
        multinomial_nll_none(y, pi_norm),
        atol=1e-5,
        rtol=1e-5,
    )


def test_hierarchical_multinomial_equals_flat_leaf():
    """Mathematical identity: Hierarchical tree Multinomial collapse to flat leaf Multinomial.

    Under a single conserved mass map with consistent probabilities:
    P(Y64 | N) * P(Y32 | Y64) * P(Y16 | Y32) == P(Y16 | N)
    """
    torch.manual_seed(42)
    # Multi-batch Mass maps at stride 4: (B=3, 1, 64, 64)
    mass = torch.rand(3, 1, 64, 64) + 0.1
    pred = sum_pool_mass_pyramid(mass)

    # Ground truth targets consistent with tree
    y16 = torch.poisson(pred[16]).round()
    y32 = F.avg_pool2d(y16, 2, stride=2, divisor_override=1)
    y64 = F.avg_pool2d(y32, 2, stride=2, divisor_override=1)
    N = y64.sum(dim=(-1, -2))

    # Flat Multinomial at level 16:
    pi_flat16 = probs_from_positive_mass(pred[16].flatten(1))
    flat_nll = multinomial_nll_none(y16.flatten(1), pi_flat16)

    # Hierarchical tree terms:
    # 1. Root -> 64
    pi64 = probs_from_positive_mass(pred[64].flatten(1))
    term_root64 = multinomial_nll_none(y64.flatten(1), pi64)

    # 2. 64 -> 32
    target_32_groups = group_2x2_flat(y32)
    pi_32_groups = probs_from_positive_mass(group_2x2_flat(pred[32]))
    term_64_32 = multinomial_nll_none(target_32_groups, pi_32_groups).sum(dim=-1)

    # 3. 32 -> 16
    target_16_groups = group_2x2_flat(y16)
    pi_16_groups = probs_from_positive_mass(group_2x2_flat(pred[16]))
    term_32_16 = multinomial_nll_none(target_16_groups, pi_16_groups).sum(dim=-1)

    hierarchical_nll = term_root64 + term_64_32 + term_32_16

    torch.testing.assert_close(
        hierarchical_nll,
        flat_nll,
        atol=1e-4,
        rtol=1e-4,
    )


def test_dm_prefers_correct_allocation():
    """Dirichlet-Multinomial NLL must be strictly lower for correct allocation."""
    y = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    correct_mass = torch.tensor([[100.0, 1.0, 1.0, 1.0]])
    wrong_mass = torch.tensor([[1.0, 100.0, 1.0, 1.0]])

    nll_correct = dm_from_mass(y, correct_mass, kappa=20.0)
    nll_wrong = dm_from_mass(y, wrong_mass, kappa=20.0)

    assert nll_correct < nll_wrong, f"Correct NLL {nll_correct} must be < wrong NLL {nll_wrong}"


def test_zero_parent_dm_nll_is_zero_and_no_gradient():
    """Zero-count parent must yield exactly 0 NLL and 0 gradient."""
    mass = torch.rand(1, 4, requires_grad=True)
    y = torch.zeros(1, 4)
    nll = dm_from_mass(y, mass, kappa=20.0)
    assert nll.item() == 0.0
    nll.backward()
    assert mass.grad is not None
    assert torch.allclose(mass.grad, torch.zeros_like(mass))


def test_hierarchical_multinomial_gradient_equals_flat_gradient():
    """Mathematical identity: Hierarchical Multinomial and Flat Leaf Multinomial must yield identical backward gradients."""
    torch.manual_seed(42)
    # Mass map with requires_grad=True
    mass_h = (torch.rand(2, 1, 64, 64) + 0.1).requires_grad_(True)
    mass_f = mass_h.clone().detach().requires_grad_(True)

    pred_h = sum_pool_mass_pyramid(mass_h)
    pred_f = sum_pool_mass_pyramid(mass_f)

    # Targets consistent with tree
    y16 = torch.poisson(pred_f[16].detach()).round()
    y32 = F.avg_pool2d(y16, 2, stride=2, divisor_override=1)
    y64 = F.avg_pool2d(y32, 2, stride=2, divisor_override=1)

    # Flat Multinomial
    pi_flat16 = probs_from_positive_mass(pred_f[16].flatten(1))
    flat_nll = multinomial_nll_none(y16.flatten(1), pi_flat16).sum()
    flat_nll.backward()

    # Hierarchical Multinomial
    pi64 = probs_from_positive_mass(pred_h[64].flatten(1))
    term_root64 = multinomial_nll_none(y64.flatten(1), pi64)
    target_32_groups = group_2x2_flat(y32)
    pi_32_groups = probs_from_positive_mass(group_2x2_flat(pred_h[32]))
    term_64_32 = multinomial_nll_none(target_32_groups, pi_32_groups).sum(dim=-1)
    target_16_groups = group_2x2_flat(y16)
    pi_16_groups = probs_from_positive_mass(group_2x2_flat(pred_h[16]))
    term_32_16 = multinomial_nll_none(target_16_groups, pi_16_groups).sum(dim=-1)

    hierarchical_nll = (term_root64 + term_64_32 + term_32_16).sum()
    hierarchical_nll.backward()

    torch.testing.assert_close(
        mass_h.grad,
        mass_f.grad,
        atol=1e-5,
        rtol=1e-4,
        msg="Gradients of Hierarchical Multinomial and Flat Leaf Multinomial must match identically",
    )


def test_nwpu_scale_numerical_stability():
    """Negative-Binomial and Dirichlet-Multinomial NLL and gradients must be finite at NWPU scale (N = 20,033)."""
    # 1. NB NLL at NWPU max count
    n_nwpu = torch.tensor([20033.0])
    mu_nwpu = torch.tensor([19850.0], requires_grad=True)
    dispersion = 50.0

    loss_nb = negative_binomial_nll_mean_dispersion(n_nwpu, mu_nwpu, dispersion)
    assert torch.isfinite(loss_nb).item()
    loss_nb.backward()
    assert mu_nwpu.grad is not None and torch.isfinite(mu_nwpu.grad).item()

    # 2. DM NLL with parent N = 20,033 and skewed distribution
    child_mass = torch.tensor([[100.0, 1.0, 0.1, 19931.9]], requires_grad=True)
    child_gt = torch.tensor([[105.0, 0.0, 0.0, 19928.0]])
    loss_dm = dm_from_mass(child_gt, child_mass, kappa=20.0).sum()
    assert torch.isfinite(loss_dm).item()
    loss_dm.backward()
    assert child_mass.grad is not None and torch.isfinite(child_mass.grad).all()
