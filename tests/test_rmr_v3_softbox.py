import torch
import pytest
from rmr_v3.solver_ops import (
    make_gaussian_softbox_kernel,
    gaussian_softbox_forward,
    gaussian_softbox_adjoint_step,
)


def test_gaussian_softbox_kernel_normalization():
    """Verify that Gaussian kernels are strictly normalized to sum to 1.0."""
    for s in [1.0, 2.0, 4.0, 8.0]:
        k_size = int(4 * s) | 1
        k = make_gaussian_softbox_kernel(k_size, s)
        assert pytest.approx(k.sum().item(), abs=1e-5) == 1.0
        assert (k >= 0.0).all()


def test_gaussian_softbox_forward_and_adjoint():
    """Verify forward softbox convolution and adjoint backprojection."""
    y = torch.zeros(2, 1, 64, 64)
    y[:, :, 20:30, 20:30] = 5.0
    sigmas = (2.0, 4.0, 8.0)

    # Forward
    observations = gaussian_softbox_forward(y, sigmas=sigmas)
    assert len(observations) == len(sigmas)
    for obs in observations:
        assert obs.shape == y.shape
        # Total mass conserved for internal point masses
        assert pytest.approx(obs.sum().item(), rel=1e-2) == y.sum().item()

    # Adjoint step
    residuals = [obs - 1.0 for obs in observations]
    weights = torch.ones(2, len(sigmas), 64, 64) / len(sigmas)
    adj = gaussian_softbox_adjoint_step(residuals, scale_routing_weights=weights, sigmas=sigmas)
    assert adj.shape == y.shape
    assert torch.isfinite(adj).all()
