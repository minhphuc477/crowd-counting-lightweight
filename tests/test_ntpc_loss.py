import pytest
import torch

from hpc.data.point_counts import build_exact_count_pyramid
from hpc.losses.negative_binomial import negative_binomial_nll_mean_dispersion
from hpc.losses.ntpc import NTPCConfig, NTPCLoss, sum_pool_mass_pyramid


def _case():
    pts = torch.tensor([[12.0, 15.0], [45.0, 80.0], [120.0, 150.0], [200.0, 210.0]])
    tree = build_exact_count_pyramid([pts], height=256, width=256)
    mass = (torch.rand(1, 1, 64, 64) + 0.1).detach().requires_grad_(True)
    targets = {k: tree[k].clone() for k in (4, 8, 16, 32, 64)}
    targets["N"] = tree["N"].clone()
    return mass, targets


def test_dtm_scale_invariance():
    """DTM spatial allocation losses must be scale-invariant: L_DTM(a * D) == L_DTM(D)."""
    mass, targets = _case()
    criterion = NTPCLoss(NTPCConfig(mode="r4_dtm_tree16"))

    loss_1, logs_1 = criterion(mass, targets)
    loss_2, logs_2 = criterion(mass * 5.0, targets)

    # Spatial tree terms must match exactly
    for term in ("root_to_64", "64_to_32", "32_to_16"):
        torch.testing.assert_close(logs_1[term], logs_2[term], atol=1e-5, rtol=1e-5)

    # But root magnitude loss changes
    assert not torch.allclose(logs_1["root_magnitude"], logs_2["root_magnitude"])


def test_is_exact_joint_nll():
    """is_exact_joint_nll must be True ONLY for r4_dtm_tree16 with unit weights."""
    assert NTPCLoss(NTPCConfig(mode="r4_dtm_tree16", root_loss="nb")).is_exact_joint_nll
    assert not NTPCLoss(NTPCConfig(mode="r4_dtm_tree16", w_64_32=2.0)).is_exact_joint_nll
    assert not NTPCLoss(NTPCConfig(mode="r4_dtm_tree16", root_loss="l1")).is_exact_joint_nll
    assert not NTPCLoss(NTPCConfig(mode="r0_exact")).is_exact_joint_nll
    assert not NTPCLoss(NTPCConfig(mode="r5_full_ntpc")).is_exact_joint_nll


def test_r0_has_root_nb_and_regional():
    mass, targets = _case()
    _, logs = NTPCLoss(NTPCConfig(mode="r0_exact"))(mass, targets)
    assert logs["root_magnitude"].item() > 0.0
    assert logs["exact_regression"].item() > 0.0


def test_ntpc_config_validation_catches_invalids():
    """NTPCConfig must reject negative/non-finite weights, invalid dispersions and thresholds."""
    with pytest.raises(ValueError):
        NTPCConfig(w_root64=-1.0)
    with pytest.raises(ValueError):
        NTPCConfig(w_64_32=float("nan"))
    with pytest.raises(ValueError):
        NTPCConfig(root_dispersion=0.0)
    with pytest.raises(ValueError):
        NTPCConfig(root_dispersion=20000.0)  # Exceeds max 10000
    with pytest.raises(ValueError):
        NTPCConfig(dense_threshold_16=-1.0)
    with pytest.raises(ValueError):
        NTPCConfig(kappa_root64=-5.0)


def test_negative_binomial_dispersion_validation():
    """NB NLL must reject non-positive or overly large dispersion parameters."""
    y = torch.tensor([10.0])
    mu = torch.tensor([10.0])
    with pytest.raises(ValueError):
        negative_binomial_nll_mean_dispersion(y, mu, dispersion=0.0)
    with pytest.raises(ValueError):
        negative_binomial_nll_mean_dispersion(y, mu, dispersion=-1.0)
    with pytest.raises(ValueError):
        negative_binomial_nll_mean_dispersion(y, mu, dispersion=50000.0)


def test_all_modes_backward_pass():
    """All R0-R5 modes must be finite and provide non-zero gradients to the mass map."""
    modes = [
        "r0_exact",
        "r1_deterministic",
        "r2_flat_dm",
        "r3_multinomial_tree",
        "r4_dtm_tree16",
        "r4_dtm_tree8",
        "r4_dtm_tree4",
        "r5_full_ntpc",
    ]
    for mode in modes:
        mass, targets = _case()
        loss, _ = NTPCLoss(NTPCConfig(mode=mode, dense_threshold_16=1))(mass, targets)
        assert torch.isfinite(loss), f"Mode {mode} produced non-finite loss"
        loss.backward()
        assert mass.grad is not None
        assert torch.isfinite(mass.grad).all()
        assert mass.grad.abs().sum() > 0
