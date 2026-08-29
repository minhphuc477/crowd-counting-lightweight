import pytest
import torch
import torch.nn.functional as F

from hpc.data.point_counts import build_exact_count_pyramid
from hpc.losses.ntpc import NTPCConfig, NTPCLoss


def _case(seed=7, count=600):
    generator = torch.Generator().manual_seed(seed)
    points = torch.rand(count, 2, generator=generator) * 255.999
    targets = build_exact_count_pyramid(
        [points], 256, 256, block_sizes=(4, 8, 16, 32, 64)
    )
    logits = torch.randn(1, 1, 64, 64, generator=generator, requires_grad=True)
    return logits, F.softplus(logits) + 1e-8, targets


def test_r4_core_and_r5_dense_extension_are_distinct():
    _, mass, targets = _case()
    r4, r4_logs = NTPCLoss(NTPCConfig(mode="r4_dtm_tree16"))(mass, targets)
    r5, r5_logs = NTPCLoss(NTPCConfig(
        mode="r5_full_ntpc", dense_threshold_16=1.0
    ))(mass, targets)
    assert r4_logs["16_to_8"].item() == 0.0
    assert r4_logs["16_to_8_dense"].item() == 0.0
    assert r4_logs["8_to_4"].item() == 0.0
    assert r5_logs["16_to_8"].item() == 0.0
    assert r5_logs["16_to_8_dense"].item() > 0.0
    assert r5_logs["8_to_4"].item() == 0.0
    assert r5 > r4


def test_r5_threshold_is_used_and_empty_dense_set_reduces_to_r4():
    _, mass, targets = _case()
    r4, _ = NTPCLoss(NTPCConfig(mode="r4_dtm_tree16"))(mass, targets)
    r5, logs = NTPCLoss(NTPCConfig(
        mode="r5_full_ntpc", dense_threshold_16=1e9
    ))(mass, targets)
    assert logs["16_to_8_dense"].item() == 0.0
    assert torch.allclose(r4, r5, atol=1e-6, rtol=0)


def test_stride4_term_is_only_in_explicit_depth_study():
    _, mass, targets = _case()
    _, full_logs = NTPCLoss(NTPCConfig(mode="r5_full_ntpc", dense_threshold_16=1))(mass, targets)
    _, depth_logs = NTPCLoss(NTPCConfig(mode="r4_dtm_tree4"))(mass, targets)
    assert full_logs["8_to_4"].item() == 0.0
    assert depth_logs["8_to_4"].item() > 0.0


def test_r1_contains_root_to_64_deterministic_allocation():
    points = torch.tensor([[2.0, 2.0], [4.0, 3.0], [7.0, 8.0]])
    targets = build_exact_count_pyramid([points], 256, 256, (16, 32, 64))
    mass = torch.ones(1, 1, 64, 64, requires_grad=True)
    _, logs = NTPCLoss(NTPCConfig(mode="r1_deterministic"))(mass, targets)
    assert logs["deterministic_alloc"].item() > 0.0


def test_each_core_component_has_a_real_gradient():
    logits, mass, targets = _case(count=900)
    _, _, components = NTPCLoss(NTPCConfig(
        mode="r5_full_ntpc", dense_threshold_16=1.0
    ))(mass, targets, return_components=True)
    for name in ("root_magnitude", "root_to_64", "64_to_32", "32_to_16", "16_to_8_dense"):
        grad = torch.autograd.grad(components[name], logits, retain_graph=True)[0]
        assert torch.isfinite(grad).all(), name
        assert grad.norm().item() > 0.0, name


@pytest.mark.parametrize("root_loss", ["nb", "poisson", "l1"])
def test_root_magnitude_ablation_modes_are_finite(root_loss):
    _, mass, targets = _case(count=80)
    loss, logs = NTPCLoss(NTPCConfig(
        mode="r4_dtm_tree16", root_loss=root_loss
    ))(mass, targets)
    assert torch.isfinite(loss)
    assert logs["root_magnitude"].item() > 0.0


def test_modes_and_required_targets_fail_fast():
    with pytest.raises(ValueError):
        NTPCConfig(mode="typo")
    _, mass, targets = _case()
    targets.pop(8)
    targets.pop("y8")
    with pytest.raises(KeyError):
        NTPCLoss(NTPCConfig(mode="r5_full_ntpc"))(mass, targets)


def test_validate_targets_catches_non_integer():
    """_validate_targets must raise ValueError when count targets contain fractional values."""
    _, mass, targets = _case()
    # Inject a fractional value into the level-16 target
    targets[16] = targets[16] + 0.5
    with pytest.raises(ValueError, match="non-integer"):
        NTPCLoss(NTPCConfig(mode="r4_dtm_tree16"))(mass, targets)


def test_validate_targets_catches_conservation_violation():
    """_validate_targets must raise ValueError when child counts don't sum to N."""
    _, mass, targets = _case()
    # Corrupt the level-32 target by adding extra mass in one cell
    targets[32] = targets[32].clone()
    targets[32].view(-1)[0] += 100.0   # now sum(Y32) != N
    with pytest.raises(ValueError, match="conservation"):
        NTPCLoss(NTPCConfig(mode="r4_dtm_tree16"))(mass, targets)


def test_validate_targets_catches_negative_values():
    """_validate_targets must raise ValueError when any count target is negative."""
    _, mass, targets = _case()
    targets[16] = targets[16].clone()
    targets[16].view(-1)[0] = -1.0
    with pytest.raises(ValueError, match="negative"):
        NTPCLoss(NTPCConfig(mode="r4_dtm_tree16"))(mass, targets)


def test_r0_has_root_nb_component():
    """R0 must now log root_magnitude > 0 (Root-NB is shared with R1–R5)."""
    _, mass, targets = _case()
    _, logs = NTPCLoss(NTPCConfig(mode="r0_exact"))(mass, targets)
    assert logs["root_magnitude"].item() > 0.0, "R0 root_magnitude must be positive"
    assert logs["exact_regression"].item() > 0.0, "R0 exact_regression must be positive"


def test_multinomial_renormalization_stable():
    """multinomial_nll_none must remain finite even when pi values before renorm sum > 1."""
    from hpc.losses.ntpc import multinomial_nll_none
    # All-ones pi: before renorm, sum = 4 per row (> 1); after renorm each = 0.25
    pi = torch.ones(5, 4) * 10.0   # very large, clamp_min has no effect; renorm fixes sum
    y = torch.tensor([[2., 1., 1., 0.]] * 5)
    nll = multinomial_nll_none(y, pi)
    assert torch.isfinite(nll).all(), "multinomial_nll must be finite after renormalization"

