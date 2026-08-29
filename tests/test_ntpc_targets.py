import pytest
import torch
import torch.nn.functional as F

from hpc.data.point_counts import build_exact_count_pyramid, points_to_y4
from hpc.losses.ntpc import NTPCConfig, NTPCLoss, sum_pool_mass_pyramid


def _dummy_case():
    pts = torch.tensor([[10.2, 14.8], [100.0, 50.0], [200.5, 220.1]])
    tree = build_exact_count_pyramid([pts], height=256, width=256, block_sizes=(4, 8, 16, 32, 64))
    mass = torch.rand(1, 1, 64, 64, requires_grad=True) + 0.1
    targets = {k: tree[k].clone() for k in (4, 8, 16, 32, 64)}
    targets["N"] = tree["N"].clone()
    return mass, targets


def test_exact_count_pyramid_conservation():
    """All levels of exact count pyramid must conserve count and be exact integers."""
    pts = [
        torch.tensor([[0.0, 0.0], [255.0, 255.0], [128.0, 128.0]]),
        torch.empty(0, 2),
    ]
    tree = build_exact_count_pyramid(pts, height=256, width=256)
    assert tree["N"].tolist() == [3.0, 0.0]
    for b in (4, 8, 16, 32, 64):
        tensor = tree[b]
        assert torch.equal(tensor, tensor.round())
        assert tensor.min() >= 0
        assert torch.allclose(tensor.flatten(1).sum(dim=1), tree["N"])


def test_parent_child_adjacent_exact_conservation():
    """Adjacent levels must reconstruct parent via 2x2 sum pooling."""
    pts = [torch.rand(50, 2) * 255.0]
    tree = build_exact_count_pyramid(pts, height=256, width=256)

    for parent_b, child_b in ((64, 32), (32, 16), (16, 8), (8, 4)):
        parent = tree[parent_b]
        child = tree[child_b]
        reconstructed = F.avg_pool2d(child, 2, stride=2, divisor_override=1)
        assert torch.equal(parent, reconstructed), f"Level {child_b} does not reconstruct {parent_b}"


def test_points_to_y4_known_boundaries():
    """Points on boundary transitions must be assigned to floor((coord + 0.5) / 4) cells."""
    pts = torch.tensor([
        [0.0, 0.0],      # cell (0, 0)
        [3.49, 3.49],    # cell (0, 0)
        [3.51, 3.51],    # cell (1, 1)
        [255.0, 255.0],  # cell (63, 63)
    ])
    y4 = points_to_y4(pts, 256, 256)
    assert y4.shape == (1, 64, 64)
    assert y4[0, 0, 0] == 2.0
    assert y4[0, 1, 1] == 1.0
    assert y4[0, 63, 63] == 1.0
    assert y4.sum() == 4.0


def test_validate_targets_catches_non_integer():
    mass, targets = _dummy_case()
    targets[16] = targets[16] + 0.5
    with pytest.raises(ValueError, match="non-integer"):
        NTPCLoss(NTPCConfig(mode="r4_dtm_tree16"))(mass, targets)


def test_validate_targets_catches_negative():
    mass, targets = _dummy_case()
    targets[16] = targets[16].clone()
    targets[16].view(-1)[0] = -1.0
    with pytest.raises(ValueError, match="negative"):
        NTPCLoss(NTPCConfig(mode="r4_dtm_tree16"))(mass, targets)


def test_validate_targets_catches_conservation_violation():
    mass, targets = _dummy_case()
    targets[32] = targets[32].clone()
    targets[32].view(-1)[0] += 10.0
    with pytest.raises(ValueError, match="conservation"):
        NTPCLoss(NTPCConfig(mode="r4_dtm_tree16"))(mass, targets)


def test_validate_targets_catches_shuffled_child():
    """Parent-child validator must reject shuffled child cells that preserve total N."""
    mass, targets = _dummy_case()
    # Shuffle level 32 cells (total sum remains N, but spatial tree is broken)
    flat32 = targets[32].flatten(1)
    if flat32.shape[1] > 1:
        targets[32] = torch.roll(flat32, shifts=1, dims=1).reshape_as(targets[32])
        with pytest.raises(ValueError, match="child counts do not reconstruct parent"):
            NTPCLoss(NTPCConfig(mode="r4_dtm_tree16"))(mass, targets)


def test_validate_targets_catches_invalid_n():
    mass, targets = _dummy_case()
    targets["N"] = torch.tensor([float("nan")])
    with pytest.raises(ValueError, match="NaN/Inf"):
        NTPCLoss(NTPCConfig(mode="r4_dtm_tree16"))(mass, targets)

    targets["N"] = torch.tensor([-1.0])
    with pytest.raises(ValueError, match="non-negative"):
        NTPCLoss(NTPCConfig(mode="r4_dtm_tree16"))(mass, targets)


def test_validate_targets_catches_missing_keys():
    mass, targets = _dummy_case()
    targets.pop(64)
    with pytest.raises(KeyError, match="requires targets"):
        NTPCLoss(NTPCConfig(mode="r4_dtm_tree16"))(mass, targets)


def test_build_exact_count_pyramid_catches_oob_before_padding():
    """Points outside original support must raise ValueError even if they fit in padded size."""
    # Image size 410x300, padded to 448x320. Point at x=430 is in padded support but invalid for original image.
    pts = torch.tensor([[430.0, 100.0]])
    with pytest.raises(ValueError, match="outside original image support"):
        build_exact_count_pyramid([pts], height=300, width=410, pad_multiple=64)


def test_build_exact_count_pyramid_invalid_inputs():
    with pytest.raises(ValueError, match="points_batch cannot be empty"):
        build_exact_count_pyramid([], height=256, width=256)

    pts = torch.tensor([[50.0, 50.0]])
    with pytest.raises(ValueError, match="positive multiple of 64"):
        build_exact_count_pyramid([pts], height=256, width=256, pad_multiple=32)

    with pytest.raises(ValueError, match="Unsupported block sizes"):
        build_exact_count_pyramid([pts], height=256, width=256, block_sizes=(12,))


def test_sum_pool_mass_pyramid_invalid_levels():
    mass = torch.rand(1, 1, 64, 64)
    with pytest.raises(ValueError, match="Unsupported mass levels"):
        sum_pool_mass_pyramid(mass, block_sizes=(12,))

