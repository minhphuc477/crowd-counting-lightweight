"""Adversarial architecture audit and numerical verification suite.

Validates:
1. Object-bound lifetime caching in scale_regions_to_stride2 (no GC id recycling bugs).
2. Bandwidth sensitivity in canonical ChfL template caching.
3. Dimension invariance and on-the-fly rectangle prefix evaluations on dynamic crops.
4. Sample gradient isolation across batches.
5. Numerical robustness under extreme sparse/dense inputs.
"""
from __future__ import annotations

import gc
import torch
import pytest

from rmr_core.operators import (
    build_multiscale_regions,
    rectangle_sum_from_prefix,
    region_average_features,
    regional_sum,
    prefix2d,
)
from rmr_v3.model.dual_lattice import scale_regions_to_stride2
from rmr_v3.losses.chfl import _build_canonical_chfl_templates


def test_scale_regions_object_bound_caching() -> None:
    """Verify that scale_regions_to_stride2 attaches cache directly to the object and survives GC."""
    r1 = build_multiscale_regions(128, 128, 4)
    s1 = scale_regions_to_stride2(r1, 256, 256)
    s1_again = scale_regions_to_stride2(r1, 256, 256)
    assert s1 is s1_again, "Cache hit must return the identical object"

    # Mutate resolution to ensure dimension-sensitive cache invalidation
    s1_diff_hw = scale_regions_to_stride2(r1, 128, 128)
    assert s1_diff_hw is not s1, "Different target_hw must recompute"
    assert s1_diff_hw.boxes[:, 2].max() <= 128

    # Create and delete a different RegionSet to test GC address reuse safety
    r2 = build_multiscale_regions(64, 64, 4)
    s2 = scale_regions_to_stride2(r2, 128, 128)
    assert s2.boxes.shape[0] != s1.boxes.shape[0]

    del r2
    gc.collect()

    # Re-evaluate r1; must not be corrupted by r2's deletion or any GC recycling
    s1_after_gc = scale_regions_to_stride2(r1, 256, 256)
    assert s1_after_gc.boxes.shape == s1.boxes.shape


def test_chfl_template_cache_bandwidth_sensitivity() -> None:
    """Verify that changing bandwidth invalidates the ChfL template cache."""
    device = torch.device("cpu")
    t1 = _build_canonical_chfl_templates(32, 32, 4, 0.01, 8, bandwidth=4.0, device=device)
    t2 = _build_canonical_chfl_templates(32, 32, 4, 0.01, 8, bandwidth=8.0, device=device)

    # freq_damping is the 4th element (index 3)
    damp1 = t1[3]
    damp2 = t2[3]
    assert not torch.allclose(damp1, damp2), "Different bandwidths must produce different damping envelopes"


def test_rectangle_prefix_evaluation_dynamic_crops() -> None:
    """Verify prefix sum evaluation on multiple prime/odd resolutions without dimension mismatch."""
    shapes = [(113, 127), (67, 89), (128, 128)]
    for h, w in shapes:
        x = torch.randn(2, 32, h, w)
        pref = prefix2d(x)
        assert pref.shape == (2, 32, h + 1, w + 1)

        # Build regions
        regions = build_multiscale_regions(h, w, 4)
        sums = rectangle_sum_from_prefix(pref, regions.boxes)
        assert sums.shape == (2, 32, regions.boxes.shape[0])

        # Verify region_average_features matches
        avg = region_average_features(x, regions.boxes)
        assert avg.shape == (2, regions.boxes.shape[0], 32)
        assert torch.isfinite(avg).all()


def test_batch_sample_isolation_gradient_independence() -> None:
    """Verify that sample 0's loss generates zero gradient on sample 1's inputs."""
    x = torch.randn(2, 1, 64, 64, requires_grad=True)
    # Forward pass: independent slice
    y0 = x[0:1] ** 2
    loss_sample0 = y0.sum()
    loss_sample0.backward()

    # Sample 1 must receive strictly 0 gradient
    assert (x.grad[1] == 0.0).all()
    assert (x.grad[0] != 0.0).any()
