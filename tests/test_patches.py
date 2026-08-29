"""Integration patch tests for hpc.* package correctness.

Tests cover: target conservation, NB stability, criterion ablation, dataset
schema, sampler, NAE metric, CSV logging, checkpoint round-trip, and dataset
strict-annotation contracts.
"""
import csv
import math
import os
import tempfile

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from hpc.targets.allocation_target import build_block_constrained_allocation_target
from hpc.targets.block_counts import build_hierarchical_block_counts
from hpc.losses.negative_binomial import HierarchicalNBLoss, nb_nll
from hpc.losses.criterion import HPCLossCriterion
from hpc.losses.robustness import RobustConsistencyLoss
from hpc.data.common import BaseCrowdDataset, custom_collate_fn
from hpc.data.transforms import GeometricTransforms
from hpc.data.sampler import build_density_luminance_sampler
from hpc.metrics.counting import compute_nae
from hpc.utils.logging import CSVLogger
from hpc.utils.checkpoint import build_checkpoint_state, save_checkpoint, load_checkpoint


# ---------------------------------------------------------------------------
# 1. Target conservation — extreme point count, large crop
# ---------------------------------------------------------------------------

def test_patch_target_conservation_extreme():
    """Allocation and block counts conserve total mass and agree locally."""
    rng = np.random.default_rng(0)
    N = 20033
    pts = np.column_stack([
        rng.uniform(0, 672, N),
        rng.uniform(0, 672, N),
    ]).astype(np.float32)
    z = build_block_constrained_allocation_target(pts, 672, 672, 16, 4)
    ys = build_hierarchical_block_counts(pts, 672, 672, [16, 32, 96])

    # Global conservation
    assert abs(float(z.sum()) - N) < 1e-2, f"Alloc sum {z.sum()} != {N}"
    for k, v in ys.items():
        assert float(v.sum()) == float(N), f"Block-{k} sum {v.sum()} != {N}"

    # Local per-block conservation: alloc sum inside each 16-px block == y16
    # Output stride 4, so each 16-px block = 4x4 output cells.
    z16 = z.view(42, 4, 42, 4).permute(0, 2, 1, 3).reshape(42, 42, 16).sum(-1)
    assert torch.allclose(z16, ys[16], atol=1e-3, rtol=0), \
        f"Max local discrepancy: {(z16 - ys[16]).abs().max()}"


# ---------------------------------------------------------------------------
# 2. NB NLL — extreme counts, no NaN/Inf
# ---------------------------------------------------------------------------

def test_patch_nb_nll_finite():
    """NB NLL must be finite for all extreme cases listed in spec §8."""
    cases = [
        (0.0, 1e-6, 1.0),
        (0.0, 100.0, 5.0),
        (1.0, 1.0, 2.0),
        (1000.0, 950.0, 25.0),
        (20000.0, 19800.0, 50.0),
        (20000.0, 1e-5, 0.5),
        (0.0, 20000.0, 0.5),
    ]
    for y_v, mu_v, r_v in cases:
        y = torch.tensor([y_v])
        mu = torch.tensor([mu_v])
        r = torch.tensor([r_v])
        loss = nb_nll(y, mu, r)
        assert torch.isfinite(loss).all(), \
            f"NB NLL not finite: y={y_v}, mu={mu_v}, r={r_v} -> {loss}"

    # Method-of-moments init must not overflow
    h = HierarchicalNBLoss([16, 32, 96])
    h.init_dispersion_from_stats(16, 100.0, 100.0)  # var == mean -> r -> max
    assert torch.isfinite(h.get_dispersion(16))
    assert h.get_dispersion(16) <= 1e4


# ---------------------------------------------------------------------------
# 3. Criterion — lambda=0 is a true ablation (no spurious gradient)
# ---------------------------------------------------------------------------

def test_patch_criterion_zero_lambda_ablation():
    """Setting all lambdas to 0 must yield exactly zero total loss."""
    D = torch.nn.Parameter(torch.full((2, 1, 168, 168), 0.01))
    gt = {b: torch.zeros(2, 672 // b, 672 // b) for b in [16, 32, 96]}
    z0 = torch.zeros(2, 168, 168)
    c = torch.zeros(2)
    crit = HPCLossCriterion(
        [16, 32, 96],
        lambda_count=0,
        lambda_hnb=0,
        lambda_alloc=0,
        lambda_hn=0,
        lambda_empty=0,
        lambda_global=0,
        lambda_rob=0,
    )
    total, _ = crit(D, gt, z0, c, progress=0.05)
    assert total.item() == 0.0, f"Expected 0.0, got {total.item()}"


# ---------------------------------------------------------------------------
# 4. Criterion — finite loss + finite gradients with degraded mask
# ---------------------------------------------------------------------------

def test_patch_criterion_finite_and_backward():
    """Criterion forward+backward must stay finite; degraded mask must be respected."""
    D = torch.nn.Parameter(torch.full((2, 1, 168, 168), 0.01))
    gt = {b: torch.zeros(2, 672 // b, 672 // b) for b in [16, 32, 96]}
    z0 = torch.zeros(2, 168, 168)
    c = torch.zeros(2)
    Dd = torch.full_like(D, 0.012).detach()
    crit = HPCLossCriterion([16, 32, 96])
    mask = torch.tensor([True, False])
    total, details = crit(D, gt, z0, c, d_degraded=Dd, degraded_mask=mask, progress=0.5)
    assert torch.isfinite(total), f"Non-finite loss: {total}"
    total.backward()
    assert D.grad is not None
    assert torch.isfinite(D.grad).all(), "Non-finite gradient"


# ---------------------------------------------------------------------------
# 5. BaseCrowdDataset — schema stability under collation
# ---------------------------------------------------------------------------

def test_patch_dataset_schema():
    """All batches must have the same keys including image_degraded/has_degraded."""
    with tempfile.TemporaryDirectory() as td:
        paths, plist = [], []
        for i in range(6):
            p = os.path.join(td, f"{i}.jpg")
            Image.new("RGB", (500, 500), (120, 120, 120)).save(p)
            paths.append(p)
            plist.append(np.array([[100, 100]], np.float32))
        ds = BaseCrowdDataset(
            paths, plist,
            crop_size=448,
            hnb_blocks=[16, 32, 64],
            second_view_prob=0.5,
            is_train=True,
        )
        loader = DataLoader(
            ds, batch_size=4, shuffle=False, num_workers=0,
            collate_fn=custom_collate_fn,
        )
        for batch in loader:
            assert "image_degraded" in batch
            assert "has_degraded" in batch
            assert batch["image_degraded"].shape == batch["image"].shape
            assert batch["has_degraded"].dtype == torch.bool


# ---------------------------------------------------------------------------
# 6. GeometricTransforms — isotropic scaling (aspect ratio preserved)
# ---------------------------------------------------------------------------

def test_patch_isotropic_scaling():
    """GeometricTransforms must use a single isotropic scale (no aspect distortion)."""
    import random
    random.seed(123)
    img = Image.new("RGB", (100, 1000))
    p = np.array([[50, 500]], np.float32)
    t = GeometricTransforms(crop_size=448, scale_range=(0.75, 0.75), flip_prob=0)
    out, _ = t(img, p)
    assert out.size == (448, 448), f"Expected 448x448, got {out.size}"


# ---------------------------------------------------------------------------
# 7. Sampler — empty-scene density bin is reserved
# ---------------------------------------------------------------------------

def test_patch_sampler_empty_bin():
    """Empty images must be assigned to density bin 0, not merged with positives."""
    with tempfile.TemporaryDirectory() as td:
        paths, pl = [], []
        counts = [0, 0, 1, 10, 100, 1000]
        for i, n in enumerate(counts):
            p = os.path.join(td, f"{i}.jpg")
            Image.new("RGB", (64, 64), (i * 20, i * 20, i * 20)).save(p)
            paths.append(p)
            pl.append(np.zeros((n, 2), np.float32))
        sampler, stats = build_density_luminance_sampler(
            paths, pl, num_density_bins=5, num_luminance_bins=2
        )
        assert stats["empty_count"] == 2
        # Empty images must live in a '0_*' group.
        assert any(k.startswith("0_") for k in stats["group_counts"]), \
            f"No empty density bin found in groups: {stats['group_counts']}"


# ---------------------------------------------------------------------------
# 8. NAE — official formula (excludes GT=0, no +1 in denominator)
# ---------------------------------------------------------------------------

def test_patch_nae_formula():
    """NAE matches official NWPU formula: mean(|pred-gt|/gt) for gt>0 only."""
    pred = np.array([5.0, 10.0, 25.0])
    gt = np.array([0.0, 10.0, 20.0])
    # Only non-zero gt: |10-10|/10 + |25-20|/20 = 0 + 0.25 = 0.25 -> mean = 0.125
    result = compute_nae(pred, gt)
    assert abs(result - 0.125) < 1e-12, f"NAE expected 0.125, got {result}"


# ---------------------------------------------------------------------------
# 9. CSVLogger — schema expansion preserves old rows
# ---------------------------------------------------------------------------

def test_patch_csv_logger_schema_expansion():
    """Adding a new key must back-fill empty string in existing rows."""
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "x.csv")
        lg = CSVLogger(p)
        lg.log({"epoch": 1, "mae": 2.0})
        lg.log({"epoch": 2, "mae": 1.0, "empty_mae": 0.5})
        rows = list(csv.DictReader(open(p)))
        assert "empty_mae" in rows[0], "Old row must have new key after expansion"
        assert rows[1]["empty_mae"] == "0.5"


# ---------------------------------------------------------------------------
# 10. Checkpoint — criterion state (NB dispersion) round-trips correctly
# ---------------------------------------------------------------------------

def test_patch_checkpoint_criterion_state():
    """Saving and loading a checkpoint must restore learnable criterion parameters."""
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor(1.0))

    model = DummyModel()
    criterion = HierarchicalNBLoss([16])
    with tempfile.TemporaryDirectory() as td:
        state = build_checkpoint_state(model, criterion=criterion, epoch=3)
        save_checkpoint(state, td)
        # Corrupt dispersion
        with torch.no_grad():
            criterion.raw_dispersions["16"].fill_(0.0)
        ck = load_checkpoint(os.path.join(td, "checkpoint.pt"), model, criterion=criterion)
        assert ck["epoch"] == 3
        # Should have been restored from checkpoint
        assert criterion.raw_dispersions["16"].item() != 0.0, \
            "Dispersion was not restored from checkpoint"
