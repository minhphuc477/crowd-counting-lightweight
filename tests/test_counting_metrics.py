import math

import numpy as np

from hpc.metrics.counting import (
    benchmark_count_summary,
    compute_mae,
    compute_nae,
    compute_rmse,
)


def test_mae_rmse_exact():
    pred = np.asarray(
        [1.0, 4.0, 7.0]
    )
    gt = np.asarray(
        [2.0, 2.0, 8.0]
    )

    # errors = [-1, +2, -1]
    assert math.isclose(
        compute_mae(pred, gt),
        4.0 / 3.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )

    assert math.isclose(
        compute_rmse(pred, gt),
        math.sqrt(2.0),
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_nae_excludes_zero_gt():
    pred = np.asarray(
        [100.0, 8.0, 12.0]
    )
    gt = np.asarray(
        [0.0, 10.0, 10.0]
    )

    # zero-GT first sample is excluded
    # remaining relative errors: .2, .2
    assert math.isclose(
        compute_nae(pred, gt),
        0.2,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_nae_all_zero_returns_nan():
    pred = np.asarray(
        [0.0, 1.0]
    )
    gt = np.asarray(
        [0.0, 0.0]
    )

    assert math.isnan(
        compute_nae(pred, gt)
    )


def test_public_summary_has_only_canonical_keys():
    out = benchmark_count_summary(
        np.asarray([1.0, 2.0]),
        np.asarray([1.0, 3.0]),
    )

    assert set(out.keys()) == {
        "mae",
        "rmse",
        "nae",
        "num_images",
    }
