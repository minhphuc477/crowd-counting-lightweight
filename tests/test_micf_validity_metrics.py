import math

import torch

from hpc.metrics.micf_validity import (
    aggregate_validity,
    recovered_measure_validity,
)


def test_valid_measure_has_zero_invalidity():
    y = torch.tensor(
        [[[[0.0, 1.0],
           [2.0, 3.0]]]],
        dtype=torch.float32,
    )

    row = recovered_measure_validity(
        y
    )

    assert row.violating_cells == 0
    assert row.vr_tau == 0.0
    assert row.negative_variation == 0.0
    assert row.nvr == 0.0


def test_nvr_is_negative_over_positive_variation():
    y = torch.tensor(
        [[[[2.0, -1.0],
           [0.0, 3.0]]]],
        dtype=torch.float32,
    )

    row = recovered_measure_validity(
        y
    )

    # P = 2 + 3 = 5
    # Q = 1
    assert math.isclose(
        row.positive_variation,
        5.0,
        abs_tol=1e-12,
    )

    assert math.isclose(
        row.negative_variation,
        1.0,
        abs_tol=1e-12,
    )

    assert math.isclose(
        row.nvr,
        1.0 / 5.0,
        abs_tol=1e-12,
    )


def test_vr_uses_tau():
    y = torch.tensor(
        [[[[0.0, -1e-7],
           [-2e-6, 1.0]]]],
        dtype=torch.float64,
    )

    row = recovered_measure_validity(
        y,
        tau=1e-6,
    )

    assert row.violating_cells == 1
    assert math.isclose(
        row.vr_tau,
        0.25,
        abs_tol=1e-12,
    )


def test_micro_aggregation():
    a = recovered_measure_validity(
        torch.tensor(
            [[[[1.0, -1.0]]]]
        )
    )

    b = recovered_measure_validity(
        torch.tensor(
            [[[[3.0, 0.0]]]]
        )
    )

    out = aggregate_validity(
        [a, b]
    )

    # pooled P = 4
    # pooled Q = 1
    assert math.isclose(
        out["nvr_micro"],
        0.25,
        abs_tol=1e-12,
    )

    # one violating cell out of four
    assert math.isclose(
        out["vr_tau_micro"],
        0.25,
        abs_tol=1e-12,
    )
