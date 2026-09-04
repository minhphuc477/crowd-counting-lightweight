import math

import numpy as np

from hpc.metrics.decomposition import (
    direct_tiled_discrepancy,
)


def test_zero_discrepancy():
    d = np.asarray(
        [10.0, 20.0]
    )

    t = np.asarray(
        [10.0, 20.0]
    )

    g = np.asarray(
        [10.0, 20.0]
    )

    out = direct_tiled_discrepancy(
        d,
        t,
        g,
    )

    assert (
        out[
            "mean_abs_prediction_discrepancy"
        ]
        == 0.0
    )

    assert (
        out[
            "mean_normalized_prediction_discrepancy"
        ]
        == 0.0
    )


def test_normalized_discrepancy():
    d = np.asarray(
        [12.0, 20.0]
    )

    t = np.asarray(
        [10.0, 25.0]
    )

    g = np.asarray(
        [10.0, 20.0]
    )

    out = direct_tiled_discrepancy(
        d,
        t,
        g,
    )

    # absolute discrepancies = [2,5]
    assert math.isclose(
        out[
            "mean_abs_prediction_discrepancy"
        ],
        3.5,
        abs_tol=1e-12,
    )

    # normalized = [.2,.25]
    assert math.isclose(
        out[
            "mean_normalized_prediction_discrepancy"
        ],
        0.225,
        abs_tol=1e-12,
    )
