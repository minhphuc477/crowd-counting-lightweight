import math

import numpy as np
import torch

from hpc.metrics.game import (
    game_errors_one_image,
    region_count_from_cell_measure,
)


def test_full_image_region_preserves_all_edge_mass():
    # image is not divisible by stride
    # H=W=10, stride=8 => output grid 2x2
    y = torch.tensor(
        [[[[1.0, 2.0],
           [3.0, 4.0]]]],
        dtype=torch.float32,
    )

    total = region_count_from_cell_measure(
        y,
        x0=0.0,
        y0=0.0,
        x1=10.0,
        y1=10.0,
        image_h=10,
        image_w=10,
        stride=8,
    )

    assert math.isclose(
        total,
        10.0,
        abs_tol=1e-12,
    )


def test_game0_equals_absolute_count_error():
    y = torch.tensor(
        [[[[1.0, 2.0],
           [3.0, 4.0]]]],
        dtype=torch.float32,
    )

    # 8 GT points anywhere inside image
    points = np.asarray(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
            [6.0, 6.0],
            [7.0, 7.0],
            [9.0, 9.0],
        ],
        dtype=np.float64,
    )

    out = game_errors_one_image(
        y,
        points,
        image_h=10,
        image_w=10,
        stride=8,
        levels=(0,),
    )

    # predicted total = 10
    # GT total = 8
    assert math.isclose(
        out[0],
        2.0,
        abs_tol=1e-12,
    )


def test_game_mass_partition_conservation():
    y = torch.tensor(
        [[[[1.0, 2.0],
           [3.0, 4.0]]]],
        dtype=torch.float32,
    )

    # Check predicted region counts across 2x2 GAME
    total = 0.0

    for r in range(2):
        for c in range(2):
            total += region_count_from_cell_measure(
                y,
                x0=5.0 * c,
                y0=5.0 * r,
                x1=5.0 * (c + 1),
                y1=5.0 * (r + 1),
                image_h=10,
                image_w=10,
                stride=8,
            )

    assert math.isclose(
        total,
        10.0,
        abs_tol=1e-10,
    )
