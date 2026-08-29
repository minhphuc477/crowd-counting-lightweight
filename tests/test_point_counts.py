import torch

from hpc.data.point_counts import build_exact_count_pyramid


def test_exact_point_counts():
    points = [
        torch.tensor([
            [1.0, 1.0],
            [7.0, 7.0],
            [20.0, 5.0],
            [40.0, 40.0],
        ])
    ]

    target = build_exact_count_pyramid(
        points_batch=points,
        height=64,
        width=64,
        block_sizes=(8, 16, 32, 64),
        device=torch.device("cpu"),
    )

    assert int(target["N"][0].item()) == 4

    for block in (8, 16, 32, 64):
        assert int(target[block][0].sum().item()) == 4
    print("test_exact_point_counts passed!")


if __name__ == "__main__":
    test_exact_point_counts()
