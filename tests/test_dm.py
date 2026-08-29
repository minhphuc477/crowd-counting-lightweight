import torch

from hpc.losses.dirichlet_multinomial import (
    dirichlet_multinomial_nll,
)


def test_dm_prefers_correct_allocation():
    y = torch.tensor([[8.0, 1.0, 1.0, 0.0]])

    good = torch.tensor([[0.78, 0.10, 0.10, 0.02]])
    bad = torch.tensor([[0.05, 0.30, 0.30, 0.35]])

    lg = dirichlet_multinomial_nll(
        y, good, concentration=20.0
    )

    lb = dirichlet_multinomial_nll(
        y, bad, concentration=20.0
    )

    assert lg < lb
    print("test_dm_prefers_correct_allocation passed!")


if __name__ == "__main__":
    test_dm_prefers_correct_allocation()
