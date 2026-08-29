import torch

from hpc.losses.count_tree import (
    build_predicted_count_pyramid,
    group_four_children,
)


def test_mass_conservation():
    torch.manual_seed(0)

    mass = torch.rand(2, 1, 64, 64)

    p = build_predicted_count_pyramid(
        mass,
        block_sizes=(8, 16, 32, 64),
        output_stride=4,
    )

    assert torch.allclose(
        group_four_children(p[8]).sum(dim=-1),
        p[16],
        atol=1e-5,
        rtol=1e-5,
    )

    assert torch.allclose(
        group_four_children(p[16]).sum(dim=-1),
        p[32],
        atol=1e-5,
        rtol=1e-5,
    )

    assert torch.allclose(
        group_four_children(p[32]).sum(dim=-1),
        p[64],
        atol=1e-5,
        rtol=1e-5,
    )

    assert torch.allclose(
        p[64].sum(dim=(1, 2)),
        p["N"],
        atol=1e-5,
        rtol=1e-5,
    )
    print("test_mass_conservation passed!")


if __name__ == "__main__":
    test_mass_conservation()
