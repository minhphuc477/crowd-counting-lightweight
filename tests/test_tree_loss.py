import torch

from hpc.losses.count_tree import AdaptiveProbabilisticCountTreeLoss, CountTreeConfig
from hpc.losses.hpc_adaptive import AdaptiveHPCLoss, HPCLossConfig


def test_tree_loss_gradient_flow():
    torch.manual_seed(0)

    cfg = HPCLossConfig(
        tree=CountTreeConfig(
            root_dispersion=50.0,
            kappa_root64=20.0,
            kappa_64_32=20.0,
            kappa_32_16=20.0,
            kappa_16_8=20.0,
            dense_threshold_16=4,
            use_dirichlet_multinomial=True,
        ),
        hard_zero_weight=0.10,
        local_contrast_weight=0.05,
    )

    criterion = AdaptiveHPCLoss(cfg, feature_dim=32)

    mass = torch.rand(2, 1, 64, 64, requires_grad=True)
    p4 = torch.rand(2, 32, 64, 64, requires_grad=True)

    target_pyramid = {
        64: torch.randint(0, 10, (2, 4, 4)).float(),
        32: torch.randint(0, 5, (2, 8, 8)).float(),
        16: torch.randint(0, 3, (2, 16, 16)).float(),
        8: torch.randint(0, 2, (2, 32, 32)).float(),
        "N": torch.tensor([50.0, 75.0]),
    }

    loss, logs = criterion(mass, p4, target_pyramid)

    assert torch.isfinite(loss), "Loss is NaN or Inf!"
    loss.backward()

    assert mass.grad is not None and torch.isfinite(mass.grad).all(), "Mass gradients invalid!"
    assert p4.grad is not None and torch.isfinite(p4.grad).all(), "P4 gradients invalid!"

    print("test_tree_loss_gradient_flow passed successfully!")


if __name__ == "__main__":
    test_tree_loss_gradient_flow()
