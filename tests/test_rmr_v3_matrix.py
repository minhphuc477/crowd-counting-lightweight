import pytest
import torch

from rmr_v3.model import RMRv3, RMRv3Config
from rmr_v3.losses import RMRv3LossConfig, compute_rmr_v3_losses


@pytest.mark.parametrize(
    "variant_name, neck_type, enable_solver, region_sizes, alloc_loss",
    [
        ("C0", "additive", False, (32, 64, 128), "flat_dm16"),
        ("C1", "additive", True, (32, 64, 128), "flat_dm16"),
        ("C2", "rep_weighted", False, (16, 32, 64, 128), "bayesian"),
        ("C3", "rep_weighted", True, (16, 32, 64, 128), "bayesian"),
    ],
)
def test_rmr_v3_c0_to_c3_matrix(variant_name, neck_type, enable_solver, region_sizes, alloc_loss):
    torch.manual_seed(42)
    cfg = RMRv3Config(
        pretrained=False,
        neck_type=neck_type,
        enable_solver=enable_solver,
        region_sizes_px=region_sizes,
    )
    model = RMRv3(cfg)
    model.train()

    # Input: 2 dummy images of size 256x256
    x = torch.randn(2, 3, 256, 256)
    out = model(x)

    assert "y" in out
    assert "y0" in out
    assert out["y"].shape == (2, 1, 64, 64)
    assert out["y0"].shape == (2, 1, 64, 64)

    if not enable_solver:
        # In direct mode, final y must exactly be y0
        assert torch.allclose(out["y"], out["y0"])

    # Test loss computation
    loss_cfg = RMRv3LossConfig(
        allocation_loss_type=alloc_loss,
    )
    target_y = torch.randint(0, 3, (2, 1, 64, 64)).float()
    points = [
        torch.tensor([[50.0, 50.0], [100.0, 120.0]]),
        torch.tensor([[30.0, 80.0]]),
    ]

    losses = compute_rmr_v3_losses(out, target_y, cfg=loss_cfg, points=points)
    total_loss = losses["total"]
    assert torch.isfinite(total_loss)
    assert total_loss.item() > 0.0

    total_loss.backward()

    # Verify gradients
    assert any(p.grad is not None and torch.isfinite(p.grad).any() for p in model.fine_head.parameters())

    # Test deploy switch for rep_weighted neck
    if neck_type == "rep_weighted":
        model.eval()
        model.switch_to_deploy()
        assert model.fusion.ref16.is_deployed
        out_deploy = model(x)
        assert out_deploy["y0"].shape == (2, 1, 64, 64)
