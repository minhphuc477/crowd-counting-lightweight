from __future__ import annotations

import math
import pytest
import torch
import torch.nn.functional as F

from rmr_core.losses import count_magnitude_loss
from rmr_v3.config import validate_v3_config
from rmr_v3.losses.config import RMRv3LossConfig
from rmr_v3.losses.orchestration import compute_rmr_v3_losses
from rmr_v3.model import RMRv3, RMRv3Config
from rmr_v3.model.dual_lattice import SubpixelAllocationHead


def test_subpixel_allocation_head_properties():
    """Verify SubpixelAllocationHead parameter count, Step 0 uniform parity, and mass conservation."""
    head = SubpixelAllocationHead(in_channels=32)
    params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    assert params == 452, f"SubpixelAllocationHead must have exactly 452 params, got {params}"

    # Verify Step 0 Parity: zero init produces uniform [0.25, 0.25, 0.25, 0.25]
    b, c, h, w = 2, 32, 16, 16
    p4 = torch.randn(b, c, h, w)
    y4 = torch.rand(b, 1, h, w) * 10.0 + 1.0  # Positive counts
    y2 = head(p4, y4)

    assert y2.shape == (b, 1, 2 * h, 2 * w), f"Expected shape {(b, 1, 32, 32)}, got {y2.shape}"

    # Step 0 uniform parity check: each 2x2 fine block must equal y4 * 0.25
    y4_expanded = F.interpolate(y4, scale_factor=2, mode="nearest") * 0.25
    assert torch.allclose(y2, y4_expanded, atol=1e-5), "At initialization, each sub-pixel must be exactly 0.25 * y4"

    # Strict mass conservation check: sum(y2) == sum(y4)
    m4 = y4.sum(dim=(-2, -1))
    m2 = y2.sum(dim=(-2, -1))
    assert torch.allclose(m4, m2, atol=1e-5), f"Mass must be conserved: sum(y4)={m4} vs sum(y2)={m2}"


def test_anscombe_count_loss():
    """Verify Anscombe variance-stabilized count magnitude loss behavior and autograd."""
    pred = torch.tensor([10.0, 500.0, 2000.0], requires_grad=True)
    target = torch.tensor([12.0, 480.0, 1950.0])

    loss = count_magnitude_loss(pred, target, mode="anscombe")
    assert torch.isfinite(loss), "Anscombe loss must be finite"
    assert loss.item() > 0.0, "Loss must be positive for non-identical counts"

    loss.backward()
    assert pred.grad is not None, "Gradient must be computed"
    assert torch.all(torch.isfinite(pred.grad)), "Gradients must be finite"

    # Perfect prediction must yield zero loss
    loss_zero = count_magnitude_loss(target, target, mode="anscombe")
    assert torch.allclose(loss_zero, torch.tensor(0.0), atol=1e-6)


def test_subpixel_dm_model_forward_backward():
    """Verify complete forward pass and backward pass with subpixel_dm enabled."""
    cfg = RMRv3Config(
        backbone_name="mobilenetv4_conv_small_050",
        pretrained=False,
        output_stride=2,
        subpixel_dm=True,
        iterations=2,
        max_trainable_params=105000,
    )
    model = RMRv3(cfg)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params <= 105000, f"Model params {total_params} exceeds 105,000 ceiling"

    x = torch.randn(2, 3, 256, 256)
    out = model(x)

    # Output y must be at Stride 2 (128x128)
    assert out.y.shape == (2, 1, 128, 128), f"Expected Stride 2 (128x128), got {out.y.shape}"
    # Carrier y_carrier must be at Stride 4 (64x64)
    assert out["y_carrier"].shape == (2, 1, 64, 64), f"Expected carrier Stride 4 (64x64), got {out['y_carrier'].shape}"

    # Verify mass conservation between fine y and carrier y
    assert torch.allclose(out.y.sum(dim=(-2, -1)), out["y_carrier"].sum(dim=(-2, -1)), atol=1e-5)

    # Compute loss
    target_stride2 = torch.rand(2, 1, 128, 128).abs()
    loss_cfg = RMRv3LossConfig(
        output_stride=2,
        count_loss_mode="anscombe",
        cell_loss_mode="count_harmonized",
        lambda_carrier_cell=0.25,
        lambda_fine_cell=0.25,
    )
    losses = compute_rmr_v3_losses(dict(out), target_stride2, loss_cfg)
    assert "total" in losses and torch.isfinite(losses["total"])
    assert losses["total"].item() > 0.0

    # Backward autograd check
    losses["total"].backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0.0, "Total gradient norm must be non-zero"


def test_subpixel_dm_odd_dimensions_and_trajectory_diagnostics():
    """Verify exact mass conservation on non-divisible odd dimensions and trajectory diagnostics."""
    from rmr_v3.diagnostics.trajectory import compute_solver_trajectory_diagnostics

    cfg = RMRv3Config(
        backbone_name="mobilenetv4_conv_small_050",
        pretrained=False,
        output_stride=2,
        subpixel_dm=True,
        iterations=2,
        max_trainable_params=105000,
    )
    model = RMRv3(cfg)

    for h_odd, w_odd in [(255, 257), (513, 515)]:
        x = torch.randn(1, 3, h_odd, w_odd)
        out = model(x)
        expected_h2 = (h_odd + 1) // 2
        expected_w2 = (w_odd + 1) // 2
        assert out.y.shape == (1, 1, expected_h2, expected_w2)

        # Exact mass conservation check on odd dimensions
        m_fine = out.y.sum().item()
        m_carrier = out["y_carrier"].sum().item()
        assert abs(m_fine - m_carrier) < 1e-4, (
            f"Mass mismatch on {h_odd}x{w_odd}: fine {m_fine} vs carrier {m_carrier}"
        )

        # Trajectory diagnostics check
        target_odd = torch.rand(1, 1, expected_h2, expected_w2).abs()
        diag = compute_solver_trajectory_diagnostics(dict(out), target_odd)
        assert "mae_reg_y0" in diag and math.isfinite(diag["mae_reg_y0"])
        assert "mae_reg_y1" in diag and math.isfinite(diag["mae_reg_y1"])
        assert diag["mae_reg_y0"] > 0.0



@pytest.mark.parametrize(
    "config_name,expected_params",
    [
        ("sub60_unified_t8.yaml", 104441),
        ("sub60_quad_scale_t8.yaml", 104474),
        ("sub60_anscombe_count_t8.yaml", 104474),
        ("sub60_quad_scale_fg75_t8.yaml", 104474),
        ("sub60_dual_lattice_t8.yaml", 104926),
    ],
)
def test_all_sub60_configs_validation_and_budgets(config_name: str, expected_params: int):
    """Verify that every Sub-60 config passes validation and satisfies strict parameter ceiling."""
    import yaml
    from pathlib import Path
    yaml_path = Path("configs/rmr_sub60") / config_name
    assert yaml_path.exists(), f"{yaml_path} must exist"
    yaml_dict = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    validate_v3_config(yaml_dict)
    m_cfg = dict(yaml_dict["model"])
    m_cfg["pretrained"] = False
    cfg = RMRv3Config.from_dict(m_cfg, pretrained=False)
    m = RMRv3(cfg)
    total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert total_params <= 105000, f"Config {config_name} params {total_params} must be <= 105,000"
    assert total_params == expected_params, f"Config {config_name} expected {expected_params} params, got {total_params}"
