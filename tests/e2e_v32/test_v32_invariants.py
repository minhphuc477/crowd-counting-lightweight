from __future__ import annotations

import pathlib
import pytest
import torch
import torch.nn.functional as F

from rmr_v3.config import load_config, validate_v3_config
from rmr_v3.model import RMRv3, RMRv3Config
from rmr_v3.model.perspective import ContinuousPerspectiveCarrierModulation
from rmr_core.heads import _density_activate


V32_CONFIG_PATHS = [
    "configs/rmr_v32/rmr_v32_step0_anchor.yaml",
    "configs/rmr_v32/rmr_v32_control_no_solver.yaml",
    "configs/rmr_v32/rmr_v32_h1_cpcm.yaml",
    "configs/rmr_v32/rmr_v32_h2_floor_suppression.yaml",
    "configs/rmr_v32/rmr_v32_h3_composite.yaml",
    "configs/rmr_v32/rmr_v32_h4_dense_loss_scaling.yaml",
    "configs/rmr_v32/rmr_v32_h5_conservative_solver.yaml",
]


@pytest.mark.parametrize("cfg_path", V32_CONFIG_PATHS)
def test_v32_config_and_parameter_ceiling(cfg_path: str) -> None:
    """Verify each RMR-v32 configuration is valid and strictly <= 105,000 trainable parameters."""
    raw_cfg = load_config(cfg_path)
    validate_v3_config(raw_cfg)
    model_cfg = RMRv3Config(**raw_cfg["model"])
    model = RMRv3(model_cfg)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert num_params <= 105000, f"{cfg_path} exceeded parameter ceiling: {num_params} > 105000"
    assert raw_cfg.get("train", {}).get("deterministic") is True, f"{cfg_path} must have deterministic: true by default"

    if "h1_cpcm" in cfg_path or "h3_composite" in cfg_path:
        assert num_params == 104753, f"Expected exactly 104,753 params for {cfg_path}, got {num_params}"
    else:
        assert num_params == 104441, f"Expected exactly 104,441 params for {cfg_path}, got {num_params}"


def test_python_source_line_counts() -> None:
    """Verify all Python source files in rmr_v3/ and rmr_core/ are strictly <= 450 lines."""
    violating_files: list[tuple[str, int]] = []
    for pkg in ["rmr_v3", "rmr_core"]:
        pkg_dir = pathlib.Path(pkg)
        for py_file in pkg_dir.rglob("*.py"):
            lines = py_file.read_text(encoding="utf-8").splitlines()
            if len(lines) > 450:
                violating_files.append((str(py_file), len(lines)))

    assert len(violating_files) == 0, f"Files exceeding 450 lines: {violating_files}"


def test_v32_step0_bitwise_parity_with_v31_anchor() -> None:
    """Verify rmr_v32_step0_anchor is bitwise identical to rmr_v31_step0_v19_anchor."""
    raw_v31 = load_config("configs/rmr_v31/rmr_v31_step0_v19_anchor.yaml")
    raw_v32 = load_config("configs/rmr_v32/rmr_v32_step0_anchor.yaml")
    cfg_v31 = RMRv3Config(**raw_v31["model"])
    cfg_v32 = RMRv3Config(**raw_v32["model"])

    torch.manual_seed(42)
    m31 = RMRv3(cfg_v31).eval()
    torch.manual_seed(42)
    m32 = RMRv3(cfg_v32).eval()

    m32.load_state_dict(m31.state_dict())

    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        out31 = m31(x)
        out32 = m32(x)

    diff_y = (out31.y - out32.y).abs().max().item()
    diff_y0 = (out31.y0 - out32.y0).abs().max().item()
    assert diff_y == 0.0, f"Bitwise discrepancy in y: {diff_y}"
    assert diff_y0 == 0.0, f"Bitwise discrepancy in y0: {diff_y0}"


def test_cpcm_zero_init_identity_warmstart() -> None:
    """Verify CPCM at initialization applies exactly multiplier 1.0 everywhere."""
    cpcm = ContinuousPerspectiveCarrierModulation(channels=32, hidden=8)
    # Total parameter check
    num_params = sum(p.numel() for p in cpcm.parameters() if p.requires_grad)
    assert num_params == 312, f"Expected 312 params for CPCM, got {num_params}"

    # Forward pass on arbitrary tensor
    x = torch.randn(2, 32, 128, 128)
    out = cpcm(x)
    diff = (out - x).abs().max().item()
    assert diff == 0.0, f"CPCM at init should be exact identity, got max diff {diff}"


def test_floor_suppression_non_saturating_dynamics() -> None:
    """Verify Shifted ReLU-Softplus zeroes out background and preserves full gradient on heads."""
    z = torch.tensor([-6.0, -5.0, -3.0, 0.0, 2.0], requires_grad=True)
    floor_tau = 0.008

    y = _density_activate(
        z,
        temp_softplus=False,
        tau=None,
        density_curvature=False,
        curvature_alpha=None,
        gated_density_curvature=False,
        curvature_dense_threshold=0.15,
        curvature_gate_beta=0.03,
        curvature_pool_kernel=8,
        floor_tau=floor_tau,
    )

    # 1. Background suppression: softplus(-6.0) = 0.00247 < 0.008 -> quadratically suppressed without dead-zone
    assert y[0].item() < 0.001, f"Expected < 0.001 for background, got {y[0].item()}"
    assert y[1].item() < 0.003, f"Expected < 0.003 for background, got {y[1].item()}"
    assert y[0].item() > 0.0, "Smooth floor must maintain strictly positive mass to avoid dying-ReLU trap"

    # 2. Foreground preservation: softplus(2.0) = 2.1269 > 0.008 -> preserved minus half floor
    assert y[4].item() > 2.1, f"Expected > 2.1 for foreground, got {y[4].item()}"

    # 3. Gradient dynamics: backprop from foreground must yield unattenuated sigmoid gradient
    loss_fg = y[4]
    loss_fg.backward(retain_graph=True)
    expected_grad = torch.sigmoid(torch.tensor(2.0)).item()
    actual_grad = z.grad[4].item()
    assert abs(actual_grad - expected_grad) < 1e-6, f"Expected unattenuated grad {expected_grad}, got {actual_grad}"

    # 4. Non-zero gradient in background (recovers from dying ReLU trap)
    loss_bg = y[0]
    loss_bg.backward()
    assert z.grad[0].item() > 0.0, "Background cell must have non-zero gradient for recovery"


def test_v32_forward_backward_gradient_flow() -> None:
    """Verify end-to-end forward and backward pass with CPCM and Floor Suppression active."""
    raw_cfg = load_config("configs/rmr_v32/rmr_v32_h3_composite.yaml")
    model_cfg = RMRv3Config(**raw_cfg["model"])
    model = RMRv3(model_cfg).train()

    x = torch.randn(2, 3, 256, 256)
    out = model(x)

    assert out.y.shape[-2:] == (64, 64)
    assert not torch.isnan(out.y).any(), "NaN in predicted y"
    assert not torch.isinf(out.y).any(), "Inf in predicted y"

    loss = out.y.sum()
    loss.backward()

    # Check that CPCM parameters receive healthy non-zero gradients
    assert model.cpcm is not None
    cpcm_grad_norms = [p.grad.norm().item() for p in model.cpcm.parameters() if p.grad is not None]
    assert len(cpcm_grad_norms) > 0
    assert all(not torch.isnan(p.grad).any() for p in model.cpcm.parameters() if p.grad is not None)
    assert model.cpcm.mlp[2].weight.grad.norm().item() > 0.0, "CPCM final weight received zero gradient"


def test_v32_dense_loss_scaling_exactness_and_isolation() -> None:
    """Verify that density loss scaling operates elementwise without batch cross-talk and works on B=1."""
    from rmr_core.operators import build_multiscale_regions, regional_sum
    from rmr_v3.losses import RMRv3LossConfig, compute_rmr_v3_losses

    h, w = 64, 64
    regions = build_multiscale_regions(h, w, 4, [32], 0.5, include_full_image=False, device="cpu")
    m = regions.boxes.shape[0]

    target_stadium = torch.full((1, 1, h, w), 1500.0 / (h * w), dtype=torch.float32)
    target_empty = torch.zeros((1, 1, h, w), dtype=torch.float32)
    pred_stadium = torch.full((1, 1, h, w), 0.1, dtype=torch.float32, requires_grad=True)
    pred_empty = torch.full((1, 1, h, w), 0.1, dtype=torch.float32, requires_grad=True)

    out_stadium = {
        "y": pred_stadium, "y0": pred_stadium.clone(),
        "regions": regions, "b_region": regional_sum(pred_stadium, regions.boxes),
        "region_dispersion": torch.full((1, 1, m), 50.0),
    }
    out_empty = {
        "y": pred_empty, "y0": pred_empty.clone(),
        "regions": regions, "b_region": regional_sum(pred_empty, regions.boxes),
        "region_dispersion": torch.full((1, 1, m), 50.0),
    }

    cfg_unscaled = RMRv3LossConfig(
        density_loss_scaling=False, elementwise_dense_scaling=False,
        lambda_count=1.0, lambda_cell=0.5, lambda_region_nb=0.0,
        lambda_flat_dm16=0.0, lambda_scale_align=0.0, lambda_curvature=0.0,
    )
    cfg_scaled = RMRv3LossConfig(
        density_loss_scaling=True, dense_loss_thresh=100.0, dense_loss_norm=150.0,
        dense_loss_alpha=1.0, dense_loss_max_boost=2.0,
        lambda_count=1.0, lambda_cell=0.5, lambda_region_nb=0.0,
        lambda_flat_dm16=0.0, lambda_scale_align=0.0, lambda_curvature=0.0,
    )

    # 1. Test B=1 stadium scaling (must receive exact 3.0x boost)
    l_unscaled = compute_rmr_v3_losses(out_stadium, target_stadium, cfg_unscaled)
    l_scaled = compute_rmr_v3_losses(out_stadium, target_stadium, cfg_scaled)

    expected_boost = 1.0 + min(2.0, (1500.0 - 100.0) / 150.0)  # = 3.0
    assert abs(l_scaled["dense_loss_scale"].item() - expected_boost) < 1e-4
    assert abs(l_scaled["total"].item() - expected_boost * l_unscaled["total"].item()) < 1e-4

    # 2. Test B=1 empty scaling (must receive exact 1.0x boost)
    l_empty_scaled = compute_rmr_v3_losses(out_empty, target_empty, cfg_scaled)
    assert abs(l_empty_scaled["dense_loss_scale"].item() - 1.0) < 1e-4
    l_empty_unscaled = compute_rmr_v3_losses(out_empty, target_empty, cfg_unscaled)
    assert abs(l_empty_scaled["total"].item() - l_empty_unscaled["total"].item()) < 1e-4

    # 3. Test B=2 batch isolation (average scale = (3.0 + 1.0)/2 = 2.0)
    target_b2 = torch.cat([target_stadium, target_empty], dim=0)
    pred_b2 = torch.cat([pred_stadium, pred_empty], dim=0)
    out_b2 = {
        "y": pred_b2, "y0": pred_b2.clone(),
        "regions": regions, "b_region": regional_sum(pred_b2, regions.boxes),
        "region_dispersion": torch.full((2, 1, m), 50.0),
    }
    l_b2 = compute_rmr_v3_losses(out_b2, target_b2, cfg_scaled)
    assert abs(l_b2["dense_loss_scale"].item() - 2.0) < 1e-4
    expected_b2_loss = 0.5 * (3.0 * l_unscaled["total"].item() + 1.0 * l_empty_unscaled["total"].item())
    assert abs(l_b2["total"].item() - expected_b2_loss) < 1e-4

