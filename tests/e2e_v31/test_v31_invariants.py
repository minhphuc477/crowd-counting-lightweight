from __future__ import annotations

import pathlib
import pytest
import torch
import torch.nn.functional as F

from rmr_v3.config import load_config, validate_v3_config
from rmr_v3.model import RMRv3, RMRv3Config
from rmr_v3.model.dual_lattice import (
    push_forward_stride2_to_stride4,
    pullback_stride4_to_stride2_rn,
    check_mass_conservation,
)
from rmr_v3.solver_ops import (
    density_gated_anscombe_discrepancy,
    perona_malik_anisotropic_diffusion,
)


V31_CONFIG_PATHS = [
    "configs/rmr_v31/rmr_v31_step0_v19_anchor.yaml",
    "configs/rmr_v31/rmr_v31_h1_stride2_area_norm.yaml",
    "configs/rmr_v31/rmr_v31_h2_density_gated_anscombe.yaml",
    "configs/rmr_v31/rmr_v31_h3_anisotropic_sirt.yaml",
    "configs/rmr_v31/rmr_v31_h4_composite_sub50.yaml",
]


@pytest.mark.parametrize("cfg_path", V31_CONFIG_PATHS)
def test_v31_config_and_parameter_ceiling(cfg_path: str) -> None:
    """Verify each RMR-v31 configuration is valid and strictly <= 105,000 trainable parameters."""
    raw_cfg = load_config(cfg_path)
    validate_v3_config(raw_cfg)
    model_cfg = RMRv3Config(**raw_cfg["model"])
    model = RMRv3(model_cfg)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert num_params <= 105000, f"{cfg_path} exceeded parameter ceiling: {num_params} > 105000"
    assert raw_cfg.get("train", {}).get("deterministic") is True, f"{cfg_path} must have deterministic: true by default"


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


def test_v31_step0_bitwise_parity_with_v30_anchor() -> None:
    """Verify rmr_v31_step0_v19_anchor is bitwise identical to rmr_v30_step0_v19_anchor."""
    raw_v30 = load_config("configs/rmr_v30/rmr_v30_step0_v19_anchor.yaml")
    raw_v31 = load_config("configs/rmr_v31/rmr_v31_step0_v19_anchor.yaml")
    cfg_v30 = RMRv3Config(**raw_v30["model"])
    cfg_v31 = RMRv3Config(**raw_v31["model"])

    torch.manual_seed(42)
    m30 = RMRv3(cfg_v30).eval()
    torch.manual_seed(42)
    m31 = RMRv3(cfg_v31).eval()

    # Load identical state dict
    m31.load_state_dict(m30.state_dict())

    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        out30 = m30(x)
        out31 = m31(x)

    diff_y = (out30.y - out31.y).abs().max().item()
    diff_y0 = (out30.y0 - out31.y0).abs().max().item()
    assert diff_y == 0.0, f"Bitwise discrepancy in y: {diff_y}"
    assert diff_y0 == 0.0, f"Bitwise discrepancy in y0: {diff_y0}"


def test_perona_malik_anisotropic_diffusion_invariants() -> None:
    """Verify Perona-Malik preserves sharp edges and conserves discrete mass."""
    torch.manual_seed(42)
    # Synthetic head peak: sharp Dirac impulse surrounded by background
    y = torch.zeros(1, 1, 32, 32)
    y[0, 0, 16, 16] = 1.0  # Sharp head peak
    y[0, 0, 16, 17] = 0.8

    initial_mass = y.sum().item()
    y_diff = perona_malik_anisotropic_diffusion(y, tv_lambda=0.02, kappa=0.05)

    # 1. Mass conservation: total mass must be conserved before/after
    diff_mass = abs(y_diff.sum().item() - initial_mass)
    assert diff_mass < 1e-6, f"Mass drift in Perona-Malik: {diff_mass}"

    # 2. Peak preservation: the peak should remain distinct
    assert y_diff[0, 0, 16, 16].item() > 0.85, "Peak was over-smoothed by Perona-Malik"


def test_density_gated_anscombe_discrepancy() -> None:
    """Verify Density-Gated Anscombe VST only activates in dense regions."""
    torch.manual_seed(42)
    # Background region: q = 0.001, b = 0.0, area = 64 cells
    q_bg = torch.tensor([[[0.001]]])
    b_bg = torch.tensor([[[0.0]]])
    area = torch.tensor([[[64.0]]])

    res_bg = density_gated_anscombe_discrepancy(q_bg, b_bg, area, tau_dense=0.08)
    expected_linear_bg = (q_bg - b_bg) / area
    assert torch.allclose(res_bg, expected_linear_bg, atol=1e-6), "Anscombe VST should be disabled on background"

    # Dense region: q = 20.0, b = 15.0, area = 64 cells (rate = 20/64 = 0.3125 > 0.08)
    q_dense = torch.tensor([[[20.0]]])
    b_dense = torch.tensor([[[15.0]]])
    res_dense = density_gated_anscombe_discrepancy(q_dense, b_dense, area, tau_dense=0.08)
    # Should NOT equal linear discrepancy
    linear_dense = (q_dense - b_dense) / area
    assert not torch.allclose(res_dense, linear_dense, atol=1e-4), "Anscombe VST must be active on dense crowds"


def test_dual_lattice_mass_conservation() -> None:
    """Verify push-forward and pull-back conserve mass to machine precision."""
    torch.manual_seed(42)
    y_fine = torch.rand(2, 1, 128, 128)
    y_carrier = push_forward_stride2_to_stride4(y_fine)

    assert check_mass_conservation(y_fine, y_carrier, eps=1e-6)

    y_prolong = pullback_stride4_to_stride2_rn(y_carrier, y_fine)
    assert check_mass_conservation(y_prolong, y_carrier, eps=1e-6)
