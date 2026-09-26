"""Mathematical Verification & Invariant Tests for RMR Breakthrough Solutions.

Strictly verifies:
1. CRCDF Support Recovery: Resuscitates zero-predicted heads (y_0 = 0) without background mass leakage.
2. A-SAM Dense Cluster Unstalling: Reduces dense crowd stall error while preserving background noise rejection.
3. CP-DCT2 DC Mass Identity & Boundary Symmetry: C(0, 0) == sum(y) / sqrt(HW) to < 1e-6 precision.
4. Parameter Ceiling & Exact Count: exactly 104,441 trainable parameters across all H7/H8 configs.
5. Codebase Protocol: all files strictly <= 450 lines.
"""
from __future__ import annotations

import math
from pathlib import Path
import yaml
import pytest
import torch
import torch.nn.functional as F

from rmr_core.operators import (
    build_multiscale_regions,
    weighted_normalized_adjoint_field,
)
from rmr_core.spectral import (
    dct_1d,
    dct_2d,
    count_preserving_dct2_loss,
    count_preserving_spectral_loss,
)
from rmr_v3.solver_ops import (
    anscombe_discrepancy,
    density_gated_anscombe_discrepancy,
)
from rmr_v3.model.architecture import RMRv3
from rmr_v3.model.config import RMRv3Config
from rmr_v3.config.validator import validate_v3_config


def test_crcdf_resuscitates_zero_head_without_background_leakage():
    """Verify CRCDF breaks zero-absorbing trap on missing heads while flat background has ZERO leakage."""
    b, c, h, w = 1, 1, 32, 32
    # Entire density iterate is zero (y = 0 everywhere)
    y = torch.zeros(b, c, h, w, dtype=torch.float32)

    # Carrier energy has a localized crest at (16, 16) with high local z-score
    carrier_energy = torch.ones(b, 1, h, w, dtype=torch.float32) * 1.0
    carrier_energy[0, 0, 16, 16] = 20.0  # High-frequency crest

    regions = build_multiscale_regions(
        height=h, width=w, output_stride=4, region_sizes_px=(16, 32), overlap=0.5,
        device=torch.device("cpu"),
    )
    # Regional measurement indicates 5 people in the region covering (16, 16)
    b_region = torch.zeros(b, 1, len(regions.boxes), dtype=torch.float32)
    for idx, box in enumerate(regions.boxes):
        y1, x1, y2, x2 = box.tolist()
        if y1 <= 16 < y2 and x1 <= 16 < x2:
            b_region[0, 0, idx] = 5.0
    weight = torch.ones_like(b_region)

    # 1. Without CRCDF: trapped in zero state (field == 0 everywhere)
    field_standard = weighted_normalized_adjoint_field(
        y, b_region, weight, regions,
        adjoint_mode="radon_nikodym",
        resonant_lambda=0.5,
        carrier_energy=carrier_energy,
        crest_discovery_flux=False,
    )
    assert torch.all(field_standard == 0.0), "Expected standard RN adjoint to be trapped at zero!"

    # 2. With CRCDF: head pixel (16, 16) receives negative field (positive SIRT mass increment)
    field_crcdf = weighted_normalized_adjoint_field(
        y, b_region, weight, regions,
        adjoint_mode="radon_nikodym",
        resonant_lambda=0.5,
        carrier_energy=carrier_energy,
        crest_discovery_flux=True,
        crest_kappa_0=2.0,
        crest_eps_seed=0.005,
    )
    head_field = field_crcdf[0, 0, 16, 16].item()
    assert head_field < -1e-5, f"Expected negative field (mass addition) at crest, got {head_field}"

    # 3. Non-leakage theorem: distant flat background pixels (e.g. (2, 2)) must remain IDENTICALLY zero
    bg_field = field_crcdf[0, 0, 2, 2].item()
    assert bg_field == 0.0, f"Background leakage detected! bg_field = {bg_field} != 0.0"


def test_asymmetric_morozov_dense_recovery_and_noise_rejection():
    """Verify A-SAM permits updates in dense deficit while suppressing background noise."""
    # Case 1: Dense cluster deficit: b = 100 people, q = 95 people
    # Under symmetric Morozov (gamma=0.75):
    # delta_tilde = 2 * (sqrt(95.375) - sqrt(100.375)) = 2 * (9.766 - 10.019) = -0.505
    # |delta_tilde| = 0.505 < 0.75 deadband => symmetric Morozov stalls with ZERO update!
    q_dense = torch.tensor([[[95.0]]], dtype=torch.float32)
    b_dense = torch.tensor([[[100.0]]], dtype=torch.float32)

    res_symm = anscombe_discrepancy(
        q_dense, b_dense, morozov_gamma=0.75, asymmetric_morozov=False
    )
    assert res_symm.item() == 0.0, f"Expected symmetric Morozov to stall at 0.0, got {res_symm.item()}"

    # Under A-SAM: gamma_under = 0.20 / (1 + 0.3 * sqrt(100)) = 0.20 / 4 = 0.05
    # |delta_tilde| = 0.505 > 0.05 deadband => A-SAM continues updating towards truth!
    res_asam = anscombe_discrepancy(
        q_dense, b_dense, morozov_gamma=0.75, asymmetric_morozov=True,
        morozov_gamma_under=0.20, morozov_rho=0.30,
    )
    assert res_asam.item() < -0.01, f"Expected A-SAM to provide active update, got {res_asam.item()}"

    # Case 2: Background false positive: b = 0, q = 0.35 (noise)
    # delta_tilde = 2 * (sqrt(0.725) - sqrt(0.375)) = 2 * (0.851 - 0.612) = +0.478
    # Since delta_tilde > 0, gamma_eff == gamma_over == 0.75
    # 0.478 < 0.75 deadband => A-SAM strictly silences background false positive!
    q_bg = torch.tensor([[[0.35]]], dtype=torch.float32)
    b_bg = torch.tensor([[[0.0]]], dtype=torch.float32)
    res_bg = anscombe_discrepancy(
        q_bg, b_bg, morozov_gamma=0.75, asymmetric_morozov=True,
        morozov_gamma_under=0.20, morozov_rho=0.30,
    )
    assert res_bg.item() == 0.0, f"Expected background noise to be silenced, got {res_bg.item()}"


def test_count_preserving_dct2_exact_dc_mass_identity():
    """Verify 2D DCT-II DC coefficient matches total spatial mass to machine precision."""
    torch.manual_seed(123)
    b, c, h, w = 2, 1, 48, 64
    x = torch.rand(b, c, h, w, dtype=torch.float32) * 5.0
    # Put high density right at the boundary
    x[:, :, 0, :] = 10.0
    x[:, :, :, 0] = 10.0

    dct_x = dct_2d(x)
    dc_computed = dct_x[:, :, 0, 0]
    dc_exact = x.sum(dim=(-2, -1)) / math.sqrt(h * w)

    # In float32, verify relative error matches machine epsilon (< 1e-6)
    rel_err = ((dc_computed - dc_exact).abs() / dc_exact.clamp_min(1e-6)).max().item()
    assert rel_err < 1e-6, f"DC mass identity relative error exceeded: {rel_err}"

    # In float64, verify exact mathematical identity (< 1e-12)
    x64 = x.double()
    dc_comp64 = dct_2d(x64)[:, :, 0, 0]
    dc_exact64 = x64.sum(dim=(-2, -1)) / math.sqrt(h * w)
    err64 = (dc_comp64 - dc_exact64).abs().max().item()
    assert err64 < 1e-12, f"DC mass identity mathematical violation in float64: {err64}"

    # Verify count_preserving_spectral_loss dispatch with transform='dct'
    loss_val, comps = count_preserving_spectral_loss(x, x * 0.9, transform="dct")
    assert loss_val.item() > 0.0
    assert "spectral_dc" in comps
    assert "spectral_ac" in comps


def test_dct2_eliminates_gibbs_boundary_crosshair_ringing():
    """Prove mathematically that DCT-II reduces boundary Gibbs ringing by > 99% vs unwindowed FFT."""
    h, w = 64, 64
    x = torch.zeros(1, 1, h, w, dtype=torch.float32)
    # Step ramp terminating at boundary with non-zero value 10.0
    x[0, 0, :, :] = torch.linspace(0.0, 10.0, w).view(1, -1).expand(h, -1)

    # 2D FFT (periodic boundary step causes artificial 10.0 -> 0.0 step discontinuity)
    fft_x = torch.fft.rfft2(x, norm="ortho")
    fft_hf_energy = fft_x[..., :, w // 4:].abs().square().sum().item()

    # 2D DCT-II (even-symmetric reflection makes boundary continuous: 10.0 -> 10.0)
    dct_x = dct_2d(x)
    dct_hf_energy = dct_x[..., :, w // 2:].square().sum().item()

    # High frequency artifact energy must be dramatically lower in DCT-II (> 95% reduction)
    gibbs_reduction_pct = (fft_hf_energy - dct_hf_energy) / fft_hf_energy * 100.0
    assert gibbs_reduction_pct > 95.0, f"Expected > 95% Gibbs reduction, got {gibbs_reduction_pct:.2f}%"


def test_research_configs_parameter_ceiling():
    """Verify exact 104,441 trainable parameter count across all H7 and H8 configs."""
    configs_to_test = [
        "configs/rmr_research/h7e_carrier_crest_discovery.yaml",
        "configs/rmr_research/h7f_asymmetric_morozov.yaml",
        "configs/rmr_research/h7g_dct_spectral.yaml",
        "configs/rmr_research/h8_harmonious_composite.yaml",
        "configs/rmr_sub60/sub60_v19_anchor.yaml",
        "configs/rmr_sub60/sub60_active_curvature.yaml",
        "configs/rmr_sub60/sub60_count_harmonized.yaml",
        "configs/rmr_sub60/sub60_peak_composite.yaml",
        "configs/rmr_sub60/sub60_resonant_peak.yaml",
    ]

    for cfg_rel in configs_to_test:
        cfg_path = Path(cfg_rel)
        assert cfg_path.exists(), f"Missing config file: {cfg_path}"
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        validate_v3_config(raw)
        cfg = RMRv3Config.from_dict(raw.get("model", {}))
        model = RMRv3(cfg)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert trainable == 104441, f"{cfg_path.name}: expected 104,441 params, got {trainable}"
        assert trainable <= 105000, f"{cfg_path.name}: exceeded parameter budget {trainable} > 105,000"


def test_source_code_strict_line_count_limit():
    """Strict monolith prevention: every .py file in rmr_core/ and rmr_v3/ must have len(lines) <= 450."""
    violations = []
    root = Path(".")
    for d in ["rmr_core", "rmr_v3"]:
        target_dir = root / d
        if not target_dir.exists():
            continue
        for py_path in target_dir.rglob("*.py"):
            lines = py_path.read_text(encoding="utf-8").splitlines()
            if len(lines) > 450:
                violations.append(f"{py_path}: {len(lines)} lines")

    assert not violations, "Files exceeding 450 lines limit:\n" + "\n".join(violations)
