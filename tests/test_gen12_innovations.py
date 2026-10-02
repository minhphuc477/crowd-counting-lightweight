import pytest
import yaml
import torch
import torch.nn.functional as F

from rmr_core.operators import build_multiscale_regions, weighted_normalized_adjoint_field
from rmr_v3.model.architecture import RMRv3
from rmr_v3.model.config import RMRv3Config
from rmr_v3.solver import unrolled_sirt_solver


def test_gen12_exact_trainable_parameter_budget():
    """Verify strictly 104,441 trainable parameters with all Gen 12 innovations enabled."""
    with open("configs/rmr_research/sub60_e5_pure_flat_dm16_no_cell.yaml") as f:
        model_cfg = yaml.safe_load(f)["model"]
    model_cfg.update(
        spatial_morozov=True,
        scale_routed_trust=True,
        scale_seeded_carrier=True,
    )
    cfg = RMRv3Config(**model_cfg)
    model = RMRv3(cfg)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_trainable == 104441, f"Expected exactly 104441 trainable parameters, got {n_trainable}"


def test_scale_routed_spatial_morozov_deadband():
    """Verify that Scale-Routed Spatial Morozov narrows deadband in dense regions."""
    b, h, w = 1, 64, 64
    regions = build_multiscale_regions(h, w, 4, region_sizes_px=(32, 64, 128), overlap=0.5, include_full_image=False)
    m = regions.boxes.shape[0]

    y = torch.full((b, 1, h, w), 0.1, dtype=torch.float32)
    b_val = torch.full((b, 1, m), 100.0, dtype=torch.float32)
    b_var = torch.full((b, 1, m), 100.0, dtype=torch.float32)  # sigma_b = 10.0
    weight = torch.ones((b, 1, m), dtype=torch.float32)

    # 1. Uniform routing where scale 0 (dense 32px) dominates (pi_0 = 1.0)
    sc_dense = torch.zeros((b, 3, h, w), dtype=torch.float32)
    sc_dense[:, 0, :, :] = 1.0

    # 2. Uniform routing where scale 2 (coarse 128px) dominates (pi_2 = 1.0)
    sc_sparse = torch.zeros((b, 3, h, w), dtype=torch.float32)
    sc_sparse[:, 2, :, :] = 1.0

    field_dense = weighted_normalized_adjoint_field(
        y, b_val, weight, regions,
        b_variance=b_var,
        scale_routing_weights=sc_dense,
        spatial_morozov=True,
        morozov_gamma_scales=(0.25, 0.50, 0.75),
    )

    field_sparse = weighted_normalized_adjoint_field(
        y, b_val, weight, regions,
        b_variance=b_var,
        scale_routing_weights=sc_sparse,
        spatial_morozov=True,
        morozov_gamma_scales=(0.25, 0.50, 0.75),
    )

    # In dense scale, gamma = 0.25 -> deadband = 2.5
    # In sparse scale, gamma = 0.75 -> deadband = 7.5
    # Residual |q - b| gets less shrinkage in dense scale, so |field_dense| > |field_sparse|
    assert field_dense.abs().mean() > field_sparse.abs().mean(), (
        f"Dense field {field_dense.abs().mean().item()} should be larger than sparse {field_sparse.abs().mean().item()}"
    )


def test_scale_routed_continuous_trust_region():
    """Verify that dense regions allow up to trust_pos_kappa expansion while sparse stays bounded."""
    b, h, w = 1, 32, 32
    regions = build_multiscale_regions(h, w, 4, region_sizes_px=(32,), overlap=0.5, include_full_image=False)
    m = regions.boxes.shape[0]

    y0 = torch.full((b, 1, h, w), 0.5, dtype=torch.float32)
    b_solver = torch.full((b, 1, m), 1000.0, dtype=torch.float32)  # Demands strong upward expansion
    weight_solver = torch.ones((b, 1, m), dtype=torch.float32)

    # Half image dense (pi_0 = 1.0), half image sparse (pi_0 = 0.0)
    sc_weights = torch.zeros((b, 3, h, w), dtype=torch.float32)
    sc_weights[:, 0, :, :w // 2] = 1.0   # Left side: dense
    sc_weights[:, 2, :, w // 2:] = 1.0   # Right side: sparse

    res = unrolled_sirt_solver(
        y0=y0,
        b_solver=b_solver,
        weight_solver=weight_solver,
        regions=regions,
        iterations=1,
        omega=10.0,  # Large omega to hit trust region ceiling
        scale_routing_weights=sc_weights,
        trust_region_kappa=0.35,
        scale_routed_trust=True,
        trust_pos_kappa=0.70,
    )

    y_out = res["y"]
    # In left half (dense), step should reach ~ (1 + 0.70) * y0 = 1.70 * 0.5 = 0.85
    # In right half (sparse), step should reach ~ (1 + 0.35) * y0 = 1.35 * 0.5 = 0.675
    dense_growth = (y_out[:, :, :, :w // 2] - y0[:, :, :, :w // 2]).mean().item()
    sparse_growth = (y_out[:, :, :, w // 2:] - y0[:, :, :, w // 2:]).mean().item()

    assert dense_growth > sparse_growth * 1.5, (
        f"Dense growth {dense_growth} should significantly exceed sparse growth {sparse_growth}"
    )


def test_scale_seeded_carrier_support_recovery():
    """Verify that scale-seeded carrier breaks the zero-support trapping state in dense regions."""
    b, h, w = 1, 32, 32
    regions = build_multiscale_regions(h, w, 4, region_sizes_px=(32,), overlap=0.5, include_full_image=False)
    m = regions.boxes.shape[0]

    # Carrier identically zero (fine head missed the crowd)
    y_zero = torch.zeros((b, 1, h, w), dtype=torch.float32)
    b_val = torch.full((b, 1, m), 50.0, dtype=torch.float32)  # Undercounting: q=0 < b=50
    weight = torch.ones((b, 1, m), dtype=torch.float32)

    # Scale router detects dense crowd features
    sc_weights = torch.zeros((b, 3, h, w), dtype=torch.float32)
    sc_weights[:, 0, :, :] = 1.0  # High fine-scale probability

    # Without seeding: y=0 -> m_eff=0 -> field identically 0.0 (trapped!)
    field_trapped = weighted_normalized_adjoint_field(
        y_zero, b_val, weight, regions,
        scale_routing_weights=sc_weights,
        adjoint_mode="radon_nikodym",
        scale_seeded_carrier=False,
    )
    assert field_trapped.abs().max().item() == 0.0, "Standard Radon-Nikodym should be trapped at y=0"

    # With seeding: y=0 -> m_eff = eps_seed * pi_fine > 0 -> field < 0 (negative field increases y!)
    field_seeded = weighted_normalized_adjoint_field(
        y_zero, b_val, weight, regions,
        scale_routing_weights=sc_weights,
        adjoint_mode="radon_nikodym",
        scale_seeded_carrier=True,
        scale_seed_eps=0.05,
    )
    assert field_seeded.min().item() < -1e-4, "Scale-seeded carrier must produce non-zero recovery field"


def test_gen12_configs_pass_validator():
    """Verify that all 5 Gen 12 YAML configs pass validate_v3_config without error."""
    from rmr_v3.config.validator import validate_v3_config
    configs = [
        "configs/rmr_research/sub60_e23_asym_morozov.yaml",
        "configs/rmr_research/sub60_e24_asym_trust.yaml",
        "configs/rmr_research/sub60_e25_morozov_asym_trust_synthesis.yaml",
        "configs/rmr_research/sub60_e26_scale_seeded_carrier.yaml",
        "configs/rmr_research/sub60_e27_morozov_trust_pair.yaml",
    ]
    for p in configs:
        with open(p) as f:
            cfg = yaml.safe_load(f)
        validate_v3_config(cfg)

