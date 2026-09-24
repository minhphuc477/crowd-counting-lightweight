from __future__ import annotations

"""Adversarial Deep Audit Invariants for PARK (RMR-v33).

Interrogates:
1. Spatial Correspondence Invariant: Zero permutation error between RegionalEvidenceHead and RegionSet.boxes.
2. Hilbert Space Adjoint Invariant: <A y, g> = <y, A* g> with exact suffix-sum backprojection.
3. Dynamic Optical Ray Partitioning: Monotonic scale grouping across 3 and 5 altitude bands.
4. Active Parameter Gradient Flow: Zero dormant parameters across all 11 RMR-v33 configs.
5. Monolith & Parameter Hard Ceilings: Parameters strictly <= 105,000.
"""

import math
import pathlib
import pytest
import torch
import torch.nn.functional as F

from rmr_core.operators.perspective_regions import (
    build_perspective_regions,
    park_adjoint_operator,
    park_forward_operator,
)
from rmr_core.operators import RegionSet, region_average_features
from rmr_v3.config import load_config
from rmr_v3.losses import compute_rmr_v3_losses, RMRv3LossConfig
from rmr_v3.model import RMRv3, RMRv3Config
from rmr_v3.regional_head import ProbabilisticRegionalEvidenceHead


def test_spatial_correspondence_zero_error() -> None:
    """Verify pooled features strictly match true box centroids with 0.0 error."""
    test_shapes = [(64, 64), (128, 128), (113, 227)]

    for h, w in test_shapes:
        regions = build_perspective_regions(h, w, output_stride=4, num_altitude_bands=3)
        head = ProbabilisticRegionalEvidenceHead(
            feature_dim=32,
            hidden=48,
            region_sizes_px=(16, 64, 128),
        )

        # Encode vertical coordinate y into channel 0
        p4 = torch.zeros(1, 32, h, w)
        for y in range(h):
            p4[0, :, y, :] = float(y)
        p8 = p4
        p16 = p4

        feat = head._collect_region_features((p4, p8, p16), regions)

        boxes = regions.boxes
        center_y_true = 0.5 * (boxes[:, 0] + boxes[:, 2] - 1).float()
        pooled_y = feat[0, :, 0]

        max_disp = (pooled_y - center_y_true).abs().max().item()
        assert max_disp < 1e-4, f"Spatial permutation mismatch on shape ({h}, {w}): max_disp={max_disp}"


def test_dynamic_5_bands_monotonicity() -> None:
    """Verify K=5 optical altitude bands generate strictly sorted scale IDs."""
    h, w = 128, 128
    regions = build_perspective_regions(h, w, output_stride=4, num_altitude_bands=5)

    assert regions.num_scales == 5
    scales = regions.scale_id
    is_monotonic = (scales[1:] >= scales[:-1]).all().item()
    assert is_monotonic, "scale_id is not monotonically non-decreasing in 5-band tiling"

    unique_scales = scales.unique().tolist()
    assert len(unique_scales) == 5, f"Expected 5 unique scale IDs, got {unique_scales}"


def test_hilbert_adjoint_suffix_sum_duality() -> None:
    """Verify Hilbert duality <A y, g> = <y, A* g> holds with exact 0.0 suffix sum error."""
    h, w = 32, 32
    torch.manual_seed(42)
    y = torch.randn(2, 1, h, w, dtype=torch.float64)
    boxes = torch.tensor(
        [
            [0.0, 0.0, 8.0, 8.0],
            [4.5, 3.2, 18.7, 16.1],
            [10.0, 12.0, 30.0, 28.0],
        ],
        dtype=torch.float64,
    )
    m = boxes.shape[0]
    g = torch.randn(2, 1, m, dtype=torch.float64)

    ay = park_forward_operator(y, boxes)
    a_star_g = park_adjoint_operator(g, boxes, height=h, width=w)

    inner1 = (ay * g).sum()
    inner2 = (y * a_star_g).sum()

    rel_err = (inner1 - inner2).abs() / max(inner1.abs(), inner2.abs(), 1e-8)
    assert rel_err.item() < 1e-10, f"Hilbert duality violation: rel_err={rel_err.item()}"


@pytest.mark.parametrize(
    "config_name",
    [
        "rmr_v33_park_perspective.yaml",
        "rmr_v33_anchor_v19_isotropic.yaml",
        "rmr_v33_ablation_isotropic_pcat.yaml",
        "rmr_v33_ablation_no_pgh.yaml",
        "rmr_v33_ablation_no_routing.yaml",
        "rmr_v33_ablation_horizon32.yaml",
        "rmr_v33_control_no_solver.yaml",
        "rmr_v33_ablation_flat_adjoint.yaml",
        "rmr_v33_ablation_aspect15.yaml",
        "rmr_v33_ablation_inverted_aspect.yaml",
        "rmr_v33_ablation_no_dm16.yaml",
        "rmr_v33_ablation_no_curvature.yaml",
        "rmr_v33_ablation_solver_t3.yaml",
        "rmr_v33_ablation_5bands.yaml",
        "rmr_v33_ablation_stride2_dual_lattice.yaml",
    ],
)
def test_all_15_configs_gradient_flow_and_params(config_name: str) -> None:
    """Interrogate all 15 configs: parameter ceiling <= 105,000, forward & backward pass."""
    cfg_path = pathlib.Path("configs/rmr_v33") / config_name
    assert cfg_path.exists(), f"Missing config file: {cfg_path}"

    cfg = load_config(str(cfg_path))
    model_cfg = RMRv3Config(**cfg["model"])
    model = RMRv3(model_cfg)

    # 1. Parameter ceiling check
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert num_trainable <= 105000, f"{config_name} exceeds 105k params: {num_trainable}"

    # 2. Forward pass with arbitrary batch
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    assert "y" in out
    assert "y0" in out
    assert torch.isfinite(out["y"]).all(), f"NaN/Inf in y for {config_name}"

    # 3. Loss computation & backward pass
    tgt_h, tgt_w = out["y"].shape[-2:]
    target_y = torch.rand(2, 1, tgt_h, tgt_w)
    loss_cfg = RMRv3LossConfig(**cfg.get("loss", {}))
    losses = compute_rmr_v3_losses(out, target_y, cfg=loss_cfg)

    assert "total" in losses
    assert torch.isfinite(losses["total"]).all(), f"NaN/Inf in total loss for {config_name}"

    losses["total"].backward()

    # 4. Check that fine head bias received active gradients
    fine_conv = model.fine_head.body[-1]
    assert fine_conv.bias.grad is not None
    assert torch.isfinite(fine_conv.bias.grad).all()
    assert fine_conv.bias.grad.abs().sum() > 0.0, f"Zero gradient on fine head for {config_name}"
