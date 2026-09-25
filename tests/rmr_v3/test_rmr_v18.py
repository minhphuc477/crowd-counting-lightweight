from __future__ import annotations

import math
import pytest
import torch
import yaml

from rmr_core.operators import build_multiscale_regions, _canonicalize_region_size
from rmr_core.scale_routing import ScaleRoutingHead
from rmr_v3.model import RMRv3, RMRv3Config
from rmr_v3.regional_head import apply_scale_consistency_gating
from rmr_v3.losses import RMRv3LossConfig, compute_rmr_v3_losses


def test_rmr_v18_parameter_budget():
    with open('configs/rmr_v18/rmr_v18_canonical.yaml') as f:
        cfg = yaml.safe_load(f)

    model_cfg = RMRv3Config(**cfg['model'])
    model = RMRv3(model_cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert n_params == 104507, f'Expected 104507 params, got {n_params}'
    assert n_params <= 105000, f'Exceeded budget: {n_params} > 105000'
    assert 105000 - n_params == 493, 'Headroom should be exactly 493 params'


def test_rmr_v18_perspective_scale_bias_step0_identity():
    head_base = ScaleRoutingHead(in_channels=32, num_scales=5, perspective_bias=False)
    head_persp = ScaleRoutingHead(in_channels=32, num_scales=5, perspective_bias=True)

    # Copy identical weights
    head_persp.dw.load_state_dict(head_base.dw.state_dict())
    head_persp.norm.load_state_dict(head_base.norm.state_dict())
    head_persp.pw.load_state_dict(head_base.pw.state_dict())

    x = torch.randn(2, 32, 64, 64)
    pi_base = head_base(x)
    pi_persp = head_persp(x)

    # At step 0, outputs must be identical
    assert torch.allclose(pi_base, pi_persp, atol=1e-7), 'Step 0 identity violated in ScaleRoutingHead'


def test_rmr_v18_perspective_scale_bias_gradient_flow():
    head = ScaleRoutingHead(in_channels=32, num_scales=5, perspective_bias=True)
    x = torch.randn(2, 32, 32, 32, requires_grad=True)
    pi = head(x)

    loss = pi[:, 3].sum()
    loss.backward()

    assert head.pw.weight.grad is not None
    assert torch.isfinite(head.pw.weight.grad).all()
    assert (head.pw.weight.grad != 0).any(), 'Gradient to pw.weight is identically zero'


def test_rmr_v18_perspective_horizon_gating():
    h, w = 64, 64
    stride = 4
    region_sizes = [16, 32, 64, (64, 32), 128]
    regions = build_multiscale_regions(h, w, stride, region_sizes_px=region_sizes)

    # Fake uniform scale weights (K=5)
    scale_weights = torch.ones(1, 5, h, w) / 5.0
    initial_weights = torch.ones(1, 1, len(regions.boxes))

    canonical_sizes = tuple(_canonicalize_region_size(s) for s in region_sizes)
    gated_weights = apply_scale_consistency_gating(
        initial_weights,
        regions,
        scale_weights,
        power=1.0,
        perspective_horizon_gate=True,
        horizon_cutoff=0.35,
        region_sizes_px=canonical_sizes,
        grid_h=h,
    )

    # Scale 3 is the anisotropic box (64, 32)
    mask_scale3 = (regions.scale_id == 3)
    assert mask_scale3.any()

    boxes_s3 = regions.boxes[mask_scale3]
    w_s3 = gated_weights[0, 0, mask_scale3]

    y_centers = 0.5 * (boxes_s3[:, 0].float() + boxes_s3[:, 2].float()) / float(h)

    # Boxes near horizon (y_center < 0.25) should have significantly lower weight than near bottom (y_center > 0.70)
    horizon_boxes = y_centers < 0.25
    ground_boxes = y_centers > 0.70

    if horizon_boxes.any() and ground_boxes.any():
        mean_horizon_w = w_s3[horizon_boxes].mean().item()
        mean_ground_w = w_s3[ground_boxes].mean().item()
        assert mean_horizon_w < 0.5 * mean_ground_w, (
            f'Horizon weight ({mean_horizon_w:.4f}) not suppressed relative to ground ({mean_ground_w:.4f})'
        )

    # Isotropic scale (Scale 0: 16x16) should NOT have horizon attenuation
    mask_scale0 = (regions.scale_id == 0)
    w_s0 = gated_weights[0, 0, mask_scale0]
    # In uniform scale routing, isotropic weights remain uniform across y
    assert (w_s0.max() - w_s0.min()).item() < 1e-4, 'Isotropic boxes should not experience horizon attenuation'


def test_rmr_v18_anisotropic_region_dictionary():
    h, w = 64, 64
    stride = 4
    region_sizes = [16, 32, 64, (64, 32), 128]
    regions = build_multiscale_regions(h, w, stride, region_sizes_px=region_sizes)

    unique_scales = torch.unique(regions.scale_id)
    # Scales 0, 1, 2, 3, 4
    for sid in range(5):
        assert sid in unique_scales, f'Scale ID {sid} missing in generated RegionSet'

    # Check that scale 3 boxes have height=16 (64/4) and width=8 (32/4)
    mask_s3 = (regions.scale_id == 3)
    boxes_s3 = regions.boxes[mask_s3]
    hy = boxes_s3[:, 2] - boxes_s3[:, 0]
    wx = boxes_s3[:, 3] - boxes_s3[:, 1]
    assert (hy == 16).all(), 'Scale 3 boxes should have height 16'
    assert (wx == 8).all(), 'Scale 3 boxes should have width 8'


def test_rmr_v18_end_to_end_forward_backward():
    with open('configs/rmr_v18/rmr_v18_canonical.yaml') as f:
        cfg = yaml.safe_load(f)

    model = RMRv3(RMRv3Config(**cfg['model']))
    # Non-square input to verify geometric generalization
    x = torch.randn(2, 3, 192, 256)
    target = torch.zeros(2, 1, 48, 64)
    target[0, 0, 12, 16] = 1.0
    target[1, 0, 30, 40] = 2.0

    out = model(x)
    assert 'y' in out and 'y0' in out and 'scale_weights' in out
    assert out['scale_weights'].shape[1] == 5, 'Expected 5 scale channels in scale_weights'

    losses = compute_rmr_v3_losses(out, target, RMRv3LossConfig(**cfg['loss']))
    loss = losses['total']
    assert torch.isfinite(loss), f'Loss is non-finite: {loss.item()}'

    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all(), f'Non-finite grad in {name}'


def test_rmr_v18_unrolled_solver_energy_reduction():
    with open('configs/rmr_v18/rmr_v18_canonical.yaml') as f:
        cfg = yaml.safe_load(f)

    model = RMRv3(RMRv3Config(**cfg['model']))
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)

    e_trace = out['energy_trace']
    assert len(e_trace) == 6, f'Expected 6 solver iterations, got {len(e_trace)}'
    e_first = e_trace[0]['before'].item()
    e_last = e_trace[-1]['after'].item()
    assert e_last <= e_first + 1e-4, f'Energy should decrease: initial {e_first} -> final {e_last}'


def test_rmr_v18_diagnostic_tracker_anisotropic_telemetry():
    from rmr_v3.tracking import DiagnosticTracker
    with open('configs/rmr_v18/rmr_v18_canonical.yaml') as f:
        cfg = yaml.safe_load(f)

    model = RMRv3(RMRv3Config(**cfg['model']))
    tracker = DiagnosticTracker(model)

    assert tracker.scale_map == {0: 16, 1: 32, 2: 64, 3: '64_32', 4: 128}

    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    tracker.update(out)
    summary = tracker.summarize()

    assert 'weight_mean_64_32' in summary
    assert 'scale_pi_64_32' in summary
    assert summary['scale_pi_64_32'] >= 0.0

    # Total scale_pi must sum to 1.0 across all 5 scales
    all_pi = (
        summary['scale_pi_16']
        + summary['scale_pi_32']
        + summary['scale_pi_64']
        + summary['scale_pi_64_32']
        + summary['scale_pi_128']
    )
    assert abs(all_pi - 1.0) < 1e-3


def test_rmr_v18_all_ablation_configs():
    import glob
    from rmr_v3.config import validate_v3_config

    configs = sorted(glob.glob('configs/rmr_v18/*.yaml'))
    assert len(configs) == 4, f'Expected 4 v18 configs, got {len(configs)}'

    for fpath in configs:
        with open(fpath) as f:
            c = yaml.safe_load(f)
        validate_v3_config(c)
        m = RMRv3(RMRv3Config(**c['model']))
        n = sum(p.numel() for p in m.parameters() if p.requires_grad)
        assert n <= 105000, f'{fpath} exceeds 105,000 params ({n})'


def test_rmr_v18_diagnostics_resolve_scale_map():
    from rmr_v3.diagnostics import resolve_scale_map

    sids_5 = torch.tensor([0, 1, 2, 3, 4])
    smap_5 = resolve_scale_map(sids_5)
    assert smap_5 == {0: 16, 1: 32, 2: 64, 3: '64_32', 4: 128}


def test_rmr_v18_fine_head_forward_density_curvature():
    from rmr_core.heads import FineMeasureHead

    head = FineMeasureHead(width=32, density_curvature=True)
    f = torch.randn(2, 32, 16, 16)
    y = head(f)
    # Output must be activated density (strictly >= 0)
    assert (y >= 0.0).all()
    assert y.shape == (2, 1, 16, 16)
