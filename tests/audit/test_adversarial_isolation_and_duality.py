from __future__ import annotations

import math
import pytest
import torch

from rmr_core.operators import (
    RegionSet,
    build_multiscale_regions,
    regional_sum,
    regional_adjoint,
)
from rmr_v3.config.provenance import load_config
from rmr_v3.losses.config import RMRv3LossConfig
from rmr_v3.losses.orchestration import compute_rmr_v3_losses
from rmr_v3.model.architecture import RMRv3
from rmr_v3.model.config import RMRv3Config


def test_adversarial_sample_isolation() -> None:
    """Verify that every loss term satisfies strict sample isolation:
    perturbing Sample 1's target produces identically 0 gradient change on Sample 0.
    """
    raw_cfg = load_config("configs/rmr_research/h11_harmonious_peak.yaml")
    loss_cfg = RMRv3LossConfig(**raw_cfg.get("loss", {}))

    y = torch.randn(2, 1, 128, 128, requires_grad=True)
    y0 = torch.randn(2, 1, 128, 128, requires_grad=True)

    regions = build_multiscale_regions(128, 128, 4, [32, 64, 128], 0.5, False, y.device)
    num_boxes = regions.boxes.shape[0]

    outputs = {
        "y": y,
        "y0": y0,
        "regions": regions,
        "b_region": torch.ones(2, 1, num_boxes, requires_grad=True),
        "region_dispersion": torch.ones(2, 1, num_boxes) * 50.0,
        "pi_scale": torch.softmax(torch.randn(2, 3, 128, 128), dim=1).requires_grad_(True),
        "hurdle_logit": torch.randn(2, 1, num_boxes, requires_grad=True),
    }

    target = torch.zeros(2, 1, 128, 128)
    target[0, 0, 30:35, 30:35] = 2.0
    target[1, 0, 50:70, 50:70] = 5.0

    losses = compute_rmr_v3_losses(outputs, target, loss_cfg)

    # Perturbed target for sample 1
    outputs_perturbed = {
        "y": y,
        "y0": y0,
        "regions": regions,
        "b_region": outputs["b_region"],
        "region_dispersion": outputs["region_dispersion"],
        "pi_scale": outputs["pi_scale"],
        "hurdle_logit": outputs["hurdle_logit"],
    }
    target_perturbed = target.clone()
    target_perturbed[1, 0, 10:110, 10:110] = 50.0
    losses_perturbed = compute_rmr_v3_losses(outputs_perturbed, target_perturbed, loss_cfg)

    for k, v in losses.items():
        if isinstance(v, torch.Tensor) and v.requires_grad:
            g_y = torch.autograd.grad(v, y, retain_graph=True, allow_unused=True)[0]
            if g_y is not None:
                g_y_p = torch.autograd.grad(
                    losses_perturbed[k], y, retain_graph=True, allow_unused=True
                )[0]
                diff_sample0 = (g_y[0] - g_y_p[0]).abs().max().item()
                assert (
                    diff_sample0 == 0.0
                ), f"Cross-sample leakage detected in loss '{k}': diff={diff_sample0}"


def test_mathematical_adjointness_duality() -> None:
    """Verify that <Ay, b> == <y, A^T b> within float64 precision (< 1e-10 relative error)."""
    shapes = [(64, 64), (128, 128), (103, 226), (29, 57)]

    for h, w in shapes:
        regions = build_multiscale_regions(
            h, w, 4, [32, 64], 0.5, False, torch.device("cpu")
        )
        num_m = regions.boxes.shape[0]

        y = torch.randn(1, 1, h, w, dtype=torch.float64)
        b = torch.randn(1, 1, num_m, dtype=torch.float64)

        ay = regional_sum(y, regions.boxes, out_dtype=torch.float64)
        atb = regional_adjoint(b, regions.boxes, h, w, out_dtype=torch.float64)

        ip1 = (ay * b).sum().item()
        ip2 = (y * atb).sum().item()

        rel_err = abs(ip1 - ip2) / (abs(ip1) + abs(ip2) + 1e-12)
        assert rel_err < 1e-10, f"Adjoint duality broken on shape ({h}, {w}): rel_err={rel_err}"


def test_arbitrary_odd_resolutions() -> None:
    """Verify that RMRv3 forward and loss computation execute without assertion errors on odd shapes."""
    raw_cfg = load_config("configs/rmr_research/h11_harmonious_peak.yaml")
    model_cfg = RMRv3Config(**raw_cfg.get("model", {}))
    loss_cfg = RMRv3LossConfig(**raw_cfg.get("loss", {}))

    model = RMRv3(model_cfg)
    model.eval()

    test_shapes = [
        (1, 3, 512, 512),
        (1, 3, 409, 902),
        (1, 3, 113, 227),
        (2, 3, 257, 383),
    ]

    for s in test_shapes:
        x = torch.randn(*s)
        with torch.no_grad():
            out = model(x)
        exp_h = (s[2] + 3) // 4
        exp_w = (s[3] + 3) // 4
        assert out.y.shape[-2] == exp_h
        assert out.y.shape[-1] == exp_w

        target = torch.zeros(s[0], 1, exp_h, exp_w)
        target[..., :min(5, exp_h), :min(5, exp_w)] = 1.0
        losses = compute_rmr_v3_losses(out, target, loss_cfg)
        assert not torch.isnan(losses["total"])
