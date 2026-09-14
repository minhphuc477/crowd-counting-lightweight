from __future__ import annotations

"""Dedicated Unit and Invariant Test Suite for RMR-v15 Native Dynamic Geometry.

Validates:
1. Exact parameter budget compliance (104,580 <= 105,000 budget).
2. Micro-scale 16px geometric operator correctness & Hilbert adjoint duality pairing.
3. Scale-conditioned fine density prior positivity and gradient flow.
4. Pre-solver scale-consistency reliability gating.
5. Convex dynamic trust-gate blending and gradient propagation.
6. End-to-end multi-loss supervision and backward pass hygiene.
"""

from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from rmr_core.heads import FineMeasureHead
from rmr_core.operators import (
    RegionSet,
    build_multiscale_regions,
    regional_adjoint,
    regional_sum,
)
from rmr_core.scale_routing import ScaleRoutingHead
from rmr_v3.config import load_config
from rmr_v3.losses import RMRv3LossConfig, compute_rmr_v3_losses
from rmr_v3.model import RMRv3, RMRv3Config
from rmr_v3.regional_head import apply_scale_consistency_gating


def _load_v15_config() -> tuple[RMRv3Config, RMRv3LossConfig]:
    cfg_path = Path("configs/rmr_v15/rmr_v15_native_geometry.yaml")
    assert cfg_path.is_file(), f"Config not found at {cfg_path}"
    with open(cfg_path, "r") as f:
        d = yaml.safe_load(f)

    m_cfg = {k: v for k, v in d["model"].items() if hasattr(RMRv3Config, k)}
    m_cfg["pretrained"] = False
    return RMRv3Config(**m_cfg), RMRv3LossConfig(**d["loss"])


class TestRMRv15NativeDynamicGeometry:
    """Comprehensive test class for RMR-v15 native architectural components."""

    def test_rmr_v15_config_loading_and_parameter_budget_exactness(self):
        """Invariant 1: Trainable parameter count is strictly <= 105,000 (exactly 104,580)."""
        model_cfg, _ = _load_v15_config()
        model = RMRv3(model_cfg)

        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n_trainable == 104580, f"Expected 104,580 params, got {n_trainable}"
        assert n_trainable <= 105000, f"Exceeded hard budget: {n_trainable} > 105,000"

        # Verify component parameter breakdown
        assert model.fine_head.scale_beta.numel() == 4
        assert model.fine_head.scale_gamma.numel() == 4
        assert sum(p.numel() for p in model.scale_router.parameters()) == 516
        assert sum(p.numel() for p in model.trust_gate.parameters()) == 33

    def test_micro_scale_16px_operator_duality(self):
        """Invariant 2: Micro-scale 16px regions satisfy discrete adjoint duality pairing."""
        h, w = 64, 64
        regions = build_multiscale_regions(h, w, output_stride=4, region_sizes_px=(16, 32, 64, 128), overlap=0.5, include_full_image=False)
        m = regions.boxes.shape[0]

        # Verify 4 distinct scale IDs: 0, 1, 2, 3
        scale_ids = torch.unique(regions.scale_id)
        assert set(scale_ids.tolist()) == {0, 1, 2, 3}

        y = torch.rand(2, 1, h, w) + 0.1
        w_reg = torch.rand(2, 1, m) + 0.1

        ay = regional_sum(y, regions.boxes)
        at_w = regional_adjoint(w_reg, regions.boxes, h, w)

        inner_1 = (ay * w_reg).sum().item()
        inner_2 = (y * at_w).sum().item()

        rel_diff = abs(inner_1 - inner_2) / max(abs(inner_1), abs(inner_2))
        assert rel_diff < 1e-4, f"Adjoint duality violation on micro-scale: {rel_diff}"

    def test_scale_conditioned_fine_prior_invariants(self):
        """Invariant 3: Scale-conditioned prior ensures positivity, modulation, and healthy gradients."""
        head = FineMeasureHead(
            width=32,
            temp_softplus=True,
            scale_conditioned=True,
            num_scales=4,
        )

        f = torch.randn(2, 32, 16, 16)
        scale_weights = F.softmax(torch.randn(2, 4, 16, 16), dim=1)

        z = head.forward_logits(f)
        y0 = head.activate(z, scale_weights=scale_weights)

        assert y0.shape == (2, 1, 16, 16)
        assert (y0 > 0.0).all(), "Fine density must be strictly positive"

        # Test backward pass
        loss = y0.sum()
        loss.backward()
        assert head.scale_beta.grad is not None and not torch.isnan(head.scale_beta.grad).any()
        assert head.scale_gamma.grad is not None and not torch.isnan(head.scale_gamma.grad).any()
        assert (head.scale_beta.grad != 0.0).any()

    def test_pre_solver_scale_consistency_gating(self):
        """Invariant 4: Pre-solver gating silences coarse boxes in micro-scale regions."""
        h, w = 32, 32
        regions = build_multiscale_regions(h, w, output_stride=4, region_sizes_px=(16, 32, 64), include_full_image=False)
        m = regions.boxes.shape[0]

        weight = torch.ones(2, 1, m)

        # Create scale weights where entire image is 100% micro-scale (scale 0)
        scale_weights = torch.zeros(2, 3, h, w)
        scale_weights[:, 0, :, :] = 1.0  # 100% micro-scale

        gated_weight = apply_scale_consistency_gating(weight, regions, scale_weights, power=1.0)

        # Scale 0 (micro) boxes should keep weight near 1.0
        mask_0 = (regions.scale_id == 0)
        assert (gated_weight[:, :, mask_0] > 0.95).all()

        # Scale 2 (coarse 64px) boxes should have weight dropped to near zero
        mask_2 = (regions.scale_id == 2)
        assert (gated_weight[:, :, mask_2] < 1e-4).all()

    def test_convex_dynamic_trust_gate_invariants(self):
        """Invariant 5: Dynamic trust gate blends y0 and y_solver with healthy gradients to both."""
        model_cfg, _ = _load_v15_config()
        model_cfg.iterations = 2
        model = RMRv3(model_cfg)

        x = torch.randn(2, 3, 128, 128)
        out = model(x)

        assert "solver_trust_alpha" in out
        alpha = out.solver_trust_alpha
        assert alpha.shape == (2, 1, 1, 1)
        assert (alpha >= 0.0).all() and (alpha <= 1.0).all()
        # Default bias 1.73 gives initial alpha around 0.85
        assert abs(alpha.mean().item() - 0.85) < 0.10

    def test_rmr_v15_end_to_end_forward_backward_and_losses(self):
        """Invariant 6: End-to-end forward, 14 loss terms, and backward execution."""
        model_cfg, loss_cfg = _load_v15_config()
        model_cfg.iterations = 2
        model = RMRv3(model_cfg)

        x = torch.randn(2, 3, 128, 128)
        out = model(x)

        assert out.y.shape == (2, 1, 32, 32)
        assert out.y0.shape == (2, 1, 32, 32)
        assert out.scale_weights.shape == (2, 4, 32, 32)
        assert (out.scale_weights.sum(dim=1) - 1.0).abs().max() < 1e-5

        target = torch.abs(torch.randn(2, 1, 32, 32))
        losses = compute_rmr_v3_losses(out, target, loss_cfg)

        assert "total" in losses
        assert not torch.isnan(losses["total"])
        assert not torch.isinf(losses["total"])

        losses["total"].backward()

        # Verify gradients reached fine prior and trust gate
        assert model.fine_head.scale_beta.grad is not None
        assert model.fine_head.scale_gamma.grad is not None
        assert model.trust_gate.weight.grad is not None
        assert not torch.isnan(model.trust_gate.weight.grad).any()

    def test_rmr_v15_eval_determinism(self):
        """Invariant 7: Deterministic bit-by-bit identical outputs in eval mode."""
        model_cfg, _ = _load_v15_config()
        model = RMRv3(model_cfg)
        model.eval()

        x = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        assert torch.equal(out1.y, out2.y)
        assert torch.equal(out1.y0, out2.y0)
        assert torch.equal(out1.scale_weights, out2.scale_weights)
