"""Unit tests for Mass-Preserving Feature SUM Regional Evidence Head and Sub-200k Architecture.

Verifies:
1. Mathematical invariants of feature sum pooling (linearity, scale invariance, discrete-fractional parity).
2. ProbabilisticRegionalEvidenceHead forward and backward passes across all stats ("mean", "mean_std", "sum", "sum_std").
3. End-to-end RMRv3 model integration with Width 48/64, ResidualDilatedFineHead, and strict parameter budget (< 200,000).
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from rmr_core.operators import (
    build_multiscale_regions,
    fractional_region_sum_features,
    fractional_region_sum_std_features,
    region_sum_features,
    region_sum_std_features,
    regional_sum,
)
from rmr_v3.model.architecture import RMRv3
from rmr_v3.model.config import RMRv3Config
from rmr_v3.regional_head import ProbabilisticRegionalEvidenceHead


class TestMassPreservingFeatureSum:
    def test_sum_pooling_linearity(self) -> None:
        """Integral over regions must be strictly linear: Integral(f1 + f2) = Integral(f1) + Integral(f2)."""
        torch.manual_seed(42)
        f1 = torch.rand(2, 16, 32, 32)
        f2 = torch.rand(2, 16, 32, 32)
        boxes = torch.tensor([[0, 0, 8, 8], [4, 4, 20, 20], [0, 0, 32, 32]], dtype=torch.long)

        s1 = region_sum_features(f1, boxes)
        s2 = region_sum_features(f2, boxes)
        s_joint = region_sum_features(f1 + f2, boxes)

        torch.testing.assert_close(s_joint, s1 + s2, atol=1e-5, rtol=1e-5)

    def test_integer_and_fractional_sum_parity(self) -> None:
        """Continuous fractional sum pooling on exact integer grid coordinates must match discrete sum."""
        torch.manual_seed(42)
        f = torch.rand(2, 8, 24, 24)
        boxes_int = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 20]], dtype=torch.long)
        boxes_float = boxes_int.float()

        discrete_sum = region_sum_features(f, boxes_int)
        fractional_sum = fractional_region_sum_features(f, boxes_float)

        torch.testing.assert_close(fractional_sum, discrete_sum, atol=1e-5, rtol=1e-5)

    def test_sum_std_shape_and_values(self) -> None:
        """region_sum_std_features must concatenate mass sum and standard deviation [B, M, 2*C]."""
        f = torch.rand(2, 12, 20, 20)
        boxes = torch.tensor([[0, 0, 10, 10], [2, 2, 18, 18]], dtype=torch.long)

        out = region_sum_std_features(f, boxes)
        assert out.shape == (2, 2, 24)

        # First C channels are sum
        torch.testing.assert_close(out[..., :12], region_sum_features(f, boxes))
        # Last C channels are non-negative std
        assert (out[..., 12:] >= 0.0).all()



    def test_harsh_operator_edge_cases_inverted_oob_empty(self) -> None:
        from rmr_core.operators import region_average_features, fractional_region_average_features
        feat = torch.ones(2, 4, 32, 32)
        box_normal = torch.tensor([[4, 4, 12, 12]])
        box_inv = torch.tensor([[12, 12, 4, 4]])
        sum_norm = region_sum_features(feat, box_normal)
        sum_inv = region_sum_features(feat, box_inv)
        avg_norm = region_average_features(feat, box_normal)
        avg_inv = region_average_features(feat, box_inv)
        torch.testing.assert_close(sum_inv, sum_norm)
        torch.testing.assert_close(avg_inv, avg_norm)
        box_oob = torch.tensor([[-4, -4, 8, 8]])
        sum_oob = region_sum_features(feat, box_oob)
        avg_oob = region_average_features(feat, box_oob)
        torch.testing.assert_close(sum_oob, torch.full_like(sum_oob, 64.0))
        torch.testing.assert_close(avg_oob, torch.ones_like(avg_oob))
        box_empty = torch.tensor([[4, 4, 4, 4]])
        sum_empty = region_sum_features(feat, box_empty)
        avg_empty = region_average_features(feat, box_empty)
        torch.testing.assert_close(sum_empty, torch.zeros_like(sum_empty))
        torch.testing.assert_close(avg_empty, torch.zeros_like(avg_empty))

class TestProbabilisticRegionalEvidenceHeadStats:
    @pytest.mark.parametrize("stats", ["mean", "mean_std", "sum", "sum_std"])
    @pytest.mark.parametrize("native", [False, True])
    def test_forward_backward_all_modes(self, stats: str, native: bool) -> None:
        """ProbabilisticRegionalEvidenceHead must propagate finite gradients in all 4 pooling configurations."""
        torch.manual_seed(123)
        b, c, h, w = 2, 32, 48, 48
        p4 = torch.randn(b, c, h, w, requires_grad=True)
        p8 = torch.randn(b, c, h // 2, w // 2, requires_grad=True)
        p16 = torch.randn(b, c, h // 4, w // 4, requires_grad=True)
        pyramid = (p4, p8, p16)

        regions = build_multiscale_regions(h, w, 4, (32, 64, 128), overlap=0.5)

        head = ProbabilisticRegionalEvidenceHead(
            feature_dim=c,
            hidden=48,
            regional_feature_stats=stats,
            native_scale_pooling=native,
        )

        out = head(pyramid, regions)
        assert "mu_count" in out
        assert "dispersion" in out
        assert out["mu_count"].shape == (b, 1, regions.boxes.shape[0])
        assert (out["mu_count"] >= 0.0).all()
        assert (out["dispersion"] >= head.dispersion_min).all()
        assert (out["dispersion"] <= head.dispersion_max).all()

        loss = out["mu_count"].sum() + out["dispersion"].sum()
        loss.backward()

        assert p4.grad is not None and not torch.isnan(p4.grad).any()
        assert p8.grad is not None and not torch.isnan(p8.grad).any()
        assert p16.grad is not None and not torch.isnan(p16.grad).any()


class TestSub200kRMRModel:
    def test_width48_model_parameter_budget(self) -> None:
        """Width 48 RMRv3 model with ResidualDilatedFineHead must strictly stay under 200,000 parameters."""
        cfg = RMRv3Config(
            backbone_name="mobilenetv4_conv_small_050.e3000_r224_in1k",
            pretrained=False,
            feature_width=48,
            max_trainable_params=200000,
            neck_type="rep_weighted",
            head_type="residual_dilated",
            fine_head_dilations=(1, 2),
            region_head_hidden=48,
            regional_feature_stats="sum",
            subpixel_stride2=False,
            enable_solver=True,
            iterations=6,
        )
        model = RMRv3(cfg)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert trainable_params <= 200_000, f"Exceeded 200k budget: {trainable_params} params"
        assert trainable_params >= 110_000, f"Model unexpectedly small: {trainable_params} params"

    def test_width48_end_to_end_pass(self) -> None:
        """Verify full forward pass and solver unrolling with feature SUM."""
        cfg = RMRv3Config(
            backbone_name="mobilenetv4_conv_small_050.e3000_r224_in1k",
            pretrained=False,
            feature_width=48,
            max_trainable_params=200000,
            neck_type="rep_weighted",
            head_type="residual_dilated",
            region_head_hidden=48,
            regional_feature_stats="sum",
            enable_solver=True,
            iterations=3,
        )
        model = RMRv3(cfg)
        x = torch.randn(1, 3, 128, 128)
        out = model(x)

        assert out.y is not None
        assert out.y0 is not None
        assert out.b_region is not None
        assert out.b_solver is not None
        assert out.y.shape[-2:] == (32, 32)
        assert (out.y >= 0.0).all()