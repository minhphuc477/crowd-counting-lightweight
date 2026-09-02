"""Unit tests for Objective Mechanism Audit v2."""
import math
import pytest
import torch
import torch.nn as nn

from hpc.losses.ntpc import NTPCConfig, NTPCLoss
from hpc.diagnostics.objective_mechanism_audit import (
    cancellation_ratio,
    compute_audit_for_mode_v2,
    compute_component_gradients,
    compute_mass_gradient_metrics,
    compute_pairwise_cosine,
    compute_parameter_space_metrics,
    stratify_by_local_crop_count,
    summarize_audit_group_v2,
    sweep_kappa_on_crop_v2,
)


class DummyCounter(nn.Module):
    """Simple dummy counter for testing parameter-space gradients."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=4, stride=4)

    def forward_mass(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(self.conv(x))


def _make_dummy_data(crop_size: int = 64, total_count: int = 25):
    h4, w4 = crop_size // 4, crop_size // 4
    mass = torch.ones(1, 1, h4, w4, dtype=torch.float32) * (float(total_count) / (h4 * w4))
    targets = {
        "N": torch.tensor([float(total_count)]),
        4: torch.zeros(1, h4, w4),
        8: torch.zeros(1, h4 // 2, w4 // 2),
        16: torch.zeros(1, h4 // 4, w4 // 4),
        32: torch.zeros(1, h4 // 8, w4 // 8),
        64: torch.zeros(1, h4 // 16, w4 // 16),
    }
    targets[4][0, 0, 0] = total_count
    targets[8][0, 0, 0] = total_count
    targets[16][0, 0, 0] = total_count
    targets[32][0, 0, 0] = total_count
    targets[64][0, 0, 0] = total_count
    return mass, targets


class TestObjectiveMechanismAuditV2:
    def test_cancellation_ratio(self):
        a = torch.randn(1, 1, 16, 16)
        # Perfectly aligned => C = 0
        assert abs(cancellation_ratio(a, a) - 0.0) < 1e-6
        # Perfectly opposite => C = 1.0
        assert abs(cancellation_ratio(a, -a) - 1.0) < 1e-6
        # Orthogonal => C = 1 - sqrt(2)/2 approx 0.29289
        b = torch.zeros_like(a)
        b[:, :, :8, :] = a[:, :, 8:, :]
        b[:, :, 8:, :] = -a[:, :, :8, :]
        expected_ortho = 1.0 - math.sqrt(2.0) / 2.0
        assert abs(cancellation_ratio(a, b) - expected_ortho) < 1e-4

    def test_euler_scale_projection(self):
        m = torch.ones(1, 1, 4, 4) * 2.0
        # g aligns with m => euler_cos = 1.0
        res = compute_mass_gradient_metrics(m, m)
        assert abs(res["euler_cos"] - 1.0) < 1e-6
        assert res["euler_dot"] > 0

        # g orthogonal to m
        g_ortho = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]).reshape(1, 1, 2, 2)
        m_flat = torch.ones(1, 1, 2, 2) * 3.0
        res_ortho = compute_mass_gradient_metrics(g_ortho, m_flat)
        assert abs(res_ortho["euler_dot"]) < 1e-6
        assert abs(res_ortho["euler_cos"]) < 1e-6

    def test_total_gradient_no_double_counting(self):
        mass, targets = _make_dummy_data(crop_size=64, total_count=20)
        cfg = NTPCConfig(mode="r2_flat_dm", root_loss="nb", kappa_flat16=20.0)
        crit = NTPCLoss(cfg)

        grads, g_total = compute_component_gradients(mass, targets, crit)
        # True total should match root_magnitude + flat_16 exactly
        # NTPCLoss logs both root_magnitude and root_nb. If we summed components.values(), it would have 2*root!
        correct_sum = grads["root_magnitude"] + grads["flat_16"]
        assert torch.allclose(correct_sum, g_total, atol=1e-5), "g_total must match true total loss gradient without root duplication"

    def test_parameter_space_count_direction(self):
        model = DummyCounter()
        crop_img = torch.randn(1, 3, 64, 64)
        _, targets = _make_dummy_data(crop_size=64, total_count=5)
        crit = NTPCLoss(NTPCConfig(mode="r2_flat_dm", root_loss="nb", kappa_flat16=20.0))

        param_metrics = compute_parameter_space_metrics(
            model, crop_img, targets, crit, ("root_magnitude", "flat_16")
        )
        assert "root_magnitude" in param_metrics
        assert "flat_16" in param_metrics
        assert "total" in param_metrics
        for name, m in param_metrics.items():
            assert "norm_theta" in m
            assert "count_dot_theta" in m
            assert "count_cos_theta" in m
            assert -1.0 <= m["count_cos_theta"] <= 1.0

    def test_sweep_kappa_v2(self):
        mass, targets = _make_dummy_data(crop_size=64, total_count=10)
        kappas = [5.0, 20.0, 50.0]
        res = sweep_kappa_on_crop_v2(mass, targets, kappas=kappas)
        assert "r2_flat" in res and "r4_tree" in res
        for k in ["k_5", "k_20", "k_50"]:
            assert "cancellation_64_32_vs_32_16" in res["r4_tree"][k]
            assert 0.0 <= res["r4_tree"][k]["cancellation_64_32_vs_32_16"] <= 1.0

    def test_stratify_by_local_crop_count(self):
        records = [
            {"gt_count": 20.0, "signed_error": 1.0, "pred_count": 21.0},
            {"gt_count": 80.0, "signed_error": -5.0, "pred_count": 75.0},
            {"gt_count": 250.0, "signed_error": -40.0, "pred_count": 210.0},
        ]
        bins = stratify_by_local_crop_count(records, threshold_low=50.0, threshold_high=150.0)
        assert len(bins["low (<50)"]) == 1
        assert len(bins["medium (50-150)"]) == 1
        assert len(bins["high (>=150)"]) == 1
