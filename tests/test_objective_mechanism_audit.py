"""Unit tests for Objective Mechanism Audit."""
import math
import pytest
import torch

from hpc.losses.ntpc import NTPCConfig, NTPCLoss
from hpc.diagnostics.objective_mechanism_audit import (
    compute_audit_for_mode,
    compute_component_gradients,
    compute_gradient_metrics,
    compute_pairwise_cosine,
    stratify_by_density,
    summarize_audit_group,
    sweep_kappa_on_crop,
)


def _make_dummy_data(crop_size: int = 64, total_count: int = 25):
    """Create synthetic crop data (batch=1, crop_size x crop_size)."""
    h4, w4 = crop_size // 4, crop_size // 4
    mass = torch.ones(1, 1, h4, w4, dtype=torch.float32) * (float(total_count) / (h4 * w4))
    # Count targets
    targets = {
        "N": torch.tensor([float(total_count)]),
        4: torch.zeros(1, h4, w4),
        8: torch.zeros(1, h4 // 2, w4 // 2),
        16: torch.zeros(1, h4 // 4, w4 // 4),
        32: torch.zeros(1, h4 // 8, w4 // 8),
        64: torch.zeros(1, h4 // 16, w4 // 16),
    }
    # Distribute count: place points in top-left
    targets[4][0, 0, 0] = total_count
    # Sum-pool up the pyramid
    targets[8][0, 0, 0] = total_count
    targets[16][0, 0, 0] = total_count
    targets[32][0, 0, 0] = total_count
    targets[64][0, 0, 0] = total_count
    return mass, targets


class TestObjectiveMechanismAudit:
    def test_gradient_metrics_directional_signs(self):
        # Gradient that pushes mass UP (grad < 0 => count_push > 0, mag_cos = -1.0)
        g_neg = -torch.ones(1, 1, 8, 8)
        m_neg = compute_gradient_metrics(g_neg)
        assert abs(m_neg["magnitude_cosine"] - (-1.0)) < 1e-6
        assert m_neg["count_push"] > 0

        # Gradient that pushes mass DOWN (grad > 0 => count_push < 0, mag_cos = +1.0)
        g_pos = torch.ones(1, 1, 8, 8)
        m_pos = compute_gradient_metrics(g_pos)
        assert abs(m_pos["magnitude_cosine"] - 1.0) < 1e-6
        assert m_pos["count_push"] < 0

        # Zero-mean gradient (pure spatial redistribution => mag_cos = 0.0)
        g_zero_mean = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]).reshape(1, 1, 2, 2)
        m_zm = compute_gradient_metrics(g_zero_mean)
        assert abs(m_zm["magnitude_cosine"]) < 1e-6
        assert abs(m_zm["count_push"]) < 1e-6

    def test_pairwise_cosine_extremes(self):
        a = torch.randn(1, 1, 16, 16)
        # Self-cosine
        assert abs(compute_pairwise_cosine(a, a) - 1.0) < 1e-6
        # Opposite cosine
        assert abs(compute_pairwise_cosine(a, -a) - (-1.0)) < 1e-6
        # Orthogonal cosine
        b = torch.zeros_like(a)
        b[:, :, :8, :] = a[:, :, 8:, :]
        b[:, :, 8:, :] = -a[:, :, :8, :]
        assert abs(compute_pairwise_cosine(a, b)) < 1e-5

    def test_component_gradients_sum_to_total_r2(self):
        mass, targets = _make_dummy_data(crop_size=64, total_count=20)
        cfg = NTPCConfig(mode="r2_flat_dm", root_loss="nb", kappa_flat16=20.0)
        crit = NTPCLoss(cfg)

        grads = compute_component_gradients(mass, targets, crit)
        assert "root_magnitude" in grads or "root_nb" in grads
        assert "flat_16" in grads
        assert "total" in grads

        # Check sum of components matches total gradient
        active_sum = grads["root_magnitude"] + grads["flat_16"]
        assert torch.allclose(active_sum, grads["total"], atol=1e-5)

    def test_component_gradients_sum_to_total_r4(self):
        mass, targets = _make_dummy_data(crop_size=64, total_count=20)
        cfg = NTPCConfig(mode="r4_dtm_tree16", root_loss="nb", kappa_root64=20.0, kappa_64_32=20.0, kappa_32_16=20.0)
        crit = NTPCLoss(cfg)

        grads = compute_component_gradients(mass, targets, crit)
        assert "root_magnitude" in grads
        assert "root_to_64" in grads
        assert "64_to_32" in grads
        assert "32_to_16" in grads
        assert "total" in grads

        active_sum = grads["root_magnitude"] + grads["root_to_64"] + grads["64_to_32"] + grads["32_to_16"]
        assert torch.allclose(active_sum, grads["total"], atol=1e-5)

    def test_compute_audit_for_mode_r2_and_r4(self):
        mass, targets = _make_dummy_data(crop_size=64, total_count=15)
        # R2
        crit_r2 = NTPCLoss(NTPCConfig(mode="r2_flat_dm", root_loss="nb", kappa_flat16=20.0))
        audit_r2 = compute_audit_for_mode(mass, targets, crit_r2, ("root_magnitude", "flat_16"))
        assert "flat_16" in audit_r2["component_metrics"]
        assert "root_magnitude_vs_flat_16" in audit_r2["pairwise_cosines"]

        # R4
        crit_r4 = NTPCLoss(NTPCConfig(mode="r4_dtm_tree16", root_loss="nb"))
        audit_r4 = compute_audit_for_mode(mass, targets, crit_r4, ("root_magnitude", "root_to_64", "64_to_32", "32_to_16"))
        assert "32_to_16" in audit_r4["component_metrics"]
        assert "root_magnitude_vs_32_to_16" in audit_r4["pairwise_cosines"]

    def test_sweep_kappa_on_crop(self):
        mass, targets = _make_dummy_data(crop_size=64, total_count=10)
        kappas = [5.0, 20.0, 50.0]
        res = sweep_kappa_on_crop(mass, targets, kappas=kappas)
        assert "r2_flat" in res and "r4_tree" in res
        for k in ["k_5", "k_20", "k_50"]:
            assert k in res["r2_flat"]
            assert k in res["r4_tree"]
            assert math.isfinite(res["r2_flat"][k]["flat_16"]["norm"])
            assert math.isfinite(res["r4_tree"][k]["32_to_16"]["norm"])

    def test_stratify_and_summarize(self):
        dummy_records = [
            {"gt_count": 50.0, "pred_count": 48.0, "signed_error": -2.0,
             "r2": {"component_metrics": {"root_magnitude": {"magnitude_cosine": 0.5, "norm": 1.0},
                                          "flat_16": {"magnitude_cosine": 0.1, "norm": 2.0},
                                          "total": {"magnitude_cosine": 0.3}},
                    "pairwise_cosines": {"root_magnitude_vs_flat_16": 0.2}},
             "r4": {"component_metrics": {"root_magnitude": {"magnitude_cosine": 0.5},
                                          "32_to_16": {"magnitude_cosine": 0.8, "norm": 3.0},
                                          "total": {"magnitude_cosine": 0.6}},
                    "pairwise_cosines": {"root_magnitude_vs_32_to_16": -0.4}}},
            {"gt_count": 1200.0, "pred_count": 1000.0, "signed_error": -200.0,
             "r2": {"component_metrics": {"root_magnitude": {"magnitude_cosine": -0.8, "norm": 5.0},
                                          "flat_16": {"magnitude_cosine": 0.2, "norm": 4.0},
                                          "total": {"magnitude_cosine": -0.4}},
                    "pairwise_cosines": {"root_magnitude_vs_flat_16": 0.1}},
             "r4": {"component_metrics": {"root_magnitude": {"magnitude_cosine": -0.8},
                                          "32_to_16": {"magnitude_cosine": 0.9, "norm": 6.0},
                                          "total": {"magnitude_cosine": 0.1}},
                    "pairwise_cosines": {"root_magnitude_vs_32_to_16": -0.9}}},
        ]
        bins = stratify_by_density(dummy_records)
        assert len(bins["sparse"]) == 1
        assert len(bins["dense"]) == 1
        assert len(bins["medium"]) == 0

        summary = summarize_audit_group(bins["dense"])
        assert summary["count"] == 1
        assert summary["mean_gt_count"] == 1200.0
        assert summary["r4"]["conflict_root_vs_32_16"] == 1.0  # -0.9 < 0 => conflict
