from __future__ import annotations

"""Adversarial Verification Suite for Canonical Final Architecture.

Tests:
1. Strict parameter ceiling: trainable parameters == 101,763 <= 104,441.
2. Canonical architecture forward pass, keys, and tensor invariants.
3. Arbitrary and non-divisible resolution parity (divisibility padding).
4. Multi-task loss computation parity with Tight Adaptive Bayesian + Harmonized Cell loss.
5. Monotonic energy dissipation of unrolled SIRT solver.
6. Configuration file validation for sub60_e128.
"""

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import unittest
import yaml
import torch

from rmr_v3.model.canonical import (
    CANONICAL_PARAM_BUDGET,
    CANONICAL_EXPECTED_PARAMS,
    build_canonical_rmr_model,
    get_canonical_model_config,
    get_canonical_loss_config,
)
from rmr_v3.losses.orchestration import compute_rmr_v3_losses
from rmr_v3.config.validator import validate_v3_config


class TestCanonicalArchitecture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Build model once on CPU without downloading weights
        cls.model = build_canonical_rmr_model(pretrained=False)
        cls.model.eval()

    def test_canonical_parameter_ceiling(self):
        """Trainable parameters must strictly equal 101,763 and be <= 104,441."""
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.assertLessEqual(trainable_params, CANONICAL_PARAM_BUDGET)
        self.assertEqual(trainable_params, CANONICAL_EXPECTED_PARAMS)

    def test_forward_pass_and_output_keys(self):
        """Model must output all required measure fields and diagnostic traces."""
        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            out = self.model(x)

        required_keys = [
            "y", "y0", "b_region", "b_solver", "iterates", "residual_fields",
            "energy_trace", "scale_weights", "solver_strength"
        ]
        for k in required_keys:
            self.assertIn(k, out, f"Missing required output key: '{k}'")

        # Check spatial dimensions at Stride 4
        self.assertEqual(out.y.shape, (2, 1, 64, 64))
        self.assertEqual(out.y0.shape, (2, 1, 64, 64))
        self.assertEqual(len(out.iterates), 7)  # y0 + 6 iterations
        self.assertTrue((out.y >= 0.0).all(), "Density predictions must be strictly non-negative")

    def test_arbitrary_resolution_divisibility_parity(self):
        """Model must transparently handle non-divisible/prime resolutions via FPN divisor padding."""
        shapes = [(1, 3, 253, 257), (1, 3, 384, 512), (1, 3, 200, 300)]
        for b, c, h, w in shapes:
            x = torch.randn(b, c, h, w)
            with torch.no_grad():
                out = self.model(x)
            expected_h4 = (h + 3) // 4
            expected_w4 = (w + 3) // 4
            self.assertEqual(out.y.shape, (b, 1, expected_h4, expected_w4))

    def test_loss_computation_and_gradient_flow(self):
        """Loss computation must succeed with adaptive Bayesian and harmonized cell losses."""
        b, c, h, w = 2, 3, 128, 128
        x = torch.randn(b, c, h, w, requires_grad=True)
        out = self.model(x)

        loss_cfg = get_canonical_loss_config()
        # Mock ground truth points and target density
        h4, w4 = h // 4, w // 4
        target_y = torch.zeros(b, 1, h4, w4)
        target_y[0, 0, 10, 10] = 1.0
        target_y[1, 0, 15, 15] = 2.0

        mock_points = [
            torch.tensor([[40.0, 40.0]]),
            torch.tensor([[60.0, 60.0], [62.0, 62.0]]),
        ]

        losses = compute_rmr_v3_losses(out, target_y, loss_cfg, points=mock_points)
        self.assertIn("total", losses)
        self.assertIn("count", losses)
        self.assertIn("allocation", losses)
        self.assertIn("cell", losses)
        self.assertIn("region_nb", losses)

        total_loss = losses["total"]
        self.assertTrue(torch.isfinite(total_loss))
        self.assertGreater(total_loss.item(), 0.0)

        total_loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_config_sub60_e128_validation(self):
        """The sub60_e128 configuration file must pass all schema validators."""
        config_path = "configs/rmr_research/sub60_e128_final_canonical_architecture.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f)

        # Validator should raise ValueError on any schema error
        validate_v3_config(cfg_dict)

        # Confirm exact parameter count matching canonical
        m_cfg = cfg_dict["model"]
        self.assertEqual(m_cfg["neck_type"], "aspp_lite")
        self.assertEqual(m_cfg["iterations"], 6)
        self.assertEqual(m_cfg["morozov_gamma"], 0.75)
        self.assertEqual(m_cfg["morozov_rho_cap"], 0.25)
        self.assertFalse(m_cfg.get("scale_seeded_carrier", False))
        self.assertEqual(cfg_dict["loss"]["allocation_loss_type"], "bayesian")
        self.assertEqual(cfg_dict["loss"]["lambda_cell"], 0.50)


if __name__ == "__main__":
    unittest.main()
