from __future__ import annotations

import unittest
import torch
import torch.nn.functional as F

from rmr_core.operators.regions import build_multiscale_regions, partition_regions_by_scale
from rmr_core.operators.carrier import compute_deficit_carrier_seed, modulate_carrier_field
from rmr_core.operators.prefix_sums import regional_sum
from rmr_v3.losses.regional import scale_balanced_regional_count_loss
from rmr_v3.solver import unrolled_sirt_solver
from rmr_v3.model.canonical import build_canonical_rmr_model, CANONICAL_PARAM_BUDGET, CANONICAL_EXPECTED_PARAMS


class TestV46SolverAndRegionalCalibration(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.h, self.w = 64, 64
        self.regions = build_multiscale_regions(
            self.h, self.w, output_stride=4, region_sizes_px=[32, 64, 128], include_full_image=False, device=self.device
        )

    def test_deficit_carrier_seed_unattenuated(self):
        """Deficit carrier seeding must preserve deficit mass without 20,000x attenuation."""
        y = torch.full((1, 1, self.h, self.w), 0.001)
        b_solver = torch.zeros((1, 1, len(self.regions.boxes)))
        # Put 40 people deficit on one box
        b_solver[0, 0, 0] = 40.0
        
        psi = compute_deficit_carrier_seed(
            y, b_solver, self.regions, output_stride=4
        )
        self.assertGreater(psi.max().item(), 0.0)
        # Integral should reflect the deficit, not be 0.0001
        self.assertGreater(psi.sum().item(), 1.0)

    def test_trust_region_deficit_removal_damping(self):
        """In deficit cells, bound_removal must be damped to prevent worsening undercounts."""
        y0 = torch.full((1, 1, self.h, self.w), 0.5)
        # Target has deficit relative to y0: b_solver > Ay0
        b_gt = regional_sum(y0, self.regions.boxes, out_dtype=torch.float32) + 20.0
        weight = torch.ones_like(b_gt)
        
        res = unrolled_sirt_solver(
            y0=y0, b_solver=b_gt, weight_solver=weight, regions=self.regions,
            iterations=3, omega=1.0, trust_region_kappa=0.35, deficit_carrier_seeding=True,
            deficit_seed_eps=1.0, output_stride=4,
        )
        # Final count must not decrease when there is a positive deficit
        self.assertGreaterEqual(res["y"].sum().item(), y0.sum().item() - 1e-3)

    def test_scale_balanced_regional_count_loss(self):
        """Regional count loss must be 0.0 on perfect sum match and strictly positive on drift."""
        target_reg = torch.full((1, 1, len(self.regions.boxes)), 10.0)
        mean_reg_perfect = target_reg.clone()
        loss_zero = scale_balanced_regional_count_loss(target_reg, mean_reg_perfect, self.regions)
        self.assertAlmostEqual(loss_zero.item(), 0.0, places=5)

        # Scale 0 has +50% drift
        mean_reg_drift = target_reg.clone()
        mask0 = (self.regions.scale_id == 0)
        mean_reg_drift[..., mask0] = mean_reg_drift[..., mask0] * 1.5
        loss_drift = scale_balanced_regional_count_loss(target_reg, mean_reg_drift, self.regions)
        self.assertGreater(loss_drift.item(), 0.05)

    def test_canonical_parameter_ceiling(self):
        """Model parameters must remain strictly <= 104,441."""
        model = build_canonical_rmr_model(pretrained=False)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertLessEqual(params, CANONICAL_PARAM_BUDGET)
        self.assertEqual(params, CANONICAL_EXPECTED_PARAMS)
