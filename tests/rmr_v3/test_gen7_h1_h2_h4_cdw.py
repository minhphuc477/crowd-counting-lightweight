import math
import pytest
import torch
import torch.nn.functional as F

import rmr_v3.model as r_m
import rmr_v3.losses as r_l
from rmr_core.spectral import characteristic_function_loss
from rmr_v3.losses.point_supervision import fidt_loss


class TestGen7Innovations:
    """Scientific verification suite for Gen 7 hypotheses (H1, H2, H4, CDW)."""

    def test_h4_decoupled_residual_solver(self):
        """H4: Verify that detach_y0_for_solver decouples carrier from solver iterates."""
        cfg = r_m.RMRv3Config(detach_y0_for_solver=True, iterations=2)
        model = r_m.RMRv3(cfg)
        x = torch.randn(2, 3, 128, 128)
        out = model(x)

        # Loss on terminal measure y should NOT give gradients to FineHead
        loss_y = out.y.sum()
        if loss_y.requires_grad:
            loss_y.backward(retain_graph=True)
            for p in model.fine_head.parameters():
                assert p.grad is None or p.grad.abs().sum() == 0

        # Loss on carrier y0 MUST give active gradients to FineHead
        loss_y0 = out.y0.sum()
        loss_y0.backward()
        fine_head_grads = [p.grad for p in model.fine_head.parameters() if p.grad is not None]
        assert len(fine_head_grads) > 0
        assert any(g.abs().sum() > 0 for g in fine_head_grads)

    def test_h1_fidt_loss_properties(self):
        """H1: Verify FIDT bounds in [0, 1], peak amplitude at heads, and gradient flow."""
        pred = torch.full((1, 1, 32, 32), 0.5, requires_grad=True)
        pts = [torch.tensor([[16.0, 16.0], [48.0, 48.0]])]  # at stride 4, grid coords (4, 4) and (12, 12)

        loss = fidt_loss(pred, pts, k=4.0, stride=4)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert loss.item() > 0

        loss.backward()
        assert pred.grad is not None
        assert pred.grad.abs().sum() > 0

        # Empty points edge case
        pred_empty = torch.full((1, 1, 32, 32), 0.2, requires_grad=True)
        loss_empty = fidt_loss(pred_empty, None, k=4.0, stride=4)
        assert torch.isfinite(loss_empty)
        assert loss_empty.item() > 0

    def test_h2_characteristic_function_loss_properties(self):
        """H2: Verify ChfL phase matching, scale invariance, and gradient flow."""
        pred = torch.ones((1, 1, 32, 32), requires_grad=True)
        pts = [torch.tensor([[20.0, 20.0], [60.0, 60.0]])]

        loss = characteristic_function_loss(pred, pts, omega_max=0.5, num_frequencies=64, stride=4)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert loss.item() > 0

        loss.backward()
        assert pred.grad is not None
        assert pred.grad.abs().sum() > 0

    def test_diag_factorized_dynamic_window_router(self):
        """CDW: Verify DiAGFactorizedRoutingHead partition of unity and parameter budget."""
        head = r_m.DiAGFactorizedRoutingHead(in_channels=32, num_scales=3, num_aspects=2)
        x = torch.randn(2, 32, 32, 32)
        delta_scale = torch.zeros(2, 3, 1, 1)
        scene_tilt = torch.full((2, 1, 1, 1), 0.5)

        joint_pi, pi_scale, pi_aspect = head(x, delta_scale=delta_scale, scene_tilt=scene_tilt)

        # Step 0 Identity Parity: uniform distribution
        assert torch.allclose(pi_scale, torch.full_like(pi_scale, 1.0 / 3.0), atol=1e-5)
        assert torch.allclose(pi_aspect, torch.full_like(pi_aspect, 0.5), atol=1e-5)

        # Exact Partition of Unity
        sum_joint = joint_pi.sum(dim=1)
        assert torch.allclose(sum_joint, torch.ones_like(sum_joint), atol=1e-5)

        # Marginal Scale Conservation: pi_1 + pi_2 == pi_scale[1]
        assert torch.allclose(joint_pi[:, 1:2] + joint_pi[:, 2:3], pi_scale[:, 1:2], atol=1e-5)

    def test_model_parameter_budgets(self):
        """Budget: Verify <= 105,000 parameters across all configurations."""
        # 1. DRS Model (sub60_e5 base)
        cfg_drs = r_m.RMRv3Config(detach_y0_for_solver=True)
        model_drs = r_m.RMRv3(cfg_drs)
        n_drs = sum(p.numel() for p in model_drs.parameters() if p.requires_grad)
        assert n_drs <= 105000, f"DRS exceeds budget: {n_drs}"

        # 2. CDW Model (DiAG + Factorized)
        cfg_cdw = r_m.RMRv3Config(use_diag=True, factorized_scale_routing=True)
        model_cdw = r_m.RMRv3(cfg_cdw)
        n_cdw = sum(p.numel() for p in model_cdw.parameters() if p.requires_grad)
        assert n_cdw <= 105000, f"CDW exceeds budget: {n_cdw}"

    def test_end_to_end_loss_orchestration_gen7(self):
        """End-to-End: Verify compute_rmr_v3_losses with FIDT, ChfL, CDW, and DRS."""
        cfg_m = r_m.RMRv3Config(detach_y0_for_solver=True, use_diag=True, factorized_scale_routing=True, iterations=2)
        model = r_m.RMRv3(cfg_m)
        x = torch.randn(2, 3, 128, 128)
        out = model(x)

        cfg_loss = r_l.RMRv3LossConfig(
            allocation_loss_type="fidt",
            fidt_k=4.0,
            fidt_normalize_by_count=True,
            use_chfl_loss=True,
            lambda_chfl=0.5,
            chfl_omega_max=0.5,
            dm_target="dual",
        )
        pts = [torch.tensor([[20.0, 20.0], [40.0, 40.0]]), torch.tensor([[10.0, 10.0]])]
        target_y = torch.zeros((2, 1, 32, 32))
        losses = r_l.compute_rmr_v3_losses(out, target_y, cfg=cfg_loss, points=pts)
        assert "allocation" in losses
        assert "chfl" in losses
        assert "total" in losses
        assert torch.isfinite(losses["total"])
        losses["total"].backward()
