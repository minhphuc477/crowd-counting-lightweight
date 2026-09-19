from __future__ import annotations

import glob
from pathlib import Path
import pytest
import torch
import yaml

from rmr_v3.config import RMRv3Config
from rmr_v3.losses import RMRv3LossConfig, compute_rmr_v3_losses
from rmr_v3.model import RMRv3


class TestRMRv27Suite:
    """Rigorous verification suite for RMR-v27 mathematical models and configurations."""

    @pytest.fixture
    def v27_config_paths(self) -> list[Path]:
        paths = sorted(Path("configs/rmr_v27").glob("*.yaml"))
        assert len(paths) == 11, f"Expected 11 configs, got {len(paths)}: {[p.name for p in paths]}"
        return paths

    def test_all_v27_configs_validate_cleanly(self, v27_config_paths: list[Path]) -> None:
        """Assert all 6 configs instantiate both RMRv3Config and RMRv3LossConfig cleanly."""
        for p in v27_config_paths:
            with open(p, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            model_cfg = RMRv3Config.from_dict(raw.get("model", {}), pretrained=False)
            loss_cfg = RMRv3LossConfig.from_dict(raw.get("loss", {}))

            assert model_cfg.output_stride == 4
            assert model_cfg.region_sizes_px == (32, 64, 128)
            assert model_cfg.use_perspective_elevation is False
            assert model_cfg.scale_conditioned_fine_head is False

    def test_parameter_budget_ceiling(self, v27_config_paths: list[Path]) -> None:
        """Verify trainable parameters are strictly <= 105,000 for every v27 model."""
        for p in v27_config_paths:
            with open(p, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            model_cfg = RMRv3Config.from_dict(raw.get("model", {}), pretrained=False)
            model = RMRv3(model_cfg)
            trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
            assert trainable <= 105000, f"{p.name} exceeds 105k budget: {trainable:,} params"
            assert trainable == 104441, f"{p.name} unexpected parameter count: {trainable:,}"

    def test_loss_triad_active_in_canonical(self) -> None:
        """Verify that the winning loss triad (Curvature, Scale Align, Hard BG) is fully active in canonical."""
        with open("configs/rmr_v27/rmr_v27_canonical_restored.yaml", "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        loss_cfg = RMRv3LossConfig.from_dict(raw["loss"])
        assert loss_cfg.lambda_curvature == 0.50, "Curvature power loss must be 0.50 in canonical restored"
        assert loss_cfg.lambda_scale_align == 0.05, "Scale alignment loss must be 0.05 in canonical restored"
        assert loss_cfg.lambda_hard_bg == 0.15, "Hard background loss must be 0.15 in canonical restored"

    def test_full_forward_backward_gradient_flow(self) -> None:
        """Assert complete forward-backward computational graph integrity on RMR-v27 canonical."""
        with open("configs/rmr_v27/rmr_v27_canonical_restored.yaml", "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        model_cfg = RMRv3Config.from_dict(raw["model"], pretrained=False)
        loss_cfg = RMRv3LossConfig.from_dict(raw["loss"])

        model = RMRv3(model_cfg)
        model.train()

        x = torch.randn(2, 3, 128, 128, requires_grad=False)
        target = torch.zeros(2, 1, 32, 32, dtype=torch.float32)
        target[0, 0, 10, 10] = 1.0
        target[0, 0, 10, 11] = 1.0
        target[1, 0, 20, 20] = 1.0

        out = model(x)
        losses = compute_rmr_v3_losses(out, target, loss_cfg)

        assert torch.isfinite(losses["total"]).item()
        assert losses["total"].item() > 0.0
        assert losses["curvature"].item() > 0.0 or losses["curvature"].requires_grad
        assert losses["count"].item() > 0.0

        losses["total"].backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} did not receive gradients!"
                assert torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite gradients!"

    def test_rmr_v27_sota_push_config_spec(self) -> None:
        """Verify SOTA push model hyperparameter blueprint."""
        with open("configs/rmr_v27/rmr_v27_sota_push.yaml", "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        model_cfg = RMRv3Config.from_dict(raw["model"], pretrained=False)
        loss_cfg = RMRv3LossConfig.from_dict(raw["loss"])

        assert model_cfg.cyclic_bb_length == 2
        assert model_cfg.use_barzilai_borwein is True
        assert model_cfg.bb_clamp_min == 0.2
        assert model_cfg.bb_clamp_max == 2.0
        assert model_cfg.morozov_gamma == 0.50
        assert loss_cfg.lambda_curvature == 0.35
        assert loss_cfg.lambda_scale_align == 0.05
        assert loss_cfg.lambda_hard_bg == 0.15

    def test_cyclic_bb_step_reuse(self) -> None:
        """Verify Cyclic BB-1 computes step at iter 1, reuses at iter 2, and recomp at iter 3."""
        from rmr_core.operators import build_multiscale_regions
        from rmr_v3.solver import unrolled_sirt_solver

        h, w = 32, 32
        regions = build_multiscale_regions(
            height=h,
            width=w,
            output_stride=4,
            region_sizes_px=(32, 64),
            overlap=0.5,
            include_full_image=False,
            device=torch.device("cpu"),
        )
        m = regions.boxes.shape[0]
        b = 1

        torch.manual_seed(42)
        y0 = torch.rand(b, 1, h, w, dtype=torch.float32) * 0.1
        b_solver = torch.rand(b, 1, m, dtype=torch.float32) * 5.0
        weight_solver = torch.ones(b, 1, m, dtype=torch.float32)

        # Run 4 iterations with cyclic_bb_length = 2
        res = unrolled_sirt_solver(
            y0=y0,
            b_solver=b_solver,
            weight_solver=weight_solver,
            regions=regions,
            iterations=4,
            omega=1.0,
            use_barzilai_borwein=True,
            cyclic_bb_length=2,
            bb_clamp_min=0.2,
            bb_clamp_max=2.0,
        )

        omegas = res["step_omegas"]
        assert len(omegas) == 4

        # Iteration 0 (first step): uses effective_omega (1.0)
        assert omegas[0] == 1.0

        # Iteration 1 (first BB step): newly computed BB-1 step
        omega_iter1 = omegas[1]
        assert isinstance(omega_iter1, torch.Tensor)

        # Iteration 2 (cyclic reuse): must be IDENTICAL to Iteration 1!
        omega_iter2 = omegas[2]
        assert torch.equal(omega_iter2, omega_iter1), "Iter 2 must reuse cached BB omega from Iter 1!"

        # Iteration 3 (new cycle): must be RECOMPUTED from fresh differences!
        omega_iter3 = omegas[3]
        assert isinstance(omega_iter3, torch.Tensor)
        assert not torch.equal(omega_iter3, omega_iter1), "Iter 3 must compute a fresh BB omega!"

