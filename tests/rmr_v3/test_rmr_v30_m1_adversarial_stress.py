from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
import yaml

from rmr_core.operators import build_multiscale_regions
from rmr_v3.solver_ops import (
    anscombe_discrepancy,
    anscombe_transform,
    compute_adaptive_tau,
    proximal_firm_threshold,
)
from rmr_v3.solver import unrolled_sirt_solver
from rmr_v3.model import RMRv3, RMRv3Config
from rmr_v3.losses import compute_rmr_v3_losses, RMRv3LossConfig


class TestAnscombeDiscrepancyAdversarialStress:
    """Stress tests for anscombe_discrepancy under extreme adversarial conditions."""

    @pytest.mark.parametrize(
        "b_val, q_val",
        [
            (10000.0, 0.001),     # extreme undercount
            (0.0, 10000.0),       # extreme overcount
            (0.0, 0.0),           # zero count
            (10000.0, 10000.0),   # high-count exact match
            (1e-7, 1e-7),         # near-zero match
            (-5.0, -5.0),         # negative count input (clamp test)
            (-1000.0, 500.0),     # negative b
            (1e6, 1e-6),          # massive scale difference
            (0.001, 10000.0),     # massive overcount
        ],
    )
    def test_anscombe_discrepancy_extreme_counts(self, b_val: float, q_val: float):
        b = torch.tensor([b_val], dtype=torch.float32, requires_grad=False)
        q = torch.tensor([q_val], dtype=torch.float32, requires_grad=True)

        res = anscombe_discrepancy(q, b, c=0.375)

        # 1. Output must be strictly finite (no NaN, no Inf)
        assert torch.isfinite(res).all(), f"res contains NaN or Inf for b={b_val}, q={q_val}: {res}"

        # 2. Backward autograd gradient w.r.t. q must be strictly finite
        res.backward()
        assert q.grad is not None
        assert torch.isfinite(q.grad).all(), f"q.grad contains NaN or Inf for b={b_val}, q={q_val}: {q.grad}"

        # 3. Rate residual must be bounded in magnitude
        assert abs(res.item()) < 10000.0, f"Residual exploded: {res.item()}"

    def test_anscombe_discrepancy_morozov_deadband_stress(self):
        """Stress-test Morozov deadband shrinkage under various noise regimes."""
        gammas = [0.0, 0.5, 1.0, 3.0]
        variances = [None, torch.tensor([0.0]), torch.tensor([5.0]), torch.tensor([1000.0])]

        for gamma in gammas:
            for b_var in variances:
                q = torch.tensor([50.0], requires_grad=True)
                b = torch.tensor([55.0], requires_grad=False)
                res = anscombe_discrepancy(q, b, morozov_gamma=gamma, b_variance=b_var)
                assert torch.isfinite(res).all()
                res.backward()
                assert torch.isfinite(q.grad).all()

    def test_anscombe_discrepancy_large_tensor_broadcasting(self):
        """Verify numerical stability on large spatial batches with mixed scales."""
        q = torch.empty(4, 1, 1000).uniform_(0.0, 5000.0).requires_grad_(True)
        b = torch.empty(4, 1, 1000).uniform_(0.0, 5000.0)
        res = anscombe_discrepancy(q, b)
        assert res.shape == (4, 1, 1000)
        assert torch.isfinite(res).all()
        loss = res.sum()
        loss.backward()
        assert torch.isfinite(q.grad).all()


class TestComputeAdaptiveTauAdversarialStress:
    """Stress tests for compute_adaptive_tau under extreme spatial and density conditions."""

    def test_zero_density_full_deadband(self):
        """Empty background must receive identically base_tau_step."""
        y_zero = torch.zeros(2, 1, 128, 128)
        tau_eff = compute_adaptive_tau(0.015, y_zero, stride=4, mode="density_adaptive", rho0=0.05)
        assert isinstance(tau_eff, torch.Tensor)
        assert torch.allclose(tau_eff, torch.tensor(0.015), atol=1e-6)

    def test_hyperdense_cluster_vanishing_tau(self):
        """Hyperdense crowd (>10,000 count/cell) must attenuate tau_eff -> 0."""
        y_dense = torch.full((1, 1, 64, 64), 10000.0)
        tau_eff = compute_adaptive_tau(0.015, y_dense, stride=4, mode="density_adaptive", rho0=0.05)
        assert isinstance(tau_eff, torch.Tensor)
        assert tau_eff.max().item() < 1e-4, f"tau_eff did not vanish on hyperdense cluster: {tau_eff.max().item()}"
        assert (tau_eff >= 0.0).all()

    @pytest.mark.parametrize(
        "h, w",
        [
            (1024, 256),   # tall non-square
            (256, 1024),   # wide non-square
            (5, 5),        # equal to pool_kernel
            (1, 1),        # degenerate 1x1
            (128, 512),    # typical surveillance panorama
        ],
    )
    def test_nonsquare_and_degenerate_resolutions(self, h: int, w: int):
        y = torch.rand(1, 1, h, w) * 20.0
        tau_eff = compute_adaptive_tau(0.015, y, stride=4, mode="density_adaptive", rho0=0.05)
        assert isinstance(tau_eff, torch.Tensor)
        assert tau_eff.shape == (1, 1, h, w)
        assert torch.isfinite(tau_eff).all()
        assert (tau_eff >= 0.0).all()
        assert (tau_eff <= 0.015 + 1e-6).all()

    def test_spatial_bifurcation_firm_threshold(self):
        """Verify that spatial tau eliminates noise on background while preserving dense clump."""
        y = torch.zeros(1, 1, 64, 64)
        # Low noise on left half
        y[0, 0, :, :32] = 0.010
        # Real dense crowd on right half
        y[0, 0, :, 32:] = 50.0

        tau_eff = compute_adaptive_tau(0.015, y, stride=4, mode="density_adaptive", rho0=0.05)
        y_proc = proximal_firm_threshold(y, tau=tau_eff, mu=3.0)

        # Background noise must be strictly 0.0
        assert (y_proc[0, 0, :, :20] == 0.0).all(), "Background noise was not eliminated by firm threshold"
        # Dense crowd peaks must suffer ZERO shrinkage
        assert torch.allclose(y_proc[0, 0, :, 40:], y[0, 0, :, 40:], atol=1e-5), "Dense crowd was shrunk"


class TestUnrolledSIRTSolverAdversarialStress:
    """Stress tests for unrolled SIRT solver backward flow and invariants."""

    @pytest.mark.parametrize("T", [6, 8])
    @pytest.mark.parametrize("mode", ["anscombe", "standard"])
    def test_solver_direct_gradient_stability(self, T: int, mode: str):
        h, w = 64, 64
        regions = build_multiscale_regions(
            h, w, output_stride=4, region_sizes_px=[32, 64], overlap=0.5, include_full_image=False
        )
        m = len(regions.boxes)

        y0 = (torch.rand(1, 1, h, w) * 0.1 + 0.01).requires_grad_(True)
        b_solver = torch.rand(1, 1, m) * 5.0
        weight_solver = torch.ones(1, 1, m)

        res = unrolled_sirt_solver(
            y0=y0,
            b_solver=b_solver,
            weight_solver=weight_solver,
            regions=regions,
            iterations=T,
            omega=1.0,
            use_anscombe=(mode == "anscombe"),
            adaptive_tau=True,
            proximal_tau=0.015,
            proximal_mode="firm",
            use_barzilai_borwein=True,
            output_stride=4,
        )

        loss = res["y"].sum()
        loss.backward()

        assert y0.grad is not None
        assert torch.isfinite(y0.grad).all(), f"NaN or Inf gradient at T={T}, mode={mode}"
        assert (y0.grad != 0).any(), f"Gradient vanished to zero at T={T}, mode={mode}"
        gnorm = y0.grad.norm().item()
        assert 0.1 < gnorm < 1000.0, f"Gradient norm unstable: {gnorm} at T={T}, mode={mode}"

    @pytest.mark.parametrize("T", [6, 8])
    def test_anscombe_support_invariance_adversarial_b(self, T: int):
        """Under huge adversarial b=10000, empty background must not leak any positive mass."""
        h, w = 64, 64
        regions = build_multiscale_regions(
            h, w, output_stride=4, region_sizes_px=[32, 64], overlap=0.5, include_full_image=False
        )
        m = len(regions.boxes)

        y0 = torch.zeros(1, 1, h, w)
        y0[0, 0, 30:34, 30:34] = 1.0  # 16 foreground pixels

        b_solver = torch.full((1, 1, m), 10000.0)
        weight_solver = torch.ones(1, 1, m)

        res = unrolled_sirt_solver(
            y0=y0.clone(),
            b_solver=b_solver,
            weight_solver=weight_solver,
            regions=regions,
            iterations=T,
            omega=1.0,
            use_anscombe=True,
            adaptive_tau=True,
            proximal_tau=0.015,
            proximal_mode="firm",
            use_barzilai_borwein=True,
            tv_lambda=0.0,
            output_stride=4,
        )

        bg_mask = (y0 == 0.0)
        bg_max = res["y"][bg_mask].max().item()
        assert bg_max == 0.0, f"Background support violated! Max background count: {bg_max}"

    @pytest.mark.parametrize("T", [6, 8])
    def test_full_model_e2e_backward_flow(self, T: int):
        """Verify end-to-end gradient propagation through full RMRv3 model at T=6 and T=8."""
        raw_model = yaml.safe_load(open("configs/rmr_v29/rmr_v29_h2_subpixel2.yaml"))["model"]
        raw_loss = yaml.safe_load(open("configs/rmr_v29/rmr_v29_h2_subpixel2.yaml"))["loss"]
        raw_model["iterations"] = T
        raw_model["use_anscombe_sirt"] = True
        raw_model["adjoint_mode"] = "anscombe_vst"
        raw_model["adaptive_tau"] = True

        cfg_m = RMRv3Config.from_dict(raw_model)
        cfg_l = RMRv3LossConfig.from_dict(raw_loss)
        model = RMRv3(cfg_m)

        # Unmask scale router by giving pw non-zero test weights (simulating post-init step)
        model.scale_router.pw.weight.data.normal_(0, 0.01)

        x = torch.randn(2, 3, 256, 256, requires_grad=True)
        out = model(x)

        target_y = torch.zeros(2, 1, 128, 128)
        target_y[:, :, 40:60, 40:60] = 0.5  # realistic crowd region

        losses = compute_rmr_v3_losses(out, target_y, cfg_l)
        losses["total"].backward()

        none_grads = []
        zero_grads = []
        nan_grads = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    none_grads.append(name)
                elif not torch.isfinite(param.grad).all():
                    nan_grads.append(name)
                elif param.grad.norm().item() == 0.0:
                    zero_grads.append(name)

        assert not nan_grads, f"NaN gradients in parameters: {nan_grads}"
        assert not none_grads, f"Parameters missing gradients: {none_grads}"
        assert not zero_grads, f"Dead parameters with zero gradients: {zero_grads}"


class TestMemoryAndResourceStress:
    """Stress tests for memory safety during forward and backward passes on 512x512 batches."""

    def test_cuda_memory_headroom_512x512(self):
        """Verify that 512x512 batch passes comfortably fit within strict VRAM limits."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        raw = yaml.safe_load(open("configs/rmr_v29/rmr_v29_h2_subpixel2.yaml"))["model"]
        raw["iterations"] = 8
        raw["use_anscombe_sirt"] = True
        raw["adjoint_mode"] = "anscombe_vst"
        raw["adaptive_tau"] = True

        cfg = RMRv3Config.from_dict(raw)
        model = RMRv3(cfg).to(device)
        model.train()

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        # Batch size 2 on 512x512 crop
        x = torch.randn(2, 3, 512, 512, device=device, requires_grad=True)
        out = model(x)
        loss = out.y.sum()
        loss.backward()

        if device.type == "cuda":
            peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            # Must remain under 1000 MB for batch size 2 on 512x512 (empirical: ~330 MB)
            assert peak_mb < 1000.0, f"Excessive peak VRAM usage: {peak_mb:.1f} MB > 1000 MB"
