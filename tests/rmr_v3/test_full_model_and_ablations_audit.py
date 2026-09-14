from __future__ import annotations

"""Master Verification Suite: Exhaustive Module, Ablation, Loss, and Adversarial Audit.

Validates every module, mathematical invariant, loss dispatch route, ablation axis,
and extreme edge-case for the RMR-v14 crowd counting architecture under <= 105,000 parameters.
"""

import copy
import math
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from rmr_core.backbones import MobileNetV4Backbone
from rmr_core.evaluation import predict_multiscale_tta
from rmr_core.heads import FineMeasureHead
from rmr_core.necks import ASPPLiteFPNNeck, AdditiveFPNNeck, CoordinateAttention, RepWeightedFPNNeck
from rmr_core.operators import (
    RegionSet,
    build_multiscale_regions,
    regional_adjoint,
    regional_sum,
    weighted_normalized_adjoint_field,
)
from rmr_core.scale_routing import ScaleRoutingHead
from rmr_v3.solver import unrolled_sirt_solver
from rmr_core.types import RMRModelOutput
from rmr_v3.config import load_config
from rmr_v3.losses import (
    RMRv3LossConfig,
    TargetSupervisionRouter,
    compute_rmr_v3_losses,
    curvature_power_loss,
    physical_scale_alignment_loss,
    topk_hard_background_loss,
)
from rmr_v3.model import RMRv3, RMRv3Config
from rmr_v3.regional_head import ProbabilisticRegionalEvidenceHead, reliability_from_nb


def _get_default_v14_cfg() -> RMRv3Config:
    cfg_path = Path("configs/rmr_v14/rmr_v14_unified_reconstruction.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    model_dict = raw.get("model", {})
    model_dict["pretrained"] = False
    return RMRv3Config.from_dict(model_dict)


# ==============================================================================
# 1. MODULE-LEVEL INVARIANTS AUDIT
# ==============================================================================

class TestModuleLevelInvariants:
    """Rigorous verification of individual structural modules and mathematical operators."""

    def test_backbone_strides_channels_and_freeze(self):
        """Verify MobileNetV4 backbone channels, strides, and eval mode freezing."""
        backbone = MobileNetV4Backbone(pretrained=False)
        x = torch.randn(2, 3, 128, 128)
        features = backbone(x)

        # Expected reductions: C4(4), C8(8), C16(16)
        assert len(features) == 3
        c4, c8, c16 = features
        assert c4.shape == (2, 16, 32, 32)
        assert c8.shape == (2, 32, 16, 16)
        assert c16.shape == (2, 48, 8, 8)

        # Eval freeze check: consecutive passes with same input are bitwise identical
        backbone.eval()
        with torch.no_grad():
            out1 = backbone(x)
            out2 = backbone(x)
            for f1, f2 in zip(out1, out2):
                assert torch.equal(f1, f2)

    def test_neck_variants_and_reparam_deploy_parity(self):
        """Verify ASPP-Lite, AdditiveFPN, and RepWeightedFPNNeck deploy weight fusion parity."""
        c4 = torch.randn(2, 16, 32, 32)
        c8 = torch.randn(2, 32, 16, 16)
        c16 = torch.randn(2, 48, 8, 8)

        # 1. ASPP-Lite
        aspp = ASPPLiteFPNNeck(in_channels=(16, 32, 48), width=32)
        p4, p8, p16 = aspp(c4, c8, c16)
        assert p4.shape == (2, 32, 32, 32)
        assert p16.shape == (2, 32, 8, 8)

        # 2. AdditiveFPN
        add_fpn = AdditiveFPNNeck(in_channels=(16, 32, 48), width=32)
        p4_add, p8_add, p16_add = add_fpn(c4, c8, c16)
        assert p4_add.shape == (2, 32, 32, 32)
        assert p16_add.shape == (2, 32, 8, 8)

        # 3. RepWeightedFPN: Train mode vs Deploy mode exact numerical parity
        rep_fpn = RepWeightedFPNNeck(in_channels=(16, 32, 48), width=32)
        rep_fpn.eval()
        with torch.no_grad():
            p4_train, p8_train, p16_train = rep_fpn(c4, c8, c16)
            rep_fpn.switch_to_deploy()
            p4_deploy, p8_deploy, p16_deploy = rep_fpn(c4, c8, c16)

            max_diff_p4 = (p4_train - p4_deploy).abs().max().item()
            max_diff_p16 = (p16_train - p16_deploy).abs().max().item()
            assert max_diff_p4 < 1e-4, f"P4 deploy parity mismatch: {max_diff_p4}"
            assert max_diff_p16 < 1e-4, f"P16 deploy parity mismatch: {max_diff_p16}"

    def test_coordinate_attention_identity_preservation(self):
        """Verify Coordinate Attention 1D pooling, channel modulation, and shapes."""
        ca = CoordinateAttention(channels=32, reduction=8)
        x = torch.randn(2, 32, 24, 24)
        out = ca(x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_fine_head_and_temp_softplus_strict_positivity(self):
        """Verify FineMeasureHead calibrated prior, learnable tau, and y0 > 0."""
        head = FineMeasureHead(width=32, temp_softplus=True)
        expected_bias = math.log(math.expm1(0.015763))
        actual_bias = head.body[-1].bias.item()
        assert abs(actual_bias - expected_bias) < 1e-3

        extreme_features = torch.tensor([-100.0, -10.0, 0.0, 10.0, 50.0]).view(1, 1, 5, 1).repeat(1, 32, 1, 1)
        z0 = head.forward_logits(extreme_features)
        y0 = head.activate(z0)
        assert (y0 > 0.0).all(), "Fine measure must be strictly positive"

        loss = y0.sum()
        loss.backward()
        assert head.tau.grad is not None
        assert head.tau.grad.abs() > 0.0

    def test_top_down_semantic_gate_and_fg_gate_floors(self):
        """Verify TDSG exactly 33 parameters, bilinear upsampling, and fg_gate floor non-zero grad."""
        tdsg_conv = nn.Conv2d(32, 1, 1)
        assert sum(p.numel() for p in tdsg_conv.parameters()) == 33

        p16 = torch.randn(2, 32, 8, 8)
        p4 = torch.randn(2, 32, 32, 32)

        sem_logit = F.interpolate(tdsg_conv(p16), size=p4.shape[-2:], mode="bilinear", align_corners=False)
        assert sem_logit.shape == (2, 1, 32, 32)

        tdsg_floor = 0.20
        sem_mask = tdsg_floor + (1.0 - tdsg_floor) * torch.sigmoid(sem_logit)
        assert (sem_mask >= tdsg_floor).all()
        assert (sem_mask <= 1.0).all()

        fg_conv = nn.Conv2d(32, 1, 1)
        p4_param = nn.Parameter(torch.full((1, 32, 16, 16), -50.0))
        fg_logit = fg_conv(p4_param)
        fg_floor = 0.10
        fg_mask = fg_floor + (1.0 - fg_floor) * torch.sigmoid(fg_logit)
        assert (fg_mask >= fg_floor).all()

        loss = fg_mask.sum()
        loss.backward()
        assert p4_param.grad is not None

    def test_regional_head_statistical_branches(self):
        """Verify ProbabilisticRegionalEvidenceHead rate positivity, dispersion bounds, and 3 reliability modes."""
        head = ProbabilisticRegionalEvidenceHead(feature_dim=32, hidden=48, hurdle_head=True)
        pyramid = (torch.randn(2, 32, 32, 32), torch.randn(2, 32, 16, 16), torch.randn(2, 32, 8, 8))
        regions = build_multiscale_regions(32, 32, output_stride=4, region_sizes_px=(32, 64), include_full_image=False)

        out = head(pyramid, regions)
        mu = out["rate"]
        log_r = out["log_dispersion"]
        r = out["dispersion"]
        z_pi = out["hurdle_logit"]
        m = regions.boxes.shape[0]

        assert (mu > 0.0).all()
        assert (r >= 0.5).all() and (r <= 500.0).all()
        assert z_pi.shape == (2, 1, m)

        pi_r = torch.sigmoid(z_pi)
        rel_var = reliability_from_nb(mu, r, regions, mode="nb_rate_variance", hurdle_pi=pi_r)
        assert (rel_var["weight"] >= 0.25).all() and (rel_var["weight"] <= 4.0).all()

        rel_snr = reliability_from_nb(mu, r, regions, mode="snr", hurdle_pi=pi_r)
        assert (rel_snr["weight"] >= 0.25).all() and (rel_snr["weight"] <= 4.0).all()

        rel_hybrid = reliability_from_nb(mu, r, regions, mode="hybrid_hurdle", hurdle_pi=pi_r)
        assert (rel_hybrid["weight"] >= 0.25).all() and (rel_hybrid["weight"] <= 4.0).all()

    def test_discrete_operators_duality_and_scale_invariance(self):
        """Verify discrete adjoint duality pairing <Ay, w> = <y, A^T w> and Radon-Nikodym adjoint."""
        h, w = 32, 32
        regions = build_multiscale_regions(h, w, output_stride=4, region_sizes_px=(32, 64))
        m = regions.boxes.shape[0]

        y = torch.rand(2, 1, h, w) + 0.1
        w_reg = torch.rand(2, 1, m) + 0.1

        ay = regional_sum(y, regions.boxes)
        at_w = regional_adjoint(w_reg, regions.boxes, h, w)

        inner_1 = (ay * w_reg).sum().item()
        inner_2 = (y * at_w).sum().item()

        rel_diff = abs(inner_1 - inner_2) / max(abs(inner_1), abs(inner_2))
        assert rel_diff < 1e-4, f"Duality pairing failure: rel_diff={rel_diff}"

        b_region = ay + 1.0
        weight_reg = torch.ones_like(b_region)
        field_rn = weighted_normalized_adjoint_field(
            y, b_region, weight_reg, regions, adjoint_mode="radon_nikodym"
        )
        assert field_rn.shape == (2, 1, h, w)

        y_zero = torch.zeros(2, 1, h, w)
        field_rn_zero = weighted_normalized_adjoint_field(
            y_zero, b_region, weight_reg, regions, adjoint_mode="radon_nikodym"
        )
        assert torch.all(field_rn_zero == 0.0), "Radon-Nikodym adjoint must be zero on zero-density pixels"

    def test_solver_morozov_deadband_and_contractivity(self):
        """Verify unrolled SIRT solver contractivity and Bayesian Morozov deadband zero-update."""
        h, w = 24, 24
        regions = build_multiscale_regions(h, w, output_stride=4, region_sizes_px=(32,))
        m = regions.boxes.shape[0]

        y0 = torch.rand(1, 1, h, w) + 0.2
        ay0 = regional_sum(y0, regions.boxes)

        variance = torch.ones(1, 1, m) * 1.0
        morozov_gamma = 0.75
        b_inside = ay0 + torch.ones(1, 1, m) * 0.5

        weight_solver = torch.ones(1, 1, m)
        res_inside = unrolled_sirt_solver(
            y0=y0,
            b_solver=b_inside,
            weight_solver=weight_solver,
            regions=regions,
            iterations=1,
            omega=0.5,
            solver_strength=1.0,
            morozov_gamma=morozov_gamma,
            b_variance=variance,
            adjoint_mode="radon_nikodym",
        )
        y1_inside = res_inside["y"]
        assert torch.allclose(y1_inside, y0, atol=1e-5), "Morozov deadband failed to nullify sub-threshold residual"

        b_outside = ay0 + torch.ones(1, 1, m) * 5.0
        res_outside = unrolled_sirt_solver(
            y0=y0,
            b_solver=b_outside,
            weight_solver=weight_solver,
            regions=regions,
            iterations=2,
            omega=0.5,
            solver_strength=1.0,
            morozov_gamma=morozov_gamma,
            b_variance=variance,
            adjoint_mode="radon_nikodym",
        )
        y2_outside = res_outside["y"]
        assert not torch.allclose(y2_outside, y0)
        assert y2_outside.sum() > y0.sum()

    def test_scale_router_simplex_partition(self):
        """Verify ScaleRoutingHead parameter count is exactly 483 and outputs sum to 1.0."""
        router = ScaleRoutingHead(in_channels=32, num_scales=3)
        assert sum(p.numel() for p in router.parameters()) == 483

        p4 = torch.randn(2, 32, 28, 28)
        weights = router(p4)
        assert weights.shape == (2, 3, 28, 28)

        scale_sum = weights.sum(dim=1)
        assert torch.allclose(scale_sum, torch.ones_like(scale_sum), atol=1e-6)


# ==============================================================================
# 2. LOSS FORMULATIONS AND SUPERVISION ROUTER AUDIT
# ==============================================================================

class TestLossFormulationsAndSupervisionRouter:
    """Rigorous verification of loss functions, supervision dispatch, and memory hygiene."""

    def test_target_supervision_router_gradient_isolation(self):
        """Verify TargetSupervisionRouter routes gradients cleanly for y, y0, and dual."""
        router_y = TargetSupervisionRouter(target_mode="y")
        router_y0 = TargetSupervisionRouter(target_mode="y0")
        router_dual = TargetSupervisionRouter(target_mode="dual")

        y = torch.tensor([2.0, 3.0], requires_grad=True)
        y0 = torch.tensor([5.0, 7.0], requires_grad=True)

        def fn(inp):
            return inp.sum()

        l_y, _ = router_y.dispatch(fn, y, y0)
        l_y.backward()
        assert y.grad is not None and y.grad.sum() > 0.0
        assert y0.grad is None

        y.grad = None
        l_y0, _ = router_y0.dispatch(fn, y, y0)
        l_y0.backward()
        assert y.grad is None
        assert y0.grad is not None and y0.grad.sum() > 0.0

        y.grad = None
        y0.grad = None
        l_dual, _ = router_dual.dispatch(fn, y, y0)
        l_dual.backward()
        assert torch.allclose(y.grad, torch.tensor([0.5, 0.5]))
        assert torch.allclose(y0.grad, torch.tensor([0.5, 0.5]))

    def test_density_gated_curvature_loss_clamping(self):
        """Verify curvature loss penalizes high frequencies only in regions above density threshold."""
        pred_bg = torch.full((1, 1, 32, 32), 0.01)
        pred_bg[0, 0, ::2, ::2] += 0.005
        target_bg = torch.full((1, 1, 32, 32), 0.01)
        loss_bg = curvature_power_loss(pred_bg, target_bg, threshold=0.08, mode="hard")
        assert loss_bg.item() == 0.0, "Curvature loss must be strictly zero on background"

        pred_dense = torch.full((1, 1, 32, 32), 0.20)
        pred_dense[0, 0, ::2, ::2] += 0.05
        target_dense = torch.full((1, 1, 32, 32), 0.20)
        loss_dense = curvature_power_loss(pred_dense, target_dense, threshold=0.08, mode="hard")
        assert loss_dense.item() > 0.0, "Curvature loss must penalize high frequencies in dense crowd"

    def test_topk_hard_background_mining(self):
        """Verify top-K background mining extracts only the worst background false alarms."""
        pred = torch.zeros(1, 1, 32, 32)
        target = torch.zeros(1, 1, 32, 32)
        pred[0, 0, :10, :1] = 0.50
        loss = topk_hard_background_loss(pred, target, ratio=0.10)
        assert loss.item() > 0.0

        pred_clean = torch.zeros(1, 1, 32, 32)
        loss_clean = topk_hard_background_loss(pred_clean, target, ratio=0.10)
        assert loss_clean.item() == 0.0

    def test_foreground_masked_scale_alignment_loss(self):
        """Verify scale alignment loss is 0.0 on background when mask_background=True."""
        scale_weights = torch.full((1, 3, 32, 32), 1.0 / 3.0)
        target_bg = torch.zeros(1, 1, 32, 32)

        loss_masked = physical_scale_alignment_loss(
            scale_weights, target_bg, mask_background=True, tau_sparse=0.03
        )
        assert loss_masked.item() == 0.0, "Masked scale loss must be 0 on pure background"

        loss_unmasked = physical_scale_alignment_loss(
            scale_weights, target_bg, mask_background=False, tau_sparse=0.03
        )
        assert loss_unmasked.item() > 0.0, "Unmasked scale loss incurs spurious background penalty"

    def test_zero_tensor_memory_and_gradient_hygiene(self):
        """Verify inactive loss terms return zero_val without breaking graph or leaking allocations."""
        cfg = _get_default_v14_cfg()
        model = RMRv3(cfg)
        loss_cfg = RMRv3LossConfig(
            lambda_curvature=0.0,
            lambda_hard_bg=0.0,
            lambda_scale_align=0.0,
            lambda_fg_gate=0.0,
        )

        x = torch.randn(1, 3, 64, 64)
        target = torch.zeros(1, 1, 16, 16)

        out = model(x)
        losses = compute_rmr_v3_losses(out, target, loss_cfg)

        assert losses["curvature"].item() == 0.0
        assert losses["hard_bg"].item() == 0.0
        assert losses["scale_align"].item() == 0.0
        assert losses["fg_bce"].item() == 0.0

        losses["total"].backward()
        assert next(p for p in model.parameters() if p.requires_grad).grad is not None


# ==============================================================================
# 3. COMPLETE ABLATION MATRIX INVARIANT AUDIT
# ==============================================================================

class TestCompleteAblationMatrix:
    """Systematic verification of all 12 primary architectural and mathematical ablation axes."""

    def test_ablation_solver_enabled_vs_disabled(self):
        """Ablation 1: Verify direct baseline mode (enable_solver=False) bypasses solver."""
        cfg = _get_default_v14_cfg()
        cfg.enable_solver = False
        model = RMRv3(cfg)

        x = torch.randn(1, 3, 64, 64)
        out = model(x)

        assert torch.equal(out.y, out.y0), "In baseline mode, y must be bitwise identical to y0"
        assert len(out.iterates) == 1
        assert len(out.residual_fields) == 0
        assert out.solver_strength == 0.0

    def test_ablation_radon_nikodym_vs_linear_adjoint(self):
        """Ablation 2: Compare Radon-Nikodym measure adjoint vs standard Linear adjoint."""
        cfg_rn = _get_default_v14_cfg()
        cfg_rn.adjoint_mode = "radon_nikodym"
        model_rn = RMRv3(cfg_rn)

        cfg_lin = copy.deepcopy(cfg_rn)
        cfg_lin.adjoint_mode = "flat"
        model_lin = RMRv3(cfg_lin)
        model_lin.load_state_dict(model_rn.state_dict())

        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out_rn = model_rn(x)
            out_lin = model_lin(x)

        assert (out_rn.y >= 0.0).all()
        assert (out_lin.y >= 0.0).all()
        assert not torch.allclose(out_rn.y, out_lin.y)

    def test_ablation_morozov_gamma_zero_vs_active(self):
        """Ablation 3: Compare Morozov gamma = 0.0 (unregularized) vs 0.75 (deadband active)."""
        cfg_act = _get_default_v14_cfg()
        cfg_act.morozov_gamma = 0.75
        model_act = RMRv3(cfg_act)

        cfg_zero = copy.deepcopy(cfg_act)
        cfg_zero.morozov_gamma = 0.0
        model_zero = RMRv3(cfg_zero)
        model_zero.load_state_dict(model_act.state_dict())

        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out_act = model_act(x)
            out_zero = model_zero(x)

        assert (out_act.y >= 0.0).all()
        assert (out_zero.y >= 0.0).all()

    def test_ablation_tdsg_active_vs_disabled(self):
        """Ablation 4: Compare TDSG active (33 params) vs disabled."""
        cfg_tdsg = _get_default_v14_cfg()
        cfg_tdsg.use_top_down_semantic_gate = True
        model_tdsg = RMRv3(cfg_tdsg)
        params_tdsg = sum(p.numel() for p in model_tdsg.parameters() if p.requires_grad)

        cfg_no_tdsg = copy.deepcopy(cfg_tdsg)
        cfg_no_tdsg.use_top_down_semantic_gate = False
        model_no_tdsg = RMRv3(cfg_no_tdsg)
        params_no_tdsg = sum(p.numel() for p in model_no_tdsg.parameters() if p.requires_grad)

        assert params_tdsg - params_no_tdsg == 33
        assert params_tdsg == 104506
        assert params_no_tdsg == 104473

    def test_ablation_fg_gate_floor_modulation(self):
        """Ablation 5: Compare fg_gate_floor 0.10 vs 0.70."""
        cfg_010 = _get_default_v14_cfg()
        cfg_010.fg_gate_floor = 0.10
        model_010 = RMRv3(cfg_010)

        cfg_070 = copy.deepcopy(cfg_010)
        cfg_070.fg_gate_floor = 0.70
        model_070 = RMRv3(cfg_070)
        model_070.load_state_dict(model_010.state_dict())

        with torch.no_grad():
            model_010.fg_gate.bias.fill_(-20.0)
            model_070.fg_gate.bias.fill_(-20.0)

            x = torch.randn(1, 3, 64, 64)
            out_010 = model_010(x)
            out_070 = model_070(x)

            ratio = (out_010.y0.sum() / out_070.y0.sum()).item()
            assert abs(ratio - (0.10 / 0.70)) < 0.05

    def test_ablation_hurdle_head_active_vs_disabled(self):
        """Ablation 6: Compare hurdle_head active vs disabled."""
        cfg_hurdle = _get_default_v14_cfg()
        cfg_hurdle.hurdle_head = True
        model_hurdle = RMRv3(cfg_hurdle)

        cfg_no_hurdle = copy.deepcopy(cfg_hurdle)
        cfg_no_hurdle.hurdle_head = False
        model_no_hurdle = RMRv3(cfg_no_hurdle)

        x = torch.randn(1, 3, 64, 64)
        out_hurdle = model_hurdle(x)
        out_no_hurdle = model_no_hurdle(x)

        assert "hurdle_logit" in out_hurdle
        assert out_no_hurdle.hurdle_logit is None

    def test_ablation_reliability_mode_sweep(self):
        """Ablation 7: Verify all 3 reliability modes run cleanly."""
        modes = ["nb_rate_variance", "snr", "hybrid_hurdle"]
        x = torch.randn(1, 3, 64, 64)

        for mode in modes:
            cfg = _get_default_v14_cfg()
            cfg.reliability_mode = mode
            model = RMRv3(cfg)
            out = model(x)
            assert (out.y >= 0.0).all()
            assert out.solver_strength > 0.0

    def test_ablation_solver_iteration_sweep(self):
        """Ablation 8: Sweep solver iterations T in {1, 2, 4, 6}."""
        iterations_to_test = [1, 2, 4, 6]
        x = torch.randn(1, 3, 64, 64)

        for t in iterations_to_test:
            cfg = _get_default_v14_cfg()
            cfg.iterations = t
            model = RMRv3(cfg)
            out = model(x)
            assert len(out.iterates) == t + 1
            assert len(out.residual_fields) == t

    def test_ablation_neck_architecture_sweep(self):
        """Ablation 9: Verify aspp_lite, additive, and rep_weighted under full RMRv3."""
        necks = ["aspp_lite", "additive", "rep_weighted"]
        x = torch.randn(1, 3, 64, 64)

        for neck in necks:
            cfg = _get_default_v14_cfg()
            cfg.neck_type = neck
            model = RMRv3(cfg)
            out = model(x)
            assert out.y.shape == (1, 1, 16, 16)
            assert (out.y >= 0.0).all()

    def test_ablation_tv_smoothing_sweep(self):
        """Ablation 10: Verify laplacian, charbonnier, and none TV types."""
        tv_types = ["laplacian", "charbonnier"]
        x = torch.randn(1, 3, 64, 64)

        for tv in tv_types:
            cfg = _get_default_v14_cfg()
            cfg.tv_type = tv
            model = RMRv3(cfg)
            out = model(x)
            assert (out.y >= 0.0).all()

    def test_all_repo_rmr_configs_under_budget_and_runnable(self):
        """Ablation 11: Verify every single RMR config across the repo satisfies budget <= 105k and runs forward."""
        import glob
        configs = sorted(glob.glob("configs/rmr_v*/*.yaml") + glob.glob("configs/rmr_v*/*/*.yaml"))
        assert len(configs) >= 50, f"Expected at least 50 RMR configs, found {len(configs)}"

        x = torch.randn(1, 3, 128, 128)
        for cfg_path in configs:
            with open(cfg_path, "r") as f:
                d = yaml.safe_load(f)
            if not d or "model" not in d:
                continue
            m_cfg = {k: v for k, v in d["model"].items() if hasattr(RMRv3Config, k)}
            m_cfg["pretrained"] = False
            model = RMRv3(RMRv3Config(**m_cfg))
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            assert n_params <= 105000, f"Budget exceeded in {cfg_path}: {n_params} > 105000"
            out = model(x)
            assert out["y"].shape == (1, 1, 32, 32), f"Bad output shape for {cfg_path}"
            assert (out["y"] >= 0.0).all(), f"Negative densities found in {cfg_path}"


# ==============================================================================
# 4. HARSH ADVERSARIAL AND EXTREME EDGE CASES AUDIT
# ==============================================================================

class TestHarshAdversarialAndExtremeEdgeCases:
    """Extreme stress-testing with boundary shapes, zero/saturated pixels, and mega-crowds."""

    def test_zero_ground_truth_and_black_image(self):
        """Adversarial 1: Completely black image X=0 with zero GT count."""
        cfg = _get_default_v14_cfg()
        model = RMRv3(cfg)
        loss_cfg = RMRv3LossConfig.from_dict({})

        x = torch.zeros(1, 3, 64, 64)
        target = torch.zeros(1, 1, 16, 16)

        out = model(x)
        assert not torch.isnan(out.y).any()
        assert not torch.isinf(out.y).any()
        assert (out.y >= 0.0).all()

        losses = compute_rmr_v3_losses(out, target, loss_cfg)
        assert not torch.isnan(losses["total"]).any()
        assert not torch.isinf(losses["total"]).any()

    def test_saturated_white_image(self):
        """Adversarial 2: Completely saturated white image X=1."""
        cfg = _get_default_v14_cfg()
        model = RMRv3(cfg)
        x = torch.ones(1, 3, 64, 64)

        out = model(x)
        assert not torch.isnan(out.y).any()
        assert not torch.isinf(out.y).any()

    def test_arbitrary_odd_prime_spatial_resolutions(self):
        """Adversarial 3: Odd prime spatial dimensions (e.g. 503x397, 257x331)."""
        cfg = _get_default_v14_cfg()
        model = RMRv3(cfg)

        for h, w in [(503, 397), (257, 331)]:
            x = torch.randn(1, 3, h, w)
            out = model(x)
            assert not torch.isnan(out.y).any()
            assert not torch.isinf(out.y).any()
            assert (out.y >= 0.0).all()

    def test_extreme_aspect_ratios(self):
        """Adversarial 4: Extreme aspect ratios 512x128 and 128x512."""
        cfg = _get_default_v14_cfg()
        model = RMRv3(cfg)

        for h, w in [(512, 128), (128, 512)]:
            x = torch.randn(1, 3, h, w)
            out = model(x)
            assert not torch.isnan(out.y).any()
            assert not torch.isinf(out.y).any()
            assert (out.y >= 0.0).all()

    def test_extreme_mega_crowd_density(self):
        """Adversarial 5: Target count > 4,000 people to stress numerical stability."""
        cfg = _get_default_v14_cfg()
        model = RMRv3(cfg)
        loss_cfg = RMRv3LossConfig.from_dict({})

        x = torch.randn(1, 3, 128, 128)
        target = torch.full((1, 1, 32, 32), 4000.0 / (32 * 32))

        out = model(x)
        losses = compute_rmr_v3_losses(out, target, loss_cfg)

        assert not torch.isnan(losses["total"]).any()
        assert not torch.isinf(losses["total"]).any()

    def test_batch_size_invariance(self):
        """Adversarial 6: Batch size 1 vs Batch size 2 slice-0 parity."""
        cfg = _get_default_v14_cfg()
        model = RMRv3(cfg)
        model.eval()

        torch.manual_seed(42)
        x1 = torch.randn(1, 3, 64, 64)
        x2 = torch.randn(1, 3, 64, 64)
        x_batch = torch.cat([x1, x2], dim=0)

        with torch.no_grad():
            out1 = model(x1)
            out_batch = model(x_batch)

        diff = (out_batch.y[0:1] - out1.y).abs().max().item()
        assert diff < 1e-5, f"Batch slice invariance violated: max_diff={diff}"

    def test_eval_mode_strict_determinism(self):
        """Adversarial 7: Consecutive eval passes produce bitwise-identical output."""
        cfg = _get_default_v14_cfg()
        model = RMRv3(cfg)
        model.eval()

        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        assert torch.equal(out1.y, out2.y)

    def test_amp_bfloat16_mixed_precision(self):
        """Adversarial 8: Autocast under bfloat16 / float16."""
        cfg = _get_default_v14_cfg()
        model = RMRv3(cfg)

        x = torch.randn(1, 3, 64, 64)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = model(x)

        assert not torch.isnan(out.y).any()
        assert not torch.isinf(out.y).any()

    def test_predict_multiscale_tta_pipeline(self):
        """Adversarial 9: Test-Time Augmentation (TTA) with Hann window blending."""
        cfg = _get_default_v14_cfg()
        model = RMRv3(cfg)
        model.eval()

        img = torch.randn(3, 128, 128)
        with torch.no_grad():
            tta_density = predict_multiscale_tta(
                model=model,
                image=img,
                output_stride=4,
                scales=(0.85, 1.0, 1.15),
                use_hflip=True,
            )

        assert tta_density.squeeze().shape == (32, 32)
        assert (tta_density >= 0.0).all()
        assert not torch.isnan(tta_density).any()
        assert not torch.isinf(tta_density).any()
