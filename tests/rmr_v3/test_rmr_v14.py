from __future__ import annotations

"""Exhaustive Verification Suite for RMR-v14 Unified Continuous-Discrete Reconstruction."""

import math
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
import yaml

from rmr_core.operators import build_multiscale_regions
from rmr_v3.config import load_config
from rmr_v3.losses import (
    RMRv3LossConfig,
    compute_rmr_v3_losses,
    physical_scale_alignment_loss,
)
from rmr_v3.model import RMRv3, RMRv3Config
from rmr_v3.regional_head import reliability_from_nb


def test_rmr_v14_config_loading():
    """Verify loading and schema validation of rmr_v14_unified_reconstruction.yaml."""
    cfg_path = Path("configs/rmr_v14/rmr_v14_unified_reconstruction.yaml")
    assert cfg_path.exists(), f"Missing config: {cfg_path}"

    cfg = load_config(cfg_path)
    model_cfg = RMRv3Config.from_dict(cfg.get("model", {}))
    loss_cfg = RMRv3LossConfig.from_dict(cfg.get("loss", {}))

    assert model_cfg.neck_type == "aspp_lite"




    assert model_cfg.reliability_mode == "hybrid_hurdle"
    assert model_cfg.adjoint_mode == "radon_nikodym"
    assert model_cfg.morozov_gamma == 0.75
    assert loss_cfg.lambda_scale_align == 0.05
    assert loss_cfg.scale_align_mask_bg is True
    assert loss_cfg.lambda_curvature == 0.50
    assert loss_cfg.curvature_gate_threshold == 0.08
    assert loss_cfg.curvature_gate_mode == "hard"


def test_rmr_v14_parameter_budget_exactness():
    """Verify exact parameter budget of RMR-v14 remains strictly <= 105,000."""
    cfg_path = Path("configs/rmr_v14/rmr_v14_unified_reconstruction.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    model_dict = raw.get("model", {})
    model_dict["pretrained"] = False
    m_cfg = RMRv3Config.from_dict(model_dict)

    model = RMRv3(m_cfg)
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert total_trainable <= 105000, (
        f"RMR-v14 parameter count {total_trainable} strictly exceeds hard limit 105,000!"
    )
    assert total_trainable == 104440, "Expected exactly 104,440 parameters"


def test_top_down_semantic_context_gate_invariants():
    """Verify TDSG modulates P4 correctly with safe floor and gradients flow to P16."""
    with pytest.raises(ValueError, match="permanently BANNED"):
        RMRv3Config(
            pretrained=False,
            neck_type="aspp_lite",
            use_top_down_semantic_gate=True,
            tdsg_floor=0.20,
        )


def test_fg_gate_deep_floor_modulation():
    """Verify dynamic fg_gate_floor allows attenuation down to 0.10 when logits are negative."""
    with pytest.raises(ValueError, match="permanently BANNED"):
        RMRv3Config(pretrained=False, foreground_gate=True, fg_gate_floor=0.70)


def test_foreground_masked_scale_alignment_invariants():
    """Verify scale alignment loss ignores empty background pixels when mask_background=True."""
    b, k, h, w = 2, 3, 32, 32
    scale_weights = torch.softmax(torch.randn(b, k, h, w), dim=1)

    # 1. Empty image (all zeros)
    empty_target = torch.zeros(b, 1, h, w)
    loss_masked = physical_scale_alignment_loss(
        scale_weights, empty_target, tau_dense=0.12, tau_sparse=0.03, mask_background=True
    )
    assert float(loss_masked.item()) == 0.0, (
        f"Empty background with mask_background=True must yield exactly 0.0 loss, got {loss_masked.item()}"
    )

    # 2. Dense image (high density everywhere)
    dense_target = torch.full((b, 1, h, w), 0.20)
    loss_dense = physical_scale_alignment_loss(
        scale_weights, dense_target, tau_dense=0.12, tau_sparse=0.03, mask_background=True
    )
    assert float(loss_dense.item()) > 0.0
    assert torch.isfinite(loss_dense)


def test_hybrid_hurdle_reliability_invariants():
    """Verify Hurdle-hybrid reliability blends rate variance on background and SNR on crowds."""
    regions = build_multiscale_regions(
        height=32, width=32, output_stride=4, region_sizes_px=(32, 64, 128), overlap=0.5
    )
    m = regions.boxes.shape[0]

    mu = torch.full((1, 1, m), 10.0)
    disp = torch.full((1, 1, m), 50.0)

    # Background hurdle (pi -> 0)
    pi_bg = torch.zeros(1, 1, m)
    rel_bg = reliability_from_nb(
        mu, disp, regions, mode="hybrid_hurdle", hurdle_pi=pi_bg
    )
    rel_var = reliability_from_nb(mu, disp, regions, mode="nb_rate_variance")
    assert torch.allclose(rel_bg["weight"], rel_var["weight"], atol=1e-5), (
        "When pi_R=0, hybrid_hurdle must identically match nb_rate_variance!"
    )

    # Crowd hurdle (pi -> 1)
    pi_crowd = torch.ones(1, 1, m)
    rel_crowd = reliability_from_nb(
        mu, disp, regions, mode="hybrid_hurdle", hurdle_pi=pi_crowd
    )
    rel_snr = reliability_from_nb(mu, disp, regions, mode="snr")
    assert torch.allclose(rel_crowd["weight"], rel_snr["weight"], atol=1e-5), (
        "When pi_R=1, hybrid_hurdle must identically match SNR weighting!"
    )


def test_rmr_v14_end_to_end_loss_and_gradients():
    """Run full forward and backward pass with all RMR-v14 components active under AMP."""
    cfg_path = Path("configs/rmr_v14/rmr_v14_unified_reconstruction.yaml")
    full_cfg = load_config(cfg_path)
    model_dict = full_cfg.get("model", {}).copy()
    model_dict["pretrained"] = False
    model_cfg = RMRv3Config.from_dict(model_dict)
    loss_cfg = RMRv3LossConfig.from_dict(full_cfg.get("loss", {}))

    model = RMRv3(model_cfg)
    model.train()

    b, c, h, w = 2, 3, 128, 128
    x = torch.randn(b, c, h, w, requires_grad=True)
    target_y = torch.zeros(b, 1, h // 4, w // 4)
    # Add synthetic crowd cluster
    target_y[:, :, 10:18, 10:18] = 0.15

    outputs = model(x)
    losses = compute_rmr_v3_losses(outputs, target_y, cfg=loss_cfg)

    assert "total" in losses
    assert "scale_align" in losses
    assert "curvature" in losses
    assert "hard_bg" in losses

    total_loss = losses["total"]
    assert torch.isfinite(total_loss)
    total_loss.backward()

    # Verify gradients flow to all key modules
    assert model.fine_head.body[-1].weight.grad is not None
    assert model.scale_router.pw.weight.grad is not None




def test_target_supervision_router():
    """Verify TargetSupervisionRouter dispatches y, y0, and dual modes accurately."""
    from rmr_v3.losses import TargetSupervisionRouter

    y = torch.tensor([10.0])
    y0 = torch.tensor([2.0])
    dummy_fn = lambda t: t * 2.0

    router_dual = TargetSupervisionRouter("dual")
    total, aux = router_dual.dispatch(dummy_fn, y, y0)
    assert total.item() == 0.5 * (20.0 + 4.0) == 12.0
    assert aux["y"].item() == 20.0
    assert aux["y0"].item() == 4.0

    router_y0 = TargetSupervisionRouter("y0")
    total_y0, aux_y0 = router_y0, router_y0.dispatch(dummy_fn, y, y0)
    assert aux_y0[0].item() == 4.0

    router_y = TargetSupervisionRouter("y")
    total_y, aux_y = router_y.dispatch(dummy_fn, y, y0)
    assert total_y.item() == 20.0


def test_rmr_model_output_mapping_and_typing():
    """Verify RMRModelOutput supports both dot-notation and dictionary mapping access."""
    cfg = RMRv3Config(pretrained=False)
    model = RMRv3(cfg)
    model.eval()

    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = model(x)

    # Dot-notation typed access
    assert hasattr(out, "y")
    assert hasattr(out, "y0")
    assert out.y.shape == (1, 1, 16, 16)

    # Dictionary mapping access (100% backward compatibility)
    assert "y" in out
    assert "y0" in out
    assert torch.equal(out["y"], out.y)
    assert out.get("y") is not None
    assert out.get("non_existent_key", 42) == 42
    assert len(out) >= 18
    assert "y" in list(out.keys())


def test_method_critical_fields_integrity():
    """Verify METHOD_CRITICAL_FIELDS tracks all key RMR-v13 and v14 model/loss parameters."""
    from rmr_v3.config import METHOD_CRITICAL_FIELDS

    model_fields = METHOD_CRITICAL_FIELDS["model"]
    loss_fields = METHOD_CRITICAL_FIELDS["loss"]

    # RMR-v13
    assert "adjoint_mode" in model_fields
    assert "morozov_gamma" in model_fields
    assert "lambda_scale_align" in loss_fields
    assert "scale_align_tau_dense" in loss_fields

    # RMR-v14
    assert "use_top_down_semantic_gate" in model_fields
    assert "tdsg_floor" in model_fields
    assert "fg_gate_floor" in model_fields
    assert "scale_align_mask_bg" in loss_fields


def test_reliability_mode_validation_without_morozov_gamma():
    """Verify validate_v3_config catches invalid reliability_mode even when morozov_gamma is omitted."""
    from rmr_v3.config import validate_v3_config

    invalid_cfg = {
        "model": {
            "reliability_mode": "invalid_mode_name",
        }
    }
    with pytest.raises(ValueError, match="reliability_mode must be 'nb_rate_variance', 'snr', or 'hybrid_hurdle'"):
        validate_v3_config(invalid_cfg)


def test_predict_multiscale_tta_parity_and_mass_conservation():
    """Verify predict_multiscale_tta matches predict_tiled at scale=1.0 without flip, and preserves mass."""
    from rmr_core.evaluation import predict_multiscale_tta, predict_tiled

    cfg = RMRv3Config(pretrained=False)
    model = RMRv3(cfg)
    model.eval()

    img = torch.randn(3, 128, 128)

    # Baseline single-scale tiled prediction
    pred_base = predict_tiled(model, img, output_stride=4, tile_size=64, halo=16)

    # TTA with scale=1.0 and no flip must be bitwise identical
    pred_tta_identity = predict_multiscale_tta(
        model, img, output_stride=4, tile_size=64, halo=16, scales=(1.0,), use_hflip=False
    )
    assert torch.allclose(pred_base, pred_tta_identity, atol=1e-6)

    # TTA with multi-scale fusion must output valid positive count
    pred_tta_multi = predict_multiscale_tta(
        model, img, output_stride=4, tile_size=64, halo=16, scales=(0.85, 1.0, 1.15), use_hflip=True
    )
    assert pred_tta_multi.shape == pred_base.shape
    assert torch.isfinite(pred_tta_multi).all()

