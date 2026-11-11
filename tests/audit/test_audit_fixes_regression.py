"""Regression tests verifying Group 1 (tooling/tiling/optim) and Group 2 (Rep necks, reporting, init_m0, YAML) audit fixes."""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml

from rmr_core.data import CrowdManifestDataset
from rmr_core.necks import (
    RLKHDCFPNNeck,
    RepASPPLiteNeck,
    RepDWBlock7x7,
    RepWeightedFPNNeck,
)
from rmr_core.tiling import predict_tiled
from rmr_v3.config import validate_v3_config
from rmr_v3.losses import RMRv3LossConfig
from rmr_v3.model import RMRv3, RMRv3Config
from rmr_v3.optim.builder import build_v3_optimizer, maybe_run_safe_lr_finder
from rmr_v3.optim.lr_finder import SafeLRFinder
from rmr_v3.reporting import (
    _is_regional_head_active,
    format_epoch_row,
    format_eval_block,
)


class _DictPointsDataset(Dataset):
    def __init__(self, n: int = 8) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        img = torch.rand(3, 64, 64)
        tgt = torch.zeros(1, 16, 16)
        tgt[0, 4, 5] = 1.0
        tgt[0, 10, 11] = 1.0
        pts = torch.tensor([[22.0, 18.0], [46.0, 42.0]], dtype=torch.float32)
        return {"image": img, "target_y": tgt, "points": pts}


def _collate_dict_points(batch):
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "target_y": torch.stack([b["target_y"] for b in batch], dim=0),
        "points": [b["points"] for b in batch],
    }


def test_safe_lr_finder_forwards_points_and_preserves_legacy_2arg() -> None:
    """Verify SafeLRFinder and maybe_run_safe_lr_finder pass `points` cleanly and support legacy 2-arg callables."""
    cfg_dict = {
        "model": {"backbone_lr_scale": 0.1},
        "train": {
            "auto_lr_finder": True,
            "lr_finder_min_lr": 1e-6,
            "lr_finder_max_lr": 1e-3,
            "lr_finder_num_iter": 10,
            "lr_finder_max_safe_lr": 5e-4,
            "grad_clip": 1.0,
        },
    }
    m_cfg = RMRv3Config(
        pretrained=False,
        enable_regional_head=False,
        enable_solver=False,
    )
    model = RMRv3(m_cfg)
    optimizer = build_v3_optimizer(model, cfg_dict, lr_init=1e-4)
    loader = DataLoader(_DictPointsDataset(8), batch_size=2, collate_fn=_collate_dict_points)

    loss_cfg = RMRv3LossConfig(
        allocation_loss_type="dual_bayesian_radon_ot",
        lambda_count=1.0,
        lambda_cell=0.5,
        lambda_region_nb=0.0,
    )
    suggested_lr = maybe_run_safe_lr_finder(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        loss_fn=loss_cfg,
        cfg=cfg_dict,
        device=torch.device("cpu"),
    )
    assert suggested_lr is not None
    assert math.isfinite(suggested_lr) and 1e-6 <= suggested_lr <= 5e-4

    # Verify points argument actually arrives and non-trivial radon_ot signal is non-zero
    seen_points = []
    def spy_loss(out, tgt, points=None):
        seen_points.append(points)
        from rmr_v3.losses import compute_rmr_v3_losses
        return compute_rmr_v3_losses(out, tgt, loss_cfg, points=points)

    finder = SafeLRFinder(model, optimizer, spy_loss, torch.device("cpu"))
    lr_spy = finder.range_test(loader, min_lr=1e-6, max_lr=1e-3, num_iter=4, max_safe_lr=5e-4)
    assert math.isfinite(lr_spy)
    assert len(seen_points) == 4 and all(p is not None and len(p) == 2 for p in seen_points)

    # Backward compatibility: legacy 2-arg callable without points parameter must still work
    def legacy_2arg_loss(out, tgt):
        return (out["y"].sum() - tgt.sum()).abs()

    finder_legacy = SafeLRFinder(model, optimizer, legacy_2arg_loss, torch.device("cpu"))
    lr_legacy = finder_legacy.range_test(loader, min_lr=1e-6, max_lr=1e-3, num_iter=4, max_safe_lr=5e-4)
    assert math.isfinite(lr_legacy)


def test_predict_tiled_odd_dims_floor_stride2_model() -> None:
    """Verify predict_tiled handles floor(h/s) stride-2 models on odd input dimensions in core and hann modes."""
    class _FloorStride2Model(nn.Module):
        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            return {
                "y": torch.ones(
                    x.shape[0], 1, x.shape[-2] // 2, x.shape[-1] // 2,
                    device=x.device, dtype=x.dtype,
                )
            }

    model = _FloorStride2Model()
    image = torch.randn(3, 257, 259)

    y_core = predict_tiled(model, image, output_stride=2, tile_size=128, halo=0, blend_mode="core")
    assert tuple(y_core.shape) == (1, 129, 130)
    assert torch.isfinite(y_core).all()
    assert torch.allclose(y_core[:, :-1, :-1], torch.ones_like(y_core[:, :-1, :-1]))

    y_hann = predict_tiled(model, image, output_stride=2, tile_size=128, halo=16, blend_mode="hann")
    assert tuple(y_hann.shape) == (1, 129, 130)
    assert torch.isfinite(y_hann).all()
    assert torch.allclose(y_hann, torch.ones_like(y_hann), atol=1e-4)


@pytest.mark.parametrize(
    "neck_type,neck_cls",
    [
        ("rep_weighted", RepWeightedFPNNeck),
        ("rep_aspp_lite", RepASPPLiteNeck),
        ("rlk_hdc", RLKHDCFPNNeck),
    ],
)
def test_rep_necks_active_gradients_channel_mix_deploy_and_budgets(
    neck_type: str,
    neck_cls: type[nn.Module],
) -> None:
    """Verify zero dead-grad params, cross-channel PW mixing, deploy equivalence, and Track A/B budgets."""
    torch.manual_seed(7)
    neck = neck_cls()
    neck.train()
    for _ in range(2):
        c4 = torch.randn(2, 16, 32, 32)
        c8 = torch.randn(2, 32, 16, 16)
        c16 = torch.randn(2, 48, 8, 8)
        p4, p8, p16 = neck(c4, c8, c16)
        (p4.sum() + p8.sum() + p16.sum()).backward()
        neck.zero_grad(set_to_none=True)

    c4 = torch.randn(2, 16, 32, 32, requires_grad=True)
    c8 = torch.randn(2, 32, 16, 16, requires_grad=True)
    c16 = torch.randn(2, 48, 8, 8, requires_grad=True)
    p4, p8, p16 = neck(c4, c8, c16)
    loss = (p4.pow(2).sum() + p8.pow(2).sum() + p16.pow(2).sum())
    loss.backward()

    zero_grad_names = [
        name for name, p in neck.named_parameters()
        if p.requires_grad and (p.grad is None or p.grad.abs().sum().item() == 0.0)
    ]
    assert not zero_grad_names, f"{neck_type} has zero-grad params: {zero_grad_names}"

    # Verify cross-channel interaction in PW + shuffle mixer
    neck.eval()
    c4_base = torch.randn(1, 16, 16, 16)
    c8_base = torch.randn(1, 32, 8, 8)
    c16_base = torch.randn(1, 48, 4, 4)
    c4_pert = c4_base.clone()
    c4_pert[:, 0:1, :, :] += 5.0
    with torch.no_grad():
        p4_base, _, _ = neck(c4_base, c8_base, c16_base)
        p4_pert, _, _ = neck(c4_pert, c8_base, c16_base)
    ch_diffs = (p4_pert - p4_base).abs().amax(dim=(0, 2, 3))
    changed_channels = int((ch_diffs > 1e-4).sum().item())
    assert changed_channels >= 16, f"{neck_type} cross-channel mixing too narrow: {changed_channels}/32"

    # Deploy algebraic equivalence
    with torch.no_grad():
        p4_t, p8_t, p16_t = neck(c4_base, c8_base, c16_base)
    neck.switch_to_deploy()
    with torch.no_grad():
        p4_d, p8_d, p16_d = neck(c4_base, c8_base, c16_base)
    assert (p4_d - p4_t).abs().max().item() < 1e-4
    assert (p8_d - p8_t).abs().max().item() < 1e-4
    assert (p16_d - p16_t).abs().max().item() < 1e-4

    # Track A (<= 104,441) and Track B (<= 200,000) full model parameter budgets
    m_track_a = RMRv3(RMRv3Config(neck_type=neck_type, feature_width=32, max_trainable_params=104441, pretrained=False))
    n_a = sum(p.numel() for p in m_track_a.parameters() if p.requires_grad)
    assert n_a <= 104441

    m_track_b = RMRv3(RMRv3Config(neck_type=neck_type, feature_width=64, max_trainable_params=200000, pretrained=False))
    n_b = sum(p.numel() for p in m_track_b.parameters() if p.requires_grad)
    assert n_b <= 200000


def test_rep_dw_block_general_odd_kernels() -> None:
    """Verify RepDWBlock7x7 supports any odd kernel >= 3 with zero dead grads and exact deploy fusion."""
    block = RepDWBlock7x7(channels=16, kernel_size=5)
    x = torch.randn(2, 16, 16, 16)
    block.train()
    for _ in range(2):
        block(x).pow(2).sum().backward()
        block.zero_grad(set_to_none=True)
    block(x).pow(2).sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum().item() > 0 for p in block.parameters())

    block.eval()
    with torch.no_grad():
        y_train = block(x)
    block.switch_to_deploy()
    with torch.no_grad():
        y_deploy = block(x)
    assert torch.allclose(y_train, y_deploy, atol=1e-4)


def test_reporting_auto_detects_disabled_regional_head() -> None:
    """Verify reporting detects region_head=None cleanly without false positives on partial dicts."""
    loss_cfg = RMRv3LossConfig(lambda_region_nb=0.0, lambda_hurdle=0.0)
    loss_avgs = {"allocation": 0.12, "cell": 0.05, "region_nb": 0.0}
    val_metrics = {
        "MAE": 67.5, "RMSE": 110.2, "NAE": 0.21, "Bias": -2.1,
        "GAME0": 67.5, "GAME1": 72.0, "GAME2": 78.0, "GAME3": 89.0,
        "coverage_50": 0.54, "coverage_80": 0.72, "coverage_95": 0.81,
    }
    diag = {"solver_energy_reduction": 0.0}

    row_disabled = {
        "train_total": 1.23, "train_count": 0.8, "train_cell": 0.05, "train_region_nb": 0.0,
        "train_dm32": 0.0, "train_dm64": 0.0,
        "region_mu_mean": 0.0, "region_weight_mean": 1.0, "region_weight_std": 0.0,
        "solver_weight_mean": 1.0, "region_dispersion_p10": 50.0, "region_dispersion_p50": 50.0,
        "region_dispersion_p90": 50.0,
    }
    assert _is_regional_head_active(None, row_disabled, loss_cfg, solver_strength=0.0) is False
    epoch_str = format_epoch_row(0, 10, row_disabled, loss_avgs, loss_cfg, solver_strength=0.0)
    eval_str = format_eval_block(0, 10, row_disabled, val_metrics, 67.5, 67.5, "BEST", 0.0, diag, loss_avgs, loss_cfg)
    assert "RegHead: off" in epoch_str
    assert "disabled (enable_regional_head=false)" in eval_str
    assert "Coverage:" not in eval_str

    row_active = dict(row_disabled, region_mu_mean=1.4, region_weight_std=0.22, region_dispersion_p10=12.0, region_dispersion_p90=90.0)
    assert _is_regional_head_active(None, row_active, loss_cfg, solver_strength=0.0) is True
    assert "Uncertainty   : Coverage:" in format_eval_block(0, 10, row_active, val_metrics, 67.5, 67.5, "BEST", 0.0, diag, loss_avgs, loss_cfg)

    row_minimal_unit = {
        "train_total": 1.0, "train_count": 0.5, "region_dispersion_p50": 50.0,
        "region_weight_mean": 1.0, "solver_weight_mean": 1.0,
    }
    assert _is_regional_head_active(None, row_minimal_unit, loss_cfg, solver_strength=0.0) is True


def test_dual_lattice_single_count_and_step0_init_m0_and_active_yamls() -> None:
    """Verify no cell loss double counting, Step-0 init_m0 parity across strides, and all 124 active YAMLs."""
    from rmr_v3.losses import compute_rmr_v3_losses

    m0_s4 = 0.015796
    for subpixel_s2 in (False, True):
        cfg = RMRv3Config(
            pretrained=False,
            enable_regional_head=False,
            enable_solver=False,
            subpixel_stride2=subpixel_s2,
            output_stride=(2 if subpixel_s2 else 4),
            init_m0=m0_s4,
        )
        m = RMRv3(cfg).eval()
        with torch.no_grad():
            out = m(torch.randn(2, 3, 128, 128))
        expected = m0_s4 / (4.0 if subpixel_s2 else 1.0)
        rel_err = abs(out.y0.mean().item() - expected) / expected
        assert rel_err < 0.05, f"subpixel_stride2={subpixel_s2}: mean={out.y0.mean().item()} vs expected={expected}"

        if subpixel_s2:
            target = torch.zeros(2, 1, 64, 64)
            pts = [torch.zeros((0, 2)), torch.zeros((0, 2))]
            loss_cfg_no_fine = RMRv3LossConfig(
                output_stride=2,
                lambda_count=1.0,
                lambda_cell=0.5,
                lambda_carrier_cell=0.25,
                lambda_fine_cell=0.0,
                lambda_region_nb=0.0,
            )
            loss_cfg_with_fine = RMRv3LossConfig(
                output_stride=2,
                lambda_count=1.0,
                lambda_cell=0.5,
                lambda_carrier_cell=0.25,
                lambda_fine_cell=1.0,
                lambda_region_nb=0.0,
            )
            losses_a = compute_rmr_v3_losses(out, target, loss_cfg_no_fine, points=pts)
            losses_b = compute_rmr_v3_losses(out, target, loss_cfg_with_fine, points=pts)
            assert torch.allclose(losses_b["cell_fine"], losses_b["cell"])
            assert torch.allclose(losses_a["total"], losses_b["total"])

    # Verify eval_jhu_attributes module & auto norm resolution
    import scripts.eval_jhu_attributes as eval_jhu_mod
    assert hasattr(eval_jhu_mod, "main")
    assert CrowdManifestDataset.resolve_norm("mobilenetv4_conv_small_050.e3000_r224_in1k", "auto") == "inception"

    # Verify all 124 active benchmark and sweep YAMLs validate cleanly
    active_files = sorted(Path("configs/benchmarks").rglob("*.yaml")) + sorted(Path("configs/training_sweeps").rglob("*.yaml"))
    assert len(active_files) == 124
    for p in active_files:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        validate_v3_config(raw)
        m_cfg = raw["model"]
        l_cfg = raw["loss"]
        is_solver = bool(m_cfg.get("enable_solver", True))
        if not is_solver:
            assert m_cfg.get("enable_regional_head") is False, f"Direct config {p} must disable regional head"
            assert float(l_cfg.get("lambda_region_nb", 0.0)) == 0.0
        else:
            assert m_cfg.get("enable_regional_head", True) is True, f"Solver config {p} must keep regional head"
