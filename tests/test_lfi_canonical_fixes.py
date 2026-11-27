from __future__ import annotations

import math
import random
from pathlib import Path

import torch

from rmr_core.heads import FineMeasureHead, _FINE_HEAD_BIAS_INIT
from rmr_core.necks import SFILiteFPNNeck
from rmr_core.transforms import point_preserving_random_erasing
from rmr_v3.config import load_config, validate_v3_config
from rmr_v3.losses import RMRv3LossConfig, compute_rmr_v3_losses
from rmr_v3.model import RMRv3, RMRv3Config


def test_random_erasing_drop_points_behavior() -> None:
    img = torch.rand(3, 128, 128)
    xs = torch.arange(4, 128, 8, dtype=torch.float32)
    ys = torch.arange(4, 128, 8, dtype=torch.float32)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    pts = torch.stack([gx.flatten(), gy.flatten()], dim=1)

    random.seed(12345)
    img_keep, pts_keep = point_preserving_random_erasing(
        img, pts, prob=1.0, scale=(0.15, 0.15), ratio=(1.0, 1.0), max_patches=1, drop_points=False
    )
    assert pts_keep.shape[0] == pts.shape[0]
    assert img_keep.shape == img.shape

    random.seed(12345)
    img_drop, pts_drop = point_preserving_random_erasing(
        img, pts, prob=1.0, scale=(0.15, 0.15), ratio=(1.0, 1.0), max_patches=1, drop_points=True
    )
    assert torch.allclose(img_drop, img_keep)
    assert 0 < pts_drop.shape[0] < pts.shape[0]

    c_mean = img.mean(dim=(-2, -1), keepdim=True)
    erased_mask = (img_drop - c_mean).abs().sum(dim=0) < 1e-6
    px = pts_drop[:, 0].long()
    py = pts_drop[:, 1].long()
    assert not erased_mask[py, px].any()


def test_dsimr_init_z_s_parity_and_gradient() -> None:
    torch.manual_seed(7)
    m0 = math.log1p(math.exp(_FINE_HEAD_BIAS_INIT))
    feat = torch.zeros(2, 32, 16, 16)

    head_0 = FineMeasureHead(
        width=32,
        subpixel_stride2=True,
        dsimr_mode=True,
        dsimr_init_z_s=0.0,
        floor_tau=0.001,
        density_adaptive_scale=True,
        density_scale_gamma=0.15,
    )
    head_2 = FineMeasureHead(
        width=32,
        subpixel_stride2=True,
        dsimr_mode=True,
        dsimr_init_z_s=2.0,
        floor_tau=0.001,
        density_adaptive_scale=True,
        density_scale_gamma=0.15,
    )

    assert head_0.dsimr_proj is not None
    assert math.isclose(float(head_0.dsimr_proj.bias[0].item()), 0.0, abs_tol=1e-6)
    assert head_2.dsimr_proj is not None
    assert math.isclose(float(head_2.dsimr_proj.bias[0].item()), 2.0, abs_tol=1e-6)

    with torch.no_grad():
        _, y0 = head_0.forward_with_logits(feat, force_activate=True)
        _, y2 = head_2.forward_with_logits(feat, force_activate=True)
        assert abs(float(y0.mean().item()) - m0 / 4.0) < 1e-3
        assert abs(float(y2.mean().item()) - m0 / 4.0) < 1e-3

    feat_rand = torch.randn(2, 32, 16, 16)
    head_0.zero_grad()
    _, y0_r = head_0.forward_with_logits(feat_rand, force_activate=True)
    y0_r.sum().backward()
    grad_0 = float(head_0.dsimr_proj.bias.grad[0].abs().item())

    head_2.zero_grad()
    _, y2_r = head_2.forward_with_logits(feat_rand, force_activate=True)
    y2_r.sum().backward()
    grad_2 = float(head_2.dsimr_proj.bias.grad[0].abs().item())

    assert grad_0 > grad_2


def test_sfi_direct_dw_and_fixed_hf_controls() -> None:
    torch.manual_seed(11)
    c4 = torch.randn(2, 16, 32, 32)
    c8 = torch.randn(2, 32, 16, 16)
    c16 = torch.randn(2, 48, 8, 8)

    kw = dict(
        in_channels=(16, 32, 48),
        width=32,
        hdc_dilations=(1, 2, 3),
        use_hdc_proj=True,
        hdc_proj_groups=2,
        use_selective_gate=False,
        norm_free_hf=True,
        hf_act_mode="linear",
        pad_aware_laplacian=True,
    )
    torch.manual_seed(99)
    neck_lap = SFILiteFPNNeck(decomp_mode="laplacian", **kw)
    torch.manual_seed(99)
    neck_ddw = SFILiteFPNNeck(decomp_mode="direct_dw", **kw)
    neck_fix = SFILiteFPNNeck(decomp_mode="fixed_hf", **kw)

    p_lap = sum(p.numel() for p in neck_lap.parameters())
    p_ddw = sum(p.numel() for p in neck_ddw.parameters())
    p_fix = sum(p.numel() for p in neck_fix.parameters())
    assert p_lap == p_ddw
    assert p_fix < p_lap

    p4_lap = neck_lap(c4, c8, c16)[0]
    p4_ddw = neck_ddw(c4, c8, c16)[0]
    p4_fix = neck_fix(c4, c8, c16)[0]
    assert not torch.allclose(p4_lap, p4_ddw)
    assert p4_fix.shape == p4_lap.shape

    for mod, out in ((neck_ddw, p4_ddw), (neck_fix, p4_fix)):
        mod.zero_grad()
        out.sum().backward(retain_graph=True)
        dead = [n for n, p in mod.named_parameters() if p.requires_grad and p.grad is None]
        assert not dead, f"Dead grads in {mod.decomp_mode}: {dead}"


def test_lfi_canonical_configs_valid_sub100k_and_zero_dead_grads() -> None:
    repo = Path(__file__).resolve().parents[1]
    cfg_paths = [
        repo / "configs/benchmarks/lfi_sub100k/exp_sha_lfi_dsimr_99k.yaml",
        repo / "configs/benchmarks/lfi_sub100k/exp_sha_sc_lfi_98k.yaml",
        repo / "configs/benchmarks/lfi_sub100k/exp_sha_lfi_s2_99k.yaml",
        repo / "configs/benchmarks/lfi_sub100k/exp_nwpu_lfi_s2_99k.yaml",
        repo / "configs/benchmarks/lfi_sub100k/ablations/fair_baseline_sha_additive_99k.yaml",
        repo / "configs/benchmarks/lfi_sub100k/ablations/fair_baseline_sha_plain_fpn_98k.yaml",
        repo / "configs/benchmarks/lfi_sub100k/ablations/fair_baseline_sha_pafpn_lite_99k.yaml",
        repo / "configs/benchmarks/lfi_sub100k/ablations/fair_baseline_sha_bifpn_99k.yaml",
        repo / "configs/benchmarks/lfi_sub100k/ablations/fair_baseline_sha_pafpn_100k.yaml",
        repo / "configs/benchmarks/lfi_sub100k/ablations/abl_sha_lfi_no_laplacian_99k.yaml",
        repo / "configs/benchmarks/lfi_sub100k/ablations/abl_sha_lfi_direct_dw_99k.yaml",
        repo / "configs/benchmarks/lfi_sub100k/ablations/abl_sha_lfi_fixed_hf_99k.yaml",
        repo / "configs/benchmarks/lfi_sub100k/ablations/abl_sha_lfi_dw_subpixel_99k.yaml",
        repo / "configs/benchmarks/lfi_sub100k/ablations/abl_sha_lfi_with_radon_ot_99k.yaml",
        repo / "configs/benchmarks/lfi_sub100k/ablations/abl_nwpu_lfi_dw_subpixel_99k.yaml",
    ]
    for p in cfg_paths:
        cfg = load_config(p)
        validate_v3_config(cfg)
        m_dict = dict(cfg["model"])
        m_dict["pretrained"] = False
        m_cfg = RMRv3Config.from_dict(m_dict)
        model = RMRv3(m_cfg)
        n_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
        assert n_params <= 104_441, f"{p.name} params={n_params}"
        if "pafpn" not in p.name:
            assert n_params < 100_000, f"{p.name} params={n_params}"

        x = torch.randn(2, 3, 128, 128)
        pts = [torch.tensor([[20.0, 20.0], [60.0, 60.0]]), torch.tensor([[30.0, 40.0]])]
        target_y = torch.zeros(2, 1, 64, 64)
        out = model(x)
        loss_cfg = RMRv3LossConfig.from_dict(cfg["loss"])
        loss_dict = compute_rmr_v3_losses(
            outputs=out,
            target_y=target_y,
            cfg=loss_cfg,
            points=pts,
        )
        loss = loss_dict["total"]
        assert torch.isfinite(loss)
        model.zero_grad()
        loss.backward()
        dead = [n for n, param in model.named_parameters() if param.requires_grad and param.grad is None]
        assert not dead, f"{p.name} dead grads: {dead}"
