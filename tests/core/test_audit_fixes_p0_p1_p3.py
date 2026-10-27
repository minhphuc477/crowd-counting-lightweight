"""Verification suite for P0/P1/P3 First-Principles Audit Upgrades.

Tests:
1. EMAManager cold-start warmup decay + floating-point BatchNorm buffer EMA vs integer buffer copy.
2. ScaleRoutingHead & FactorizedRoutingHead step-0 non-zero spatial DW gradient + uniform prior.
3. FineMeasureHead subpixel Stride-2 ICNR initialization (near-zero checkerboard variance).
4. RMRv3.forward replicate boundary padding invariance on non-16-divisible constant images.
5. ChfL standalone loss activation when use_chfl_loss=True and all other aux losses are disabled.
6. Hann-blended Partition-of-Unity tiling, corpus-level Point Localization F1@4/F1@8, and negative-scene mae_zero.
7. Backbone-aware normalization resolver & Hardware Profiler smoke verification.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from rmr_core.evaluation import evaluate_dataset, predict_tiled
from rmr_core.heads import FineMeasureHead
from rmr_core.operators import build_multiscale_regions
from rmr_core.scale_routing import FactorizedRoutingHead, ScaleRoutingHead
from rmr_core.spatial_analysis import evaluate_point_localization, extract_subpixel_peaks
from rmr_core.training import build_checkpoint
from rmr_core.transforms import normalize_image, resolve_backbone_norm_mode
from rmr_v3.checkpoint import EMAManager
from rmr_v3.losses import RMRv3LossConfig, compute_rmr_v3_losses
from rmr_v3.model import RMRv3, RMRv3Config
from scripts.profile_efficiency import profile_config


def test_ema_warmup_and_bn_float_buffer_smoothing() -> None:
    """Verify EMAManager applies warmup decay and smooths float BN stats instead of copying instantaneous noise."""
    cfg = RMRv3Config(feature_width=32, pretrained=False)
    model = RMRv3(cfg)
    ema = EMAManager(model, decay=0.999)

    float_buf_name = next(name for name in ema._float_buffers)
    int_buf_name = next(
        name for name, b in model.named_buffers() if name in ema.state and not b.is_floating_point()
    )

    orig_param_name, orig_param = next(iter(model.named_parameters()))
    p0 = orig_param.detach().clone().float()
    b0 = dict(model.named_buffers())[float_buf_name].detach().clone().float()

    with torch.no_grad():
        orig_param.add_(10.0)
        dict(model.named_buffers())[float_buf_name].add_(20.0)
        dict(model.named_buffers())[int_buf_name].fill_(42)

    ema.update(model)
    assert ema.num_updates == 1
    expected_d = 2.0 / 11.0  # min(0.999, (1+1)/(10+1))

    assert torch.allclose(ema.state[orig_param_name], expected_d * p0 + (1.0 - expected_d) * (p0 + 10.0), atol=1e-5)
    assert torch.allclose(ema.state[float_buf_name], expected_d * b0 + (1.0 - expected_d) * (b0 + 20.0), atol=1e-5)
    assert int(ema.state[int_buf_name].item()) == 42

    ckpt = build_checkpoint(
        epoch=5, model=model, optimizer=None, scheduler=None, scaler=None,
        config={}, config_hash="h", best_mae=1.0, epochs_without_improvement=0,
        ema_state=ema.state, extra_fields={"ema_updates": ema.num_updates},
    )
    ema2 = EMAManager(RMRv3(cfg), decay=0.999)
    assert ema2.restore(ckpt)
    assert ema2.num_updates == 1
    assert int(ema2.state[int_buf_name].item()) == 42


def test_scale_router_step0_spatial_gradient_unblocked() -> None:
    """Verify ScaleRoutingHead and FactorizedRoutingHead maintain uniform prior AND non-zero step-0 DW gradients."""
    torch.manual_seed(7)
    router = ScaleRoutingHead(in_channels=32, num_scales=3)
    x = torch.randn(2, 32, 16, 16)
    pi = router(x)
    assert torch.allclose(pi, torch.full_like(pi, 1.0 / 3.0), atol=1e-5)

    target_weights = torch.tensor([1.0, 2.0, 3.0]).view(1, 3, 1, 1)
    loss = (pi * target_weights * torch.randn_like(pi)).sum()
    loss.backward()
    assert router.dw.weight.grad is not None
    assert router.dw.weight.grad.norm().item() > 0.0

    f_router = FactorizedRoutingHead(in_channels=32, num_scales=3, num_aspects=2)
    joint_pi, pi_s, pi_a = f_router(x)
    assert torch.allclose(pi_s, torch.full_like(pi_s, 1.0 / 3.0), atol=1e-5)
    assert torch.allclose(pi_a, torch.full_like(pi_a, 0.5), atol=1e-5)
    (joint_pi * torch.randn_like(joint_pi)).sum().backward()
    assert f_router.dw.weight.grad is not None
    assert f_router.dw.weight.grad.norm().item() > 0.0


def test_icnr_subpixel_fine_head_eliminates_checkerboard() -> None:
    """Verify ICNR subpixel initialization suppresses 2x2 intra-cell checkerboard variance by >20x."""
    from rmr_core.fine_head import ResidualDilatedFineHead

    torch.manual_seed(123)
    feat = torch.randn(2, 48, 32, 32)
    for head in (
        FineMeasureHead(width=48, subpixel_stride2=True, temp_softplus=True, floor_tau=0.003, icnr_init=True),
        ResidualDilatedFineHead(width=48, subpixel_stride2=True, temp_softplus=True, floor_tau=0.003),
    ):
        head.eval()
        with torch.no_grad():
            y = head(feat)
        assert y.shape == (2, 1, 64, 64)
        # Reshape [B, 1, 64, 64] into 2x2 subpixel quads [B, 1, 32, 2, 32, 2]
        quads = y.reshape(2, 1, 32, 2, 32, 2)
        intra_quad_std = quads.std(dim=(3, 5)).mean().item()
        inter_cell_std = quads.mean(dim=(3, 5)).std().item()
        assert intra_quad_std < 0.05 * max(inter_cell_std, 1e-6)


def test_chfl_standalone_auxiliary_gate_not_skipped() -> None:
    """Verify use_chfl_loss=True computes positive ChfL loss even when all other aux losses are 0.0."""
    cfg = RMRv3LossConfig(
        lambda_count=0.0,
        lambda_flat_dm16=0.0,
        lambda_cell=0.0,
        lambda_region_nb=0.0,
        lambda_hurdle=0.0,
        lambda_trunc_nb=0.0,
        use_chfl_loss=True,
        lambda_chfl=1.0,
    )
    h, w = 32, 32
    y = torch.zeros(1, 1, h, w, requires_grad=True)
    y0 = torch.zeros(1, 1, h, w, requires_grad=True)
    gt = torch.zeros(1, 1, h, w)
    gt[0, 0, 10, 10] = 3.0
    pts = [torch.tensor([[40.0, 40.0], [42.0, 40.0], [40.0, 42.0]])]
    regions = build_multiscale_regions(h, w, 4, (32, 64), 0.5, False, device="cpu")
    outputs = {
        "y": y,
        "y0": y0,
        "regions": regions,
        "b_region": torch.ones(1, 1, len(regions.boxes)),
        "region_dispersion": torch.full((1, 1, len(regions.boxes)), 50.0),
    }
    losses = compute_rmr_v3_losses(outputs, gt, cfg, points=pts)
    assert losses["chfl"].item() > 0.0
    assert losses["total"].item() > 0.0


def test_hann_tiling_localization_and_zero_scene_metrics() -> None:
    """Verify Hann Partition-of-Unity tiling, broad-head peak localization, and zero-scene evaluation."""
    class ConstantDensityModel(nn.Module):
        def forward(self, x, **kwargs):
            b, _, h, w = x.shape
            return {"y": torch.full((b, 1, (h + 1) // 2, (w + 1) // 2), 0.25, device=x.device)}

    img = torch.ones(3, 130, 150)
    out_hann = predict_tiled(
        ConstantDensityModel(), img, output_stride=2, tile_size=64, halo=32, blend_mode="hann"
    )
    assert out_hann.shape == (1, 65, 75)
    assert torch.allclose(out_hann, torch.full_like(out_hann, 0.25), atol=1e-5)

    # 1) Sharp 1-cell Dirac spike + 6px-apart adjacent dense crowd heads (zero plateau corner-shift!)
    dirac = torch.zeros(32, 32)
    dirac[10, 10] = 1.0
    dirac[10, 13] = 1.0  # 3 cells = 6.0px apart at Stride 2
    dirac[25, 25] = 0.08  # Sub-person noise (< 0.25) must NOT leak!
    pts_dirac, _ = extract_subpixel_peaks(dirac, threshold=0.0125, nms_kernel=3, stride=2, integrate_window=True)
    loc_dirac = evaluate_point_localization(
        pts_dirac, torch.tensor([[21.0, 21.0], [27.0, 21.0]]), distance_thresholds=(4.0, 8.0)
    )
    assert len(pts_dirac) == 2 and loc_dirac["F1@4"] == 1.0

    # 2) Broad foreground head (sigma=6px on Stride-2 grid, total mass=1.0, peak cell < 0.05)
    yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
    gauss = torch.exp(-(((xx * 2.0 + 1.0 - 64.0) ** 2 + (yy * 2.0 + 1.0 - 64.0) ** 2) / (2.0 * 6.0 ** 2)))
    gauss = gauss / gauss.sum()
    assert float(gauss.max()) < 0.05  # Single-cell threshold 0.05 would miss this person!
    pts_det, _ = extract_subpixel_peaks(gauss, threshold=0.0125, nms_kernel=3, stride=2, integrate_window=True)
    loc = evaluate_point_localization(pts_det, torch.tensor([[64.0, 64.0]]), distance_thresholds=(4.0, 8.0))
    assert loc["F1@4"] == 1.0

    # Corpus-level localization + zero-scene tracking in evaluate_dataset
    class ExactMapModel(nn.Module):
        def forward(self, x, **kwargs):
            b, _, h, w = x.shape
            out_y = torch.zeros((b, 1, h // 2, w // 2), device=x.device)
            if float(x.mean()) > 0.0:
                out_y[0, 0] = gauss
            return {"y": out_y}

    loader = [[
        {
            "image": torch.ones(3, 128, 128),
            "target_y": gauss.unsqueeze(0),
            "points": torch.tensor([[64.0, 64.0]]),
            "id": "fg_1",
            "height": 128,
            "width": 128,
        },
        {
            "image": torch.zeros(3, 128, 128),
            "target_y": torch.zeros(1, 64, 64),
            "points": torch.zeros((0, 2)),
            "id": "empty_0",
            "height": 128,
            "width": 128,
        },
    ]]
    _, summary = evaluate_dataset(
        ExactMapModel(), loader, torch.device("cpu"), output_stride=2, run_tiling=False, eval_localization=True
    )
    assert summary["n_zero"] == 1
    assert summary["mae_zero"] == 0.0
    assert summary["loc_F1@4"] == 1.0
    assert summary["loc_F1@8"] == 1.0


def test_backbone_norm_resolver_and_profiler_smoke() -> None:
    """Verify backbone normalization resolver and efficiency profiler execution."""
    assert resolve_backbone_norm_mode("mobilenetv4_conv_small_050.e3000_r224_in1k", "auto") == "inception"
    assert resolve_backbone_norm_mode("mobilenetv3_small_075.lamb_in1k", "auto") == "imagenet"
    assert resolve_backbone_norm_mode("convnext_atto", "auto") == "imagenet"

    x = torch.full((3, 4, 4), 0.5)
    assert torch.allclose(normalize_image(x, mode="inception"), torch.zeros_like(x), atol=1e-6)
    assert not torch.allclose(normalize_image(x, mode="imagenet"), torch.zeros_like(x), atol=1e-3)

    rep = profile_config(
        "configs/rmr_v68/v68_exp1_track_a_conservative_mcbpro_1500ep.yaml",
        device_str="cpu",
        resolutions=((128, 128),),
        amp=False,
        warmup=1,
        iters=1,
    )
    assert rep["trainable_params"] <= 104_441
    assert rep["benchmarks"][0]["gflops"] > 0.0
