"""Unit tests for Layer-4 mechanism metrics and Axis A/B/C/D ablation variants."""
from __future__ import annotations

import torch

from rmr_core.mechanism_metrics import (
    compute_erf,
    gate_response_analysis,
    hf_preservation_ratio,
    peak_sharpness,
    per_size_mae,
    spatial_entropy,
)
from rmr_v3.model import RMRv3, RMRv3Config


def test_mechanism_metrics_suite_on_sfi_and_frankenstein() -> None:
    """Verify all 6 Layer-4 mechanism functions execute cleanly with valid mathematical bounds."""
    torch.manual_seed(42)
    cfg = RMRv3Config(
        pretrained=False,
        output_stride=2,
        subpixel_stride2=True,
        subpixel_kernel_size=3,
        feature_width=32,
        max_trainable_params=104441,
        neck_type="sfi_lite",
        sfi_norm_free_hf=True,
        sfi_use_hdc_proj=True,
        head_type="dsimr_subpixel",
        density_adaptive_scale=True,
        density_scale_gamma=0.15,
        enable_regional_head=False,
        enable_solver=False,
        solver_mode="direct",
    )
    model = RMRv3(cfg).eval()

    # 1. hf_preservation_ratio
    f_in = torch.randn(1, 32, 32, 32)
    f_out = f_in + 0.5 * (f_in - torch.nn.functional.avg_pool2d(f_in, 3, 1, 1))
    hf_res = hf_preservation_ratio(f_in, f_out)
    assert hf_res["hf_preservation_ratio"] > 1.0

    # 2. peak_sharpness & 3. spatial_entropy (sharper Gaussian -> higher sharpness, lower entropy)
    yy, xx = torch.meshgrid(torch.arange(64, dtype=torch.float32), torch.arange(64, dtype=torch.float32), indexing="ij")
    pts = torch.tensor([[64.0, 64.0]], dtype=torch.float32)  # center at grid (32, 32) for stride 2
    dm_sharp = torch.exp(-((yy - 32.0).square() + (xx - 32.0).square()) / (2.0 * 1.0 ** 2))
    dm_sharp = dm_sharp / dm_sharp.sum()
    dm_blur = torch.exp(-((yy - 32.0).square() + (xx - 32.0).square()) / (2.0 * 4.0 ** 2))
    dm_blur = dm_blur / dm_blur.sum()

    ps_sharp = peak_sharpness(dm_sharp, pts, stride=2)
    ps_blur = peak_sharpness(dm_blur, pts, stride=2)
    assert ps_sharp["mean_sharpness"] > ps_blur["mean_sharpness"]

    ent_sharp = spatial_entropy(dm_sharp)
    ent_blur = spatial_entropy(dm_blur)
    assert ent_sharp["spatial_entropy"] < ent_blur["spatial_entropy"]
    assert 0.0 <= ent_sharp["normalized_entropy"] <= 1.0

    # 4. gate_response_analysis
    imgs = [torch.randn(3, 128, 128), torch.randn(3, 128, 128)]
    pts_list = [torch.empty((0, 2)), torch.rand(40, 2) * 128.0]
    gate_res = gate_response_analysis(model, imgs, pts_list, sparse_max=10.0, dense_min=30.0)
    assert gate_res["available"] is True
    assert gate_res["image_level"]["sparse"]["n"] == 1
    assert gate_res["image_level"]["dense"]["n"] == 1
    assert 0.0 <= gate_res["image_level"]["sparse"]["mean"] <= 1.0

    # 5. per_size_mae (both gt_boxes and knn_proxy paths)
    boxes = [None, torch.tensor([[10.0, 10.0, 20.0, 20.0], [50.0, 50.0, 95.0, 95.0]])]
    pts_sz = [torch.tensor([[30.0, 30.0], [80.0, 80.0]]), torch.tensor([[15.0, 15.0], [72.0, 72.0]])]
    dms_sz = [dm_sharp, dm_sharp]
    sz_proxy = per_size_mae([dm_sharp], [pts_sz[0]], boxes_list=None, stride=2)
    assert sz_proxy["size_source"] == "knn_proxy"
    sz_gt = per_size_mae([dm_sharp], [pts_sz[1]], boxes_list=[boxes[1]], stride=2)
    assert sz_gt["size_source"] == "gt_boxes"
    assert sz_gt["small_count"] == 1 and sz_gt["large_count"] == 1

    # 6. compute_erf
    erf_res = compute_erf(model, input_shape=(1, 3, 128, 128), target_layer="p4", num_samples=2)
    assert 0.0 < erf_res["r50_px"] <= erf_res["r95_px"]
    assert erf_res["erf_map"].shape == (128, 128)


def test_sfi_complementary_hf_gate_invariants() -> None:
    """Verify complementary HF gating adds 0 params, preserves exact zero fixed point, and decouples LF/HF polarity."""
    import pytest
    from rmr_core.necks.sfi import SFILiteFPNNeck
    from rmr_v3.config.validator import validate_v3_config

    torch.manual_seed(7)
    base_kwargs = dict(
        pretrained=False,
        output_stride=2,
        subpixel_stride2=True,
        subpixel_kernel_size=3,
        feature_width=32,
        max_trainable_params=104441,
        neck_type="sfi_lite",
        sfi_norm_free_hf=True,
        sfi_use_hdc_proj=True,
        sfi_hdc_proj_groups=2,
        use_aspp_gap=False,
        head_type="dsimr_subpixel",
        density_adaptive_scale=True,
        density_scale_gamma=0.15,
        enable_regional_head=False,
        enable_solver=False,
        solver_mode="direct",
    )
    m_coupled = RMRv3(RMRv3Config(**base_kwargs, sfi_hf_gate_mode="coupled"))
    m_comp = RMRv3(RMRv3Config(**base_kwargs, sfi_hf_gate_mode="complementary"))
    p_coupled = sum(p.numel() for p in m_coupled.parameters() if p.requires_grad)
    p_comp = sum(p.numel() for p in m_comp.parameters() if p.requires_grad)
    assert p_coupled == p_comp == 101174

    # 1. Exact zero fixed point on flat spatial inputs (zero HF residual -> hf_term == 0)
    neck = SFILiteFPNNeck(
        in_channels=(16, 32, 48),
        width=32,
        norm_free_hf=True,
        use_hdc_proj=True,
        hf_gate_mode="complementary",
    ).eval()
    with torch.no_grad():
        assert torch.tanh(neck.hf_enhance(torch.zeros(1, 32, 32, 32))).abs().max().item() == 0.0  # type: ignore[arg-type]
        _, flat_hf = neck.decompose_frequency(torch.ones(1, 32, 32, 32) * 0.37)
        assert flat_hf[:, :, 1:-1, 1:-1].abs().max().item() == 0.0
        assert torch.tanh(neck.hf_enhance(flat_hf))[:, :, 2:-2, 2:-2].abs().max().item() == 0.0  # type: ignore[arg-type]

    # 2. Complementary polarity: gate4 -> 1 passes up8_to_4 and blocks HF; gate4 -> 0 blocks up8_to_4 and passes HF
    c4_hf = torch.randn(1, 16, 32, 32)
    c8_a, c16_a = torch.randn(1, 32, 16, 16), torch.randn(1, 48, 8, 8)
    c8_b, c16_b = torch.randn(1, 32, 16, 16), torch.randn(1, 48, 8, 8)
    gate_proj = neck.selective_gate[-2]  # type: ignore[index]

    with torch.no_grad():
        # Case gate4 -> 1
        gate_proj.weight.zero_()
        gate_proj.bias.fill_(100.0)
        neck.gamma_hf.fill_(1.0)  # type: ignore[union-attr]
        out_g1_hf1, _, _ = neck(c4_hf, c8_a, c16_a)
        neck.gamma_hf.fill_(0.0)  # type: ignore[union-attr]
        out_g1_hf0, _, _ = neck(c4_hf, c8_a, c16_a)
        out_g1_ctx_b, _, _ = neck(c4_hf, c8_b, c16_b)
        assert (out_g1_hf1 - out_g1_hf0).abs().max() < 1e-6
        assert (out_g1_hf0 - out_g1_ctx_b).abs().max() > 1e-2

        # Case gate4 -> 0
        gate_proj.bias.fill_(-100.0)
        neck.gamma_hf.fill_(1.0)  # type: ignore[union-attr]
        out_g0_hf1, _, _ = neck(c4_hf, c8_a, c16_a)
        neck.gamma_hf.fill_(0.0)  # type: ignore[union-attr]
        out_g0_hf0, _, _ = neck(c4_hf, c8_a, c16_a)
        out_g0_ctx_b, _, _ = neck(c4_hf, c8_b, c16_b)
        assert (out_g0_hf1 - out_g0_hf0).abs().max() > 1e-2
        assert (out_g0_hf0 - out_g0_ctx_b).abs().max() < 1e-6

    # 3. Validation errors for invalid hf_gate_mode
    with pytest.raises(ValueError, match="Unsupported hf_gate_mode"):
        SFILiteFPNNeck(hf_gate_mode="invalid_mode")
    with pytest.raises(ValueError, match="sfi_hf_gate_mode"):
        validate_v3_config({"model": {"sfi_hf_gate_mode": "invalid_mode"}})


def test_lfi_sub100k_no_gate_invariants() -> None:
    """Verify Option-1 LFI-Lite + DSIMR stays strictly <100k params with zero dead gradients."""
    torch.manual_seed(11)
    cfg = RMRv3Config(
        pretrained=False,
        output_stride=2,
        subpixel_stride2=True,
        subpixel_kernel_size=3,
        feature_width=32,
        max_trainable_params=104441,
        neck_type="sfi_lite",
        sfi_norm_free_hf=True,
        sfi_use_hdc_proj=True,
        sfi_hdc_proj_groups=2,
        sfi_use_selective_gate=False,
        use_aspp_gap=False,
        head_type="dsimr_subpixel",
        density_adaptive_scale=True,
        density_scale_gamma=0.15,
        enable_regional_head=False,
        enable_solver=False,
        solver_mode="direct",
    )
    model = RMRv3(cfg)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert params == 99766
    assert params < 100000
    assert model.fusion.selective_gate is None

    # Forward + backward sanity: zero dead trainable parameters
    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    y0 = out["y0"] if isinstance(out, dict) else out
    y0.sum().backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Dead gradient on {name}"
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient on {name}"

    # Gate capture reports unavailable while HF ratio stays valid
    gate_res = gate_response_analysis(
        model,
        [torch.randn(3, 128, 128), torch.randn(3, 128, 128)],
        [torch.empty((0, 2)), torch.rand(40, 2) * 128.0],
        sparse_max=10.0,
        dense_min=30.0,
    )
    assert gate_res["available"] is False
    assert torch.isfinite(torch.tensor(gate_res["mean_hf_preservation_ratio"]))
