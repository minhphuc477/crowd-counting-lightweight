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
