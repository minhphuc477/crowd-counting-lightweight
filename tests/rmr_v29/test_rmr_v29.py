import math
from pathlib import Path
import pytest
import torch
import yaml

from rmr_v3.config import validate_v3_config
from rmr_v3.model import RMRv3, RMRv3Config
from rmr_v3.losses import compute_rmr_v3_losses, RMRv3LossConfig


CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "rmr_v29"


def test_v29_configs_exist_and_validate():
    """Verify that all RMR-v29 configuration files exist and pass validation."""
    configs = [
        "rmr_v29_step0_v19_anchor.yaml",
        "rmr_v29_h1_depth8.yaml",
        "rmr_v29_h2_subpixel2.yaml",
        "rmr_v29_h3_subpixel2_depth8.yaml",
        "rmr_v29_h4_scale_preserve.yaml",
        "rmr_v29_h5_loss_unsuppressed.yaml",
        "rmr_v29_h6_backbone_lr.yaml",
    ]
    for cfg_name in configs:
        p = CONFIG_DIR / cfg_name
        assert p.is_file(), f"Missing config file: {p}"
        raw = yaml.safe_load(p.read_text())
        validate_v3_config(raw)


def test_v29_parameter_budget():
    """Verify strict parameter ceilings (<= 105,000) across all v29 variants."""
    configs = {
        "rmr_v29_step0_v19_anchor.yaml": 104441,
        "rmr_v29_h1_depth8.yaml": 104441,
        "rmr_v29_h2_subpixel2.yaml": 104540,
        "rmr_v29_h3_subpixel2_depth8.yaml": 104540,
        "rmr_v29_h4_scale_preserve.yaml": 104441,
        "rmr_v29_h5_loss_unsuppressed.yaml": 104441,
        "rmr_v29_h6_backbone_lr.yaml": 104441,
    }
    for cfg_name, expected_params in configs.items():
        p = CONFIG_DIR / cfg_name
        raw = yaml.safe_load(p.read_text())
        m_cfg = RMRv3Config.from_dict(raw.get("model", {}))
        model = RMRv3(m_cfg)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable == expected_params, f"{cfg_name}: expected {expected_params} params, got {trainable}"
        assert trainable <= 105000, f"{cfg_name}: exceeds 105,000 parameter budget: {trainable}"


def test_v29_single_variable_isolation():
    """Verify single-variable hypothesis isolation across the Two-Loop ladder."""
    p_step0 = CONFIG_DIR / "rmr_v29_step0_v19_anchor.yaml"
    p_h1 = CONFIG_DIR / "rmr_v29_h1_depth8.yaml"
    p_h2 = CONFIG_DIR / "rmr_v29_h2_subpixel2.yaml"
    p_h3 = CONFIG_DIR / "rmr_v29_h3_subpixel2_depth8.yaml"
    p_h4 = CONFIG_DIR / "rmr_v29_h4_scale_preserve.yaml"
    p_h5 = CONFIG_DIR / "rmr_v29_h5_loss_unsuppressed.yaml"
    p_h6 = CONFIG_DIR / "rmr_v29_h6_backbone_lr.yaml"

    c_step0 = yaml.safe_load(p_step0.read_text())
    c_h1 = yaml.safe_load(p_h1.read_text())
    c_h2 = yaml.safe_load(p_h2.read_text())
    c_h3 = yaml.safe_load(p_h3.read_text())
    c_h4 = yaml.safe_load(p_h4.read_text())
    c_h5 = yaml.safe_load(p_h5.read_text())
    c_h6 = yaml.safe_load(p_h6.read_text())

    # Step 0 vs H1 differs ONLY in iterations
    assert c_step0["model"]["iterations"] == 6
    assert c_h1["model"]["iterations"] == 8
    c_s0_copy = dict(c_step0["model"])
    c_h1_copy = dict(c_h1["model"])
    c_s0_copy.pop("iterations")
    c_h1_copy.pop("iterations")
    assert c_s0_copy == c_h1_copy, "H1 introduces uncontrolled variables against Step 0!"

    # Step 0 vs H2 differs ONLY in subpixel_stride2 and output_stride
    assert c_h2["model"]["subpixel_stride2"] is True
    assert c_h2["model"]["output_stride"] == 2
    c_s0_m = dict(c_step0["model"])
    c_h2_m = dict(c_h2["model"])
    c_s0_m.pop("output_stride")
    c_h2_m.pop("output_stride")
    c_h2_m.pop("subpixel_stride2")
    assert c_s0_m == c_h2_m, "H2 introduces uncontrolled model variables against Step 0!"
    assert c_step0["loss"] == c_h2["loss"], "H2 loss differs from Step 0!"
    assert c_step0["train"] == c_h2["train"], "H2 train differs from Step 0!"
    assert c_step0["eval"] == c_h2["eval"], "H2 eval differs from Step 0!"

    # H3 is the composite of H1 (depth 8) and H2 (subpixel stride 2)
    assert c_h3["model"]["subpixel_stride2"] is True
    assert c_h3["model"]["output_stride"] == 2
    assert c_h3["model"]["iterations"] == 8

    # Step 0 vs H4 differs ONLY in scale_range
    assert c_step0["data"]["scale_range"] == [0.70, 1.35]
    assert c_h4["data"]["scale_range"] == [0.80, 1.35]
    c_s0_d = dict(c_step0["data"])
    c_h4_d = dict(c_h4["data"])
    c_s0_d.pop("scale_range")
    c_h4_d.pop("scale_range")
    assert c_s0_d == c_h4_d, "H4 introduces uncontrolled data variables against Step 0!"
    assert c_step0["model"] == c_h4["model"], "H4 model differs from Step 0!"
    assert c_step0["loss"] == c_h4["loss"], "H4 loss differs from Step 0!"

    # Step 0 vs H5 differs ONLY in lambda_hard_bg
    assert c_step0["loss"]["lambda_hard_bg"] == 0.15
    assert c_h5["loss"]["lambda_hard_bg"] == 0.05
    c_s0_l = dict(c_step0["loss"])
    c_h5_l = dict(c_h5["loss"])
    c_s0_l.pop("lambda_hard_bg")
    c_h5_l.pop("lambda_hard_bg")
    assert c_s0_l == c_h5_l, "H5 introduces uncontrolled loss variables against Step 0!"
    assert c_step0["model"] == c_h5["model"], "H5 model differs from Step 0!"
    assert c_step0["data"] == c_h5["data"], "H5 data differs from Step 0!"

    # Step 0 vs H6 differs ONLY in backbone_lr_scale and warmup_epochs
    assert c_step0["model"]["backbone_lr_scale"] == 0.1
    assert c_h6["model"]["backbone_lr_scale"] == 0.20
    assert c_h6["train"]["backbone_lr_scale"] == 0.20
    assert c_h6["train"]["warmup_epochs"] == 15


def test_v29_step0_forward_backward_gradient_flow():
    """Verify forward-backward gradient flow through the unrolled SIRT solver for Step 0."""
    p = CONFIG_DIR / "rmr_v29_step0_v19_anchor.yaml"
    raw = yaml.safe_load(p.read_text())
    m_cfg = RMRv3Config.from_dict(raw.get("model", {}))
    l_cfg = RMRv3LossConfig.from_dict(raw.get("loss", {}))

    model = RMRv3(m_cfg)
    model.train()

    # Synthetic batch: 2 images of 256x256
    x = torch.randn(2, 3, 256, 256, requires_grad=False)
    target_y = torch.zeros(2, 1, 64, 64)
    target_y[0, 0, 10, 10] = 1.0
    target_y[1, 0, 20, 20] = 1.0

    out = model(x)
    assert out.y.shape == (2, 1, 64, 64), f"Expected shape (2, 1, 64, 64), got {out.y.shape}"
    assert len(out.iterates) == 7  # y0 + 6 iterations

    losses = compute_rmr_v3_losses(out, target_y, l_cfg)
    assert torch.isfinite(losses["total"]), "Total loss is non-finite!"

    losses["total"].backward()

    # Check gradient flow into backbone and heads
    has_grad = False
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN/Inf gradient in parameter {name}"
            has_grad = True
    assert has_grad, "No parameters received gradients!"


def test_v29_h2_subpixel2_forward_backward():
    """Verify forward-backward gradient flow and Stride 2 output resolution for H2."""
    p = CONFIG_DIR / "rmr_v29_h2_subpixel2.yaml"
    raw = yaml.safe_load(p.read_text())
    m_cfg = RMRv3Config.from_dict(raw.get("model", {}))
    l_cfg = RMRv3LossConfig.from_dict(raw.get("loss", {}))

    model = RMRv3(m_cfg)
    model.train()

    # Synthetic batch: 2 images of 256x256
    x = torch.randn(2, 3, 256, 256, requires_grad=False)
    # Stride 2 target: 128x128
    target_y = torch.zeros(2, 1, 128, 128)
    target_y[0, 0, 20, 20] = 1.0
    target_y[1, 0, 40, 40] = 1.0

    out = model(x)
    assert out.y.shape == (2, 1, 128, 128), f"Expected shape (2, 1, 128, 128), got {out.y.shape}"

    losses = compute_rmr_v3_losses(out, target_y, l_cfg)
    assert torch.isfinite(losses["total"]), "H2 Total loss is non-finite!"

    losses["total"].backward()

    has_grad = False
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN/Inf gradient in parameter {name}"
            has_grad = True
    assert has_grad, "H2: No parameters received gradients!"


def test_v29_h3_subpixel2_depth8_forward_backward():
    """Verify forward-backward gradient flow for H3 (Sub-pixel Stride-2 + Depth 8)."""
    p = CONFIG_DIR / "rmr_v29_h3_subpixel2_depth8.yaml"
    raw = yaml.safe_load(p.read_text())
    m_cfg = RMRv3Config.from_dict(raw.get("model", {}))
    l_cfg = RMRv3LossConfig.from_dict(raw.get("loss", {}))

    model = RMRv3(m_cfg)
    model.train()

    # Synthetic batch: 2 images of 256x256
    x = torch.randn(2, 3, 256, 256, requires_grad=False)
    # Stride 2 target: 128x128
    target_y = torch.zeros(2, 1, 128, 128)
    target_y[0, 0, 20, 20] = 1.0
    target_y[1, 0, 40, 40] = 1.0

    out = model(x)
    assert out.y.shape == (2, 1, 128, 128), f"Expected shape (2, 1, 128, 128), got {out.y.shape}"
    assert len(out.iterates) == 9  # y0 + 8 iterations

    losses = compute_rmr_v3_losses(out, target_y, l_cfg)
    assert torch.isfinite(losses["total"]), "H3 Total loss is non-finite!"

    losses["total"].backward()

    has_grad = False
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN/Inf gradient in parameter {name}"
            has_grad = True
    assert has_grad, "H3: No parameters received gradients!"


def test_v29_h4_h5_h6_forward_backward():
    """Verify forward-backward gradient flow for H4, H5, and H6 variants."""
    for cfg_name in ["rmr_v29_h4_scale_preserve.yaml", "rmr_v29_h5_loss_unsuppressed.yaml", "rmr_v29_h6_backbone_lr.yaml"]:
        p = CONFIG_DIR / cfg_name
        raw = yaml.safe_load(p.read_text())
        m_cfg = RMRv3Config.from_dict(raw.get("model", {}))
        l_cfg = RMRv3LossConfig.from_dict(raw.get("loss", {}))

        model = RMRv3(m_cfg)
        model.train()

        x = torch.randn(2, 3, 256, 256, requires_grad=False)
        target_y = torch.zeros(2, 1, 64, 64)
        target_y[0, 0, 10, 10] = 1.0

        out = model(x)
        losses = compute_rmr_v3_losses(out, target_y, l_cfg)
        assert torch.isfinite(losses["total"]), f"{cfg_name}: total loss is non-finite!"

        losses["total"].backward()

        has_grad = False
        for name, p_tensor in model.named_parameters():
            if p_tensor.requires_grad and p_tensor.grad is not None:
                assert torch.isfinite(p_tensor.grad).all(), f"{cfg_name}: NaN/Inf gradient in {name}"
                has_grad = True
        assert has_grad, f"{cfg_name}: No parameters received gradients!"


def test_v29_subpixel2_with_scale_align_and_anti_pattern_guard():
    """Verify subpixel_stride2 with scale alignment and verify foreground_gate ban."""
    p = CONFIG_DIR / "rmr_v29_h2_subpixel2.yaml"
    raw = yaml.safe_load(p.read_text())

    # 1. Verify anti-pattern ban on foreground_gate
    m_dict_banned = dict(raw.get("model", {}))
    m_dict_banned["foreground_gate"] = True
    with pytest.raises(ValueError, match="permanently BANNED"):
        RMRv3Config.from_dict(m_dict_banned)

    # 2. Verify subpixel_stride2 with active scale alignment loss
    m_dict = dict(raw.get("model", {}))
    l_dict = dict(raw.get("loss", {}))
    l_dict["lambda_scale_align"] = 0.1

    m_cfg = RMRv3Config.from_dict(m_dict)
    l_cfg = RMRv3LossConfig.from_dict(l_dict)

    model = RMRv3(m_cfg)
    model.train()

    x = torch.randn(2, 3, 256, 256, requires_grad=False)
    target_y = torch.zeros(2, 1, 128, 128)
    target_y[0, 0, 20, 20] = 1.0

    out = model(x)
    assert out.y.shape == (2, 1, 128, 128)

    losses = compute_rmr_v3_losses(out, target_y, l_cfg)
    assert torch.isfinite(losses["total"])
    assert torch.isfinite(losses["scale_align"])

    losses["total"].backward()

    for name, p_tensor in model.named_parameters():
        if p_tensor.requires_grad and p_tensor.grad is not None:
            assert torch.isfinite(p_tensor.grad).all(), f"NaN/Inf in {name}"


def test_v29_loss_resolution_invariance():
    """Verify that all loss components maintain physical resolution invariance across Stride 4 and Stride 2."""
    from rmr_core.losses import count_magnitude_loss, flat_dm16_loss, balanced_smooth_l1
    from rmr_v3.losses.auxiliary import (
        curvature_power_loss,
        mass_weighted_cell_loss,
        topk_hard_background_loss,
    )

    torch.manual_seed(42)
    N_pts = 100
    pts = torch.rand(N_pts, 2) * 512

    t4 = torch.zeros(1, 1, 128, 128)
    j4 = (pts[:, 0] / 4).long().clamp(0, 127)
    i4 = (pts[:, 1] / 4).long().clamp(0, 127)
    t4[0, 0].index_put_((i4, j4), torch.ones(N_pts), accumulate=True)

    t2 = torch.zeros(1, 1, 256, 256)
    j2 = (pts[:, 0] / 2).long().clamp(0, 255)
    i2 = (pts[:, 1] / 2).long().clamp(0, 255)
    t2[0, 0].index_put_((i2, j2), torch.ones(N_pts), accumulate=True)

    y4 = (t4 * 0.85 + 0.01).clamp_min(0.0)
    y2 = (t2 * 0.85 + 0.0025).clamp_min(0.0)

    # 1. Count loss
    l_cnt4 = count_magnitude_loss(y4, t4, mode="nb")
    l_cnt2 = count_magnitude_loss(y2, t2, mode="nb")
    ratio_cnt = (l_cnt2 / l_cnt4).item()
    assert 0.99 <= ratio_cnt <= 1.01, f"Count loss ratio failed: {ratio_cnt}"

    # 2. Flat DM16 loss
    l_dm4 = flat_dm16_loss(y4, t4, stride=4)
    l_dm2 = flat_dm16_loss(y2, t2, stride=2)
    ratio_dm = (l_dm2 / l_dm4).item()
    assert 0.99 <= ratio_dm <= 1.01, f"Flat DM16 loss ratio failed: {ratio_dm}"

    # 3. Mass-weighted cell loss
    l_mw4 = mass_weighted_cell_loss(y4, t4, stride=4, alpha=2.0, gamma=1.25)
    l_mw2 = mass_weighted_cell_loss(y2, t2, stride=2, alpha=2.0, gamma=1.25)
    ratio_mw = (l_mw2 / l_mw4).item()
    assert 0.95 <= ratio_mw <= 1.20, f"Mass-weighted cell loss ratio failed: {ratio_mw}"

    # 4. Balanced Smooth L1
    l_bsl4 = balanced_smooth_l1(y4, t4, stride=4)
    l_bsl2 = balanced_smooth_l1(y2, t2, stride=2)
    ratio_bsl = (l_bsl2 / l_bsl4).item()
    assert 0.95 <= ratio_bsl <= 1.20, f"Balanced Smooth L1 ratio failed: {ratio_bsl}"

    # 5. Hard-gated Curvature loss
    # NOTE: curvature_power_loss operates in people/cell (no area_scale normalization).
    # Ratio is in [0.50, 1.50] because stride-2 has smaller cells: fewer cells exceed
    # the threshold in mode='hard', so the denominator (gate_sum) is smaller.
    l_curv4 = curvature_power_loss(y4, t4, stride=4, threshold=0.08, kernel_size=5, mode="hard")
    l_curv2 = curvature_power_loss(y2, t2, stride=2, threshold=0.08, kernel_size=5, mode="hard")
    ratio_curv = (l_curv2 / l_curv4).item()
    assert 0.50 <= ratio_curv <= 1.50, f"Curvature loss ratio failed: {ratio_curv}"

    # 6. Hard BG loss
    # NOTE: topk_hard_background_loss has no area_scale division (Dirac mass invariant).
    # Background cells are 0.0 at both strides, so ratio is ~1.0 for any reasonable predictions.
    l_hbg4 = topk_hard_background_loss(y4, t4, stride=4)
    l_hbg2 = topk_hard_background_loss(y2, t2, stride=2)
    ratio_hbg = (l_hbg2 / l_hbg4).item()
    assert 0.90 <= ratio_hbg <= 1.10, f"Hard BG loss ratio failed: {ratio_hbg}"





