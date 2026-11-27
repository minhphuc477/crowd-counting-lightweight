from __future__ import annotations

from pathlib import Path
import pytest
import torch
import yaml

from rmr_core.backbones import TimmPyramidBackbone
from rmr_core.phase_head import S2DPhaseDSIMRHead
from rmr_v3.config.validator import validate_v3_config
from rmr_v3.engine import build_optimizer, make_loss_cfg, make_model
from rmr_v3.losses import compute_rmr_v3_losses

REPO_ROOT = Path(__file__).resolve().parent.parent
ARCH_V2_DIR = REPO_ROOT / "configs" / "benchmarks" / "arch_v2"


def test_all_core_and_v3_files_under_450_lines() -> None:
    """Enforce the 450-line hard limit across every .py file in rmr_core and rmr_v3."""
    violations = []
    for folder in ("rmr_core", "rmr_v3"):
        for py_file in sorted((REPO_ROOT / folder).rglob("*.py")):
            n_lines = len(py_file.read_bytes().splitlines())
            if n_lines > 450:
                violations.append((str(py_file.relative_to(REPO_ROOT)), n_lines))
    assert not violations, f"Files exceeding 450 lines: {violations}"


@pytest.mark.parametrize("stride_mode", ["standard", "split_s8_s16", "dilated_os8"])
def test_backbone_stride_modes_preserve_weights_and_shapes(stride_mode: str) -> None:
    """Verify 100% parameter preservation and spatial shapes on odd & even inputs."""
    bb = TimmPyramidBackbone(
        model_name="mobilenetv4_conv_small_050.e3000_r224_in1k",
        pretrained=False,
        target_reductions=(2, 4, 8, 16),
        stride_mode=stride_mode,
    )
    total_bb_params = sum(p.numel() for p in bb.parameters())
    assert total_bb_params == 87568, f"Expected 87568 backbone params, got {total_bb_params}"

    for h, w in ((512, 512), (256, 256), (253, 317)):
        x = torch.randn(1, 3, h, w)
        c2, c4, c8, c16 = bb(x)
        assert c2.shape[-2:] == ((h + 1) // 2, (w + 1) // 2)
        assert c4.shape[-2:] == ((h + 3) // 4, (w + 3) // 4)
        assert c8.shape[-2:] == ((h + 7) // 8, (w + 7) // 8)
        if stride_mode == "dilated_os8":
            assert c16.shape[-2:] == ((h + 7) // 8, (w + 7) // 8)
        else:
            assert c16.shape[-2:] == ((h + 15) // 16, (w + 15) // 16)


def test_s2d_phase_head_contrast_and_quadrant_routing() -> None:
    """Verify zero contrast on constant background and positive contrast on impulses."""
    head = S2DPhaseDSIMRHead(width=32, c2_channels=32, dsimr_init_z_s=0.0)
    f4 = torch.zeros(1, 32, 16, 16)
    c2_const = torch.ones(1, 32, 32, 32)
    c2_impulse = c2_const.clone()
    c2_impulse[:, :, 5, 7] = 5.0

    _, y_const = head.forward_with_logits(f4, c2=c2_const)
    _, y_imp = head.forward_with_logits(f4, c2=c2_impulse)
    assert y_const.shape == (1, 1, 32, 32)
    assert y_imp.shape == (1, 1, 32, 32)
    assert not torch.allclose(y_const, y_imp)
    assert head._last_support_logit is not None
    assert head._last_support_logit.shape == (1, 1, 16, 16)


@pytest.mark.parametrize(
    "cfg_name",
    [
        "exp1a_sha_split_s8s16_plainfpn_98k.yaml",
        "exp1b_sha_dilated_os8_plainfpn_98k.yaml",
        "exp2_sha_s2d_phase_plainfpn_99k.yaml",
        "exp3_sha_sup_bce_plainfpn_98k.yaml",
        "exp4_sha_full_split_s8s16_phase_sup_99k.yaml",
        "exp5_sha_full_os8_phase_sup_99k.yaml",
        "exp4_shb_full_split_s8s16_phase_sup_99k.yaml",
        "exp4_qnrf_full_split_s8s16_phase_sup_99k.yaml",
    ],
)
def test_arch_v2_configs_valid_sub100k_and_zero_dead_gradients(cfg_name: str) -> None:
    """Validate schema, <100k param budget, odd-size forward, and 0 dead grads with full loss."""
    cfg = yaml.safe_load((ARCH_V2_DIR / cfg_name).read_text(encoding="utf-8"))
    validate_v3_config(cfg)
    cfg["model"]["pretrained"] = False

    model, _ = make_model(cfg)
    loss_cfg = make_loss_cfg(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params < 100000, f"{cfg_name} exceeded 100k budget: {n_params}"

    # 1. Odd resolution forward check (simulates arbitrary SHA/QNRF test aspect ratios)
    x_odd = torch.randn(1, 3, 253, 317)
    out_odd = model(x_odd)
    assert out_odd.y.shape == (1, 1, 127, 159)
    assert torch.isfinite(out_odd.y).all()

    # 2. Full loss backward on mixed batch: sample 0 populated (N=12), sample 1 empty (N=0)
    x = torch.randn(2, 3, 128, 128)
    target_y = torch.zeros(2, 1, 64, 64)
    pts0 = torch.tensor([[20.0, 30.0], [64.0, 64.0], [90.0, 40.0], [50.0, 100.0]], dtype=torch.float32)
    pts1 = torch.zeros((0, 2), dtype=torch.float32)
    for px, py in pts0:
        target_y[0, 0, int(py // 2), int(px // 2)] += 1.0

    model.zero_grad(set_to_none=True)
    out = model(x)
    losses = compute_rmr_v3_losses(out, target_y, loss_cfg, points=[pts0, pts1])
    assert torch.isfinite(losses["total"]), f"{cfg_name} non-finite loss: {losses['total']}"
    losses["total"].backward()

    dead = [
        name for name, p in model.named_parameters()
        if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum().item() == 0.0)
    ]
    assert not dead, f"{cfg_name} had dead or non-finite gradients: {dead}"

    # When lambda_support_bce > 0, verify dsimr_proj.bias[0] receives non-zero gradient even on empty-background mix
    if loss_cfg.lambda_support_bce > 0.0:
        b_grad = model.fine_head.dsimr_proj.bias.grad
        assert b_grad is not None and abs(b_grad[0].item()) > 1e-7, f"{cfg_name} z_s bias grad starved: {b_grad}"


def test_optimizer_groups_for_split_backbone_and_gammas() -> None:
    """Verify split backbone blocks get backbone LR and scalar/1D gammas get weight_decay=0."""
    cfg = yaml.safe_load((ARCH_V2_DIR / "exp4_sha_full_split_s8s16_phase_sup_99k.yaml").read_text(encoding="utf-8"))
    cfg["model"]["pretrained"] = False
    model, _ = make_model(cfg)
    opt = build_optimizer(model, cfg, lr_init=cfg["train"]["lr"])

    groups = {g["name"]: g for g in opt.param_groups}
    bb_param_ids = {id(p) for p in groups["backbone_decay"]["params"] + groups["backbone_no_decay"]["params"]}
    no_decay_ids = {id(p) for p in groups["backbone_no_decay"]["params"] + groups["main_no_decay"]["params"]}

    for p in model.encoder.split_s8_blocks.parameters():
        assert id(p) in bb_param_ids
    for p in model.encoder.split_s16_blocks.parameters():
        assert id(p) in bb_param_ids
    assert id(model.fine_head.gamma_c2) in no_decay_ids
    assert id(model.fine_head.gamma_phase) in no_decay_ids
    assert groups["main_no_decay"]["weight_decay"] == 0.0


def test_jit_trace_exportability_exp4_and_exp5() -> None:
    """Verify forward trace compatibility (ONNX/TorchScript readiness) for Exp 4 & Exp 5."""
    class _TraceWrap(torch.nn.Module):
        def __init__(self, m: torch.nn.Module) -> None:
            super().__init__()
            self.m = m

        def forward(self, inp: torch.Tensor) -> torch.Tensor:
            return self.m(inp)["y"]

    x = torch.randn(1, 3, 256, 256)
    for cfg_name in ("exp4_sha_full_split_s8s16_phase_sup_99k.yaml", "exp5_sha_full_os8_phase_sup_99k.yaml"):
        cfg = yaml.safe_load((ARCH_V2_DIR / cfg_name).read_text(encoding="utf-8"))
        cfg["model"]["pretrained"] = False
        model, _ = make_model(cfg)
        model.eval()
        wrapper = _TraceWrap(model).eval()
        with torch.no_grad():
            y_ref = wrapper(x)
            traced = torch.jit.trace(wrapper, x, strict=False)
            y_traced = traced(x)
        assert torch.allclose(y_ref, y_traced, atol=1e-5)
