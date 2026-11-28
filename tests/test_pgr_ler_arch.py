from __future__ import annotations

import math
from pathlib import Path
import pytest
import torch
import yaml

from rmr_core.backbones import TimmPyramidBackbone
from rmr_core.ler_head import LERSubpixelHead
from rmr_v3.config.validator import validate_v3_config
from rmr_v3.engine import build_optimizer, make_loss_cfg, make_model
from rmr_v3.losses import compute_rmr_v3_losses

REPO_ROOT = Path(__file__).resolve().parent.parent
PGR_LER_DIR = REPO_ROOT / "configs" / "benchmarks" / "pgr_ler"


def test_os8_hdc_dilations_and_stage2_truncation() -> None:
    """Verify gridding-free (1, 2, 3, 1, 2) dilations on 0.5x and (1, 2, 3) on 1.0x b2[:4]."""
    bb05 = TimmPyramidBackbone(
        model_name="mobilenetv4_conv_small_050.e3000_r224_in1k",
        pretrained=False,
        target_reductions=(4, 8, 16),
        stride_mode="os8_hdc",
    )
    b2_05 = list(bb05.backbone.blocks[2].children())
    assert b2_05[0].dw_mid.conv.stride == (1, 1)
    assert [b.dw_mid.conv.dilation for b in b2_05[1:5]] == [(1, 1), (2, 2), (3, 3), (1, 1)]
    assert b2_05[5].dw_start.conv.dilation == (2, 2)

    bb10 = TimmPyramidBackbone(
        model_name="mobilenetv4_conv_small.e2400_r224_in1k",
        pretrained=False,
        target_reductions=(4, 8, 16),
        stride_mode="os8_hdc",
        max_stage2_blocks=4,
    )
    b2_10 = list(bb10.backbone.blocks[2].children())
    assert len(b2_10) == 4
    assert sum(p.numel() for p in bb10.parameters()) == 202272
    assert [b.dw_mid.conv.dilation for b in b2_10[1:4]] == [(1, 1), (2, 2), (3, 3)]


def test_ler_head_calibration_zero_anchor_and_open_tail() -> None:
    """Verify Step-0 density calibration, exact zero anchor, and unbounded upper tail."""
    m0 = 0.015763
    init_bias = math.log(math.expm1(m0))
    head = LERSubpixelHead(width=32, init_bias=init_bias, use_persp_prior=False)
    head.eval()
    with torch.no_grad():
        head.bin_proj.weight.zero_()
        head.res_proj.weight.zero_()
        head.subpixel_proj.weight.zero_()
        _, y0 = head.forward_with_logits(torch.zeros(1, 32, 8, 8))
        assert abs(float(y0.mean().item()) - (m0 / 4.0)) < 1e-5

        # Exact zero background when bin 0 dominates
        head.bin_proj.bias.fill_(-50.0)
        head.bin_proj.bias[0] = 50.0
        _, y_zero = head.forward_with_logits(torch.zeros(1, 32, 8, 8))
        assert float(y_zero.max().item()) < 1e-12

        # Open-ended residual tail when highest anchor dominates
        head.bin_proj.bias.fill_(-50.0)
        head.bin_proj.bias[-1] = 50.0
        head.res_proj.bias.fill_(5.0)
        _, y_dense = head.forward_with_logits(torch.zeros(1, 32, 8, 8))
        assert float(y_dense.mean().item()) > 5.5


def test_validator_rejects_mismatched_persp_prior_config() -> None:
    """Fail closed if ler_use_persp_prior=True is paired with non-PGR neck."""
    cfg = yaml.safe_load((PGR_LER_DIR / "testA_sha_os8_plainfpn_ler_98k.yaml").read_text(encoding="utf-8"))
    cfg["model"]["ler_use_persp_prior"] = True
    with pytest.raises(ValueError, match="ler_use_persp_prior"):
        validate_v3_config(cfg)


@pytest.mark.parametrize(
    "cfg_name",
    [
        "testA_sha_os8_plainfpn_ler_98k.yaml",
        "testB0_sha_os8hdc_plainfpn_98k.yaml",
        "testB_sha_os8hdc_pgrfpn_dsimr_103k.yaml",
        "testC1_noprior_sha_pgr_ler_nano_104k.yaml",
        "testC1_sha_pgr_ler_nano_104k.yaml",
        "baseline_sha_s100_os8_plainfpn_224k.yaml",
        "testC2_sha_pgr_ler_tiny_225k.yaml",
        "testC1_shb_pgr_ler_nano_104k.yaml",
        "testC1_qnrf_pgr_ler_nano_104k.yaml",
    ],
)
def test_pgr_ler_configs_valid_sub300k_odd_shape_and_zero_dead_grads(cfg_name: str) -> None:
    """Validate schema, <300k param budget, odd-size forward, and 0 dead grads with full loss."""
    cfg = yaml.safe_load((PGR_LER_DIR / cfg_name).read_text(encoding="utf-8"))
    validate_v3_config(cfg)
    cfg["model"]["pretrained"] = False

    model, _ = make_model(cfg)
    loss_cfg = make_loss_cfg(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params <= 300000, f"{cfg_name} exceeded 300k budget: {n_params}"

    out_odd = model(torch.randn(1, 3, 253, 317))
    assert out_odd.y.shape == (1, 1, 127, 159)
    assert torch.isfinite(out_odd.y).all()

    x = torch.randn(2, 3, 128, 128)
    target_y = torch.zeros(2, 1, 64, 64)
    pts0 = torch.tensor([[20.0, 30.0], [64.0, 64.0], [90.0, 40.0], [50.0, 100.0]], dtype=torch.float32)
    pts1 = torch.zeros((0, 2), dtype=torch.float32)
    for px, py in pts0:
        target_y[0, 0, int(py // 2), int(px // 2)] += 1.0

    model.zero_grad(set_to_none=True)
    out = model(x)
    losses = compute_rmr_v3_losses(out, target_y, loss_cfg, points=[pts0, pts1])
    assert torch.isfinite(losses["total"])
    losses["total"].backward()

    dead = [
        name for name, p in model.named_parameters()
        if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum().item() == 0.0)
    ]
    assert not dead, f"{cfg_name} had dead/non-finite gradients: {dead}"


def test_pgr_ler_optimizer_groups_and_jit_trace() -> None:
    """Verify beta_persp is weight-decay exempt and Test C1/C2 pass TorchScript trace."""
    class _Wrap(torch.nn.Module):
        def __init__(self, m: torch.nn.Module) -> None:
            super().__init__()
            self.m = m

        def forward(self, inp: torch.Tensor) -> torch.Tensor:
            return self.m(inp)["y"]

    x = torch.randn(1, 3, 256, 256)
    for cfg_name in ("testC1_sha_pgr_ler_nano_104k.yaml", "testC2_sha_pgr_ler_tiny_225k.yaml"):
        cfg = yaml.safe_load((PGR_LER_DIR / cfg_name).read_text(encoding="utf-8"))
        cfg["model"]["pretrained"] = False
        model, _ = make_model(cfg)
        opt = build_optimizer(model, cfg, lr_init=cfg["train"]["lr"])
        no_decay_ids = {
            id(p) for g in opt.param_groups if g["weight_decay"] == 0.0 for p in g["params"]
        }
        assert id(model.fine_head.beta_persp) in no_decay_ids

        wrapper = _Wrap(model).eval()
        with torch.no_grad():
            y_ref = wrapper(x)
            y_traced = torch.jit.trace(wrapper, x, strict=False)(x)
        assert torch.allclose(y_ref, y_traced, atol=1e-5)


def test_ler_head_shb_sparsity_and_fp32_output_under_fp16_input() -> None:
    """Verify SHB Step-0 parity, FP32 output under FP16 input, and active tail gradient."""
    m0_shb = 0.0022
    init_bias = math.log(math.expm1(m0_shb))
    head = LERSubpixelHead(width=32, init_bias=init_bias, use_persp_prior=False)
    head.eval()
    with torch.no_grad():
        head.bin_proj.weight.zero_()
        head.res_proj.weight.zero_()
        head.subpixel_proj.weight.zero_()
        _, y0 = head.forward_with_logits(torch.zeros(1, 32, 8, 8))
        assert abs(float(y0.mean().item()) - (m0_shb / 4.0)) < 1e-5
        probs0 = torch.softmax(head.bin_proj.bias, dim=0)
        assert float(probs0[-1].item()) > 1e-5

    f_fp16 = torch.randn(1, 32, 8, 8).half()
    head_fp16 = LERSubpixelHead(width=32, init_bias=init_bias, use_persp_prior=False).half()
    _, y_out = head_fp16.forward_with_logits(f_fp16)
    assert y_out.dtype == torch.float32
    y_out.sum().backward()
    assert head_fp16.res_proj.bias.grad is not None
    assert abs(float(head_fp16.res_proj.bias.grad.item())) > 1e-6
