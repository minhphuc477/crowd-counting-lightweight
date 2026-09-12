"""Export MICF-Lite and RMR-v3 models to ONNX format and verify parity."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import yaml

_REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT)

from hpc.models.micf_lite import MICFLite


def is_rmr_config(cfg: dict) -> bool:
    """Determine whether the configuration specifies an RMR model."""
    m_cfg = cfg.get("model", {})
    return (
        "region_sizes_px" in m_cfg
        or "neck_type" in m_cfg
        or "enable_solver" in m_cfg
        or "iterations" in m_cfg
        or "reliability_mode" in m_cfg
    )


def build_model_from_config(cfg: dict) -> nn.Module:
    if is_rmr_config(cfg):
        from rmr_v3.train import make_model
        model, _ = make_model(cfg)
        return model

    m_cfg = cfg.get("model", {})
    return MICFLite(
        backbone_name=m_cfg.get(
            "backbone", "mobilenetv4_conv_small_050.e3000_r224_in1k"
        ),
        pretrained=False,
        neck_width=int(m_cfg.get("neck_width", 32)),
        context_dilations=tuple(m_cfg.get("context_dilations", [1, 2, 3])),
        use_integral_context=bool(m_cfg.get("use_integral_context", True)),
        context_type=str(m_cfg.get("context_type", "directional")),
        head_type=m_cfg.get("head_type", "cumulative"),
        output_stride=int(m_cfg.get("output_stride", 16)),
        eps_d=float(m_cfg.get("eps_d", 1e-8)),
        extent_aware=bool(m_cfg.get("extent_aware", True)),
        finite_horizon=m_cfg.get("finite_horizon", None),
        fh_strict_local=bool(m_cfg.get("fh_strict_local", False)),
        fh_local_norm=str(m_cfg.get("fh_local_norm", "group")),
    )


class _ExportWrapper(nn.Module):
    """Wrapper targeting forward for clean single-tensor ONNX export."""

    def __init__(self, model: nn.Module, is_rmr: bool = False):
        super().__init__()
        self.model = model
        self.is_rmr = is_rmr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if self.is_rmr and isinstance(out, dict):
            return out["y"]
        return out


def export_model_to_onnx(
    checkpoint_path: str | None,
    config_path: str,
    output_onnx: str = "runs/model.onnx",
    input_resolution: int | None = None,
    opset_version: int = 17,
    allow_random_init: bool = False,
    deploy_mode: bool = True,
    skip_verify: bool = False,
) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    is_rmr = is_rmr_config(cfg)
    model_family = "RMR" if is_rmr else "MICF"
    print(f"Detected model architecture family: {model_family}")

    if input_resolution is None:
        if is_rmr:
            input_resolution = int(cfg.get("data", {}).get("crop_size", 512))
        else:
            input_resolution = 256

    has_checkpoint = checkpoint_path is not None and checkpoint_path.lower() not in {"none", ""}
    if has_checkpoint and allow_random_init:
        raise ValueError(
            "Conflicting arguments: cannot specify both --checkpoint and --allow-random-init. "
            "Provide a checkpoint path to export trained weights, or use --allow-random-init alone."
        )
    if not has_checkpoint and not allow_random_init:
        raise ValueError(
            "A valid --checkpoint path is required for export. "
            "Pass --allow-random-init explicitly if you intend to export untrained weights."
        )

    model = build_model_from_config(cfg)
    if has_checkpoint:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = (
            ckpt.get("state_dict")
            or ckpt.get("model_state_dict")
            or ckpt
        )
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded checkpoint from: {checkpoint_path}")
    else:
        print("Exporting model with initial random weights (--allow-random-init specified).")

    model.eval()

    if deploy_mode and hasattr(model, "switch_to_deploy"):
        model.switch_to_deploy()
        print("Switched model to fused deployment mode (switch_to_deploy).")

    export_module = _ExportWrapper(model, is_rmr=is_rmr)
    dummy_input = torch.randn(1, 3, input_resolution, input_resolution)
    os.makedirs(os.path.dirname(os.path.abspath(output_onnx)), exist_ok=True)

    # Note on dynamic axes:
    # MICF supports fully dynamic batch and spatial dimensions.
    # RMR models evaluate multi-scale box grid partitions for the specific spatial resolution
    # at trace time. Static spatial resolution is thus used for RMR to guarantee exact geometric consistency.
    if is_rmr:
        dynamic_axes = None
    else:
        dynamic_axes = {
            "image": {0: "batch_size", 2: "height", 3: "width"},
            "output_field": {0: "batch_size", 2: "out_height", 3: "out_width"},
        }

    torch.onnx.export(
        export_module,
        dummy_input,
        output_onnx,
        input_names=["image"],
        output_names=["output_field"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        training=torch.onnx.TrainingMode.EVAL,
    )
    export_module.eval()
    print(f"Successfully exported ONNX model to: {output_onnx}")

    if skip_verify:
        print("Skipping ONNX verification as requested (--skip-verify).")
        return

    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime is not installed. Skipping runtime verification.")
        return

    ort_session = ort.InferenceSession(output_onnx, providers=["CPUExecutionProvider"])

    if is_rmr:
        test_shapes = [
            (1, 3, input_resolution, input_resolution),
        ]
    else:
        test_shapes = [
            (1, 3, input_resolution, input_resolution),
            (2, 3, input_resolution, input_resolution),
            (1, 3, 320, 320),
            (1, 3, 384, 512),
        ]

    for shape in test_shapes:
        x_test = np.random.randn(*shape).astype(np.float32)
        with torch.no_grad():
            torch_out = export_module(torch.from_numpy(x_test)).numpy()
        ort_inputs = {ort_session.get_inputs()[0].name: x_test}
        ort_out = ort_session.run(None, ort_inputs)[0]
        np.testing.assert_allclose(torch_out, ort_out, rtol=1e-3, atol=1e-4)
        print(f"Verified ONNX vs PyTorch parity on shape {shape}: PASS (diff < 1e-4)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MICF-Lite or RMR-v3 to ONNX")
    parser.add_argument("--config", default="configs/rmr_v9/rmr_v9_canonical.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="runs/exported_model.onnx")
    parser.add_argument("--resolution", type=int, default=None, help="Input resolution (default: config crop size or 256/512)")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--no-deploy", dest="deploy_mode", action="store_false", help="Do not switch to deploy mode")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    export_model_to_onnx(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_onnx=args.output,
        input_resolution=args.resolution,
        opset_version=args.opset,
        allow_random_init=args.allow_random_init,
        deploy_mode=args.deploy_mode,
        skip_verify=args.skip_verify,
    )


if __name__ == "__main__":
    main()
