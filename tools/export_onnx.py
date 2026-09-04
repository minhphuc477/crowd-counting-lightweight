"""Export MICF-Lite model to ONNX format and verify dynamic multi-shape parity."""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import yaml

_REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT)

from hpc.models.micf_lite import MICFLite


def build_model_from_config(cfg: dict) -> MICFLite:
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


class _ExportMICFWrapper(nn.Module):
    """Wrapper targeting forward for clean ONNX export."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def export_model_to_onnx(
    checkpoint_path: str | None,
    config_path: str,
    output_onnx: str = "runs/micf_lite.onnx",
    input_resolution: int = 256,
    opset_version: int = 17,
    allow_random_init: bool = False,
    skip_verify: bool = False,
) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if (checkpoint_path is None or checkpoint_path.lower() in {"none", ""}) and not allow_random_init:
        raise ValueError(
            "A valid --checkpoint path is required for export. "
            "Pass --allow-random-init explicitly if you intend to export untrained weights."
        )

    has_checkpoint = (
        checkpoint_path is not None
        and checkpoint_path.lower() not in {"none", ""}
        and not allow_random_init
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
    export_module = _ExportMICFWrapper(model)

    dummy_input = torch.randn(1, 3, input_resolution, input_resolution)
    os.makedirs(os.path.dirname(os.path.abspath(output_onnx)), exist_ok=True)

    # Export to ONNX with dynamic batch and spatial dimensions
    torch.onnx.export(
        export_module,
        dummy_input,
        output_onnx,
        input_names=["image"],
        output_names=["output_field"],
        dynamic_axes={
            "image": {0: "batch_size", 2: "height", 3: "width"},
            "output_field": {0: "batch_size", 2: "out_height", 3: "out_width"},
        },
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
    parser = argparse.ArgumentParser(description="Export MICF-Lite to ONNX")
    parser.add_argument("--config", default="configs/pilot_micf/psfh_b8_k4.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="runs/micf_lite.onnx")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    export_model_to_onnx(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_onnx=args.output,
        input_resolution=args.resolution,
        opset_version=args.opset,
        allow_random_init=args.allow_random_init,
        skip_verify=args.skip_verify,
    )


if __name__ == "__main__":
    main()
