import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import yaml

from hpc.models.factory import assert_checkpoint_compatible, build_model_from_config


class _ExportMassWrapper(nn.Module):
    """Clean wrapper targeting forward_mass for branch-free tracing."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward_mass(x)


def export_model_to_onnx(
    checkpoint_path: str | None,
    config_path: str,
    output_onnx: str = "hpc_lite.onnx",
    input_resolution: int = 448,
    opset_version: int = 17,
):
    """Export HPC-Lite model to ONNX format and verify dynamic multi-shape parity."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = build_model_from_config(cfg)
    if checkpoint_path is not None and checkpoint_path.lower() != "none" and checkpoint_path != "":
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert_checkpoint_compatible(ckpt, cfg)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded checkpoint from: {checkpoint_path}")
    else:
        print("Exporting model with initial weights (no checkpoint specified).")

    model.eval()
    export_module = _ExportMassWrapper(model)

    dummy_input = torch.randn(1, 3, input_resolution, input_resolution)
    os.makedirs(os.path.dirname(os.path.abspath(output_onnx)), exist_ok=True)

    # Export to ONNX with dynamic batch and spatial dimensions
    torch.onnx.export(
        export_module,
        dummy_input,
        output_onnx,
        input_names=["image"],
        output_names=["mass_map"],
        dynamic_axes={
            "image": {0: "batch_size", 2: "height", 3: "width"},
            "mass_map": {0: "batch_size", 2: "out_height", 3: "out_width"},
        },
        opset_version=opset_version,
    )
    print(f"Successfully exported ONNX model to: {output_onnx}")

    # Verify dynamic rectangular shapes & batches with onnxruntime
    try:
        import onnxruntime as ort

        ort_session = ort.InferenceSession(output_onnx, providers=["CPUExecutionProvider"])
        test_shapes = [
            (1, 3, input_resolution, input_resolution),
            (1, 3, 320, 448),
            (1, 3, 317, 411),
            (2, 3, 256, 256),
        ]

        for shape in test_shapes:
            x = torch.randn(*shape)
            with torch.no_grad():
                torch_out = model.forward_mass(x).numpy()
            ort_out = ort_session.run(None, {"image": x.numpy()})[0]

            diff = float(np.max(np.abs(torch_out - ort_out)))
            torch_cnt = torch_out.reshape(torch_out.shape[0], -1).sum(axis=-1)
            ort_cnt = ort_out.reshape(ort_out.shape[0], -1).sum(axis=-1)
            cnt_diff = float(np.max(np.abs(torch_cnt - ort_cnt)))
            print(f"Parity check for shape {shape}: Max map diff = {diff:.6e}, Max count diff = {cnt_diff:.6e}")
            if not np.allclose(torch_out, ort_out, rtol=1e-3, atol=1e-3):
                raise RuntimeError(f"ONNX map parity check failed for shape {shape} with max absolute error: {diff}")
            if not np.allclose(torch_cnt, ort_cnt, rtol=1e-3, atol=0.5):
                raise RuntimeError(f"ONNX count parity check failed for shape {shape} with max count error: {cnt_diff}")

        print("Dynamic ONNX multi-shape verification PASSED.")
    except ImportError:
        print("onnxruntime not installed, skipped runtime verification.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to PyTorch checkpoint (optional)")
    parser.add_argument("--config", type=str, default="configs/ntpc_r4_neural_dtm_tree.yaml", help="Path to config YAML")
    parser.add_argument("--output", type=str, default="hpc_lite.onnx", help="Path to save ONNX file")
    parser.add_argument("--resolution", type=int, default=448, help="Input resolution")
    args = parser.parse_args()

    export_model_to_onnx(args.checkpoint, args.config, args.output, args.resolution)
