import os
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np

from hpc.models.hpc_lite import HPCLite


def export_model_to_onnx(
    checkpoint_path: str,
    config_path: str,
    output_onnx: str = "hpc_lite.onnx",
    input_resolution: int = 448,
    opset_version: int = 17,
):
    """Export HPC-Lite model to ONNX format and verify numerical parity."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    m_cfg = cfg.get("model", {})
    model = HPCLite(
        backbone_name=m_cfg.get("backbone", "mobilenetv4_conv_small_050"),
        pretrained=False,
        neck_width=m_cfg.get("neck_width", 32),
        context_dilations=tuple(m_cfg.get("context_dilations", [1, 2, 3])),
    )
    
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from: {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}, exporting uninitialized weights.")
        
    model.eval()
    
    dummy_input = torch.randn(1, 3, input_resolution, input_resolution)
    
    os.makedirs(os.path.dirname(os.path.abspath(output_onnx)), exist_ok=True)
    
    # Export to ONNX
    torch.onnx.export(
        model,
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
    
    # Numerical parity verification with onnxruntime
    try:
        import onnxruntime as ort
        ort_session = ort.InferenceSession(output_onnx, providers=["CPUExecutionProvider"])
        ort_inputs = {"image": dummy_input.numpy()}
        ort_outs = ort_session.run(None, ort_inputs)
        
        with torch.no_grad():
            torch_out = model(dummy_input).numpy()
            
        diff = np.max(np.abs(torch_out - ort_outs[0]))
        print(f"Numerical parity check: Max absolute difference = {diff:.6e}")
        assert diff < 1e-4, f"Parity error too high: {diff}"
        print("ONNX verification PASSED.")
    except ImportError:
        print("onnxruntime not installed, skipped runtime verification.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="best.pt", help="Path to PyTorch checkpoint")
    parser.add_argument("--config", type=str, default="configs/sha.yaml", help="Path to config YAML")
    parser.add_argument("--output", type=str, default="hpc_lite.onnx", help="Path to save ONNX file")
    parser.add_argument("--resolution", type=int, default=448, help="Input resolution")
    args = parser.parse_args()
    
    export_model_to_onnx(args.checkpoint, args.config, args.output, args.resolution)
