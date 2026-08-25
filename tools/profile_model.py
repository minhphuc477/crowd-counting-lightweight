import os
import time
import json
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np

from hpc.models.hpc_lite import HPCLite


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def profile_model_efficiency(
    model: nn.Module,
    input_resolution: int = 448,
    batch_size: int = 1,
    device_name: str = "cuda" if torch.cuda.is_available() else "cpu",
    warmup_iters: int = 50,
    measure_iters: int = 200,
) -> dict:
    """Compute detailed model efficiency profile."""
    device = torch.device(device_name)
    model = model.to(device)
    model.eval()
    
    # 1. Parameter Breakdown
    total_params = count_parameters(model)
    trainable_params = count_trainable_parameters(model)
    backbone_params = count_parameters(model.backbone) if hasattr(model, "backbone") else 0
    neck_params = count_parameters(model.neck) if hasattr(model, "neck") else 0
    head_params = (
        count_parameters(model.head_dw)
        + count_parameters(model.head_norm)
        + count_parameters(model.head_out)
        if hasattr(model, "head_out")
        else 0
    )
    
    # 2. FLOPs / MACs estimation (analytical or forward hooks)
    x = torch.randn(batch_size, 3, input_resolution, input_resolution, device=device)
    
    # Use torchinfo / thop or direct forward pass
    macs_est = 0
    try:
        # Approximate MACs for conv layers via hooks
        conv_macs = []
        def conv_hook(m, inp, out):
            if isinstance(m, nn.Conv2d):
                cin = m.in_channels
                cout = m.out_channels
                kh, kw = m.kernel_size
                groups = m.groups
                oh, ow = out.shape[-2:]
                mac = (cin // groups) * kh * kw * cout * oh * ow * batch_size
                conv_macs.append(mac)
        
        hooks = []
        for mod in model.modules():
            if isinstance(mod, nn.Conv2d):
                hooks.append(mod.register_forward_hook(conv_hook))
                
        with torch.no_grad():
            _ = model(x)
            
        for h in hooks:
            h.remove()
        macs_est = sum(conv_macs)
    except Exception as e:
        print(f"Hook profiling note: {e}")
        
    # 3. Peak memory & Latency benchmark
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
                
    latencies = []
    with torch.no_grad():
        for _ in range(measure_iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms
            
    latencies = np.array(latencies)
    median_latency = float(np.median(latencies))
    p90_latency = float(np.percentile(latencies, 90))
    fps = float(1000.0 / median_latency) * batch_size if median_latency > 0 else 0.0
    
    peak_mem_mb = 0.0
    if device.type == "cuda":
        peak_mem_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
        
    device_desc = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    
    profile = {
        "params_total": total_params,
        "params_trainable": trainable_params,
        "params_deploy": total_params,
        "params_backbone": backbone_params,
        "params_neck": neck_params,
        "params_head": head_params,
        "macs_resolution": f"{input_resolution}x{input_resolution}",
        "macs": macs_est,
        "gmacs": round(macs_est / 1e9, 4),
        "latency_device": device_desc,
        "latency_batch": batch_size,
        "latency_median_ms": round(median_latency, 3),
        "latency_p90_ms": round(p90_latency, 3),
        "fps": round(fps, 2),
        "peak_memory_mb": round(peak_mem_mb, 2),
    }
    return profile


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sha.yaml", help="Path to config YAML")
    parser.add_argument("--resolution", type=int, default=448, help="Input test resolution")
    parser.add_argument("--output", type=str, default="profile.json", help="Path to save output profile JSON")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
        
    m_cfg = cfg.get("model", {})
    model = HPCLite(
        backbone_name=m_cfg.get("backbone", "mobilenetv4_conv_small_050"),
        pretrained=False,
        neck_width=m_cfg.get("neck_width", 32),
        context_dilations=tuple(m_cfg.get("context_dilations", [1, 2, 3])),
        truncate_backbone=m_cfg.get("truncate_backbone", True),
    )
    
    prof = profile_model_efficiency(model, input_resolution=args.resolution)
    print(json.dumps(prof, indent=2))
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(prof, f, indent=2)
