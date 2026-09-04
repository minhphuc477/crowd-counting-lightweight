"""Profile MICF model parameters, Conv2d MACs, latency, and peak memory."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import timm
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


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def profile_model_efficiency(
    model: nn.Module,
    input_resolution: int = 256,
    batch_size: int = 1,
    device_name: str | None = None,
    warmup_iters: int = 20,
    measure_iters: int = 100,
) -> dict:
    if input_resolution <= 0 or batch_size <= 0:
        raise ValueError("input_resolution and batch_size must be positive")
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device).eval()
    sample = torch.randn(batch_size, 3, input_resolution, input_resolution, device=device)
    convolution_macs: list[int] = []
    handles = []

    def convolution_hook(module: nn.Conv2d, _inputs, output):
        out_height, out_width = output.shape[-2:]
        convolution_macs.append(int(
            batch_size * out_height * out_width * module.out_channels
            * (module.in_channels // module.groups)
            * module.kernel_size[0] * module.kernel_size[1]
        ))

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(convolution_hook))
    with torch.no_grad():
        model(sample)
    for handle in handles:
        handle.remove()
    total_macs = int(sum(convolution_macs))

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in range(warmup_iters):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency = []
        for _ in range(measure_iters):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latency.append((time.perf_counter() - start) * 1000.0)

    git_sha = None
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        pass

    runtime = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "timm": str(timm.__version__),
        "cuda": str(torch.version.cuda) if torch.cuda.is_available() else None,
        "cudnn": str(torch.backends.cudnn.version()) if torch.cuda.is_available() and torch.backends.cudnn.version() is not None else None,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "git_sha": git_sha,
    }

    data = np.asarray(latency)
    peak_allocated = (
        float(torch.cuda.max_memory_allocated(device) / (1024**2))
        if device.type == "cuda" else 0.0
    )
    peak_reserved = (
        float(torch.cuda.max_memory_reserved(device) / (1024**2))
        if device.type == "cuda" else 0.0
    )
    median = float(np.median(data))
    return {
        "params_total": count_parameters(model),
        "params_trainable": count_trainable_parameters(model),
        "input_resolution": f"{input_resolution}x{input_resolution}",
        "batch_size": batch_size,
        "conv_macs": total_macs,
        "conv_gmacs": total_macs / 1e9,
        "conv_flops_multiply_add_2": 2 * total_macs,
        "flops_note": "Conv2d only; multiply-add counted as two FLOPs",
        "latency_device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "latency_median_ms": median,
        "latency_p95_ms": float(np.percentile(data, 95)),
        "throughput_images_per_second": batch_size * 1000.0 / median,
        "peak_allocated_mb": peak_allocated,
        "peak_reserved_mb": peak_reserved,
        "runtime": runtime,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot_micf/b8.yaml")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--measure-iters", type=int, default=100)
    parser.add_argument("--output", default="runs/model_profile.json")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    model = build_model_from_config(cfg)
    profile = profile_model_efficiency(
        model,
        input_resolution=args.resolution,
        batch_size=args.batch_size,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(profile, handle, indent=2)
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
