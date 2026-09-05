import argparse
import contextlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from .model import RMRConfig, RMRCount, count_parameters
from .train import make_model


@torch.no_grad()
def profile_latency(
    model: torch.nn.Module,
    x: torch.Tensor,
    warmup: int = 50,
    iters: int = 200,
    use_amp: bool = False,
) -> dict:
    model.eval()
    device = x.device
    autocast_ctx = (
        torch.amp.autocast("cuda")
        if (use_amp and device.type == "cuda")
        else contextlib.nullcontext()
    )

    with autocast_ctx:
        for _ in range(warmup):
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with autocast_ctx:
                _ = model(x)
            end.record()
            torch.cuda.synchronize(device)
            times.append(start.elapsed_time(end))
    else:
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = model(x)
            times.append((time.perf_counter() - t0) * 1000.0)

    a = np.asarray(times)
    return {
        "latency_ms_mean": float(a.mean()),
        "latency_ms_p50": float(np.quantile(a, 0.50)),
        "latency_ms_p95": float(np.quantile(a, 0.95)),
        "fps_from_mean": float(1000.0 / a.mean()),
    }


def profiler_flops(model: torch.nn.Module, x: torch.Tensor) -> float | None:
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if x.is_cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, with_flops=True) as prof:
            _ = model(x)
        total = sum((evt.flops or 0) for evt in prof.key_averages())
        return float(total)
    except Exception:
        return None


@torch.no_grad()
def measure_clean_peak_memory(
    model: torch.nn.Module,
    x: torch.Tensor,
    use_amp: bool = False,
) -> float | None:
    if not x.is_cuda:
        return None
    model.eval()
    device = x.device
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    autocast_ctx = (
        torch.amp.autocast("cuda")
        if use_amp
        else contextlib.nullcontext()
    )
    with autocast_ctx:
        _ = model(x)
    torch.cuda.synchronize(device)
    return float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Profile RMR-Count models for latency, memory, and complexity.")
    ap.add_argument("--config", default=None, help="Path to YAML config (e.g. configs/rmr/b5_rmr_t2.yaml)")
    ap.add_argument("--variant", default="rmr", choices=["direct", "region_loss", "region_aux", "local_refine", "learned_project", "rmr"])
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default=None, help="Optional output JSON path")
    args = ap.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text())
        model = make_model(cfg).to(device).eval()
        variant = model.variant
        iterations = getattr(model.cfg, "iterations", args.iterations)
    else:
        mcfg = RMRConfig(
            iterations=args.iterations,
            region_sizes_px=(32, 64, 128),
            eta_max=0.20,
            eta_init=0.05,
            residual_clip=5.0,
        )
        model = RMRCount(mcfg, variant=args.variant).to(device).eval()
        variant = args.variant
        iterations = args.iterations

    x = torch.randn(1, 3, args.height, args.width, device=device)

    # Clean single-forward peak memory measurement (before running benchmark loops)
    peak_mem_fp32 = measure_clean_peak_memory(model, x, use_amp=False)
    peak_mem_amp = measure_clean_peak_memory(model, x, use_amp=True) if device.type == "cuda" else None

    # Profiler FLOPs
    supported_flops = profiler_flops(model, x)

    # Latency profiling: FP32 and AMP
    fp32_latency = profile_latency(model, x, warmup=args.warmup, iters=args.iters, use_amp=False)
    amp_latency = (
        profile_latency(model, x, warmup=args.warmup, iters=args.iters, use_amp=True)
        if device.type == "cuda"
        else None
    )

    result = {
        "variant": variant,
        "iterations": iterations,
        "params": count_parameters(model),
        "input_shape": [1, 3, args.height, args.width],
        "device": str(device),
        "fp32": {
            **fp32_latency,
            "peak_allocated_mb": peak_mem_fp32,
        },
        "amp": {
            **(amp_latency or {}),
            "peak_allocated_mb": peak_mem_amp,
        } if amp_latency is not None else None,
        "profiler_supported_flops": supported_flops,
        "flops_note": (
            "profiler_supported_flops captures standard PyTorch convolution, normalization, and linear layers. "
            "Custom prefix-sum (integral image) and scatter-add operations in RMR operators are O(G + M) and "
            "are not registered in torch.profiler FLOP formulas."
        ),
    }

    formatted = json.dumps(result, indent=2)
    print(formatted)
    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(formatted)


if __name__ == "__main__":
    main()
