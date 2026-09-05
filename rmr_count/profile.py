from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from .model import RMRConfig, RMRCount, count_parameters


@torch.no_grad()
def profile_latency(model: torch.nn.Module, x: torch.Tensor, warmup: int = 100, iters: int = 500) -> dict:
    model.eval()
    for _ in range(warmup):
        _ = model(x)
    if x.is_cuda:
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(x)
            end.record()
            torch.cuda.synchronize()
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="rmr", choices=["direct", "region_loss", "region_aux", "local_refine", "learned_project", "rmr"])
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = RMRCount(RMRConfig(iterations=args.iterations, region_sizes_px=(32,64,128), eta_max=0.20, eta_init=0.05, residual_clip=5.0), variant=args.variant).to(device).eval()
    x = torch.randn(1, 3, args.height, args.width, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    result = {
        "variant": args.variant,
        "iterations": args.iterations,
        "params": count_parameters(model),
        "input": [1, 3, args.height, args.width],
    }
    result.update(profile_latency(model, x))
    result["profiler_flops"] = profiler_flops(model, x)
    if device.type == "cuda":
        result["peak_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
