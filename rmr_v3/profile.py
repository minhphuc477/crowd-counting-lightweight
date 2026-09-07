from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

# Ensure repository root is on sys.path and prevent shadowing stdlib profile module
script_dir = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == script_dir:
    sys.path.pop(0)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Windows stdout encoding safety
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import torch
import yaml

from rmr_core.evaluation import predict_tiled
from rmr_v3.model import RMRv3, RMRv3Config
from rmr_v3.train import make_model


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def profile_latency(
    model: torch.nn.Module,
    x: torch.Tensor,
    warmup: int = 20,
    iters: int = 50,
    use_amp: bool = False,
    forward_fn: Callable[[torch.Tensor], Any] | None = None,
) -> dict:
    model.eval()
    device = x.device
    autocast_ctx = (
        torch.amp.autocast("cuda")
        if (use_amp and device.type == "cuda")
        else contextlib.nullcontext()
    )

    call = forward_fn if forward_fn is not None else (lambda inp: model(inp))

    with autocast_ctx:
        for _ in range(warmup):
            _ = call(x)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        times: list[float] = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with autocast_ctx:
                _ = call(x)
            end.record()
            torch.cuda.synchronize(device)
            times.append(float(start.elapsed_time(end)))
    else:
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = call(x)
            times.append(float((time.perf_counter() - t0) * 1000.0))

    a = np.asarray(times, dtype=np.float64)
    return {
        "latency_ms_mean": float(a.mean()),
        "latency_ms_p50": float(np.quantile(a, 0.50)),
        "latency_ms_p95": float(np.quantile(a, 0.95)),
        "fps_from_mean": float(1000.0 / a.mean()) if a.mean() > 0 else 0.0,
    }


def profiler_flops(
    model: torch.nn.Module,
    x: torch.Tensor,
    forward_fn: Callable[[torch.Tensor], Any] | None = None,
) -> float | None:
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if x.is_cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        call = forward_fn if forward_fn is not None else (lambda inp: model(inp))
        with torch.profiler.profile(activities=activities, with_flops=True) as prof:
            _ = call(x)
        total = sum((evt.flops or 0) for evt in prof.key_averages())
        return float(total)
    except Exception:
        return None


@torch.no_grad()
def measure_clean_peak_memory(
    model: torch.nn.Module,
    x: torch.Tensor,
    use_amp: bool = False,
    forward_fn: Callable[[torch.Tensor], Any] | None = None,
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
    call = forward_fn if forward_fn is not None else (lambda inp: model(inp))
    with autocast_ctx:
        _ = call(x)
    torch.cuda.synchronize(device)
    return float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Profile RMR-v3 models for latency, memory, and complexity.")
    ap.add_argument("--config", default=None, help="Path to YAML config (e.g. configs/rmr_v3/reliability_weighted.yaml)")
    ap.add_argument("--uniform-reliability", dest="uniform_reliability", action="store_true", default=False, help="Profile with uniform reliability (W=I)")
    ap.add_argument("--weighted-reliability", dest="uniform_reliability", action="store_false", help="Profile with weighted reliability (W=diag(w_R))")
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tiling", action="store_true", default=False, help="Profile tiled inference (tile 512 with halo 64)")
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--halo", type=int, default=64)
    ap.add_argument("--output", default=None, help="Optional output JSON path")
    args = ap.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        model, cfg_uniform = make_model(cfg)
        model = model.to(device).eval()
        uniform_reliability = cfg_uniform if not args.uniform_reliability else True
        iterations = getattr(model.cfg, "iterations", args.iterations)
    else:
        mcfg = RMRv3Config(
            iterations=args.iterations,
            pretrained=False,
            region_sizes_px=(32, 64, 128),
        )
        model = RMRv3(mcfg).to(device).eval()
        uniform_reliability = args.uniform_reliability
        iterations = args.iterations

    model.set_solver_strength(1.0)
    x = torch.randn(1, 3, args.height, args.width, device=device)

    mode_str = "uniform" if uniform_reliability else "weighted"
    stride = int(getattr(model.cfg, "output_stride", 4))

    if args.tiling:
        forward_call = lambda inp: predict_tiled(
            model,
            inp[0],
            output_stride=stride,
            tile_size=args.tile_size,
            halo=args.halo,
            forward_kwargs={"uniform_reliability": uniform_reliability, "solver_strength": 1.0},
        )
    else:
        forward_call = lambda inp: model(inp, uniform_reliability=uniform_reliability, solver_strength=1.0)

    # Clean single-forward peak memory measurement
    peak_mem_fp32 = measure_clean_peak_memory(model, x, use_amp=False, forward_fn=forward_call)
    peak_mem_amp = (
        measure_clean_peak_memory(model, x, use_amp=True, forward_fn=forward_call)
        if device.type == "cuda"
        else None
    )

    # Profiler FLOPs (only on direct pass)
    supported_flops = profiler_flops(model, x, forward_fn=forward_call) if not args.tiling else None

    # Latency profiling: FP32 and AMP
    fp32_latency = profile_latency(model, x, warmup=args.warmup, iters=args.iters, use_amp=False, forward_fn=forward_call)
    amp_latency = (
        profile_latency(model, x, warmup=args.warmup, iters=args.iters, use_amp=True, forward_fn=forward_call)
        if device.type == "cuda"
        else None
    )

    total_params = count_trainable_parameters(model)

    result = {
        "architecture": "RMR-v3",
        "reliability_mode": mode_str,
        "iterations": iterations,
        "trainable_parameters": total_params,
        "input_shape": [1, 3, args.height, args.width],
        "tiling": args.tiling,
        "tile_size": args.tile_size if args.tiling else None,
        "halo": args.halo if args.tiling else None,
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
            "profiler_supported_flops captures standard PyTorch conv, norm, and linear operations. "
            "Custom 2D prefix sums (integral images) and regional adjoint accumulations are O(G + M) and "
            "not registered in torch.profiler FLOP accounting."
        ),
    }

    formatted = json.dumps(result, indent=2)
    print(formatted)
    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(formatted, encoding="utf-8")


if __name__ == "__main__":
    main()
