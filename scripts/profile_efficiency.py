"""Hardware Efficiency & Complexity Profiler for Rank A* Benchmark Tables (CVPR/ICCV/TPAMI).

Measures exact:
  1. Trainable & Total Parameter Counts (Track A <= 104,441 | Track B <= 200,000).
  2. Forward FLOPs & MACs via PyTorch native FlopCounterMode (zero external package deps).
  3. Warmed-up Inference Latency (ms/image), Throughput (FPS), and Peak VRAM (MB)
     across canonical crowd-counting resolutions (512x512, 1024x768, 1920x1080).

Usage:
  python scripts/profile_efficiency.py --config configs/rmr_v68/v68_exp0_conservative_mcbpro_floor_tau_1500ep.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import yaml
from rmr_v3.model import RMRv3, RMRv3Config


def measure_flops(model: torch.nn.Module, x: torch.Tensor) -> tuple[float, float]:
    """Return (gflops, gmacs) for a single forward pass using PyTorch FlopCounterMode."""
    from torch.utils.flop_counter import FlopCounterMode

    model.eval()
    with torch.no_grad():
        flop_counter = FlopCounterMode(display=False)
        with flop_counter:
            _ = model(x)
        total_flops = float(flop_counter.get_total_flops())
    gflops = total_flops / 1e9
    gmacs = gflops / 2.0
    return gflops, gmacs


def benchmark_resolution(
    model: RMRv3,
    height: int,
    width: int,
    device: torch.device,
    amp: bool = True,
    warmup: int = 10,
    iters: int = 30,
) -> dict[str, Any]:
    """Measure exact GFLOPs, latency (ms), FPS, and peak allocated VRAM (MB) at (H, W)."""
    x = torch.randn(1, 3, height, width, device=device, dtype=torch.float32)
    gflops, gmacs = measure_flops(model, x)

    use_amp = bool(amp and device.type == "cuda")
    with torch.no_grad():
        for _ in range(max(1, warmup)):
            with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu", enabled=use_amp):
                _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            for _ in range(max(1, iters)):
                with torch.amp.autocast("cuda", enabled=use_amp):
                    _ = model(x)
            end_ev.record()
            torch.cuda.synchronize(device)
            elapsed_ms = float(start_ev.elapsed_time(end_ev)) / float(max(1, iters))
            peak_vram_mb = float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
        else:
            t0 = time.perf_counter()
            for _ in range(max(1, iters)):
                _ = model(x)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0 / float(max(1, iters))
            peak_vram_mb = 0.0

    fps = 1000.0 / max(elapsed_ms, 1e-6)
    return {
        "resolution": f"{width}x{height}",
        "gflops": round(gflops, 4),
        "gmacs": round(gmacs, 4),
        "latency_ms": round(elapsed_ms, 3),
        "fps": round(fps, 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "precision": "FP16_AMP" if use_amp else "FP32",
    }


def profile_config(
    config_path: str | Path,
    device_str: str | None = None,
    resolutions: tuple[tuple[int, int], ...] = ((512, 512), (768, 1024), (1080, 1920)),
    amp: bool = True,
    warmup: int = 10,
    iters: int = 30,
) -> dict[str, Any]:
    """Build model from YAML config (offline-safe pretrained=False) and profile complexity."""
    cfg_raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    m_dict = dict(cfg_raw.get("model", {}))
    if m_dict.get("init_m0") in (None, "auto"):
        m_dict["init_m0"] = 0.015763

    device = torch.device(device_str if device_str else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    m_cfg = RMRv3Config.from_dict(m_dict, pretrained=False)
    model = RMRv3(m_cfg)
    model.switch_to_deploy()
    model.set_solver_strength(1.0 if m_cfg.enable_solver else 0.0)
    model.to(device).eval()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    rows = [
        benchmark_resolution(model, h, w, device=device, amp=amp, warmup=warmup, iters=iters)
        for (h, w) in resolutions
    ]
    return {
        "config": str(config_path),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "output_stride": 2 if m_cfg.subpixel_stride2 else m_cfg.output_stride,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "benchmarks": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Profile RMRv3 Parameters, GFLOPs, FPS, and VRAM")
    ap.add_argument("--config", required=True, help="Path to YAML configuration file")
    ap.add_argument("--device", default=None, help="Execution device (e.g. cuda:0 or cpu)")
    ap.add_argument("--no-amp", dest="amp", action="store_false", default=True, help="Disable FP16 AMP")
    ap.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    ap.add_argument("--iters", type=int, default=30, help="Timed iterations")
    ap.add_argument("--output-json", default=None, help="Optional JSON output path")
    args = ap.parse_args()

    report = profile_config(
        args.config, device_str=args.device, amp=args.amp, warmup=args.warmup, iters=args.iters
    )
    print(f"\n=== RMR Hardware Efficiency Profile ({report['device_name']}) ===")
    print(f"Config          : {report['config']}")
    print(f"Trainable Params: {report['trainable_params']:,} (Output Stride {report['output_stride']})")
    for b in report["benchmarks"]:
        print(
            f"  [{b['resolution']:>9}] {b['gflops']:6.3f} GFLOPs ({b['gmacs']:6.3f} GMACs) | "
            f"{b['latency_ms']:7.2f} ms ({b['fps']:6.1f} FPS) | Peak VRAM: {b['peak_vram_mb']:6.1f} MB [{b['precision']}]"
        )
    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
