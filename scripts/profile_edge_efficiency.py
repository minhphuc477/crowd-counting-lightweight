"""Pure-PyTorch Hardware Efficiency Profiler (Params, MACs, GFLOPs, Latency, FPS).

Usage:
  python scripts/profile_edge_efficiency.py ^
      --configs configs/benchmarks/lfi_sub100k/exp_sha_lfi_dsimr_99k.yaml ^
      --sizes 512 1024 --iters 20 --device cpu ^
      --output-json results/edge_efficiency_profile_cpu.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn

from rmr_v3.config import load_config
from rmr_v3.engine import make_model


def count_macs(model: nn.Module, inp: torch.Tensor) -> int:
    """Measure exact forward MACs across Conv2d and Linear layers using forward hooks."""
    total_macs = 0
    handles = []

    def conv_hook(mod: nn.Conv2d, _in: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        nonlocal total_macs
        if not isinstance(out, torch.Tensor):
            return
        bsz, out_c, out_h, out_w = out.shape
        k_h, k_w = mod.kernel_size
        in_per_group = mod.in_channels // max(mod.groups, 1)
        macs = bsz * out_c * out_h * out_w * in_per_group * k_h * k_w
        total_macs += int(macs)

    def linear_hook(mod: nn.Linear, _in: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        nonlocal total_macs
        if not isinstance(out, torch.Tensor):
            return
        batch_elems = out.numel() // max(mod.out_features, 1)
        total_macs += int(batch_elems * mod.in_features * mod.out_features)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))

    try:
        with torch.no_grad():
            _ = model(inp)
    finally:
        for h in handles:
            h.remove()
    return total_macs


def profile_single_model(
    config_path: str,
    sizes: list[int],
    iters: int,
    warmup: int,
    device: torch.device,
) -> dict[str, Any]:
    """Build model from config and measure params, MACs, GFLOPs, latency, and FPS."""
    cfg = load_config(config_path)
    cfg.setdefault("model", {})["pretrained"] = False
    model, _ = make_model(cfg)
    model = model.to(device).eval()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    entry: dict[str, Any] = {
        "config": config_path,
        "name": Path(config_path).stem,
        "neck_type": cfg.get("model", {}).get("neck_type"),
        "head_type": cfg.get("model", {}).get("head_type"),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "device": str(device),
        "profiles": {},
    }

    for sz in sizes:
        x = torch.randn(1, 3, int(sz), int(sz), device=device)
        macs = count_macs(model, x)
        gflops = (2.0 * macs) / 1e9

        with torch.no_grad():
            for _ in range(max(warmup, 1)):
                _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(max(iters, 1)):
                _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = max(time.perf_counter() - t0, 1e-9)

        ms_per_img = (elapsed / float(max(iters, 1))) * 1000.0
        fps = float(max(iters, 1)) / elapsed
        entry["profiles"][f"{sz}x{sz}"] = {
            "macs": macs,
            "m_macs": round(macs / 1e6, 2),
            "gflops": round(gflops, 4),
            "latency_ms": round(ms_per_img, 3),
            "fps": round(fps, 2),
        }
    return entry


def main() -> None:
    ap = argparse.ArgumentParser(description="Profile Params, MACs, GFLOPs, Latency, and FPS")
    ap.add_argument("--configs", nargs="+", required=True, help="List of YAML config paths")
    ap.add_argument("--sizes", nargs="+", type=int, default=[512, 1024], help="Square input sizes")
    ap.add_argument("--iters", type=int, default=20, help="Timed iterations per size")
    ap.add_argument("--warmup", type=int, default=5, help="Warmup iterations per size")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output-json", default=None, help="Optional JSON output path")
    args = ap.parse_args()

    device = torch.device(args.device)
    results = [
        profile_single_model(c, args.sizes, args.iters, args.warmup, device)
        for c in args.configs
    ]

    print(f"\nHardware Efficiency Profile (device={device}, iters={args.iters})")
    print("-" * 92)
    header = f"{'Config':<38} | {'Params':>8}"
    for sz in args.sizes:
        header += f" | {f'{sz} GFLOPs':>10} | {f'{sz} ms/FPS':>15}"
    print(header)
    print("-" * 92)
    for r in results:
        line = f"{r['name']:<38} | {r['trainable_params']:>8,}"
        for sz in args.sizes:
            p = r["profiles"][f"{sz}x{sz}"]
            line += f" | {p['gflops']:>10.3f} | {p['latency_ms']:>6.2f}ms/{p['fps']:>5.1f}"
        print(line)

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
        print(f"\nSaved profile JSON to {out_p}")


if __name__ == "__main__":
    main()
