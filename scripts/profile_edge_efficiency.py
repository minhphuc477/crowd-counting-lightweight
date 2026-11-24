"""JRTIP Real-Time Deployment & Hardware Efficiency Profiler.

Measures:
  - Parameter count & serialized model footprint (FP32 MB, FP16 MB, ONNX MB)
  - Exact MACs & GFLOPs (Conv2d / Linear forward hooks)
  - End-to-end & stage-wise latency breakdown (Encoder, Neck, Head, Post-sum)
  - Multi-resolution CCTV profiling (e.g., 512x512, 720x1280, 1080x1920)
  - CPU thread scaling (--threads 1 vs multi-thread), CUDA FP16 (--fp16), Peak VRAM
  - ONNX Runtime export & inference benchmark (--onnx)
  - Optional FP32 vs FP16 counting fidelity audit (--verify-fp16-ckpt + --manifest)

Usage:
  python scripts/profile_edge_efficiency.py ^
      --configs configs/benchmarks/lfi_sub100k/exp_sha_lfi_dsimr_99k.yaml ^
      --resolutions 512x512 720x1280 1080x1920 --threads 1 --onnx --device cpu ^
      --output-json results/jrtip_realtime_profile_cpu1t.json
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn

from rmr_core.data import CrowdManifestDataset, collate_eval
from rmr_core.evaluation import evaluate_dataset
from rmr_v3.config import load_config
from rmr_v3.engine import make_model
from rmr_v3.eval import load_model_from_ckpt


class _OnnxY0Wrapper(nn.Module):
    """Thin export wrapper returning primary Stride-2 density map y0."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, dict):
            return out["y"] if "y" in out else out["y0"]
        return out


def parse_resolutions(resolutions: list[str] | None, sizes: list[int]) -> list[tuple[int, int]]:
    if resolutions:
        parsed: list[tuple[int, int]] = []
        for r in resolutions:
            parts = r.lower().strip().split("x")
            if len(parts) != 2:
                raise ValueError(f"Resolution must be HxW (e.g. 720x1280), got {r!r}")
            parsed.append((int(parts[0]), int(parts[1])))
        return parsed
    return [(int(s), int(s)) for s in sizes]


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
        total_macs += int(bsz * out_c * out_h * out_w * in_per_group * k_h * k_w)

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


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_fn(fn, iters: int, warmup: int, device: torch.device, ctx_factory) -> float:
    with torch.no_grad():
        for _ in range(max(warmup, 1)):
            with ctx_factory():
                _ = fn()
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(max(iters, 1)):
            with ctx_factory():
                _ = fn()
        _sync(device)
    elapsed = max(time.perf_counter() - t0, 1e-9)
    return (elapsed / float(max(iters, 1))) * 1000.0


def profile_stage_breakdown(
    model: nn.Module,
    x: torch.Tensor,
    iters: int,
    warmup: int,
    device: torch.device,
    ctx_factory,
) -> dict[str, float]:
    """Measure isolated stage latency (Encoder, Neck, Head, Post-sum)."""
    if not (hasattr(model, "encoder") and hasattr(model, "fusion") and hasattr(model, "fine_head")):
        return {}

    enc_ms = _time_fn(lambda: model.encoder(x), iters, warmup, device, ctx_factory)
    with torch.no_grad(), ctx_factory():
        feats = model.encoder(x)
    neck_ms = _time_fn(lambda: model.fusion(*feats), iters, warmup, device, ctx_factory)
    with torch.no_grad(), ctx_factory():
        p4, _, _ = model.fusion(*feats)

    def _run_head():
        return model.fine_head(p4, scale_weights=None)

    head_ms = _time_fn(_run_head, iters, warmup, device, ctx_factory)
    with torch.no_grad(), ctx_factory():
        y0 = _run_head()
    post_ms = _time_fn(lambda: y0.sum(), iters, warmup, device, ctx_factory)
    return {
        "encoder_ms": round(enc_ms, 3),
        "neck_ms": round(neck_ms, 3),
        "head_ms": round(head_ms, 3),
        "post_ms": round(post_ms, 3),
    }


def profile_onnx_runtime(
    model: nn.Module,
    h: int,
    w: int,
    iters: int,
    warmup: int,
    device: torch.device,
    threads: int | None,
) -> dict[str, Any]:
    """Export wrapper to ONNX opset 17 and benchmark ONNX Runtime inference."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        return {"onnx_error": f"onnxruntime unavailable: {exc}"}

    wrapper = _OnnxY0Wrapper(model).cpu().eval()
    x_cpu = torch.randn(1, 3, int(h), int(w), dtype=torch.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "model.onnx"
        try:
            torch.onnx.export(
                wrapper,
                x_cpu,
                str(onnx_path),
                opset_version=17,
                input_names=["input"],
                output_names=["density"],
                do_constant_folding=True,
                dynamo=False,
            )
        except TypeError:
            torch.onnx.export(
                wrapper,
                x_cpu,
                str(onnx_path),
                opset_version=17,
                input_names=["input"],
                output_names=["density"],
                do_constant_folding=True,
            )
        except Exception as exc:
            return {"onnx_error": f"ONNX export failed: {type(exc).__name__}: {exc}"}

        onnx_mb = round(onnx_path.stat().st_size / 1e6, 4)
        so = ort.SessionOptions()
        if threads is not None and threads > 0:
            so.intra_op_num_threads = int(threads)
            so.inter_op_num_threads = 1
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device.type == "cuda"
            else ["CPUExecutionProvider"]
        )
        try:
            sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers)
        except Exception as exc:
            return {"onnx_mb": onnx_mb, "onnx_error": f"ORT session failed: {exc}"}

        in_name = sess.get_inputs()[0].name
        x_np = x_cpu.numpy()
        for _ in range(max(warmup, 1)):
            _ = sess.run(None, {in_name: x_np})
        t0 = time.perf_counter()
        for _ in range(max(iters, 1)):
            _ = sess.run(None, {in_name: x_np})
        elapsed = max(time.perf_counter() - t0, 1e-9)
        ms = (elapsed / float(max(iters, 1))) * 1000.0
        return {
            "onnx_mb": onnx_mb,
            "onnx_provider": sess.get_providers()[0],
            "onnx_ms": round(ms, 3),
            "onnx_fps": round(float(max(iters, 1)) / elapsed, 2),
        }


def profile_single_model(
    config_path: str,
    resolutions: list[tuple[int, int]],
    iters: int,
    warmup: int,
    device: torch.device,
    use_fp16: bool,
    run_onnx: bool,
    threads: int | None,
) -> dict[str, Any]:
    """Build model from config and measure footprint, MACs, GFLOPs, latency, stages, and ONNX."""
    cfg = load_config(config_path)
    cfg.setdefault("model", {})["pretrained"] = False
    model, _ = make_model(cfg)
    model = model.to(device).eval()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    def _ctx():
        if use_fp16 and device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.float16)
        return nullcontext()

    entry: dict[str, Any] = {
        "config": config_path,
        "name": Path(config_path).stem,
        "neck_type": cfg.get("model", {}).get("neck_type"),
        "head_type": cfg.get("model", {}).get("head_type"),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "fp32_mb": round((trainable_params * 4) / 1e6, 4),
        "fp16_mb": round((trainable_params * 2) / 1e6, 4),
        "device": str(device),
        "fp16_mode": bool(use_fp16 and device.type == "cuda"),
        "cpu_threads": threads,
        "profiles": {},
    }

    for h, w in resolutions:
        x = torch.randn(1, 3, int(h), int(w), device=device)
        macs = count_macs(model, x)
        gflops = (2.0 * macs) / 1e9

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        e2e_ms = _time_fn(lambda: model(x), iters, warmup, device, _ctx)
        peak_vram_mb = (
            round(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2), 2)
            if device.type == "cuda"
            else None
        )
        stages = profile_stage_breakdown(model, x, iters, warmup, device, _ctx)

        prof: dict[str, Any] = {
            "macs": macs,
            "m_macs": round(macs / 1e6, 2),
            "gflops": round(gflops, 4),
            "latency_ms": round(e2e_ms, 3),
            "fps": round(1000.0 / max(e2e_ms, 1e-9), 2),
            "peak_vram_mb": peak_vram_mb,
            "stages": stages,
        }
        if run_onnx:
            prof["onnx"] = profile_onnx_runtime(model, h, w, iters, warmup, device, threads)
            model = model.to(device).eval()

        entry["profiles"][f"{h}x{w}"] = prof
    return entry


def verify_fp16_fidelity(
    ckpt_path: str,
    config_path: str | None,
    manifest_path: str,
    device: torch.device,
) -> dict[str, Any]:
    """Compare FP32 vs FP16 (CUDA autocast) evaluation accuracy on a real checkpoint and manifest."""
    if device.type != "cuda":
        return {"skipped": True, "reason": "FP16 autocast accuracy verification requires CUDA device"}

    model, _, cfg, _ = load_model_from_ckpt(Path(ckpt_path), device=device, use_ema=True, config_path=config_path)
    model.eval()
    d_cfg = cfg.get("data", {})
    m_cfg = cfg.get("model", {})
    stride = int(m_cfg.get("output_stride", 2))
    norm_mode = CrowdManifestDataset.resolve_norm(m_cfg.get("backbone_name", ""), d_cfg.get("norm_mode", "auto"))

    ds = CrowdManifestDataset(
        manifest=manifest_path,
        train=False,
        data_root=d_cfg.get("data_root", "data"),
        max_size=int(d_cfg.get("max_size", 2048) or 2048),
        output_stride=stride,
        norm_mode=norm_mode,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_eval)

    _, sum_fp32 = evaluate_dataset(model=model, loader=loader, device=device, output_stride=stride, run_tiling=False)
    with torch.autocast("cuda", dtype=torch.float16):
        _, sum_fp16 = evaluate_dataset(model=model, loader=loader, device=device, output_stride=stride, run_tiling=False)

    return {
        "checkpoint": ckpt_path,
        "manifest": manifest_path,
        "num_samples": sum_fp32["num_samples"],
        "fp32": {"MAE": round(sum_fp32["MAE"], 4), "RMSE": round(sum_fp32["RMSE"], 4), "Bias": round(sum_fp32["Bias"], 4)},
        "fp16": {"MAE": round(sum_fp16["MAE"], 4), "RMSE": round(sum_fp16["RMSE"], 4), "Bias": round(sum_fp16["Bias"], 4)},
        "delta_mae_fp16_minus_fp32": round(sum_fp16["MAE"] - sum_fp32["MAE"], 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Profile Params, MACs, GFLOPs, Stage-wise Latency, FPS, and ONNX")
    ap.add_argument("--configs", nargs="+", required=True, help="List of YAML config paths")
    ap.add_argument("--sizes", nargs="+", type=int, default=[512, 1024], help="Square input sizes (legacy fallback)")
    ap.add_argument("--resolutions", nargs="+", default=None, help="Explicit HxW resolutions (e.g. 512x512 720x1280 1080x1920)")
    ap.add_argument("--iters", type=int, default=20, help="Timed iterations per resolution")
    ap.add_argument("--warmup", type=int, default=5, help="Warmup iterations per resolution")
    ap.add_argument("--threads", type=int, default=None, help="Optional CPU thread cap (e.g. 1 for embedded simulation)")
    ap.add_argument("--fp16", action="store_true", help="Benchmark CUDA FP16 autocast inference")
    ap.add_argument("--onnx", action="store_true", help="Export ONNX (opset 17) and benchmark ONNX Runtime")
    ap.add_argument("--verify-fp16-ckpt", default=None, help="Optional checkpoint path to audit FP32 vs FP16 MAE fidelity")
    ap.add_argument("--manifest", default=None, help="Manifest path for --verify-fp16-ckpt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output-json", default=None, help="Optional JSON output path")
    args = ap.parse_args()

    if args.threads is not None and args.threads > 0:
        torch.set_num_threads(int(args.threads))

    device = torch.device(args.device)
    res_list = parse_resolutions(args.resolutions, args.sizes)
    results = [
        profile_single_model(c, res_list, args.iters, args.warmup, device, args.fp16, args.onnx, args.threads)
        for c in args.configs
    ]

    print(f"\nJRTIP Real-Time Efficiency Profile (device={device}, fp16={bool(args.fp16 and device.type=='cuda')}, threads={args.threads}, onnx={args.onnx})")
    print("-" * 112)
    for r in results:
        print(f"Model: {r['name']} | Params: {r['trainable_params']:,} | Size: {r['fp32_mb']:.3f} MB (FP32) / {r['fp16_mb']:.3f} MB (FP16)")
        for res_key, p in r["profiles"].items():
            st = p.get("stages", {})
            st_str = (
                f"Enc {st['encoder_ms']:.2f} + Neck {st['neck_ms']:.2f} + Head {st['head_ms']:.2f} ms"
                if st else "N/A"
            )
            onnx_info = p.get("onnx", {})
            onnx_str = (
                f" | ORT: {onnx_info['onnx_ms']:.2f}ms/{onnx_info['onnx_fps']:.1f}FPS ({onnx_info['onnx_mb']:.3f}MB)"
                if "onnx_ms" in onnx_info
                else (f" | ORT: {onnx_info['onnx_error']}" if onnx_info else "")
            )
            vram_str = f" | VRAM: {p['peak_vram_mb']:.1f}MB" if p.get("peak_vram_mb") is not None else ""
            print(
                f"  [{res_key:>9}] {p['gflops']:.3f} GFLOPs | E2E: {p['latency_ms']:.2f}ms ({p['fps']:.1f} FPS) "
                f"| Stages: [{st_str}]{vram_str}{onnx_str}"
            )

    fp16_audit = None
    if args.verify_fp16_ckpt:
        if not args.manifest:
            raise ValueError("--verify-fp16-ckpt requires --manifest")
        fp16_audit = verify_fp16_fidelity(args.verify_fp16_ckpt, args.configs[0], args.manifest, device)
        print("\nFP32 vs FP16 Fidelity Audit:", fp16_audit)

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"results": results}
        if fp16_audit is not None:
            payload["fp16_fidelity"] = fp16_audit
        out_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved profile JSON to {out_p}")


if __name__ == "__main__":
    main()
