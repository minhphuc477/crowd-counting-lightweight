"""Generate reproducible NTPC architecture tables from an executed forward graph."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import torch
import torch.nn as nn
import yaml

_REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT)

from hpc.models.factory import build_model_from_config, resolve_model_config


def _shape(value) -> str:
    if isinstance(value, torch.Tensor):
        return "x".join(str(int(x)) for x in value.shape)
    if isinstance(value, (tuple, list)):
        return ", ".join(_shape(x) for x in value if isinstance(x, torch.Tensor))
    return "-"


def _component(name: str) -> str:
    if name.startswith("backbone."):
        return "backbone"
    if name.startswith("neck."):
        return "neck"
    return "head"


def _details(module: nn.Module) -> str:
    if isinstance(module, nn.Conv2d):
        return (
            f"k={module.kernel_size[0]}x{module.kernel_size[1]}, "
            f"s={module.stride[0]}, d={module.dilation[0]}, g={module.groups}"
        )
    if isinstance(module, nn.GroupNorm):
        return f"groups={module.num_groups}"
    if isinstance(module, nn.BatchNorm2d):
        return f"features={module.num_features}"
    return "-"


@torch.inference_mode()
def collect_architecture(cfg: dict, height: int, width: int) -> dict:
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    model = build_model_from_config(cfg, load_pretrained=False).eval()
    selected_types = (nn.Conv2d, nn.BatchNorm2d, nn.GroupNorm, nn.ReLU, nn.SiLU)
    rows: list[dict] = []
    hooks = []
    backbone_shapes: list[str] = []

    def leaf_hook(name: str, module: nn.Module):
        def hook(_module, inputs, output):
            macs = 0
            if isinstance(module, nn.Conv2d):
                out_h, out_w = output.shape[-2:]
                macs = int(
                    out_h
                    * out_w
                    * module.out_channels
                    * (module.in_channels // module.groups)
                    * module.kernel_size[0]
                    * module.kernel_size[1]
                )
            rows.append(
                {
                    "index": len(rows) + 1,
                    "component": _component(name),
                    "module": name,
                    "type": type(module).__name__,
                    "input_shape": _shape(inputs),
                    "output_shape": _shape(output),
                    "parameters": sum(p.numel() for p in module.parameters(recurse=False)),
                    "conv_macs": macs,
                    "details": _details(module),
                }
            )

        return hook

    def backbone_hook(_module, _inputs, output):
        backbone_shapes.extend(_shape(feature) for feature in output)

    for name, module in model.named_modules():
        if name and isinstance(module, selected_types):
            hooks.append(module.register_forward_hook(leaf_hook(name, module)))
    hooks.append(model.backbone.register_forward_hook(backbone_hook))
    try:
        image = torch.zeros(1, 3, height, width)
        mass, aux = model(image, return_aux=True)
    finally:
        for handle in hooks:
            handle.remove()

    component_params = {
        "backbone": sum(p.numel() for p in model.backbone.parameters()),
        "neck": sum(p.numel() for p in model.neck.parameters()),
        "head": sum(
            p.numel() for name, p in model.named_parameters() if name.startswith("head_")
        ),
    }
    component_macs = defaultdict(int)
    for row in rows:
        component_macs[row["component"]] += row["conv_macs"]
    flow = [
        ("Input", f"1x3x{height}x{width}", "1"),
        ("Backbone C4", backbone_shapes[0], "4"),
        ("Backbone C8", backbone_shapes[1], "8"),
        ("Backbone C16", backbone_shapes[2], "16"),
        ("Neck P16", _shape(aux["p16"]), "16"),
        ("Neck P8", _shape(aux["p8"]), "8"),
        ("Neck P4", _shape(aux["p4"]), "4"),
        ("Positive mass D", _shape(mass), "4"),
        ("Count", "1", "global sum(D)"),
    ]
    return {
        "model": model,
        "resolved": resolve_model_config(cfg),
        "rows": rows,
        "flow": flow,
        "component_params": component_params,
        "component_macs": dict(component_macs),
        "total_params": sum(p.numel() for p in model.parameters()),
        "total_macs": sum(row["conv_macs"] for row in rows),
    }


def write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: str, config_path: str, result: dict, height: int, width: int) -> None:
    resolved = result["resolved"]
    lines = [
        "# NTPC executed architecture table",
        "",
        f"- Config: `{os.path.abspath(config_path)}`",
        f"- Backbone: `{resolved['backbone']}`",
        f"- Pretrained requested by training config: `{resolved['pretrained']}`",
        "- Profiling construction: pretrained loading disabled (architecture is unchanged)",
        f"- Input: `1x3x{height}x{width}`",
        f"- Backbone truncation: `{result['model'].backbone.truncated_after}`",
        f"- Total parameters: **{result['total_params']:,}**",
        f"- Conv MACs: **{result['total_macs']:,} ({result['total_macs'] / 1e9:.6f} GMAC)**",
        "- Scope: executed Conv/normalization/activation modules; functional interpolation/add/sum/Softplus are described in the flow but carry no parameters and are excluded from Conv MACs.",
        "",
        "## End-to-end flow",
        "",
        "| Stage | Tensor shape | Reduction/operation |",
        "|---|---:|---:|",
    ]
    lines.extend(f"| {name} | `{shape}` | {reduction} |" for name, shape, reduction in result["flow"])
    lines.extend([
        "",
        "## Component totals",
        "",
        "| Component | Parameters | Share | Conv MACs | GMAC |",
        "|---|---:|---:|---:|---:|",
    ])
    for component in ("backbone", "neck", "head"):
        params = result["component_params"][component]
        macs = result["component_macs"].get(component, 0)
        lines.append(
            f"| {component} | {params:,} | {100.0 * params / result['total_params']:.2f}% | "
            f"{macs:,} | {macs / 1e9:.6f} |"
        )
    lines.extend([
        f"| **Total** | **{result['total_params']:,}** | **100%** | "
        f"**{result['total_macs']:,}** | **{result['total_macs'] / 1e9:.6f}** |",
        "",
        "## Executed layer table",
        "",
        "| # | Component | Module | Type | Output | Params | Conv MACs | Details |",
        "|---:|---|---|---|---:|---:|---:|---|",
    ])
    for row in result["rows"]:
        lines.append(
            f"| {row['index']} | {row['component']} | `{row['module']}` | {row['type']} | "
            f"`{row['output_shape']}` | {row['parameters']:,} | {row['conv_macs']:,} | {row['details']} |"
        )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an executed NTPC architecture table")
    parser.add_argument("--config", required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--output-md", default="runs/architecture/ntpc_r4_256.md")
    parser.add_argument("--output-csv", default="runs/architecture/ntpc_r4_256.csv")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    result = collect_architecture(cfg, args.height, args.width)
    write_markdown(args.output_md, args.config, result, args.height, args.width)
    write_csv(args.output_csv, result["rows"])
    print(
        f"parameters={result['total_params']:,} conv_gmac={result['total_macs'] / 1e9:.6f} "
        f"layers={len(result['rows'])} md={os.path.abspath(args.output_md)} "
        f"csv={os.path.abspath(args.output_csv)}"
    )


if __name__ == "__main__":
    main()
