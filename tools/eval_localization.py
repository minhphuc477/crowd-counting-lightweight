"""Joint counting and parameter-free localization evaluation for NTPC."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import yaml

_REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT)

from hpc.data.nwpu import NWPUDataset
from hpc.data.qnrf import UCFQNRFDataset
from hpc.data.sha import ShanghaiTechDataset
from hpc.metrics.counting import evaluate_counting_metrics
from hpc.metrics.localization import evaluate_dataset_localization, extract_points_from_mass_map
from hpc.metrics.otm import (
    DEFAULT_OTM_MAX_SOURCE_POINTS,
    DEFAULT_OTM_MAX_TRANSPORT_ELEMENTS,
    otm_localize,
)
from hpc.metrics.subgroup import evaluate_subgroup_diagnostics
from hpc.models.factory import build_model_from_config
from hpc.utils.seed import seed_everything


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timing_summary(values: Sequence[float]) -> dict:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}
    data = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "p95": float(np.percentile(data, 95)),
    }


def build_evaluation_dataset(cfg: dict):
    data = cfg["dataset"]
    name = str(data.get("name", "sha")).lower().replace("-", "_")
    common = {
        "crop_size": int(data.get("crop_size", 256)),
        "is_train": False,
        "image_mean": data.get("image_mean", [0.485, 0.456, 0.406]),
        "image_std": data.get("image_std", [0.229, 0.224, 0.225]),
    }
    if "coordinate_base" in data:
        common["coordinate_base"] = int(data["coordinate_base"])

    if name in {"sha", "shanghaitech", "shanghaitech_a", "shanghaitech_b"}:
        part = data.get("part", "part_B" if name.endswith("_b") else "part_A")
        return ShanghaiTechDataset(
            root=data["root"], part=part, split="test_data", **common
        ), "test_data"
    if name in {"qnrf", "ucf_qnrf"}:
        return UCFQNRFDataset(root=data["root"], split="Test", **common), "Test"
    if name == "nwpu":
        return NWPUDataset(
            root=data["root"], split="val", split_file=data.get("val_split_file"), **common
        ), "val"
    raise ValueError(f"Localization evaluator does not support dataset '{name}'")


from hpc.models.factory import assert_checkpoint_compatible, build_model_from_config


def build_model(cfg: dict, checkpoint_path: str, device: torch.device) -> nn.Module:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    model = build_model_from_config(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    assert_checkpoint_compatible(checkpoint, cfg)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def estimate_conv_efficiency(model: nn.Module, resolution: int, device: torch.device) -> dict:
    """Deterministic Conv2d MAC/FLOP estimate at the paper's crop resolution."""
    macs: list[int] = []
    hooks = []
    def hook(module: nn.Conv2d, _inputs, output):
        out_h, out_w = output.shape[-2:]
        operations = (
            out_h * out_w * module.out_channels
            * (module.in_channels // module.groups)
            * module.kernel_size[0] * module.kernel_size[1]
        )
        macs.append(int(operations))
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(hook))
    try:
        model(torch.zeros(1, 3, resolution, resolution, device=device))
    finally:
        for handle in hooks:
            handle.remove()
    total_macs = int(sum(macs))
    return {
        "params": int(sum(parameter.numel() for parameter in model.parameters())),
        "profile_resolution": f"{resolution}x{resolution}",
        "conv_macs": total_macs,
        "conv_gmacs": total_macs / 1e9,
        "conv_flops_multiply_add_2": 2 * total_macs,
        "note": "Conv2d-only analytical estimate; interpolation/activation/normalization excluded",
    }


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate_checkpoint(
    config_path: str,
    checkpoint_path: str,
    output_json: str,
    output_csv: str | None = None,
    methods: Iterable[str] = ("local_max", "otm"),
    radii: Sequence[float] = (4.0, 8.0),
    seed: int = 42,
    max_samples: int | None = None,
    local_threshold_rel: float = 0.05,
    local_threshold_abs: float = 0.01,
    local_min_distance_px: int = 4,
    otm_max_iterations: int = 16,
    otm_scaling: float = 0.75,
    otm_blur: float = 0.01,
    otm_max_source_points: int | None = DEFAULT_OTM_MAX_SOURCE_POINTS,
    otm_max_transport_elements: int = DEFAULT_OTM_MAX_TRANSPORT_ELEMENTS,
) -> dict:
    methods = tuple(dict.fromkeys(methods))
    unknown = set(methods) - {"local_max", "otm"}
    if unknown:
        raise ValueError(f"Unknown localization methods: {sorted(unknown)}")
    if not methods:
        raise ValueError("At least one localization method is required")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    config_seed = int(cfg.get("experiment", {}).get("seed", seed))
    if config_seed != int(seed):
        raise ValueError(f"Config seed {config_seed} does not match evaluation seed {seed}")
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, checkpoint_path, device)
    dataset, split = build_evaluation_dataset(cfg)
    sample_count = len(dataset) if max_samples is None else min(len(dataset), int(max_samples))
    if sample_count <= 0:
        raise ValueError("Evaluation set is empty")

    predictions_count: list[float] = []
    ground_truth_count: list[float] = []
    predicted_points = {method: [] for method in methods}
    ground_truth_points = []
    model_times, method_times = [], {method: [] for method in methods}
    consistency_gaps = {method: [] for method in methods}
    otm_diagnostics: list[dict] = []
    image_rows: list[dict] = []

    for index in range(sample_count):
        sample = dataset[index]
        if not bool(sample.get("has_gt", True)):
            raise ValueError(f"Sample {index} in split {split} has no localization ground truth")
        image = sample["image"].unsqueeze(0).to(device)
        image_hw = tuple(int(value) for value in image.shape[-2:])
        gt_count = float(sample["gt_count"])
        gt_points = np.asarray(sample["gt_points"], dtype=np.float32).reshape(-1, 2)

        _synchronize(device)
        started = time.perf_counter()
        pred_count_tensor, pred_mass = model.predict(image, pad_multiple=None)
        _synchronize(device)
        model_ms = (time.perf_counter() - started) * 1000.0
        pred_count = float(pred_count_tensor.item())
        predictions_count.append(pred_count)
        ground_truth_count.append(gt_count)
        ground_truth_points.append(gt_points)
        model_times.append(model_ms)
        row = {
            "index": index,
            "image": sample["img_path"],
            "gt_count": gt_count,
            "pred_count": pred_count,
            "model_latency_ms": model_ms,
        }

        if "local_max" in methods:
            started = time.perf_counter()
            local_points = extract_points_from_mass_map(
                pred_mass[0, 0],
                stride=4,
                threshold_rel=local_threshold_rel,
                threshold_abs=local_threshold_abs,
                min_distance_px=local_min_distance_px,
            )
            local_ms = (time.perf_counter() - started) * 1000.0
            predicted_points["local_max"].append(local_points)
            method_times["local_max"].append(local_ms)
            gap = abs(len(local_points) - pred_count)
            consistency_gaps["local_max"].append(gap)
            row.update(local_max_points=len(local_points), local_max_gap=gap, local_max_latency_ms=local_ms)

        if "otm" in methods:
            _synchronize(device)
            started = time.perf_counter()
            otm_points, diagnostics = otm_localize(
                pred_mass[0, 0],
                output_stride=4,
                outer_iterations=otm_max_iterations,
                ot_scaling=otm_scaling,
                blur=otm_blur,
                max_source_points=otm_max_source_points,
                max_transport_elements=otm_max_transport_elements,
                seed=seed + index,
                image_hw=image_hw,
                return_diagnostics=True,
            )
            _synchronize(device)
            otm_ms = (time.perf_counter() - started) * 1000.0
            otm_numpy = otm_points.cpu().numpy()
            predicted_points["otm"].append(otm_numpy)
            method_times["otm"].append(otm_ms)
            gap = abs(len(otm_numpy) - pred_count)
            consistency_gaps["otm"].append(gap)
            otm_diagnostics.append(diagnostics)
            row.update(
                otm_points=len(otm_numpy), otm_gap=gap, otm_latency_ms=otm_ms,
                otm_iterations=diagnostics["iterations"],
                otm_retained_mass_ratio=diagnostics["source_retained_mass_ratio"],
            )
        image_rows.append(row)

    localization = {
        method: evaluate_dataset_localization(
            predicted_points[method], ground_truth_points, tuple(float(x) for x in radii)
        )
        for method in methods
    }
    consistency = {
        method: {
            "mean_abs_localized_count_minus_mass_count": float(np.mean(consistency_gaps[method])),
            "max_abs_localized_count_minus_mass_count": float(np.max(consistency_gaps[method])),
        }
        for method in methods
    }
    diagnostics_summary = {}
    if otm_diagnostics:
        diagnostics_summary = {
            "mean_iterations": float(np.mean([x["iterations"] for x in otm_diagnostics])),
            "min_retained_mass_ratio": float(min(x["source_retained_mass_ratio"] for x in otm_diagnostics)),
            "mean_retained_mass_ratio": float(
                np.mean([x["source_retained_mass_ratio"] for x in otm_diagnostics])
            ),
            "max_transport_elements": int(max(x["transport_elements"] for x in otm_diagnostics)),
        }
    result = {
        "metadata": {
            "config": os.path.abspath(config_path),
            "checkpoint": os.path.abspath(checkpoint_path),
            "seed": seed,
            "selection_split": split,
            "samples": sample_count,
            "methods": list(methods),
            "radii_px": [float(x) for x in radii],
            "matching": "distance-gated Hungarian one-to-one, micro-aggregated",
            "otm": "Lin & Chan CVPR 2023 alternating epsilon-scaling OT/M-step",
        },
        "counting": evaluate_counting_metrics(predictions_count, ground_truth_count),
        "subgroups": evaluate_subgroup_diagnostics(predictions_count, ground_truth_count),
        "localization": localization,
        "count_consistency": consistency,
        "latency_ms_per_image": {
            "model": _timing_summary(model_times),
            **{method: _timing_summary(method_times[method]) for method in methods},
        },
        "efficiency": estimate_conv_efficiency(
            model, int(cfg["dataset"].get("crop_size", 256)), device
        ),
        "otm_diagnostics": diagnostics_summary,
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    if output_csv:
        _write_csv(output_csv, image_rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NTPC counting/localization")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-image-csv")
    parser.add_argument("--methods", nargs="+", default=["local_max", "otm"])
    parser.add_argument("--radii", nargs="+", type=float, default=[4.0, 8.0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--otm-max-source-points", type=int, default=DEFAULT_OTM_MAX_SOURCE_POINTS)
    args = parser.parse_args()
    result = evaluate_checkpoint(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_json=args.output,
        output_csv=args.per_image_csv,
        methods=args.methods,
        radii=args.radii,
        seed=args.seed,
        max_samples=args.max_samples,
        otm_max_source_points=args.otm_max_source_points,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
