"""Evaluate the one-seed NTPC hierarchy-depth localization experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import yaml

_REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT)

from hpc.metrics.otm import DEFAULT_OTM_MAX_SOURCE_POINTS, DEFAULT_OTM_MAX_TRANSPORT_ELEMENTS
from tools.eval_localization import evaluate_checkpoint


def _resolve(path: str, manifest_path: str) -> str:
    if os.path.isabs(path):
        return path
    repository_root = os.path.dirname(os.path.dirname(os.path.abspath(manifest_path)))
    return os.path.abspath(os.path.join(repository_root, path))


def run_manifest(manifest_path: str, max_samples: int | None = None) -> dict:
    manifest_path = os.path.abspath(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    seed = int(manifest.get("seed", 42))
    experiments = manifest.get("experiments", {})
    if set(experiments) != {"r4_dtm16", "t1_dtm8", "t2_dtm4"}:
        raise ValueError("Manifest must contain exactly r4_dtm16, t1_dtm8 and t2_dtm4")
    evaluation = manifest.get("evaluation", {})
    output_dir = _resolve(manifest.get("output_dir", "runs/ntpc_localization_seed42"), manifest_path)
    os.makedirs(output_dir, exist_ok=True)

    resolved = {}
    missing = []
    for name, experiment in experiments.items():
        config_path = _resolve(experiment["config"], manifest_path)
        checkpoint_path = _resolve(experiment["checkpoint"], manifest_path)
        if not os.path.isfile(config_path):
            missing.append(config_path)
        if not os.path.isfile(checkpoint_path):
            missing.append(checkpoint_path)
        else:
            with open(config_path, "r", encoding="utf-8") as handle:
                cfg = yaml.safe_load(handle)
            if int(cfg.get("experiment", {}).get("seed", seed)) != seed:
                raise ValueError(f"{name} does not use the manifest seed {seed}")
        resolved[name] = (config_path, checkpoint_path)
    if missing:
        raise FileNotFoundError("Missing experiment inputs:\n" + "\n".join(missing))

    results, rows = {}, []
    radii = tuple(float(x) for x in evaluation.get("radii", [4, 8]))
    methods = tuple(evaluation.get("methods", ["local_max", "otm"]))
    effective_max_samples = max_samples if max_samples is not None else evaluation.get("max_samples")
    for name, (config_path, checkpoint_path) in resolved.items():
        result = evaluate_checkpoint(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            output_json=os.path.join(output_dir, f"{name}.json"),
            output_csv=os.path.join(output_dir, f"{name}_per_image.csv"),
            methods=methods,
            radii=radii,
            seed=seed,
            max_samples=effective_max_samples,
            local_threshold_rel=float(evaluation.get("local_threshold_rel", 0.05)),
            local_threshold_abs=float(evaluation.get("local_threshold_abs", 0.01)),
            local_min_distance_px=int(evaluation.get("local_min_distance_px", 4)),
            otm_max_iterations=int(evaluation.get("otm_max_iterations", 16)),
            otm_scaling=float(evaluation.get("otm_scaling", 0.75)),
            otm_blur=float(evaluation.get("otm_blur", 0.01)),
            otm_max_source_points=evaluation.get("otm_max_source_points", DEFAULT_OTM_MAX_SOURCE_POINTS),
            otm_max_transport_elements=int(evaluation.get("otm_max_transport_elements", DEFAULT_OTM_MAX_TRANSPORT_ELEMENTS)),
        )
        results[name] = result
        row = {
            "experiment": name,
            "seed": seed,
            "mae": result["counting"]["mae"],
            "rmse": result["counting"]["rmse"],
            "params": result["efficiency"]["params"],
            "conv_gmacs": result["efficiency"]["conv_gmacs"],
            "model_latency_median_ms": result["latency_ms_per_image"]["model"]["median"],
        }
        for method in methods:
            for radius in radii:
                key = f"sigma_{int(radius)}"
                row[f"{method}_{key}_precision"] = result["localization"][method][f"{key}_precision"]
                row[f"{method}_{key}_recall"] = result["localization"][method][f"{key}_recall"]
                row[f"{method}_{key}_f1"] = result["localization"][method][f"{key}_f1"]
            row[f"{method}_latency_median_ms"] = result["latency_ms_per_image"][method]["median"]
            row[f"{method}_mean_count_gap"] = result["count_consistency"][method][
                "mean_abs_localized_count_minus_mass_count"
            ]
        rows.append(row)

    summary = {
        "seed": seed,
        "experiments": list(experiments),
        "methods": list(methods),
        "radii": list(radii),
        "results": results,
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(os.path.join(output_dir, "summary.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="One-seed NTPC depth/localization evaluation")
    parser.add_argument("--manifest", default="configs/ntpc_localization_seed42.yaml")
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    summary = run_manifest(args.manifest, max_samples=args.max_samples)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
