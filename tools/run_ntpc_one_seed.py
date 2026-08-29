"""Sequential one-seed launcher for the initial NTPC experiment table."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import yaml

_REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT)

from hpc.models.factory import validate_pretrained_normalization


def _repo_path(path: str, repository_root: str) -> str:
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(repository_root, path))


def load_and_validate(manifest_path: str) -> tuple[int, list[str], str, str]:
    manifest_path = os.path.abspath(manifest_path)
    repository_root = os.path.dirname(os.path.dirname(manifest_path))
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    seed = int(manifest.get("seed", 42))
    configs = [_repo_path(path, repository_root) for path in manifest["training_order"]]
    localization = _repo_path(manifest["localization_manifest"], repository_root)
    seen_names = set()
    for config_path in configs:
        if not os.path.isfile(config_path):
            raise FileNotFoundError(config_path)
        with open(config_path, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        validate_pretrained_normalization(cfg)
        if int(cfg["experiment"].get("seed", seed)) != seed:
            raise ValueError(f"Seed mismatch in {config_path}")
        name = cfg["experiment"]["name"]
        if name in seen_names:
            raise ValueError(f"Duplicate experiment name: {name}")
        seen_names.add(name)
    return seed, configs, localization, repository_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the initial NTPC study with one seed")
    parser.add_argument("--manifest", default="configs/ntpc_one_seed_experiments.yaml")
    parser.add_argument("--stage", choices=["train", "localization", "all"], default="train")
    parser.add_argument("--only", nargs="*", help="Optional experiment names to train")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-localization-samples", type=int)
    args = parser.parse_args()
    seed, configs, localization, repository_root = load_and_validate(args.manifest)
    print(f"Initial NTPC study uses exactly one seed: {seed}")

    if args.stage in {"train", "all"}:
        for config_path in configs:
            with open(config_path, "r", encoding="utf-8") as handle:
                name = yaml.safe_load(handle)["experiment"]["name"]
            if args.only and name not in set(args.only):
                continue
            command = [sys.executable, "train_ntpc.py", "--config", config_path]
            print("RUN", " ".join(command))
            if not args.dry_run:
                subprocess.run(command, cwd=repository_root, check=True)

    if args.stage in {"localization", "all"}:
        command = [
            sys.executable,
            "tools/eval_ntpc_localization_depth.py",
            "--manifest",
            localization,
        ]
        if args.max_localization_samples is not None:
            command.extend(["--max-samples", str(args.max_localization_samples)])
        print("RUN", " ".join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=repository_root, check=True)


if __name__ == "__main__":
    main()
