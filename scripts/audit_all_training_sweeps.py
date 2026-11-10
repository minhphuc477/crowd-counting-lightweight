#!/usr/bin/env python3
"""Comprehensive audit of ALL configs, benchmarks, ablations, training sweeps, and cross-domain evals."""
from pathlib import Path
import json
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent


def count_epochs(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    if len(lines) <= 1:
        return 0
    try:
        return int(lines[-1].split(",")[0])
    except Exception:
        return len(lines) - 1


def inspect_config_group(title: str, yaml_paths: list[Path]) -> None:
    print("\n" + "=" * 115)
    print(f"{title}")
    print("=" * 115)
    print(f"{'Config Name':42s} | {'Status':13s} | {'Epochs':10s} | {'last.pt':7s} | {'MAE':>7s} | {'RMSE':>7s} | {'Bias':>7s}")
    print("-" * 115)

    for c_path in sorted(yaml_paths):
        with open(c_path, "r", encoding="utf-8") as f:
            c = yaml.safe_load(f)
        target_epochs = int(c.get("train", {}).get("epochs", 0))
        out_dir = _REPO_ROOT / c["output_dir"]
        name = c_path.stem
        log_p = out_dir / "train_log.csv"
        eval_p = out_dir / "eval_val" / "summary.json"
        last_p = out_dir / "last.pt"

        cur_epoch = count_epochs(log_p)
        has_last = "YES" if last_p.exists() else ("DIR_ONLY" if out_dir.exists() and any(out_dir.iterdir()) else "NO")

        mae_str, rmse_str, bias_str = "-", "-", "-"
        if eval_p.exists():
            with open(eval_p, "r", encoding="utf-8") as f:
                d = json.load(f)
            mae_str = f"{d.get('MAE', d.get('mae', 0.0)):7.2f}"
            rmse_str = f"{d.get('RMSE', d.get('rmse', 0.0)):7.2f}"
            bias_str = f"{d.get('Bias', d.get('bias', 0.0)):+7.2f}"

        if cur_epoch == 0:
            status = "DIRTY_EMPTY" if out_dir.exists() and any(out_dir.iterdir()) else "NOT_STARTED"
        elif cur_epoch < target_epochs:
            status = "INCOMPLETE"
        else:
            status = "COMPLETED"

        ep_str = f"{cur_epoch}/{target_epochs}"
        print(f"{name:42s} | {status:13s} | {ep_str:10s} | {has_last:7s} | {mae_str:>7s} | {rmse_str:>7s} | {bias_str:>7s}")


def main() -> None:
    inspect_config_group(
        "1. BENCHMARKS (Direct & UCF_CC_50)",
        list((_REPO_ROOT / "configs" / "benchmarks").glob("*.yaml")),
    )
    inspect_config_group(
        "2. ABLATIONS - JHU",
        list((_REPO_ROOT / "configs" / "benchmarks" / "ablations_jhu").glob("*.yaml")),
    )
    inspect_config_group(
        "3. ABLATIONS - QNRF",
        list((_REPO_ROOT / "configs" / "benchmarks" / "ablations_qnrf").glob("*.yaml")),
    )
    inspect_config_group(
        "4. ABLATIONS - SHA",
        list((_REPO_ROOT / "configs" / "benchmarks" / "ablations_sha").glob("*.yaml")),
    )
    inspect_config_group(
        "5. ABLATIONS - SHB",
        list((_REPO_ROOT / "configs" / "benchmarks" / "ablations_shb").glob("*.yaml")),
    )
    inspect_config_group(
        "6. TRAINING SWEEPS - JHU",
        list((_REPO_ROOT / "configs" / "training_sweeps" / "jhu").glob("*.yaml")),
    )
    inspect_config_group(
        "7. TRAINING SWEEPS - QNRF",
        list((_REPO_ROOT / "configs" / "training_sweeps" / "qnrf").glob("*.yaml")),
    )
    inspect_config_group(
        "8. TRAINING SWEEPS - SHA",
        list((_REPO_ROOT / "configs" / "training_sweeps").glob("*.yaml")),
    )

    print("\n" + "=" * 115)
    print("9. CROSS-DOMAIN & TEST EVALUATIONS IN runs/")
    print("=" * 115)
    for d in sorted((_REPO_ROOT / "runs").glob("eval_*")):
        s_p = d / "summary.json"
        if s_p.exists():
            data = json.load(open(s_p))
            print(f"{d.name:35s} | COMPLETED | MAE={data.get('MAE', 0.0):.2f} | RMSE={data.get('RMSE', 0.0):.2f}")
        else:
            print(f"{d.name:35s} | INCOMPLETE/EMPTY")

    for name in ["exp_jhu_direct_pretrained_104k", "exp_jhu_direct_scratch_104k"]:
        base_d = _REPO_ROOT / "runs" / "benchmarks" / name
        has_eval_test = (base_d / "eval_test" / "summary.json").exists()
        has_attr = (base_d / "attribute_breakdown.json").exists()
        print(f"{name:35s} | eval_test/summary.json={has_eval_test} | attribute_breakdown.json={has_attr}")


if __name__ == "__main__":
    main()
