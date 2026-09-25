"""RMR-v34 (DiAG + DSMP) Comprehensive Research & Ablation Suite Orchestrator.

Manages sequential execution, single-variable ablation validation,
and automated publication-ready table generation for the RMR-v34 series.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import yaml

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rmr_v3.config.validator import validate_v3_config
from rmr_v3.model import RMRv3, RMRv3Config

V34_CONFIG_DIR = Path("configs/rmr_v34")

ABLATION_CATALOG: dict[str, dict[str, str]] = {
    # Baseline
    "rmr_v34_diag_canonical": {
        "config": "configs/rmr_v34/rmr_v34_diag_canonical.yaml",
        "category": "Baseline",
        "desc": "Canonical Baseline (DiAG Router, DSMP Protection, T=6, SNR Weighting)",
    },
    "rmr_v34_shb_canonical": {
        "config": "configs/rmr_v34/rmr_v34_shb_canonical.yaml",
        "category": "Cross-Dataset",
        "desc": "ShanghaiTech Part B Benchmark (Sparse wide-angle surveillance)",
    },
    # DiAG Geometry Routing
    "rmr_v34_abl_no_diag": {
        "config": "configs/rmr_v34/rmr_v34_abl_no_diag.yaml",
        "category": "DiAG Routing",
        "desc": "Ablate DiAG Routing (Uniform isotropic multiscale weights)",
    },
    "rmr_v34_abl_no_dcap_tilt": {
        "config": "configs/rmr_v34/rmr_v34_abl_no_dcap_tilt.yaml",
        "category": "DiAG Routing",
        "desc": "Ablate DCAP Scene Tilt (Disable global perspective contrast scaling)",
    },
    "rmr_v34_abl_no_scale_align": {
        "config": "configs/rmr_v34/rmr_v34_abl_no_scale_align.yaml",
        "category": "DiAG Routing",
        "desc": "Ablate Scale Alignment Loss (lambda_scale_align = 0.0)",
    },
    "rmr_v34_abl_vdp_dcap": {
        "config": "configs/rmr_v34/rmr_v34_abl_vdp_dcap.yaml",
        "category": "DiAG Routing",
        "desc": "Upgrade DCAP with Vertical Differential Pooling (VDP, 104,701 params)",
    },
    # DSMP Discrete Measure Protection
    "rmr_v34_abl_no_hurdle": {
        "config": "configs/rmr_v34/rmr_v34_abl_no_hurdle.yaml",
        "category": "DSMP Protection",
        "desc": "Ablate Hurdle Head (No regional occupancy gating)",
    },
    "rmr_v34_abl_no_hard_bg": {
        "config": "configs/rmr_v34/rmr_v34_abl_no_hard_bg.yaml",
        "category": "DSMP Protection",
        "desc": "Ablate Top-K Hard Background Mining (lambda_hard_bg = 0.0)",
    },
    "rmr_v34_abl_no_ci_cell": {
        "config": "configs/rmr_v34/rmr_v34_abl_no_ci_cell.yaml",
        "category": "DSMP Protection",
        "desc": "Ablate CI-Cell v2 (Standard balanced L1 instead of count-invariance)",
    },
    "rmr_v34_abl_no_proximal": {
        "config": "configs/rmr_v34/rmr_v34_abl_no_proximal.yaml",
        "category": "DSMP Protection",
        "desc": "Ablate Proximal Thresholding (No thresholding in Landweber updates)",
    },
    "rmr_v34_abl_soft_proximal": {
        "config": "configs/rmr_v34/rmr_v34_abl_soft_proximal.yaml",
        "category": "DSMP Protection",
        "desc": "Ablate Firm Thresholding (Standard L1 soft-shrinkage vs MCP firm)",
    },
    # Inverse Problem Solver & Dynamics
    "rmr_v34_abl_no_solver": {
        "config": "configs/rmr_v34/rmr_v34_abl_no_solver.yaml",
        "category": "Inverse Solver",
        "desc": "Ablate Iterative Solver (T=0 direct FineMeasureHead prediction)",
    },
    "rmr_v34_abl_solver_t2": {
        "config": "configs/rmr_v34/rmr_v34_abl_solver_t2.yaml",
        "category": "Inverse Solver",
        "desc": "Fast Solver Contraction Depth (T=2 iterations vs canonical T=6)",
    },
    "rmr_v34_abl_no_resonant": {
        "config": "configs/rmr_v34/rmr_v34_abl_no_resonant.yaml",
        "category": "Inverse Solver",
        "desc": "Ablate Resonant Adjoint (No carrier Laplacian momentum)",
    },
    "rmr_v34_abl_no_curvature": {
        "config": "configs/rmr_v34/rmr_v34_abl_no_curvature.yaml",
        "category": "Inverse Solver",
        "desc": "Ablate Density Curvature Regularization (lambda_curvature = 0.0)",
    },
    "rmr_v34_abl_uniform_reliability": {
        "config": "configs/rmr_v34/rmr_v34_abl_uniform_reliability.yaml",
        "category": "Inverse Solver",
        "desc": "Ablate SNR Reliability Weighting (Unweighted residual updates)",
    },
    # Multi-Seed Reproducibility
    "rmr_v34_seed123": {
        "config": "configs/rmr_v34/rmr_v34_seed123.yaml",
        "category": "Multi-Seed",
        "desc": "Statistical Variance Verification (Seed: 123)",
    },
    "rmr_v34_seed456": {
        "config": "configs/rmr_v34/rmr_v34_seed456.yaml",
        "category": "Multi-Seed",
        "desc": "Statistical Variance Verification (Seed: 456)",
    },
}


def list_ablations() -> None:
    print("=" * 86)
    print(f"{'RMR-v34 ABLATION SUITE CATALOG':^86}")
    print("=" * 86)
    print(f"{'Key':<32} {'Category':<16} {'Params':<8} {'Description'}")
    print("-" * 86)

    for key, item in ABLATION_CATALOG.items():
        cfg_path = Path(item["config"])
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8-sig"))
        m_cfg = RMRv3Config.from_dict(raw.get("model", {}))
        m_cfg.pretrained = False
        m = RMRv3(m_cfg)
        params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"{key:<32} {item['category']:<16} {params:>7,}  {item['desc']}")
    print("=" * 86)


def run_single(name: str, item: dict[str, str], epochs: int | None = None, dry_run: bool = False) -> int:
    cmd = [
        sys.executable,
        "rmr_v3/train.py",
        "--config", item["config"],
    ]
    if dry_run:
        cmd.extend(["--epochs", "1", "--eval-every", "1", "--output-dir", f"runs/dryrun_{name}"])
    elif epochs is not None:
        cmd.extend(["--epochs", str(epochs)])

    print(f"\n>>> Running: {name} ({item['desc']})")
    print(f"    Command: {' '.join(cmd)}")
    start = time.time()
    res = subprocess.run(cmd)
    elapsed = time.time() - start
    status = "SUCCESS" if res.returncode == 0 else f"FAILED (code {res.returncode})"
    print(f"<<< Completed: {name} in {elapsed:.1f}s | Status: {status}")

    if dry_run and res.returncode == 0:
        import shutil
        out_dir = Path(f"runs/dryrun_{name}")
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)

    return res.returncode


def generate_table() -> None:
    print("\n### RMR-v34 (DiAG + DSMP) Ablation Matrix Results\n")
    print("| Hypothesis / Config | Category | Params | MAE | RMSE | GAME0 | GAME3 | Dense MAE |")
    print("|---|---|---|---|---|---|---|---|")

    for key, item in ABLATION_CATALOG.items():
        cfg_path = Path(item["config"])
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8-sig"))
        out_dir = Path(raw.get("output_dir", f"runs/sha_a/{key}"))
        summary_path = out_dir / "summary.json"

        m_cfg = RMRv3Config.from_dict(raw.get("model", {}))
        m_cfg.pretrained = False
        m = RMRv3(m_cfg)
        params = sum(p.numel() for p in m.parameters() if p.requires_grad)

        if summary_path.exists():
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            mae = f"{data.get('best_val_mae', 0.0):.2f}"
            rmse = f"{data.get('best_val_rmse', 0.0):.2f}"
            g0 = f"{data.get('best_val_game0', 0.0):.2f}"
            g3 = f"{data.get('best_val_game3', 0.0):.2f}"
            dense = f"{data.get('best_val_mae_dense', 0.0):.2f}"
        else:
            mae, rmse, g0, g3, dense = "Pending", "Pending", "Pending", "Pending", "Pending"

        print(f"| `{key}` | {item['category']} | {params:,} | {mae} | {rmse} | {g0} | {g3} | {dense} |")


def main() -> None:
    ap = argparse.ArgumentParser(description="RMR-v34 Ablation Suite Runner")
    ap.add_argument("--list", action="store_true", help="List all ablations in the catalog")
    ap.add_argument("--ablation", choices=list(ABLATION_CATALOG.keys()), help="Run specific ablation")
    ap.add_argument("--all", action="store_true", help="Run all ablations sequentially")
    ap.add_argument("--dry-run", action="store_true", help="Execute 1-epoch dry-run verification for all/specified")
    ap.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    ap.add_argument("--generate-table", action="store_true", help="Generate Markdown results table from outputs")
    args = ap.parse_args()

    if args.list:
        list_ablations()
        return

    if args.generate_table:
        generate_table()
        return

    if args.ablation:
        run_single(args.ablation, ABLATION_CATALOG[args.ablation], epochs=args.epochs, dry_run=args.dry_run)
        return

    if args.all:
        for name, item in ABLATION_CATALOG.items():
            rc = run_single(name, item, epochs=args.epochs, dry_run=args.dry_run)
            if rc != 0:
                print(f"[ERROR] Ablation {name} failed with code {rc}. Stopping suite.")
                sys.exit(rc)
        return

    list_ablations()


if __name__ == "__main__":
    main()
