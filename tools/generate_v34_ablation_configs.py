"""Generate authoritative, validated ablation configs for RMR-v34 (DiAG + DSMP).

Each ablation configuration cleanly modifies exactly one targeted variable
relative to the canonical baseline (rmr_v34_diag_canonical.yaml), ensuring
strict single-factor hypothesis isolation under the <= 105,000 parameter budget.
"""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import yaml

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rmr_v3.config.validator import validate_v3_config
from rmr_v3.model import RMRv3, RMRv3Config

CANONICAL_PATH = Path("configs/rmr_v34/rmr_v34_diag_canonical.yaml")
OUTPUT_DIR = Path("configs/rmr_v34")

ABLATIONS: dict[str, dict] = {
    # 1. ShanghaiTech Part B Benchmark
    "rmr_v34_shb_canonical.yaml": {
        "desc": "ShanghaiTech Part B Benchmark (316 test samples, sparse surveillance)",
        "mod": lambda c: (
            c["data"].update({
                "train_manifest": "data/shb_train_all.jsonl",
                "val_manifest": "data/shb_test.jsonl",
            }),
            c["eval"].update({"density_bins": [50.0, 200.0]}),
            c.update({"output_dir": "runs/shb/rmr_v34_shb_canonical"}),
        ),
    },
    # 2. DiAG Dynamic Routing Hypothesis Isolation
    "rmr_v34_abl_no_diag.yaml": {
        "desc": "DiAG Routing Ablation (use_diag: false, uniform isotropic multiscale weights)",
        "mod": lambda c: (
            c["model"].update({"use_diag": False}),
            c.update({"output_dir": "runs/sha_a/rmr_v34_abl_no_diag"}),
        ),
    },
    "rmr_v34_abl_no_dcap_tilt.yaml": {
        "desc": "DCAP Perspective Contrast Tilt Ablation (use_dcap_tilt: false, pure local carrier)",
        "mod": lambda c: (
            c["model"].update({"use_dcap_tilt": False}),
            c.update({"output_dir": "runs/sha_a/rmr_v34_abl_no_dcap_tilt"}),
        ),
    },
    "rmr_v34_abl_no_scale_align.yaml": {
        "desc": "Scale Alignment Supervision Ablation (lambda_scale_align: 0.0)",
        "mod": lambda c: (
            c["loss"].update({"lambda_scale_align": 0.0}),
            c.update({"output_dir": "runs/sha_a/rmr_v34_abl_no_scale_align"}),
        ),
    },
    "rmr_v34_abl_no_vdp.yaml": {
        "desc": "Ablate VDP DCAP (Revert to 2D GAP DCAP, 104,573 params)",
        "mod": lambda c: (
            c["model"].update({"use_vertical_gradient_dcap": False}),
            c.update({"output_dir": "runs/sha_a/rmr_v34_abl_no_vdp"}),
        ),
    },
    "rmr_v34_abl_scale_prior.yaml": {
        "desc": "Scale-Conditioned Fine Head Prior Coupling (+6 params, 104,707 params)",
        "mod": lambda c: (
            c["model"].update({"scale_conditioned_prior": True}),
            c.update({"output_dir": "runs/sha_a/rmr_v34_abl_scale_prior"}),
        ),
    },
    # 3. DSMP Measure Protection Pillar Isolation
    "rmr_v34_abl_no_hurdle.yaml": {
        "desc": "Hurdle Occupancy Gating Ablation (hurdle_head: false, lambda_hurdle: 0.0)",
        "mod": lambda c: (
            c["model"].update({"hurdle_head": False}),
            c["loss"].update({"lambda_hurdle": 0.0}),
            c.update({"output_dir": "runs/sha_a/rmr_v34_abl_no_hurdle"}),
        ),
    },
    "rmr_v34_abl_no_hard_bg.yaml": {
        "desc": "Top-K Hard Background Mining Ablation (lambda_hard_bg: 0.0)",
        "mod": lambda c: (
            c["loss"].update({"lambda_hard_bg": 0.0}),
            c.update({"output_dir": "runs/sha_a/rmr_v34_abl_no_hard_bg"}),
        ),
    },
    "rmr_v34_abl_no_ci_cell.yaml": {
        "desc": "CI-Cell v2 Count-Invariance Ablation (cell_loss_mode: balanced)",
        "mod": lambda c: (
            c["loss"].update({"cell_loss_mode": "balanced"}),
            c.update({"output_dir": "runs/sha_a/rmr_v34_abl_no_ci_cell"}),
        ),
    },
    "rmr_v34_abl_no_proximal.yaml": {
        "desc": "Proximal Thresholding Ablation (proximal_mode: none)",
        "mod": lambda c: (
            c["model"].update({"proximal_mode": "none"}),
            c.update({"output_dir": "runs/sha_a/rmr_v34_abl_no_proximal"}),
        ),
    },
    "rmr_v34_abl_soft_proximal.yaml": {
        "desc": "Proximal Thresholding Mode Ablation (proximal_mode: soft vs MCP firm)",
        "mod": lambda c: (
            c["model"].update({"proximal_mode": "soft"}),
            c.update({"output_dir": "runs/sha_a/rmr_v34_abl_soft_proximal"}),
        ),
    },
    # 4. Inverse Problem Solver & Dynamics Isolation
    "rmr_v34_abl_no_solver.yaml": {
        "desc": "Iterative Inverse Solver Control (enable_solver: false, T=0 direct prediction)",
        "mod": lambda c: (
            c["model"].update({"enable_solver": False}),
            c.update({"output_dir": "runs/sha_a/rmr_v34_abl_no_solver"}),
        ),
    },
    "rmr_v34_abl_solver_t2.yaml": {
        "desc": "Solver Contraction Depth Ablation (iterations: 2 vs T=6)",
        "mod": lambda c: (
            c["model"].update({"iterations": 2}),
            c.update({"output_dir": "runs/sha_a/rmr_v34_abl_solver_t2"}),
        ),
    },
    "rmr_v34_abl_solver_t8.yaml": {
        "desc": "Deep Solver Contraction Depth Ablation (iterations: 8 vs T=6)",
        "mod": lambda c: (
            c["model"].update({"iterations": 8}),
            c.update({"output_dir": "runs/sha_a/rmr_v34_abl_solver_t8"}),
        ),
    },
    "rmr_v34_abl_asym_morozov.yaml": {
        "desc": "Asymmetric Morozov Poisson Discrepancy (asymmetric_morozov: true)",
        "mod": lambda c: (
            c["model"].update({"asymmetric_morozov": True}),
            c.update({"output_dir": "runs/sha_a/rmr_v34_abl_asym_morozov"}),
        ),
    },
    "rmr_v34_abl_no_resonant.yaml": {
        "desc": "Resonant Adjoint Carrier Momentum Ablation (resonant_adjoint: false)",
        "mod": lambda c: (
            c["model"].update({"resonant_adjoint": False}),
            c.update({"output_dir": "runs/sha_a/rmr_v34_abl_no_resonant"}),
        ),
    },
    "rmr_v34_abl_no_curvature.yaml": {
        "desc": "Density Curvature Regularization Ablation (density_curvature: false, lambda_curv: 0.0)",
        "mod": lambda c: (
            c["model"].update({"density_curvature": False, "gated_density_curvature": False}),
            c["loss"].update({"lambda_curvature": 0.0}),
            c.update({"output_dir": "runs/sha_a/rmr_v34_abl_no_curvature"}),
        ),
    },
    "rmr_v34_abl_uniform_reliability.yaml": {
        "desc": "SNR Reliability Weighting Ablation (uniform_reliability: true)",
        "mod": lambda c: (
            c["model"].update({"uniform_reliability": True}),
            c.update({"output_dir": "runs/sha_a/rmr_v34_abl_uniform_reliability"}),
        ),
    },
    # 5. Multi-Seed Statistical Reproducibility
    "rmr_v34_seed123.yaml": {
        "desc": "Multi-Seed Reproducibility Check (seed: 123)",
        "mod": lambda c: (
            c.update({"seed": 123, "output_dir": "runs/sha_a/rmr_v34_seed123"}),
        ),
    },
    "rmr_v34_seed456.yaml": {
        "desc": "Multi-Seed Reproducibility Check (seed: 456)",
        "mod": lambda c: (
            c.update({"seed": 456, "output_dir": "runs/sha_a/rmr_v34_seed456"}),
        ),
    },
}


def main() -> None:
    canonical_raw = yaml.safe_load(CANONICAL_PATH.read_text(encoding="utf-8-sig"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating {len(ABLATIONS)} ablation configurations from {CANONICAL_PATH}...")

    for fname, spec in ABLATIONS.items():
        cfg = copy.deepcopy(canonical_raw)
        spec["mod"](cfg)

        # Validate syntax, fields, and paths
        validate_v3_config(cfg)

        # Validate parameter ceiling <= 105,000
        m_cfg = RMRv3Config.from_dict(cfg.get("model", {}))
        m_cfg.pretrained = False
        model = RMRv3(m_cfg)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert params <= 105000, f"{fname} parameter count {params} exceeds 105,000 ceiling!"

        out_path = OUTPUT_DIR / fname
        content = f"# {spec['desc']}\n" + yaml.safe_dump(cfg, sort_keys=False)
        out_path.write_text(content, encoding="utf-8")
        print(f"  [OK] {fname:<34} | Params: {params:>7,} | {spec['desc']}")

    print(f"\nAll {len(ABLATIONS)} ablation configurations generated and verified successfully!")


if __name__ == "__main__":
    main()
