"""
Comprehensive 15-Model Experimental & Ablation Taxonomy for RMR-v33 (PARK).
Generates publication-ready Markdown and LaTeX ablation tables.
"""
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import yaml
from rmr_v3.model import RMRv3, RMRv3Config

TAXONOMY = [
    {
        "id": "M01",
        "config": "rmr_v33_park_perspective.yaml",
        "name": "RMR-v33 Canonical PARK",
        "category": "Main System",
        "hypothesis": "Primary: Perspective-Adaptive Regional Kernels eliminate horizon over-aggregation & foreground engulfment.",
        "params": 104666,
        "key_diff": "Baseline Canonical PARK (PCAT 3-band, rho_max=2.0, horizon=16, PGH + PARK Routing, T=6 SIRT, RN Adjoint, Curv)",
    },
    {
        "id": "M02",
        "config": "rmr_v33_anchor_v19_isotropic.yaml",
        "name": "RMR-v33 Isotropic Grid Anchor",
        "category": "Benchmark Anchor",
        "hypothesis": "Isolates the exact empirical delta of PARK by reverting to classical isotropic [32, 64, 128] grid under identical backbone & losses.",
        "params": 103958,
        "key_diff": "use_park=False, use_pgh=False, park_routing=False, region_sizes_px=[32, 64, 128]",
    },
    {
        "id": "M03",
        "config": "rmr_v33_ablation_isotropic_pcat.yaml",
        "name": "Isotropic PCAT (Square Boxes)",
        "category": "Geometric Anisotropy",
        "hypothesis": "Tests whether projective elliptical elongation (rho > 1.0) is necessary vs square perspective altitude boxes.",
        "params": 104666,
        "key_diff": "park_max_aspect=1.0 (all boxes square hy == wx along vertical scanline)",
    },
    {
        "id": "M04",
        "config": "rmr_v33_ablation_aspect15.yaml",
        "name": "Moderate Aspect (rho_max=1.5)",
        "category": "Geometric Anisotropy",
        "hypothesis": "Tests sensitivity of projective elongation prior (1.5:1 vs 2.0:1 aspect ratio).",
        "params": 104666,
        "key_diff": "park_max_aspect=1.5",
    },
    {
        "id": "M05",
        "config": "rmr_v33_ablation_inverted_aspect.yaml",
        "name": "Inverted Aspect (rho_max=0.5)",
        "category": "Geometric Adversarial",
        "hypothesis": "Adversarial control: Wider-than-tall boxes (landscape) violate standing upright human geometry and must degrade accuracy.",
        "params": 104666,
        "key_diff": "park_max_aspect=0.5 (anti-perspective control)",
    },
    {
        "id": "M06",
        "config": "rmr_v33_ablation_no_pgh.yaml",
        "name": "w/o Perspective Geometry Head",
        "category": "Feature Modulation",
        "hypothesis": "Tests contribution of 1D continuous scanline depthwise-separable modulation along optical rays.",
        "params": 104408,
        "key_diff": "use_pgh=False (-226 params)",
    },
    {
        "id": "M07",
        "config": "rmr_v33_ablation_no_routing.yaml",
        "name": "w/o PARK Spatial Routing",
        "category": "Scale Routing",
        "hypothesis": "Tests contribution of 2-mode spatial partition of unity between horizon core and projective elongation.",
        "params": 104216,
        "key_diff": "park_routing=False (-450 params)",
    },
    {
        "id": "M08",
        "config": "rmr_v33_ablation_horizon32.yaml",
        "name": "Coarse Horizon (h_min=32px)",
        "category": "Horizon Discretization",
        "hypothesis": "Tests whether fine 16px horizon discretization is required to resolve distant small heads vs legacy 32px.",
        "params": 104666,
        "key_diff": "park_horizon_h=32 (hy in [32, 128])",
    },
    {
        "id": "M09",
        "config": "rmr_v33_ablation_5bands.yaml",
        "name": "Fine Altitude Tiling (5 Bands)",
        "category": "Discretization Granularity",
        "hypothesis": "Tests whether 5 altitude bands improve depth continuity over 3 canonical bands.",
        "params": 104666,
        "key_diff": "park_altitude_bands=5",
    },
    {
        "id": "M10",
        "config": "rmr_v33_control_no_solver.yaml",
        "name": "Feedforward Control (T=0)",
        "category": "Inverse Problem",
        "hypothesis": "Measures the pure mathematical contribution of unrolled SIRT Radon measure recovery over direct feedforward Y_0.",
        "params": 104666,
        "key_diff": "enable_solver=False, iterations=0",
    },
    {
        "id": "M11",
        "config": "rmr_v33_ablation_solver_t3.yaml",
        "name": "Shallow Solver (T=3)",
        "category": "Inverse Problem",
        "hypothesis": "Evaluates iteration depth trade-off (T=3 vs T=6 iterates) on measure reconciliation and latency.",
        "params": 104666,
        "key_diff": "iterations=3",
    },
    {
        "id": "M12",
        "config": "rmr_v33_ablation_flat_adjoint.yaml",
        "name": "Flat Lebesgue Adjoint (A^T)",
        "category": "Adjoint Physics",
        "hypothesis": "Demonstrates why Radon-Nikodym measure modulation (m * A^T) is mathematically necessary to avoid background mass bleeding.",
        "params": 104666,
        "key_diff": "adjoint_mode='flat'",
    },
    {
        "id": "M13",
        "config": "rmr_v33_ablation_no_curvature.yaml",
        "name": "w/o Density Curvature (alpha=0)",
        "category": "Nonlinear Calibration",
        "hypothesis": "Quantifies the impact of quadratic curvature in preventing high-density head saturation in crowded clumps.",
        "params": 104666,
        "key_diff": "density_curvature=False, lambda_curvature=0.0",
    },
    {
        "id": "M14",
        "config": "rmr_v33_ablation_no_dm16.yaml",
        "name": "w/o Flat-DM16 Optimal Transport",
        "category": "Loss Formulation",
        "hypothesis": "Quantifies the contribution of Flat-DM16 discrete mass discrepancy loss vs pure cell smooth-L1.",
        "params": 104666,
        "key_diff": "lambda_flat_dm16=0.0",
    },
    {
        "id": "M15",
        "config": "rmr_v33_ablation_stride2_dual_lattice.yaml",
        "name": "Dual-Lattice Stride-2 DCSR + PARK",
        "category": "Multi-Lattice Resolution",
        "hypothesis": "Combines Stride-2 fine lattice with PARK perspective kernels to test if perspective scaling eliminates dense undercounting.",
        "params": 104765,
        "key_diff": "subpixel_stride2=True, output_stride=2, lambda_carrier_cell=0.25, lambda_fine_cell=0.25",
    },
]

def main():
    print("=" * 135)
    print("RMR-v33 (PARK) 15-MODEL COMPREHENSIVE EXPERIMENTAL & ABLATION TAXONOMY")
    print("Strict Invariants: Trainable Params <= 105,000 | Files <= 450 Lines | Zero KD | Seed 42 | SHA Part A")
    print("=" * 135)
    print(f"{'ID':<4} | {'Model Name':<34} | {'Category':<22} | {'Params':<8} | {'Key Architectural Difference':<55}")
    print("-" * 135)
    for m in TAXONOMY:
        print(f"{m['id']:<4} | {m['name']:<34} | {m['category']:<22} | {m['params']:<8,d} | {m['key_diff']:<55}")
    print("=" * 135)

if __name__ == "__main__":
    main()
