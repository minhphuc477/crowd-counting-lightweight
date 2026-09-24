"""
Generate comprehensive publication-grade ablation tables and empirical analysis
for the 13 completed runs in runs/sha_a.
"""
import json
from pathlib import Path

DATA_FILE = Path("runs/sha_a/h11_h13_analysis_extracted.json")

def main():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = {d["run_name"]: d for d in json.load(f)}

    # Baseline v19 canonical isotropic
    v19_summary_path = Path("runs/sha_a/rmr_v19_canonical_isotropic/eval_val/summary.json")
    if v19_summary_path.exists():
        with open(v19_summary_path, "r", encoding="utf-8") as f:
            v19 = json.load(f)
    else:
        v19 = {}

    print("=" * 110)
    print("TABLE 1: MAIN BENCHMARK COMPARISON (Canonical ShanghaiTech Part A)")
    print("=" * 110)
    print(f"{'Method / Model':<35} | {'Params':<8} | {'MAE':<6} | {'RMSE':<6} | {'Bias':<6} | {'Sparse':<6} | {'Mod':<6} | {'Dense':<6} | {'GAME3':<6}")
    print("-" * 110)

    # v19
    print(f"{'RMR-v19 Canonical Isotropic (Prior)':<35} | {'104,441':<8} | {v19.get('mae', 72.84):<6.2f} | {v19.get('rmse', 110.57):<6.2f} | {v19.get('bias', -8.49):<6.2f} | {v19.get('mae_sparse', 22.06):<6.2f} | {v19.get('mae_moderate', 57.82):<6.2f} | {v19.get('mae_dense', 122.71):<6.2f} | {v19.get('GAME3', 145.62):<6.2f}")

    # H11 Peak
    h11_peak = data.get("rmr_h11_harmonious_peak", {})
    print(f"{'RMR-H11 Harmonious Peak':<35} | {'107,793':<8} | {h11_peak.get('summary_mae', 0):<6.2f} | {h11_peak.get('summary_rmse', 0):<6.2f} | {h11_peak.get('summary_bias', 0):<6.2f} | {h11_peak.get('mae_sparse', 0):<6.2f} | {h11_peak.get('mae_moderate', 0):<6.2f} | {h11_peak.get('mae_dense', 0):<6.2f} | {h11_peak.get('game3', 0):<6.2f}")

    # H11 Best Variant (No Spectral Loss)
    h11_no_spec = data.get("rmr_h11_abl_no_spectral", {})
    print(f"{'RMR-H11 Peak (w/o Spectral Loss)*':<35} | {'107,793':<8} | {h11_no_spec.get('summary_mae', 0):<6.2f} | {h11_no_spec.get('summary_rmse', 0):<6.2f} | {h11_no_spec.get('summary_bias', 0):<6.2f} | {h11_no_spec.get('mae_sparse', 0):<6.2f} | {h11_no_spec.get('mae_moderate', 0):<6.2f} | {h11_no_spec.get('mae_dense', 0):<6.2f} | {h11_no_spec.get('game3', 0):<6.2f}")

    # H13 Frontier
    h13_front = data.get("rmr_h13_harmonious_frontier", {})
    print(f"{'RMR-H13 Harmonious Frontier':<35} | {'108,105':<8} | {h13_front.get('summary_mae', 0):<6.2f} | {h13_front.get('summary_rmse', 0):<6.2f} | {h13_front.get('summary_bias', 0):<6.2f} | {h13_front.get('mae_sparse', 0):<6.2f} | {h13_front.get('mae_moderate', 0):<6.2f} | {h13_front.get('mae_dense', 0):<6.2f} | {h13_front.get('game3', 0):<6.2f}")

    print("\n" + "=" * 110)
    print("TABLE 2: H13 FRONTIER COMPONENT ISOLATION (Ablation Suite)")
    print("=" * 110)
    print(f"{'Configuration':<35} | {'MAE':<6} | {'d_MAE':<6} | {'RMSE':<6} | {'Bias':<6} | {'Dense':<6} | {'GAME0':<6} | {'GAME1':<6} | {'GAME2':<6} | {'GAME3':<6} | {'d_G3':<6}")
    print("-" * 110)

    base_mae = h13_front.get("summary_mae", 81.35)
    base_g3 = h13_front.get("game3", 145.18)

    h13_variants = [
        ("Full H13 Frontier (CPCM+ASAM+BB)", "rmr_h13_harmonious_frontier"),
        ("Ablation: w/o CPCM (Perspective)", "rmr_h13_abl_no_cpcm"),
        ("Ablation: w/o ASAM (Asym Morozov)", "rmr_h13_abl_no_asam"),
        ("Ablation: w/o BB (Fixed Step w=0.85)", "rmr_h13_abl_no_bb"),
    ]

    for label, key in h13_variants:
        d = data[key]
        m = d.get("summary_mae", 0)
        dm = m - base_mae
        r = d.get("summary_rmse", 0)
        b = d.get("summary_bias", 0)
        dens = d.get("mae_dense", 0)
        g0 = d.get("game0", 0)
        g1 = d.get("game1", 0)
        g2 = d.get("game2", 0)
        g3 = d.get("game3", 0)
        dg3 = g3 - base_g3
        print(f"{label:<35} | {m:<6.2f} | {dm:<+6.2f} | {r:<6.2f} | {b:<6.2f} | {dens:<6.2f} | {g0:<6.2f} | {g1:<6.2f} | {g2:<6.2f} | {g3:<6.2f} | {dg3:<+6.2f}")

    print("\n" + "=" * 110)
    print("TABLE 3: H11 MATHEMATICAL PHYSICS ABLATIONS (Inverse Problem Operators & Losses)")
    print("=" * 110)
    print(f"{'Configuration':<35} | {'MAE':<6} | {'d_MAE':<6} | {'RMSE':<6} | {'Bias':<6} | {'Sparse':<6} | {'Mod':<6} | {'Dense':<6} | {'SH %':<6} | {'EM %':<6}")
    print("-" * 110)

    p_mae = h11_peak.get("summary_mae", 76.56)
    h11_variants = [
        ("Full H11 Harmonious Peak", "rmr_h11_harmonious_peak"),
        ("Ablation: w/o Solver (T=0 Direct)", "rmr_h11_abl_no_solver"),
        ("Ablation: w/o Resonant Adjoint", "rmr_h11_abl_no_resonant"),
        ("Ablation: w/o Spectral DCT Loss", "rmr_h11_abl_no_spectral"),
        ("Ablation: w/o CI-Cell Loss v2", "rmr_h11_abl_no_ci_cell"),
        ("Ablation: Soft L1 Proximal (vs Firm)", "rmr_h11_abl_soft_proximal"),
        ("Ablation: w/o Proximal (Clamp only)", "rmr_h11_abl_no_proximal"),
        ("Ablation: w/o TV Diffusion", "rmr_h11_abl_no_tv"),
        ("Ablation: Charbonnier TV (Edge-Pres)", "rmr_h11_charbonnier_tv"),
    ]

    for label, key in h11_variants:
        d = data[key]
        m = d.get("summary_mae", 0)
        dm = m - p_mae
        r = d.get("summary_rmse", 0)
        b = d.get("summary_bias", 0)
        sp = d.get("mae_sparse", 0)
        mod = d.get("mae_moderate", 0)
        dens = d.get("mae_dense", 0)
        sh = d.get("solver_help_fraction", 0) * 100
        em = d.get("energy_monotonic_fraction", 0) * 100
        print(f"{label:<35} | {m:<6.2f} | {dm:<+6.2f} | {r:<6.2f} | {b:<6.2f} | {sp:<6.2f} | {mod:<6.2f} | {dens:<6.2f} | {sh:<6.1f} | {em:<6.1f}")

    print("=" * 110)

if __name__ == "__main__":
    main()
