#!/usr/bin/env bash
# ==============================================================================
# RMR V67 Complete Research Campaign (13 Closed-Loop Experiments)
# Target Venue: CVPR / ICCV / ECCV / TPAMI (Rank A* / Q1)
# Hardware: Ubuntu Linux Workstation (NVIDIA GeForce RTX 5070 Ti, 16GB GDDR7)
# ==============================================================================
set -euo pipefail

mkdir -p runs/sha_a/rmr_v67 runs/shb/rmr_v67 logs

echo "======================================================================"
echo "STARTING RMR V67 RESEARCH CAMPAIGN (13 CLOSED-LOOP EXPERIMENTS)"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Git commit: $(git rev-parse HEAD)"
echo "======================================================================"

# --- TIER 1: FLAGSHIP CHAMPIONS (Highest Priority) ---
echo "[1/13] Launching EXP 6: Flagship Track B (RD-FineHead + FiLM, 135k params)..."
python train.py --config configs/rmr_v67/v67_exp6_rd_finehead_first_principles_1500ep.yaml

echo "[2/13] Launching EXP 0: Flagship Track B (Standard Head, 130k params)..."
python train.py --config configs/rmr_v67/v67_exp0_gold_first_principles_1500ep.yaml

echo "[3/13] Launching EXP 1: Flagship Track A (Stride 4 Standard, 104k params)..."
python train.py --config configs/rmr_v67/v67_exp1_track_a_first_principles_1500ep.yaml

# --- TIER 2: ORTHOGONAL LOSS FACTORIAL SUITE (2^3 - 1 Matrix) ---
echo "[4/13] Launching EXP 4: Ablation Pure 2-Loss (Count + Bayesian, No DM16)..."
python train.py --config configs/rmr_v67/v67_exp4_abl_pure_2loss_no_dm16_1500ep.yaml

echo "[5/13] Launching EXP 7: Ablation Pure 2-Loss (Bayesian + DM16, No Count)..."
python train.py --config configs/rmr_v67/v67_exp7_abl_no_count_loss_1500ep.yaml

echo "[6/13] Launching EXP 12: Ablation Pure 2-Loss (Count + DM16, No Bayesian)..."
python train.py --config configs/rmr_v67/v67_exp12_abl_no_bayesian_1500ep.yaml

echo "[7/13] Launching EXP 2: Ablation Pure 1-Loss (Pure L1 Count Only)..."
python train.py --config configs/rmr_v67/v67_exp2_abl_pure_l1_count_only_1500ep.yaml

echo "[8/13] Launching EXP 3: Ablation Pure 1-Loss (Pure Bayesian Dirac Only)..."
python train.py --config configs/rmr_v67/v67_exp3_abl_pure_bayesian_only_1500ep.yaml

# --- TIER 3: SOLVER & ARCHITECTURE ABLATIONS ---
echo "[9/13] Launching EXP 8: Ablation Solver (Zero-Solver MC-BPRO Baseline)..."
python train.py --config configs/rmr_v67/v67_exp8_abl_no_solver_1500ep.yaml

echo "[10/13] Launching EXP 9: Ablation Arch (RD-FineHead without FiLM)..."
python train.py --config configs/rmr_v67/v67_exp9_abl_rd_finehead_no_film_1500ep.yaml

echo "[11/13] Launching EXP 5: Ablation Scale (Crop Size 256x256 Patch)..."
python train.py --config configs/rmr_v67/v67_exp5_abl_crop256_sota_patch_1500ep.yaml

# --- TIER 4: CROSS-DATASET GENERALIZATION (ShanghaiTech Part B) ---
echo "[12/13] Launching EXP 10: Track B RD-FineHead on ShanghaiTech Part B..."
python train.py --config configs/rmr_v67/v67_exp10_shb_track_b_rd_finehead_1500ep.yaml

echo "[13/13] Launching EXP 11: Track A Standard on ShanghaiTech Part B..."
python train.py --config configs/rmr_v67/v67_exp11_shb_track_a_first_principles_1500ep.yaml

echo "======================================================================"
echo "ALL 13 EXPERIMENTS IN THE V67 CAMPAIGN HAVE COMPLETED SUCCESSFULLY!"
echo "Date: $(date)"
echo "======================================================================"
