#!/usr/bin/env bash
# Bash Runner for Gen 7 Factorial Matrix & Breakthrough Ablation Suite
# 17 Experiments | Target: Breakthrough Sub-60 MAE (< 60.0)
# Constraints: <= 105,000 params, Zero KD, ShanghaiTech Part A
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

declare -a MODELS=(
    # 1. Base 4 Isolated Innovations
    "g7_h4_drs:configs/rmr_research/gen7/g7_h4_drs.yaml"
    "g7_h1_fidt:configs/rmr_research/gen7/g7_h1_fidt.yaml"
    "g7_h2_chfl:configs/rmr_research/gen7/g7_h2_chfl.yaml"
    "g7_cdw_factorized_diag:configs/rmr_research/gen7/g7_cdw_factorized_diag.yaml"

    # 2. Pairwise Interactions (2-way Orthogonal Factorials)
    "g7_h4_h1_drs_fidt:configs/rmr_research/gen7/g7_h4_h1_drs_fidt.yaml"
    "g7_h4_h2_drs_chfl:configs/rmr_research/gen7/g7_h4_h2_drs_chfl.yaml"
    "g7_h4_cdw:configs/rmr_research/gen7/g7_h4_cdw.yaml"
    "g7_h1_h2_fidt_chfl:configs/rmr_research/gen7/g7_h1_h2_fidt_chfl.yaml"
    "g7_cdw_h1_fidt:configs/rmr_research/gen7/g7_cdw_h1_fidt.yaml"
    "g7_cdw_h2_chfl:configs/rmr_research/gen7/g7_cdw_h2_chfl.yaml"

    # 3. Triads & Champion Synthesis
    "g7_triad_drs_fidt_cdw:configs/rmr_research/gen7/g7_triad_drs_fidt_cdw.yaml"
    "g7_triad_drs_chfl_cdw:configs/rmr_research/gen7/g7_triad_drs_chfl_cdw.yaml"
    "g7_champion_synthesis:configs/rmr_research/gen7/g7_champion_synthesis.yaml"

    # 4. Hyperparameter Sweeps & Sensitivity Controls
    "g7_h1_fidt_k2:configs/rmr_research/gen7/g7_h1_fidt_k2.yaml"
    "g7_h1_fidt_k6:configs/rmr_research/gen7/g7_h1_fidt_k6.yaml"
    "g7_h2_chfl_lam02:configs/rmr_research/gen7/g7_h2_chfl_lam02.yaml"
    "g7_h2_chfl_lam10:configs/rmr_research/gen7/g7_h2_chfl_lam10.yaml"
)

LOG_DIR="runs/sha_a/suite_logs_gen7"
mkdir -p "$LOG_DIR"

echo "================================================================================"
echo "  RMR GEN 7 COMPREHENSIVE FACTORIAL MATRIX & BREAKTHROUGH SUITE (Bash)"
echo "  17 Experiments | Target: Breakthrough Sub-60 MAE (< 60.0)"
echo "  Constraints: <= 105,000 params, Zero KD, ShanghaiTech Part A"
echo "================================================================================"

PYTHON_BIN="python"
if [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
fi

for ENTRY in "${MODELS[@]}"; do
    RUN_ID="${ENTRY%%:*}"
    CFG="${ENTRY##*:}"
    LOG_FILE="$LOG_DIR/${RUN_ID}.log"

    echo ""
    echo ">>> Starting training for: $RUN_ID ($CFG)"
    $PYTHON_BIN -m rmr_v3.train --config "$CFG" --run-id "$RUN_ID" --overwrite 2>&1 | tee "$LOG_FILE"
    echo ">>> Completed training: $RUN_ID"

    CKPT_PATH="runs/sha_a/${RUN_ID}/best_val_mae.pt"
    if [ -f "$CKPT_PATH" ]; then
        echo ">>> Evaluating Standard Test: $CKPT_PATH"
        EVAL_LOG="$LOG_DIR/${RUN_ID}_eval_direct.log"
        $PYTHON_BIN -m rmr_v3.eval --checkpoint "$CKPT_PATH" --no-tiling 2>&1 | tee "$EVAL_LOG"

        echo ">>> Evaluating Horizontal Flip TTA: $CKPT_PATH"
        EVAL_TTA_LOG="$LOG_DIR/${RUN_ID}_eval_tta.log"
        $PYTHON_BIN -m rmr_v3.eval --checkpoint "$CKPT_PATH" --tta --no-tiling 2>&1 | tee "$EVAL_TTA_LOG"
    fi
done

echo ""
echo ">>> Summarizing Gen 7 Suite Results:"
$PYTHON_BIN scripts/summarize_gen7_suite.py

echo ""
echo "All Gen 7 experiments completed!"
