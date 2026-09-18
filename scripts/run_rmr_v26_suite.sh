#!/usr/bin/env bash
# Bash Runner for RMR-v26 Benchmark & Ablation Suite
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

declare -a MODELS=(
    "rmr_v26_canonical:configs/rmr_v26/rmr_v26_canonical.yaml"
    "rmr_v26_ablation_no_elevation:configs/rmr_v26/rmr_v26_ablation_no_elevation.yaml"
    "rmr_v26_ablation_no_bb:configs/rmr_v26/rmr_v26_ablation_no_bb.yaml"
    "rmr_v26_ablation_no_morozov:configs/rmr_v26/rmr_v26_ablation_no_morozov.yaml"
    "rmr_v26_ablation_with_curv01:configs/rmr_v26/rmr_v26_ablation_with_curv01.yaml"
    "rmr_v26_control_no_solver:configs/rmr_v26/rmr_v26_control_no_solver.yaml"
)

LOG_DIR="runs/sha_a/suite_logs_v26"
mkdir -p "$LOG_DIR"

echo "================================================================================"
echo "  RMR-v26 BENCHMARK & ABLATION EXPERIMENT SUITE (Bash)"
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
    $PYTHON_BIN -m rmr_v3.train --config "$CFG" --run-id "$RUN_ID" 2>&1 | tee "$LOG_FILE"
    echo ">>> Completed: $RUN_ID"

    CKPT_PATH="runs/sha_a/${RUN_ID}/best_val_mae.pt"
    if [ -f "$CKPT_PATH" ]; then
        echo ">>> Evaluating with Horizontal Flip TTA: $CKPT_PATH"
        EVAL_LOG="$LOG_DIR/${RUN_ID}_eval_tta.log"
        $PYTHON_BIN -m rmr_v3.eval --checkpoint "$CKPT_PATH" --tta --no-tiling 2>&1 | tee "$EVAL_LOG"
    fi
done

echo ""
echo ">>> Summarizing RMR-v26 Suite Results:"
$PYTHON_BIN scripts/summarize_rmr_v26_suite.py

echo ""
echo "All RMR-v26 experiments completed!"
