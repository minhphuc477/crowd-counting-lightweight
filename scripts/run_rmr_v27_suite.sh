#!/usr/bin/env bash
# Bash Runner for RMR-v27 Comprehensive Benchmark & Ablation Suite
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

declare -a MODELS=(
    "rmr_v27_canonical_restored:configs/rmr_v27/rmr_v27_canonical_restored.yaml"
    "rmr_v27_sota_push:configs/rmr_v27/rmr_v27_sota_push.yaml"
    "rmr_v27_ablation_cyclic_bb:configs/rmr_v27/rmr_v27_ablation_cyclic_bb.yaml"
    "rmr_v27_ablation_no_bb:configs/rmr_v27/rmr_v27_ablation_no_bb.yaml"
    "rmr_v27_ablation_morozov05:configs/rmr_v27/rmr_v27_ablation_morozov05.yaml"
    "rmr_v27_ablation_no_morozov:configs/rmr_v27/rmr_v27_ablation_no_morozov.yaml"
    "rmr_v27_ablation_curv035:configs/rmr_v27/rmr_v27_ablation_curv035.yaml"
    "rmr_v27_ablation_no_curvature:configs/rmr_v27/rmr_v27_ablation_no_curvature.yaml"
    "rmr_v27_ablation_no_scale_align:configs/rmr_v27/rmr_v27_ablation_no_scale_align.yaml"
    "rmr_v27_ablation_no_hard_bg:configs/rmr_v27/rmr_v27_ablation_no_hard_bg.yaml"
    "rmr_v27_control_no_solver:configs/rmr_v27/rmr_v27_control_no_solver.yaml"
)

LOG_DIR="runs/sha_a/suite_logs_v27"
mkdir -p "$LOG_DIR"

echo "================================================================================"
echo "  RMR-v27 COMPREHENSIVE BENCHMARK & ABLATION EXPERIMENT SUITE (Bash)"
echo "  Gold Reference: RMR-v19 Canonical Isotropic (72.61 TTA MAE, 72.84 Direct MAE)"
echo "  Target: Sub-70 MAE (< 69.80)"
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
    echo ">>> Completed: $RUN_ID"

    CKPT_PATH="runs/sha_a/${RUN_ID}/best_val_mae.pt"
    if [ -f "$CKPT_PATH" ]; then
        echo ">>> Evaluating with Horizontal Flip TTA: $CKPT_PATH"
        EVAL_LOG="$LOG_DIR/${RUN_ID}_eval_tta.log"
        $PYTHON_BIN -m rmr_v3.eval --checkpoint "$CKPT_PATH" --tta --no-tiling 2>&1 | tee "$EVAL_LOG"
    fi
done

echo ""
echo ">>> Summarizing RMR-v27 Suite Results:"
$PYTHON_BIN scripts/summarize_rmr_v27_suite.py

echo ""
echo "All RMR-v27 experiments completed!"
