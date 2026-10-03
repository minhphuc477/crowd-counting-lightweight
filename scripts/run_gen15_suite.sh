#!/usr/bin/env bash
# ==============================================================================
# Runner for Gen 15 Architectural Breakthrough Suite (Target: Dense Unchoking & Gridding Fix)
# Baseline Anchor: sub60_e34 Champion (73.04 MAE / 110.79 RMSE / Dense MAE 123.73)
# Invariants: Exactly <= 104,441 params, Zero KD, Canonical ShanghaiTech Part A (300/182)
# ==============================================================================
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

declare -a MODELS=(
    # --- Part 1: First-Principles Architectural Fixes (Empirically Diagnosed) ---
    # E50: HDC-Lite Neck (dilations [1, 2, 3] eliminates 99.6% gridding sparsity at dilation 6)
    "sub60_e50_hdc_lite_neck:configs/rmr_research/sub60_e50_hdc_lite_neck.yaml"

    # E51: CI-Cell Dynamic Range (breaks z0 <= -1.1 barrier, restoring y0 >= 1.0 peak capacity)
    "sub60_e51_ci_cell_dynamic_range:configs/rmr_research/sub60_e51_ci_cell_dynamic_range.yaml"

    # E52: Decoupled Carrier Supervision (dm_target = y0 eliminates opposing gradient interference)
    "sub60_e52_decoupled_carrier_supervision:configs/rmr_research/sub60_e52_decoupled_carrier_supervision.yaml"

    # E53: HDC-Lite + CI-Cell Dynamic Range Synthesis
    "sub60_e53_hdc_ci_cell_synthesis:configs/rmr_research/sub60_e53_hdc_ci_cell_synthesis.yaml"

    # --- Part 2: Isolated Mathematical Solver Enhancements ---
    # E44: Detached Spatial Morozov (safely captures sparse ~12.09 MAE with zero autograd leak)
    "sub60_e44_detached_spatial_morozov:configs/rmr_research/sub60_e44_detached_spatial_morozov.yaml"

    # E45: Spatially-Routed Trust Floor (dynamic floor: 0.005 background -> 0.025 dense)
    "sub60_e45_spatially_routed_floor:configs/rmr_research/sub60_e45_spatially_routed_floor.yaml"

    # E46: Calibrated Dense Carrier Seed (eps = 0.05 carrier support)
    "sub60_e46_calibrated_dense_seed:configs/rmr_research/sub60_e46_calibrated_dense_seed.yaml"

    # E48: Gen 15 Grand Synthesis (harmonic combination of leak-free solver enhancements)
    "sub60_e48_gen15_grand_synthesis:configs/rmr_research/sub60_e48_gen15_grand_synthesis.yaml"
)

LOG_DIR="runs/sha_a/suite_logs_gen15"
mkdir -p "$LOG_DIR"

echo "================================================================================"
echo "  RMR GEN 15 ARCHITECTURAL BREAKTHROUGH SUITE (Bash)"
echo "  8 Targeted Experiments | Causal Isolation & Architectural Synthesis"
echo "  Strict Budget: Exactly <= 104,441 Trainable Params, Zero KD, Canonical 300/182"
echo "================================================================================"

PYTHON_BIN="python"
if [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
elif [ -f ".venv/Scripts/python.exe" ]; then
    PYTHON_BIN=".venv/Scripts/python.exe"
fi

for ENTRY in "${MODELS[@]}"; do
    RUN_ID="${ENTRY%%:*}"
    CFG="${ENTRY##*:}"
    LOG_FILE="$LOG_DIR/${RUN_ID}.log"

    echo ""
    echo ">>> Starting training for: $RUN_ID ($CFG)"
    $PYTHON_BIN -m rmr_v3.train \
        --config "$CFG" \
        2>&1 | tee "$LOG_FILE"
    echo ">>> Completed training for: $RUN_ID"
done

echo ""
echo "================================================================================"
echo "  All Gen 15 experiments completed successfully!"
echo "================================================================================"
