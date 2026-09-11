#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# RMR-v8 Comprehensive Multi-Model Experiment Suite for Ubuntu (16GB VRAM)
# ==============================================================================
# Models in the suite:
#   1. rmr_v8_canonical: Flagship Stage 2 (T=2, Mult-SIRT, Charbonnier TV, L1 count, Mass-weighted cell)
#   2. rmr_v8_t6_tv: Deep regularized solver (T=6, Mult-SIRT, Charbonnier TV, L1 count)
#   3. rmr_v8_no_mult_sirt: Ablation of Stage 2a (Additive SIRT)
#   4. rmr_v8_no_charbonnier: Ablation of Stage 2b (Isotropic Laplacian TV)
#   5. rmr_v8_no_l1_loss: Ablation of Stage 2c (NB count loss)
#   6. rmr_v8_no_mass_weight: Ablation of Stage 2d (Balanced cell loss)
#   7. rmr_v8_coord_attn: Stage 3 (+CoordinateAttention on P4, 784 params, GroupNorm)
#   8. rmr_v8_with_kd: Stage 3 (Knowledge Distillation placeholder)
# ==============================================================================

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED="1"

PYTHON_EXE=".venv/bin/python"
if [ ! -f "$PYTHON_EXE" ]; then
    if command -v python3 &>/dev/null; then
        PYTHON_EXE="python3"
    else
        PYTHON_EXE="python"
    fi
fi

echo "================================================================================"
echo "  RMR-v8 MULTI-MODEL EXPERIMENT SUITE (Ubuntu 16GB VRAM Optimized)"
echo "  Python executable: $PYTHON_EXE"
echo "  Date: $(date '+%Y-%m-%d %H:%M:%S')"
if command -v nvidia-smi &>/dev/null; then
    echo "  GPU Info:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
fi
echo "================================================================================"

MODE="${1:-sequential}"
echo "Execution mode: $MODE (options: sequential | batch-2 | all-parallel)"

LOG_DIR="runs/sha_a/suite_logs_v8"
mkdir -p "$LOG_DIR"

declare -A RUN_CONFIGS=(
    ["rmr_v8_canonical"]="configs/rmr_v8/rmr_v8_canonical.yaml"
    ["rmr_v8_t6_tv"]="configs/rmr_v8/rmr_v8_t6_tv.yaml"
    ["rmr_v8_no_mult_sirt"]="configs/rmr_v8/rmr_v8_no_mult_sirt.yaml"
    ["rmr_v8_no_charbonnier"]="configs/rmr_v8/rmr_v8_no_charbonnier.yaml"
    ["rmr_v8_no_l1_loss"]="configs/rmr_v8/rmr_v8_no_l1_loss.yaml"
    ["rmr_v8_no_mass_weight"]="configs/rmr_v8/rmr_v8_no_mass_weight.yaml"
    ["rmr_v8_coord_attn"]="configs/rmr_v8/rmr_v8_coord_attn.yaml"
    ["rmr_v8_with_kd"]="configs/rmr_v8/rmr_v8_with_kd.yaml"
)

ORDERED_RUNS=(
    "rmr_v8_canonical"
    "rmr_v8_t6_tv"
    "rmr_v8_no_mult_sirt"
    "rmr_v8_no_charbonnier"
    "rmr_v8_no_l1_loss"
    "rmr_v8_no_mass_weight"
    "rmr_v8_coord_attn"
    "rmr_v8_with_kd"
)

launch_single_run() {
    local run_id="$1"
    local cfg_file="${RUN_CONFIGS[$run_id]}"
    local out_dir="runs/sha_a/$run_id"
    local log_file="$LOG_DIR/${run_id}.log"
    local last_ckpt="$out_dir/last.pt"

    mkdir -p "$out_dir"

    local cmd=("$PYTHON_EXE" "-m" "rmr_v3.train" "--config" "$cfg_file" "--run-id" "$run_id")

    if [ -f "$last_ckpt" ]; then
        echo "  [$run_id] Existing checkpoint found at $last_ckpt -> RESUMING"
        cmd+=("--resume" "$last_ckpt" "--allow-cross-commit-resume")
    else
        echo "  [$run_id] Starting fresh run (config: $cfg_file)"
        cmd+=("--overwrite")
    fi

    echo "  [$run_id] Logging to: $log_file"
    "${cmd[@]}" > "$log_file" 2>&1 &
    local pid=$!
    echo "$pid"
}

run_foreground() {
    local run_id="$1"
    local cfg_file="${RUN_CONFIGS[$run_id]}"
    local out_dir="runs/sha_a/$run_id"
    local last_ckpt="$out_dir/last.pt"

    mkdir -p "$out_dir"
    local cmd=("$PYTHON_EXE" "-m" "rmr_v3.train" "--config" "$cfg_file" "--run-id" "$run_id")
    if [ -f "$last_ckpt" ]; then
        echo "  [$run_id] Existing checkpoint found at $last_ckpt -> RESUMING"
        cmd+=("--resume" "$last_ckpt" "--allow-cross-commit-resume")
    else
        echo "  [$run_id] Starting fresh run"
        cmd+=("--overwrite")
    fi
    "${cmd[@]}"
}

if [ "$MODE" = "sequential" ]; then
    echo "Running models sequentially in foreground:"
    for run_id in "${ORDERED_RUNS[@]}"; do
        echo "================================================================================"
        echo "Starting: $run_id"
        echo "================================================================================"
        run_foreground "$run_id"
    done
elif [ "$MODE" = "batch-2" ]; then
    echo "Running 2 models concurrently:"
    TOTAL=${#ORDERED_RUNS[@]}
    for ((i=0; i<TOTAL; i+=2)); do
        pids=()
        for ((j=i; j<i+2 && j<TOTAL; j++)); do
            run_id="${ORDERED_RUNS[j]}"
            pid=$(launch_single_run "$run_id")
            pids+=("$pid")
        done
        for pid in "${pids[@]}"; do
            wait "$pid"
        done
    done
elif [ "$MODE" = "all-parallel" ]; then
    echo "Launching all models concurrently:"
    pids=()
    for run_id in "${ORDERED_RUNS[@]}"; do
        pid=$(launch_single_run "$run_id")
        pids+=("$pid")
    done
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
fi

echo "All runs completed."
