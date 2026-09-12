#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# RMR-v9 Comprehensive Multi-Model Experiment Suite for Ubuntu (16GB VRAM)
# ==============================================================================
# Models in the suite (6 runs):
#   1. rmr_v9_canonical: Flagship canonical model (T=6, Additive SIRT, Isotropic Laplacian TV, Flat-DM16, tau=0.0)
#   2. rmr_v9_aq_rmr: Full Anisotropic & Quadratic model (+anisotropic [64,32] & [32,64] boxes, mean_std stats, tau=0.015)
#   3. rmr_v9_ablation_no_proximal: Ablation of proximal step (tau=0.0, with anisotropic boxes + mean_std)
#   4. rmr_v9_ablation_isotropic: Ablation of anisotropic boxes (square boxes [32,64,128], tau=0.015 + mean_std)
#   5. rmr_v9_ablation_mean_only: Ablation of variance statistics (mean only, with anisotropic boxes + tau=0.015)
#   6. rmr_v9_control_no_solver: Control baseline without SIRT solver (direct base density y0)
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
echo "  RMR-v9 MULTI-MODEL EXPERIMENT SUITE (Ubuntu 16GB VRAM Optimized)"
echo "  Python executable: $PYTHON_EXE"
echo "  Date: $(date '+%Y-%m-%d %H:%M:%S')"
if command -v nvidia-smi &>/dev/null; then
    echo "  GPU Info:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
fi
echo "================================================================================"

MODE="${1:-sequential}"
echo "Execution mode: $MODE (options: sequential | batch-2 | all-parallel)"

LOG_DIR="runs/sha_a/suite_logs_v9"
mkdir -p "$LOG_DIR"

declare -A RUN_CONFIGS=(
    ["rmr_v9_canonical"]="configs/rmr_v9/rmr_v9_canonical.yaml"
    ["rmr_v9_aq_rmr"]="configs/rmr_v9/rmr_v9_aq_rmr.yaml"
    ["rmr_v9_ablation_no_proximal"]="configs/rmr_v9/rmr_v9_ablation_no_proximal.yaml"
    ["rmr_v9_ablation_isotropic"]="configs/rmr_v9/rmr_v9_ablation_isotropic.yaml"
    ["rmr_v9_ablation_mean_only"]="configs/rmr_v9/rmr_v9_ablation_mean_only.yaml"
    ["rmr_v9_control_no_solver"]="configs/rmr_v9/rmr_v9_control_no_solver.yaml"
)

ORDERED_RUNS=(
    "rmr_v9_canonical"
    "rmr_v9_aq_rmr"
    "rmr_v9_ablation_no_proximal"
    "rmr_v9_ablation_isotropic"
    "rmr_v9_ablation_mean_only"
    "rmr_v9_control_no_solver"
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
        echo "  [$run_id] Existing checkpoint found at $last_ckpt -> RESUMING" >&2
        cmd+=("--resume" "$last_ckpt" "--allow-cross-commit-resume")
    else
        echo "  [$run_id] Starting fresh run (config: $cfg_file)" >&2
        cmd+=("--overwrite")
    fi

    echo "  [$run_id] Logging to: $log_file" >&2
    "${cmd[@]}" > "$log_file" 2>&1 &
    local pid=$!
    echo "$pid"
}

evaluate_single_run() {
    local run_id="$1"
    local out_dir="runs/sha_a/$run_id"
    local best_ckpt="$out_dir/best_val_mae.pt"
    local eval_dir="$out_dir/eval_test"
    local eval_log="$LOG_DIR/${run_id}_eval.log"

    if [ -f "$best_ckpt" ]; then
        echo "  [$run_id] Evaluating best checkpoint ($best_ckpt) on test set..."
        "$PYTHON_EXE" -m rmr_v3.eval --checkpoint "$best_ckpt" --manifest "data/sha_a_test.jsonl" --output-dir "$eval_dir" > "$eval_log" 2>&1 || true
        echo "  [$run_id] Evaluation complete. Summary saved to $eval_dir/summary.json"
    else
        echo "  [$run_id] Warning: No best_val_mae.pt found at $best_ckpt to evaluate." >&2
    fi
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
    evaluate_single_run "$run_id"
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
        batch_runs=()
        for ((j=i; j<i+2 && j<TOTAL; j++)); do
            run_id="${ORDERED_RUNS[j]}"
            pid=$(launch_single_run "$run_id")
            pids+=("$pid")
            batch_runs+=("$run_id")
        done
        for pid in "${pids[@]}"; do
            wait "$pid"
        done
        for run_id in "${batch_runs[@]}"; do
            evaluate_single_run "$run_id"
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
    for run_id in "${ORDERED_RUNS[@]}"; do
        evaluate_single_run "$run_id"
    done
else
    echo "Unknown mode: $MODE. Choose from: sequential | batch-2 | all-parallel"
    exit 1
fi

echo "================================================================================"
echo "All RMR-v9 runs completed."
echo "Generating summary table..."
"$PYTHON_EXE" scripts/summarize_rmr_v9_suite.py
echo "================================================================================"
