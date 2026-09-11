#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# RMR-v7 Comprehensive Multi-Model Experiment Suite for Ubuntu (16GB VRAM)
# ==============================================================================
# Models in the suite:
#   1. rmr_v7_canonical: Flagship (T=4, hurdle=true, temp_softplus=true, EMA=0.999)
#   2. rmr_v7_t6_tv: Deep regularized solver (T=6, tv_lambda=0.02, hurdle=true)
#   3. rmr_v7_multiscale_dm: Multi-scale DM loss (16, 32, 64 block partitions)
#   4. rmr_v7_ablation_no_hurdle: Zero-inflation ablation (hurdle=false)
#   5. rmr_v7_ablation_no_ema: Live stochastic weights ablation (ema_decay=0.0)
# ==============================================================================

# Memory fragmentation protection under concurrent execution
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
echo "  RMR-v7 MULTI-MODEL EXPERIMENT SUITE (Ubuntu 16GB VRAM Optimized)"
echo "  Python executable: $PYTHON_EXE"
echo "  Date: $(date '+%Y-%m-%d %H:%M:%S')"
if command -v nvidia-smi &>/dev/null; then
    echo "  GPU Info:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
fi
echo "================================================================================"

MODE="${1:-all-parallel}"
echo "Execution mode: $MODE (options: all-parallel | batch-2 | sequential)"

LOG_DIR="runs/sha_a/suite_logs"
mkdir -p "$LOG_DIR"

declare -A RUN_CONFIGS=(
    ["rmr_v7_canonical"]="configs/rmr_v7/rmr_v7_canonical.yaml"
    ["rmr_v7_t6_tv"]="configs/rmr_v7/rmr_v7_t6_tv.yaml"
    ["rmr_v7_multiscale_dm"]="configs/rmr_v7/rmr_v7_multiscale_dm.yaml"
    ["rmr_v7_ablation_no_hurdle"]="configs/rmr_v7/rmr_v7_ablation_no_hurdle.yaml"
    ["rmr_v7_ablation_no_ema"]="configs/rmr_v7/rmr_v7_ablation_no_ema.yaml"
)

ORDERED_RUNS=(
    "rmr_v7_canonical"
    "rmr_v7_t6_tv"
    "rmr_v7_multiscale_dm"
    "rmr_v7_ablation_no_hurdle"
    "rmr_v7_ablation_no_ema"
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

evaluate_single_run() {
    local run_id="$1"
    local out_dir="runs/sha_a/$run_id"
    local best_ckpt="$out_dir/best.pt"

    if [ -f "$best_ckpt" ]; then
        echo -e "\n---> Running full test evaluation on $best_ckpt..."
        "$PYTHON_EXE" -m rmr_v3.eval --checkpoint "$best_ckpt" --data-manifest data/sha_a_test.jsonl || true
    else
        echo "  [WARN] Checkpoint $best_ckpt not found, skipping eval."
    fi
}

if [ "$MODE" = "all-parallel" ]; then
    echo -e "\n[+] Launching all 5 experiments concurrently in background..."
    declare -A PIDS
    for r_id in "${ORDERED_RUNS[@]}"; do
        pid=$(launch_single_run "$r_id")
        PIDS["$r_id"]=$pid
        echo "    Launched $r_id with PID $pid"
        sleep 2
    done

    echo -e "\n[+] All jobs running. Waiting for completion..."
    for r_id in "${ORDERED_RUNS[@]}"; do
        pid="${PIDS[$r_id]}"
        if wait "$pid"; then
            echo "  [DONE] $r_id completed successfully."
        else
            echo "  [FAIL] $r_id exited with error code $?. Check $LOG_DIR/${r_id}.log"
        fi
        evaluate_single_run "$r_id"
    done

elif [ "$MODE" = "batch-2" ]; then
    echo -e "\n[+] Running suite in 2-job parallel batches..."
    total=${#ORDERED_RUNS[@]}
    for ((i=0; i<total; i+=2)); do
        r1="${ORDERED_RUNS[i]}"
        pid1=$(launch_single_run "$r1")
        echo "    Launched $r1 with PID $pid1"

        if [ $((i+1)) -lt "$total" ]; then
            r2="${ORDERED_RUNS[i+1]}"
            pid2=$(launch_single_run "$r2")
            echo "    Launched $r2 with PID $pid2"
            wait "$pid1" || true
            wait "$pid2" || true
            evaluate_single_run "$r1"
            evaluate_single_run "$r2"
        else
            wait "$pid1" || true
            evaluate_single_run "$r1"
        fi
    done

elif [ "$MODE" = "sequential" ]; then
    echo -e "\n[+] Running suite sequentially (one by one)..."
    for r_id in "${ORDERED_RUNS[@]}"; do
        pid=$(launch_single_run "$r_id")
        wait "$pid" || true
        evaluate_single_run "$r_id"
    done
fi

echo -e "\n================================================================================"
echo "  ALL EXPERIMENTS COMPLETE! Aggregating summary benchmark..."
echo "================================================================================"
"$PYTHON_EXE" scripts/summarize_rmr_v7_suite.py
echo -e "\nDone! See runs/sha_a/rmr_v7_benchmark_summary.md for the full comparison table.\n"
