#!/usr/bin/env bash
# ==============================================================================
# RMR V55 Experiment Suite Runner for Ubuntu Linux (RTX 5070 Ti 16GB)
# Focus: Purified Core Loss Suite & Radon-Nikodym Transport Energy Loss (5 runs)
# Auto-detects interrupted runs and automatically resumes from last.pt!
#
# Usage:
#   ./scripts/run_v55_ubuntu.sh                     # Run all 5 sequentially (auto-resume)
#   ./scripts/run_v55_ubuntu.sh parallel            # Run all 5 in 3 parallel waves
#   ./scripts/run_v55_ubuntu.sh proposed            # Run only proposed Track A & B
#   ./scripts/run_v55_ubuntu.sh ablation            # Run only 3 ablations sequentially
#   ./scripts/run_v55_ubuntu.sh parallel-ablation   # Run 3 ablations in parallel waves
#   ./scripts/run_v55_ubuntu.sh <config_key>        # Run a single specific model
# ==============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export MALLOC_ARENA_MAX=2
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PYTHON_EXE=".venv/bin/python"
if [ ! -f "$PYTHON_EXE" ]; then
    if command -v python3 &>/dev/null; then
        PYTHON_EXE="python3"
    else
        PYTHON_EXE="python"
    fi
fi

LOG_DIR="${REPO_ROOT}/logs/rmr_v55"
mkdir -p "${LOG_DIR}"

declare -A RUN_CONFIGS=(
    ["v55_track_a_purified_rn"]="configs/rmr_v55/v55_track_a_purified_rn.yaml"
    ["v55_track_b_purified_rn"]="configs/rmr_v55/v55_track_b_purified_rn.yaml"
    ["v55_track_a_no_transport_loss"]="configs/rmr_v55/v55_track_a_no_transport_loss.yaml"
    ["v55_track_a_control_bloat"]="configs/rmr_v55/v55_track_a_control_bloat.yaml"
    ["v55_track_b_control_bloat"]="configs/rmr_v55/v55_track_b_control_bloat.yaml"
)

ORDERED_RUNS=(
    "v55_track_a_purified_rn"
    "v55_track_b_purified_rn"
    "v55_track_a_no_transport_loss"
    "v55_track_a_control_bloat"
    "v55_track_b_control_bloat"
)

echo "================================================================================"
echo "  RMR-V55 EXPERIMENT SUITE (Ubuntu RTX 5070 Ti - Auto Resume Enabled)"
echo "  Python: $PYTHON_EXE"
echo "  Date:   $(date '+%Y-%m-%d %H:%M:%S')"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
fi
echo "================================================================================"

MODE="${1:-sequential}"

get_train_action() {
    local out_dir="$1"
    local cfg="$2"

    if [ ! -d "$out_dir" ]; then
        echo "FRESH"
        return
    fi

    local target_epochs=1000
    if [ -f "$cfg" ]; then
        local found
        found=$(grep -E '^[[:space:]]*epochs:[[:space:]]*[0-9]+' "$cfg" | head -n1 | awk '{print $2}')
        if [ -n "$found" ]; then
            target_epochs="$found"
        fi
    fi

    local done_epochs=0
    if [ -f "${out_dir}/train_log.csv" ]; then
        done_epochs=$(wc -l < "${out_dir}/train_log.csv" 2>/dev/null || echo 0)
        done_epochs=$((done_epochs > 0 ? done_epochs - 1 : 0))
    fi

    if [ "$done_epochs" -ge "$target_epochs" ]; then
        echo "COMPLETED"
    elif [ -f "${out_dir}/last.pt" ]; then
        echo "RESUME"
    else
        echo "FRESH"
    fi
}

run_single_model() {
    local key="$1"
    local cfg="${RUN_CONFIGS[$key]}"
    local log="${LOG_DIR}/${key}.log"
    local out_dir="runs/sha_a/rmr_v55/${key}"

    echo ""
    echo ">>> [CHECKING] $key"
    echo "    Config: $cfg"
    echo "    Log:    $log"

    local action
    action=$(get_train_action "$out_dir" "$cfg")

    if [ "$action" = "COMPLETED" ]; then
        echo "    [SKIPPED] Model already completed all epochs (${out_dir}/best_val_mae.pt exists)."
        return 0
    fi

    local extra_args=""
    if [ "$action" = "RESUME" ]; then
        echo "    [AUTO-RESUME] Interrupted run detected. Resuming from ${out_dir}/last.pt..."
        extra_args="--resume ${out_dir}/last.pt --force-resume"
    else
        echo "    [FRESH] Starting new training run..."
    fi

    local t_start
    t_start=$(date +%s)

    "$PYTHON_EXE" train.py --config "$cfg" $extra_args >> "$log" 2>&1
    local ret=$?

    local t_end
    t_end=$(date +%s)
    local duration=$((t_end - t_start))

    if [ $ret -eq 0 ]; then
        echo ">>> [SUCCESS] $key in $((duration / 60)) min"
    else
        echo ">>> [FAILED]  $key with exit code $ret after $((duration / 60)) min. See $log"
        return $ret
    fi
}

run_pair_parallel() {
    local key_a="$1"
    local key_b="${2:-}"

    echo ""
    echo "================================================================================"
    if [ -n "$key_b" ]; then
        echo ">>> LAUNCHING PARALLEL PAIR: [$key_a] AND [$key_b]"
    else
        echo ">>> LAUNCHING SINGLE ITEM: [$key_a]"
    fi
    echo "================================================================================"

    local pid_a=""
    local pid_b=""

    local out_a="runs/sha_a/rmr_v55/${key_a}"
    local cfg_a="${RUN_CONFIGS[$key_a]}"
    local action_a
    action_a=$(get_train_action "$out_a" "$cfg_a")

    if [ "$action_a" = "COMPLETED" ]; then
        echo "    [$key_a] already completed. Skipping."
    else
        local res_a=""
        if [ "$action_a" = "RESUME" ]; then
            res_a="--resume ${out_a}/last.pt --force-resume"
            echo "    [$key_a] AUTO-RESUMING from ${out_a}/last.pt..."
        fi
        "$PYTHON_EXE" train.py --config "$cfg_a" $res_a >> "${LOG_DIR}/${key_a}.log" 2>&1 &
        pid_a=$!
        echo "    [$key_a] launched with PID: $pid_a -> ${LOG_DIR}/${key_a}.log"
    fi

    if [ -n "$key_b" ]; then
        sleep 3
        local out_b="runs/sha_a/rmr_v55/${key_b}"
        local cfg_b="${RUN_CONFIGS[$key_b]}"
        local action_b
        action_b=$(get_train_action "$out_b" "$cfg_b")

        if [ "$action_b" = "COMPLETED" ]; then
            echo "    [$key_b] already completed. Skipping."
        else
            local res_b=""
            if [ "$action_b" = "RESUME" ]; then
                res_b="--resume ${out_b}/last.pt --force-resume"
                echo "    [$key_b] AUTO-RESUMING from ${out_b}/last.pt..."
            fi
            "$PYTHON_EXE" train.py --config "$cfg_b" $res_b >> "${LOG_DIR}/${key_b}.log" 2>&1 &
            pid_b=$!
            echo "    [$key_b] launched with PID: $pid_b -> ${LOG_DIR}/${key_b}.log"
        fi
    fi

    if [ -n "$pid_a" ]; then
        wait "$pid_a" || echo "WARNING: [$key_a] exited with error"
    fi
    if [ -n "$pid_b" ]; then
        wait "$pid_b" || echo "WARNING: [$key_b] exited with error"
    fi

    echo ">>> BATCH COMPLETED."
}

case "$MODE" in
    "sequential")
        echo "Executing all 5 V55 models sequentially (with auto-resume)..."
        for key in "${ORDERED_RUNS[@]}"; do
            run_single_model "$key"
        done
        ;;
    "proposed")
        echo "Executing Proposed V55 Track A & Track B sequentially..."
        run_single_model "v55_track_a_purified_rn"
        run_single_model "v55_track_b_purified_rn"
        ;;
    "ablation")
        echo "Executing all 3 V55 Ablation & Control models sequentially..."
        run_single_model "v55_track_a_no_transport_loss"
        run_single_model "v55_track_a_control_bloat"
        run_single_model "v55_track_b_control_bloat"
        ;;
    "parallel"|"parallel-all")
        echo "Executing ALL 5 V55 models in parallel waves (Proposed -> Ablations)..."
        run_pair_parallel "v55_track_a_purified_rn" "v55_track_b_purified_rn"
        run_pair_parallel "v55_track_a_no_transport_loss" "v55_track_a_control_bloat"
        run_pair_parallel "v55_track_b_control_bloat" ""
        ;;
    "parallel-proposed")
        echo "Executing only Proposed V55 Track A & Track B in parallel..."
        run_pair_parallel "v55_track_a_purified_rn" "v55_track_b_purified_rn"
        ;;
    "parallel-ablation")
        echo "Executing only Ablation V55 models in parallel waves..."
        run_pair_parallel "v55_track_a_no_transport_loss" "v55_track_a_control_bloat"
        run_pair_parallel "v55_track_b_control_bloat" ""
        ;;
    *)
        if [[ -v "RUN_CONFIGS[$MODE]" ]]; then
            run_single_model "$MODE"
        else
            echo "Unknown mode/config: $MODE"
            echo "Available models: ${ORDERED_RUNS[*]}"
            echo "Available modes:  sequential | ablation | proposed | parallel | parallel-proposed | parallel-ablation"
            exit 1
        fi
        ;;
esac

echo ""
echo "================================================================================"
echo "  V55 EXECUTION COMPLETED"
echo "================================================================================"
