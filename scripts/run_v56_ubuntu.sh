#!/usr/bin/env bash
# ==============================================================================
# RMR V56 Experiment Suite Runner for Ubuntu Linux (RTX 5070 Ti 16GB)
# Focus: Asymmetric Trust Damping & Deep Asymmetric Radon-Nikodym (4 runs)
#
# Usage:
#   ./scripts/run_v56_ubuntu.sh                # Run all 4 sequentially
#   ./scripts/run_v56_ubuntu.sh parallel       # Run Track A & B proposed in parallel
#   ./scripts/run_v56_ubuntu.sh proposed       # Run only proposed Track A & B
#   ./scripts/run_v56_ubuntu.sh <config_key>   # Run a single specific model
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

LOG_DIR="${REPO_ROOT}/logs/rmr_v56"
mkdir -p "${LOG_DIR}"

declare -A RUN_CONFIGS=(
    ["v56_track_a_proposed_asym_rn"]="configs/rmr_v56/v56_track_a_proposed_asym_rn.yaml"
    ["v56_track_b_proposed_deep_asym_rn"]="configs/rmr_v56/v56_track_b_proposed_deep_asym_rn.yaml"
    ["v56_track_a_ablation_sym_rn"]="configs/rmr_v56/v56_track_a_ablation_sym_rn.yaml"
    ["v56_track_b_ablation_sym_rn"]="configs/rmr_v56/v56_track_b_ablation_sym_rn.yaml"
)

ORDERED_RUNS=(
    "v56_track_a_proposed_asym_rn"
    "v56_track_b_proposed_deep_asym_rn"
    "v56_track_a_ablation_sym_rn"
    "v56_track_b_ablation_sym_rn"
)

echo "================================================================================"
echo "  RMR-V56 EXPERIMENT SUITE (Ubuntu RTX 5070 Ti)"
echo "  Python: $PYTHON_EXE"
echo "  Date:   $(date '+%Y-%m-%d %H:%M:%S')"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
fi
echo "================================================================================"

MODE="${1:-sequential}"

run_single_model() {
    local key="$1"
    local cfg="${RUN_CONFIGS[$key]}"
    local log="${LOG_DIR}/${key}.log"

    echo ""
    echo ">>> [STARTING] $key"
    echo "    Config: $cfg"
    echo "    Log:    $log"

    local out_dir="runs/sha_a/rmr_v56/${key}"
    if [ -f "${out_dir}/best.pth" ]; then
        echo "    [SKIPPED] Model already completed (${out_dir}/best.pth exists)."
        return 0
    fi

    local t_start
    t_start=$(date +%s)

    "$PYTHON_EXE" train.py --config "$cfg" > "$log" 2>&1
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

case "$MODE" in
    "sequential")
        echo "Executing all 4 V56 models sequentially..."
        for key in "${ORDERED_RUNS[@]}"; do
            run_single_model "$key"
        done
        ;;
    "proposed")
        echo "Executing Proposed V56 Track A & Track B sequentially..."
        run_single_model "v56_track_a_proposed_asym_rn"
        run_single_model "v56_track_b_proposed_deep_asym_rn"
        ;;
    "parallel")
        echo "Launching Proposed Track A & Track B in parallel (~5.5 GB VRAM)..."
        "$PYTHON_EXE" train.py --config "${RUN_CONFIGS['v56_track_a_proposed_asym_rn']}" \
            > "${LOG_DIR}/v56_track_a_proposed_asym_rn.log" 2>&1 &
        pid_a=$!
        echo "Track A PID: $pid_a"

        sleep 3

        "$PYTHON_EXE" train.py --config "${RUN_CONFIGS['v56_track_b_proposed_deep_asym_rn']}" \
            > "${LOG_DIR}/v56_track_b_proposed_deep_asym_rn.log" 2>&1 &
        pid_b=$!
        echo "Track B PID: $pid_b"

        echo "Waiting for parallel jobs to complete..."
        wait $pid_a
        wait $pid_b
        echo "Parallel runs completed!"
        ;;
    *)
        if [[ -v "RUN_CONFIGS[$MODE]" ]]; then
            run_single_model "$MODE"
        else
            echo "Unknown mode/config: $MODE"
            echo "Available models: ${ORDERED_RUNS[*]}"
            exit 1
        fi
        ;;
esac

echo ""
echo "================================================================================"
echo "  V56 EXECUTION COMPLETED"
echo "================================================================================"
