#!/usr/bin/env bash
# ==============================================================================
# run_c0_c3_matrix.sh: Observer-Solver 2x2 Factorial Matrix Runner (Sequential / By-ID)
# Run ID | Neck Architecture   | Spatial Loss  | Scales           | Solver (RW-SIRT)
#   C0   | AdditiveFPNNeck      | Flat-DM16     | (32, 64, 128)    | OFF (Y = Y0)
#   C1   | AdditiveFPNNeck      | Flat-DM16     | (32, 64, 128)    | ON  (T=2, w=1.0)
#   C2   | RepWeightedFPNNeck   | Bayesian Loss | (16,32,64,128)   | OFF (Y = Y0)
#   C3   | RepWeightedFPNNeck   | Bayesian Loss | (16,32,64,128)   | ON  (T=2, w=1.0)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

TARGET="${1:-all}"
FRESH="${2:-false}"

PYTHON_EXE=".venv/bin/python"
if [ ! -f "$PYTHON_EXE" ]; then
    PYTHON_EXE="python3"
fi

declare -A CONFIGS
CONFIGS["C0"]="configs/rmr_v3/c0_observer_old_direct.yaml"
CONFIGS["C1"]="configs/rmr_v3/c1_observer_old_rwsirt.yaml"
CONFIGS["C2"]="configs/rmr_v3/c2_observer_new_direct.yaml"
CONFIGS["C3"]="configs/rmr_v3/c3_observer_new_rwsirt.yaml"

declare -A OUTDIRS
OUTDIRS["C0"]="runs/sha_a/c0_observer_old_direct_seed42"
OUTDIRS["C1"]="runs/sha_a/c1_observer_old_rwsirt_seed42"
OUTDIRS["C2"]="runs/sha_a/c2_observer_new_direct_seed42"
OUTDIRS["C3"]="runs/sha_a/c3_observer_new_rwsirt_seed42"

declare -A DESCS
DESCS["C0"]="Old Observer (Additive + Flat-DM16) + Solver OFF (B0 Baseline)"
DESCS["C1"]="Old Observer (Additive + Flat-DM16) + Solver ON  (B5-P Solver)"
DESCS["C2"]="New Observer (RepWeighted + Bayesian) + Solver OFF (New Observer direct)"
DESCS["C3"]="New Observer (RepWeighted + Bayesian) + Solver ON  (Core Decoupling Test)"

RUN_LIST=()
if [ "$TARGET" == "all" ] || [ "$TARGET" == "ALL" ]; then
    RUN_LIST=("C0" "C1" "C2" "C3")
else
    UPPER_TARGET=$(echo "$TARGET" | tr '[:lower:]' '[:upper:]')
    if [[ -z "${CONFIGS[$UPPER_TARGET]+unset}" ]]; then
        echo "[ERROR] Unknown Run ID '$TARGET'. Valid choices: C0, C1, C2, C3, all"
        exit 1
    fi
    RUN_LIST=("$UPPER_TARGET")
fi

echo -e "\n================================================================================"
echo "  RMR OBSERVER-SOLVER 2x2 FACTORIAL MATRIX RUNNER"
echo "  Target Runs: ${RUN_LIST[*]}"
echo "  Python: $PYTHON_EXE"
echo "================================================================================\n"

for ID in "${RUN_LIST[@]}"; do
    CFG="${CONFIGS[$ID]}"
    OUTDIR="${OUTDIRS[$ID]}"
    DESC="${DESCS[$ID]}"
    BEST_CKPT="$OUTDIR/best_val_mae.pt"
    LAST_CKPT="$OUTDIR/last.pt"
    EVAL_DIR="$OUTDIR/eval_test"

    echo "--------------------------------------------------------------------------------"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Run $ID : $DESC"
    echo "  Config: $CFG | Output: $OUTDIR"
    echo "--------------------------------------------------------------------------------"

    mkdir -p "$OUTDIR"
    TRAIN_ARGS=("-m" "rmr_v3.train" "--config" "$CFG")

    if [ "$FRESH" != "true" ] && [ "$FRESH" != "--fresh" ] && [ -f "$LAST_CKPT" ]; then
        echo "  [RESUME] Found existing checkpoint at $LAST_CKPT. Resuming..."
        TRAIN_ARGS+=("--resume" "$LAST_CKPT" "--allow-cross-commit-resume")
    else
        TRAIN_ARGS+=("--overwrite")
    fi

    "$PYTHON_EXE" "${TRAIN_ARGS[@]}"

    # Automatic evaluation on SHA test set
    if [ -f "$BEST_CKPT" ]; then
        echo -e "\n  [EVAL] Evaluating $ID on ShanghaiTech Part A Test Set (182 images)..."
        "$PYTHON_EXE" -m rmr_v3.eval --checkpoint "$BEST_CKPT" --manifest "data/sha_a_test.jsonl" --output-dir "$EVAL_DIR"
        echo "  [DONE] Run $ID evaluation finished. Summary saved to $EVAL_DIR/summary.json"
    fi
    echo -e "--------------------------------------------------------------------------------\n"
done

echo "================================================================================"
echo "  ALL REQUESTED MATRIX RUNS HAVE COMPLETED!"
echo "================================================================================"
