#!/usr/bin/env bash
# ==============================================================================
# run_c0_c3_parallel.sh: Parallel Observer-Solver 2x2 Factorial Matrix Runner (16GB GPU)
# Runs C0, C1, C2, C3 concurrently on an Ubuntu machine with 16GB VRAM.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PYTHON_EXE=".venv/bin/python"
if [ ! -f "$PYTHON_EXE" ]; then
    PYTHON_EXE="python3"
fi

LOG_DIR="runs/sha_a/matrix_parallel_logs"
mkdir -p "$LOG_DIR"

echo -e "\n================================================================================"
echo "  OBSERVER-SOLVER 2x2 FACTORIAL MATRIX (PARALLEL 16GB VRAM MODE)"
echo "  Hardware: Ubuntu / RTX 5070 Ti (16GB VRAM)"
echo "  Matrix Targets:"
echo "    [C0] Additive + Flat-DM16    | Scales (32,64,128)  | Solver OFF (Y=Y0) | 101,763 params"
echo "    [C1] Additive + Flat-DM16    | Scales (32,64,128)  | Solver ON  (T=2)  | 101,763 params"
echo "    [C2] RepWeighted + Bayesian  | Scales (16..128)    | Solver OFF (Y=Y0) | 102,249 params"
echo "    [C3] RepWeighted + Bayesian  | Scales (16..128)    | Solver ON  (T=2)  | 102,249 params"
echo "  Logs: $LOG_DIR/"
echo -e "================================================================================\n"

launch_run() {
    local id="$1"
    local name="$2"
    local cfg="$3"
    local out_dir="$4"
    local log_file="$LOG_DIR/${id}.log"

    mkdir -p "$out_dir"
    local last_ckpt="$out_dir/last.pt"
    local train_args=("-m" "rmr_v3.train" "--config" "$cfg")

    if [ -f "$last_ckpt" ]; then
        echo "  [$id: RESUME] Checkpoint found at $last_ckpt. Resuming..."
        train_args+=("--resume" "$last_ckpt" "--allow-cross-commit-resume")
    else
        train_args+=("--overwrite")
    fi

    echo "  [LAUNCH] $id: $name -> Log: $log_file"
    "$PYTHON_EXE" "${train_args[@]}" > "$log_file" 2>&1 &
    local pid=$!
    echo "$pid"
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching all 4 matrix runs in parallel..."

PID_C0=$(launch_run "C0" "Old Observer (Additive+FlatDM) Solver OFF" "configs/rmr_v3/c0_observer_old_direct.yaml" "runs/sha_a/c0_observer_old_direct_seed42")
PID_C1=$(launch_run "C1" "Old Observer (Additive+FlatDM) Solver ON"  "configs/rmr_v3/c1_observer_old_rwsirt.yaml" "runs/sha_a/c1_observer_old_rwsirt_seed42")
PID_C2=$(launch_run "C2" "New Observer (RepWeight+Bayes) Solver OFF" "configs/rmr_v3/c2_observer_new_direct.yaml" "runs/sha_a/c2_observer_new_direct_seed42")
PID_C3=$(launch_run "C3" "New Observer (RepWeight+Bayes) Solver ON"  "configs/rmr_v3/c3_observer_new_rwsirt.yaml" "runs/sha_a/c3_observer_new_rwsirt_seed42")

echo -e "\nAll 4 processes are now running in background on GPU:"
echo "  [C0] PID $PID_C0 | Live log: tail -f $LOG_DIR/C0.log"
echo "  [C1] PID $PID_C1 | Live log: tail -f $LOG_DIR/C1.log"
echo "  [C2] PID $PID_C2 | Live log: tail -f $LOG_DIR/C2.log"
echo "  [C3] PID $PID_C3 | Live log: tail -f $LOG_DIR/C3.log"
echo -e "\nMonitoring progress... Press Ctrl+C anytime (processes continue in background).\n"

wait_and_check() {
    local pid="$1"
    local name="$2"
    if wait "$pid"; then
        echo "[SUCCESS] $name (PID $pid) completed successfully!"
        return 0
    else
        echo "[FAILED] $name (PID $pid) exited with an error. Check logs in $LOG_DIR!"
        return 1
    fi
}

FAILED=0
wait_and_check "$PID_C0" "C0" || FAILED=1
wait_and_check "$PID_C1" "C1" || FAILED=1
wait_and_check "$PID_C2" "C2" || FAILED=1
wait_and_check "$PID_C3" "C3" || FAILED=1

if [ "$FAILED" -ne 0 ]; then
    echo -e "\n[WARNING] One or more runs failed. Check the logs in $LOG_DIR."
    exit 1
fi

echo -e "\n================================================================================"
echo "  ALL 4 MATRIX RUNS FINISHED! RUNNING CANONICAL EVALUATION (TEST SET 182 IMGS)"
echo "================================================================================"

for ID in "C0" "C1" "C2" "C3"; do
    case "$ID" in
        C0) OUTDIR="runs/sha_a/c0_observer_old_direct_seed42" ;;
        C1) OUTDIR="runs/sha_a/c1_observer_old_rwsirt_seed42" ;;
        C2) OUTDIR="runs/sha_a/c2_observer_new_direct_seed42" ;;
        C3) OUTDIR="runs/sha_a/c3_observer_new_rwsirt_seed42" ;;
    esac
    BEST="$OUTDIR/best_val_mae.pt"
    EVAL_DIR="$OUTDIR/eval_test"
    if [ -f "$BEST" ]; then
        echo "Evaluating $ID: $BEST -> $EVAL_DIR"
        "$PYTHON_EXE" -m rmr_v3.eval --checkpoint "$BEST" --manifest "data/sha_a_test.jsonl" --output-dir "$EVAL_DIR"
    fi
done

echo -e "\n================================================================================"
echo "  2x2 FACTORIAL DECOUPLING MATRIX RESULTS SUMMARY"
echo "================================================================================"
"$PYTHON_EXE" -c '
import json
from pathlib import Path

runs = [
    ("C0", "runs/sha_a/c0_observer_old_direct_seed42", "Additive + Flat-DM16", "OFF (Y=Y0)"),
    ("C1", "runs/sha_a/c1_observer_old_rwsirt_seed42", "Additive + Flat-DM16", "ON (T=2)"),
    ("C2", "runs/sha_a/c2_observer_new_direct_seed42", "RepWeighted + Bayesian", "OFF (Y=Y0)"),
    ("C3", "runs/sha_a/c3_observer_new_rwsirt_seed42", "RepWeighted + Bayesian", "ON (T=2)"),
]

results = {}
print(f"{\"Run\":<6} | {\"Observer Neck\":<24} | {\"Solver\":<12} | {\"Val MAE\":<10} | {\"Test MAE\":<10} | {\"Test MSE\":<10}")
print("-" * 84)

for tag, d, neck, solver in runs:
    val_mae = "N/A"
    test_mae = "N/A"
    test_mse = "N/A"
    
    val_file = Path(d) / "eval_val" / "summary.json"
    if val_file.exists():
        with open(val_file) as f:
            val_mae = f"{json.load(f).get(\"mae\", 0):.2f}"
            
    test_file = Path(d) / "eval_test" / "summary.json"
    if test_file.exists():
        with open(test_file) as f:
            tdata = json.load(f)
            t_mae_val = tdata.get("mae", 0)
            test_mae = f"{t_mae_val:.2f}"
            test_mse = f"{tdata.get(\"mse\", 0):.2f}"
            results[tag] = t_mae_val

    print(f"{tag:<6} | {neck:<24} | {solver:<12} | {val_mae:<10} | {test_mae:<10} | {test_mse:<10}")

print("-" * 84)
if "C0" in results and "C1" in results:
    delta_old = results["C0"] - results["C1"]
    print(f"Delta_old (Legacy Gain: C0 - C1)  = {delta_old:+.2f} MAE")
if "C2" in results and "C3" in results:
    delta_new = results["C2"] - results["C3"]
    print(f"Delta_new (Decoupling Gain: C2 - C3) = {delta_new:+.2f} MAE")
print("================================================================================")
'

echo -e "\nMatrix completed! All results and checkpoints are stored under runs/sha_a/."
