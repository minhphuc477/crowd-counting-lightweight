#!/usr/bin/env bash
set -euo pipefail

# Parse optional arguments
FRESH=false
ALLOW_DIRTY=false

for arg in "$@"; do
    case "$arg" in
        --fresh) FRESH=true ;;
        --allow-dirty) ALLOW_DIRTY=true ;;
        *) echo "Unknown argument: $arg" && exit 1 ;;
    esac
done

PYTHON_EXE=".venv/bin/python"
if [ ! -f "$PYTHON_EXE" ]; then
    PYTHON_EXE="python3"
fi

CURRENT_COMMIT=$(git rev-parse HEAD | tr -d '[:space:]')
GIT_STATUS_RAW=$(git status --porcelain)
IS_DIRTY=false
if [ -n "$GIT_STATUS_RAW" ]; then
    IS_DIRTY=true
fi

if [ "$IS_DIRTY" = true ] && [ "$ALLOW_DIRTY" = false ]; then
    echo "ERROR: Working tree is dirty at git commit $CURRENT_COMMIT. Please commit or stash changes, or pass --allow-dirty." >&2
    exit 1
fi

CFG="configs/rmr_v4/mean_std_regions.yaml"
OUT_DIR="runs/sha_a/rmr_v4_mean_std_seed42"
BEST_CKPT="$OUT_DIR/best_val_mae.pt"
EVAL_DIR="$OUT_DIR/eval_test"
SUMMARY_JSON="$EVAL_DIR/summary.json"

echo -e "\n================================================================================"
echo "  RMR-V4 ABLATION RUNNER: V4-S (Mean + Std Regional Stats) (Linux/Ubuntu)"
echo "  Feature Isolated: Mean + Std 2nd-moment variance region features (65-dim)"
echo "  Git HEAD: $CURRENT_COMMIT (Dirty: $IS_DIRTY)"
echo -e "================================================================================\n"

# 1. Training Step
echo "--------------------------------------------------------------------------------"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Training: V4-S (Mean+Std)"
echo "  Config: $CFG | OutDir: $OUT_DIR"
echo "--------------------------------------------------------------------------------"

LAST_CKPT="$OUT_DIR/last.pt"
if [ "$FRESH" = false ] && [ -f "$LAST_CKPT" ]; then
    echo "  [RESUME] Found valid resume checkpoint at $LAST_CKPT. Resuming training from last epoch..."
    TRAIN_ARGS=("-m" "rmr_v3.train" "--config" "$CFG" "--resume" "$LAST_CKPT" "--allow-cross-commit-resume")
else
    TRAIN_ARGS=("-m" "rmr_v3.train" "--config" "$CFG" "--overwrite")
fi
echo "  [TRAIN] Executing: $PYTHON_EXE ${TRAIN_ARGS[*]}"
"$PYTHON_EXE" "${TRAIN_ARGS[@]}"

# 2. Evaluation Step
echo -e "\n--------------------------------------------------------------------------------"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Evaluating on ShanghaiTech Part A Test Set (182 images)..."
echo "--------------------------------------------------------------------------------"

if [ ! -f "$BEST_CKPT" ]; then
    echo "ERROR: Best checkpoint not found at $BEST_CKPT" >&2
    exit 1
fi

"$PYTHON_EXE" -m rmr_v3.eval --checkpoint "$BEST_CKPT" --manifest "data/sha_a_test.jsonl" --output-dir "$EVAL_DIR"

# 3. Paired Statistical Comparison against V3-B
PRED_V3B="runs/sha_a/historical_commit_91c0b841/rmr_v3_rw_seed42/eval_test/predictions.csv"
PRED_V4S="$EVAL_DIR/predictions.csv"
COMP_JSON_V3B="runs/sha_a/comparison_v3b_vs_v4_mean_std.json"

if [ -f "$PRED_V3B" ]; then
    echo -e "\n--------------------------------------------------------------------------------"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running Paired Statistical Comparison: V3-B vs V4-S (Mean+Std)..."
    echo "--------------------------------------------------------------------------------"
    "$PYTHON_EXE" -m rmr_count.aggregate --compare "$PRED_V3B" "$PRED_V4S" --name-a "V3-B_RW" --name-b "V4_MeanStd" --pred-col pred --output "$COMP_JSON_V3B"
fi

# 4. Optional Paired Comparison against Full Candidate if candidate has completed
PRED_V4_CAND="runs/sha_a/rmr_v4_candidate_seed42/eval_test/predictions.csv"
COMP_JSON_CAND="runs/sha_a/comparison_v4_candidate_vs_v4_mean_std.json"
if [ -f "$PRED_V4_CAND" ]; then
    echo -e "\n--------------------------------------------------------------------------------"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running Paired Statistical Comparison: V4 Candidate vs V4-S..."
    echo "--------------------------------------------------------------------------------"
    "$PYTHON_EXE" -m rmr_count.aggregate --compare "$PRED_V4_CAND" "$PRED_V4S" --name-a "V4_Candidate" --name-b "V4_MeanStd" --pred-col pred --output "$COMP_JSON_CAND"
fi

echo -e "\n================================================================================"
echo "  RMR-V4 MEAN+STD ABLATION EXPERIMENT COMPLETED SUCCESSFULLY!"
echo -e "================================================================================\n"
