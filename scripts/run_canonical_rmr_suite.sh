#!/usr/bin/env bash
set -euo pipefail

# Canonical RMR & RMR-v3 Benchmark Suite Runner for Linux/Ubuntu
# Adheres strictly to:
# 1. Zero ad-hoc splits (300 train_all, 182 test).
# 2. Sequential execution for maximum CUDA throughput and thermal stability.
# 3. Direct full-image MAE/RMSE headline metrics.
# 4. Rigorous paired comparison (t-test, Wilcoxon, bootstrap 95% CI).
# 5. Publication-grade provenance verification (matching clean git HEAD).

FRESH=false
ALLOW_CROSS_COMMIT_RESUME=false
ARCHIVE_STALE=true
ALLOW_DIRTY=false

for arg in "$@"; do
    case "$arg" in
        --fresh) FRESH=true ;;
        --allow-cross-commit-resume) ALLOW_CROSS_COMMIT_RESUME=true ;;
        --no-archive) ARCHIVE_STALE=false ;;
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
    echo "ERROR: Working tree is dirty at git commit $CURRENT_COMMIT. Canonical benchmark runs require a clean git tree." >&2
    exit 1
fi

echo -e "\n================================================================================"
echo "  CANONICAL RMR / RMR-V3 BENCHMARK SUITE (Linux/Ubuntu)"
echo "  Order: V3-A -> V3-B -> B5-P (Sequential execution on GPU)"
echo "  Git HEAD: $CURRENT_COMMIT (Dirty: $IS_DIRTY) | Fresh: $FRESH"
echo -e "================================================================================\n"

run_benchmark_entry() {
    local name="$1"
    local train_mod="$2"
    local cfg="$3"
    local out_dir="$4"
    local eval_mod="$5"

    local best_ckpt="$out_dir/best_val_mae.pt"
    local eval_dir="$out_dir/eval_test"
    local summary_json="$eval_dir/summary.json"

    echo "--------------------------------------------------------------------------------"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: $name"
    echo "  Config: $cfg | OutDir: $out_dir"
    echo "--------------------------------------------------------------------------------"

    local artifacts_valid=false
    if [ "$FRESH" = false ] && [ -f "$summary_json" ] && [ -f "$best_ckpt" ]; then
        if "$PYTHON_EXE" -c "
import json, sys
data = json.load(open('$summary_json'))
p = data.get('provenance', {})
if p.get('training_commit') == '$CURRENT_COMMIT' and not p.get('training_git_dirty', True):
    sys.exit(0)
sys.exit(1)
" 2>/dev/null; then
            artifacts_valid=true
        fi
    fi

    if [ "$artifacts_valid" = true ]; then
        echo "  [SKIP] Verified matching clean HEAD artifacts at $out_dir. Skipping training."
    else
        local can_resume=false
        local last_ckpt="$out_dir/last.pt"
        if [ "$FRESH" = false ] && [ -f "$last_ckpt" ]; then
            if "$PYTHON_EXE" -c "
import torch, sys
ckpt = torch.load('$last_ckpt', map_location='cpu', weights_only=False)
c = str(ckpt.get('git_commit', ckpt.get('provenance', {}).get('git_commit', 'unknown')))
if c == '$CURRENT_COMMIT' or '$ALLOW_CROSS_COMMIT_RESUME' == 'true':
    sys.exit(0)
sys.exit(1)
" 2>/dev/null; then
                can_resume=true
            fi
        fi

        if [ "$can_resume" = true ]; then
            echo "  [RESUME] Found valid resume checkpoint at $last_ckpt. Resuming..."
            local train_args=("-m" "$train_mod" "--config" "$cfg" "--resume" "$last_ckpt")
            if [ "$ALLOW_CROSS_COMMIT_RESUME" = true ]; then
                train_args+=("--allow-cross-commit-resume")
            fi
        else
            if [ -d "$out_dir" ] && [ "$(ls -A "$out_dir" 2>/dev/null)" ]; then
                if [ "$ARCHIVE_STALE" = true ]; then
                    local ts=$(date '+%Y%m%d_%H%M%S')
                    local clean_name=$(basename "$out_dir")
                    local archive_dir="runs/sha_a/archive_${clean_name}_$ts"
                    echo "  [ARCHIVE] Archiving stale artifacts to $archive_dir..."
                    mkdir -p "$(dirname "$archive_dir")"
                    mv "$out_dir" "$archive_dir"
                fi
            fi
            local train_args=("-m" "$train_mod" "--config" "$cfg" "--overwrite")
        fi

        echo "  [TRAIN] Executing: $PYTHON_EXE ${train_args[*]}"
        "$PYTHON_EXE" "${train_args[@]}"
    fi

    # Evaluation step
    local eval_valid=false
    if [ "$FRESH" = false ] && [ -f "$summary_json" ]; then
        if "$PYTHON_EXE" -c "
import json, sys
data = json.load(open('$summary_json'))
p = data.get('provenance', {})
if p.get('evaluation_commit') == '$CURRENT_COMMIT' and not p.get('git_dirty', True):
    sys.exit(0)
sys.exit(1)
" 2>/dev/null; then
            eval_valid=true
        fi
    fi

    if [ "$eval_valid" = false ]; then
        if [ ! -f "$best_ckpt" ]; then
            echo "ERROR: Checkpoint not found: $best_ckpt" >&2
            exit 1
        fi
        echo "  [EVAL] Evaluating $best_ckpt on data/sha_a_test.jsonl..."
        "$PYTHON_EXE" -m "$eval_mod" --checkpoint "$best_ckpt" --manifest "data/sha_a_test.jsonl" --output-dir "$eval_dir"
    fi

    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: $name\n"
}

# Run the 3 canonical variants
run_benchmark_entry "V3-A (Probabilistic Uniform Control)" "rmr_v3.train" "configs/rmr_v3/probabilistic_uniform.yaml" "runs/sha_a/rmr_v3_uniform_control_seed42" "rmr_v3.eval"
run_benchmark_entry "V3-B (Reliability Weighted - Proposed)" "rmr_v3.train" "configs/rmr_v3/reliability_weighted.yaml" "runs/sha_a/rmr_v3_rw_seed42" "rmr_v3.eval"
run_benchmark_entry "B5-P (Canonical Deterministic Baseline)" "rmr_v2.train" "configs/rmr_v2/rmr_projected_t2.yaml" "runs/sha_a/b5p_canonical_seed42" "rmr_v2.eval"

echo "================================================================================"
echo "  ALL 3 RUNS COMPLETED. RUNNING PAIRED COMPARISONS AND AGGREGATION..."
echo -e "================================================================================\n"

PRED_V3A="runs/sha_a/rmr_v3_uniform_control_seed42/eval_test/predictions.csv"
PRED_V3B="runs/sha_a/rmr_v3_rw_seed42/eval_test/predictions.csv"
PRED_B5P="runs/sha_a/b5p_canonical_seed42/eval_test/predictions.csv"

# Comparison 1: V3-A vs V3-B
echo "--- Comparison 1: V3-A (Uniform) vs V3-B (Reliability Weighted) [MAIN HYPOTHESIS] ---"
"$PYTHON_EXE" -m rmr_count.aggregate --compare "$PRED_V3A" "$PRED_V3B" --name-a "V3-A_Uniform" --name-b "V3-B_RW" --pred-col pred --output "runs/sha_a/comparison_v3a_vs_v3b.json"

# Comparison 2: B5-P vs V3-A
echo -e "\n--- Comparison 2: B5-P (Deterministic) vs V3-A (Probabilistic Uniform) ---"
"$PYTHON_EXE" -m rmr_count.aggregate --compare "$PRED_B5P" "$PRED_V3A" --name-a "B5-P_Deterministic" --name-b "V3-A_Uniform" --pred-col pred --output "runs/sha_a/comparison_b5p_vs_v3a.json"

# Comparison 3: B5-P vs V3-B
echo -e "\n--- Comparison 3: B5-P (Deterministic) vs V3-B (Reliability Weighted) ---"
"$PYTHON_EXE" -m rmr_count.aggregate --compare "$PRED_B5P" "$PRED_V3B" --name-a "B5-P_Deterministic" --name-b "V3-B_RW" --pred-col pred --output "runs/sha_a/comparison_b5p_vs_v3b.json"

echo -e "\n================================================================================"
echo "  CANONICAL SUITE EXECUTION AND COMPARISONS COMPLETED SUCCESSFULLY!"
echo -e "================================================================================\n"
