#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Comprehensive 4-Model Experiment Runner for Linux/Ubuntu
# Models:
#   1. V3-A: Probabilistic Uniform Control (configs/rmr_v3/probabilistic_uniform.yaml)
#   2. V3-B: Reliability Weighted Proposed (configs/rmr_v3/reliability_weighted.yaml)
#   3. B5-P: Canonical Deterministic Baseline (configs/rmr_v2/rmr_projected_t2.yaml)
#   4. V4:   RMR-v4 Full Candidate (configs/rmr_v4/final_candidate.yaml)
# ==============================================================================

FRESH=false
ALLOW_CROSS_COMMIT_RESUME=true
ARCHIVE_STALE=false
ALLOW_DIRTY=true

for arg in "$@"; do
    case "$arg" in
        --fresh) FRESH=true ;;
        --strict-provenance) ALLOW_CROSS_COMMIT_RESUME=false; ALLOW_DIRTY=false; ARCHIVE_STALE=true ;;
        --allow-dirty) ALLOW_DIRTY=true ;;
        *) echo "Unknown argument: $arg" && exit 1 ;;
    esac
done

PYTHON_EXE=".venv/bin/python"
if [ ! -f "$PYTHON_EXE" ]; then
    PYTHON_EXE="python3"
fi

CURRENT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
GIT_STATUS_RAW=$(git status --porcelain 2>/dev/null || true)
IS_DIRTY=false
if [ -n "$GIT_STATUS_RAW" ]; then
    IS_DIRTY=true
fi

if [ "$IS_DIRTY" = true ] && [ "$ALLOW_DIRTY" = false ]; then
    echo "ERROR: Working tree is dirty at git commit $CURRENT_COMMIT. Pass --allow-dirty to continue." >&2
    exit 1
fi

echo -e "\n================================================================================"
echo "  RMR FULL 4-MODEL BENCHMARK SUITE (Linux/Ubuntu)"
echo "  Order: V3-A -> V3-B -> B5-P -> V4-Candidate"
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
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Model: $name"
    echo "  Config: $cfg | OutDir: $out_dir"
    echo "--------------------------------------------------------------------------------"

    local artifacts_valid=false
    if [ "$FRESH" = false ] && [ -f "$summary_json" ] && [ -f "$best_ckpt" ]; then
        if [ "$ALLOW_CROSS_COMMIT_RESUME" = true ]; then
            artifacts_valid=true
        else
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
    fi

    if [ "$artifacts_valid" = true ]; then
        echo "  [SKIP] Found valid artifacts at $out_dir. Skipping training."
    else
        local can_resume=false
        local last_ckpt="$out_dir/last.pt"
        if [ "$FRESH" = false ] && [ -f "$last_ckpt" ]; then
            if [ "$ALLOW_CROSS_COMMIT_RESUME" = true ]; then
                can_resume=true
            else
                if "$PYTHON_EXE" -c "
import torch, sys
ckpt = torch.load('$last_ckpt', map_location='cpu', weights_only=False)
c = str(ckpt.get('git_commit', ckpt.get('provenance', {}).get('git_commit', 'unknown')))
if c == '$CURRENT_COMMIT':
    sys.exit(0)
sys.exit(1)
" 2>/dev/null; then
                    can_resume=true
                fi
            fi
        fi

        if [ "$can_resume" = true ]; then
            echo "  [RESUME] Resuming training from $last_ckpt..."
            local train_args=("-m" "$train_mod" "--config" "$cfg" "--resume" "$last_ckpt" "--allow-cross-commit-resume")
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
    if [ ! -f "$best_ckpt" ]; then
        echo "ERROR: Checkpoint not found: $best_ckpt" >&2
        exit 1
    fi
    echo "  [EVAL] Evaluating $best_ckpt on data/sha_a_test.jsonl..."
    "$PYTHON_EXE" -m "$eval_mod" --checkpoint "$best_ckpt" --manifest "data/sha_a_test.jsonl" --output-dir "$eval_dir"

    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] Finished Model: $name\n"
}

# 1. Model 1: V3-A (Probabilistic Uniform Control)
run_benchmark_entry "1/4: V3-A (Probabilistic Uniform Control)" "rmr_v3.train" "configs/rmr_v3/probabilistic_uniform.yaml" "runs/sha_a/rmr_v3_uniform_control_seed42" "rmr_v3.eval"

# 2. Model 2: V3-B (Reliability Weighted Proposed)
run_benchmark_entry "2/4: V3-B (Reliability Weighted - Proposed)" "rmr_v3.train" "configs/rmr_v3/reliability_weighted.yaml" "runs/sha_a/rmr_v3_rw_seed42" "rmr_v3.eval"

# 3. Model 3: B5-P (Canonical Deterministic Baseline)
run_benchmark_entry "3/4: B5-P (Canonical Deterministic Baseline)" "rmr_v2.train" "configs/rmr_v2/rmr_projected_t2.yaml" "runs/sha_a/b5p_canonical_seed42" "rmr_v2.eval"

# 4. Model 4: V4 Candidate (Full Candidate)
run_benchmark_entry "4/4: RMR-v4 Full Candidate" "rmr_v3.train" "configs/rmr_v4/final_candidate.yaml" "runs/sha_a/rmr_v4_candidate_seed42" "rmr_v3.eval"

echo "================================================================================"
echo "  ALL 4 RUNS COMPLETED! RUNNING PAIRED STATISTICAL COMPARISONS..."
echo -e "================================================================================\n"

PRED_V3A="runs/sha_a/rmr_v3_uniform_control_seed42/eval_test/predictions.csv"
PRED_V3B="runs/sha_a/rmr_v3_rw_seed42/eval_test/predictions.csv"
PRED_B5P="runs/sha_a/b5p_canonical_seed42/eval_test/predictions.csv"
PRED_V4="runs/sha_a/rmr_v4_candidate_seed42/eval_test/predictions.csv"

# 1. V3-A vs V3-B (Reliability weighting causal hypothesis)
echo "--- 1. Paired Comparison: V3-A (Uniform) vs V3-B (Reliability Weighted) ---"
"$PYTHON_EXE" -m rmr_count.aggregate --compare "$PRED_V3A" "$PRED_V3B" --name-a "V3-A_Uniform" --name-b "V3-B_RW" --pred-col pred --output "runs/sha_a/comparison_v3a_vs_v3b.json"

# 2. B5-P vs V3-B (Overall improvement from Baseline to Proposed)
echo -e "\n--- 2. Paired Comparison: B5-P (Deterministic) vs V3-B (Reliability Weighted) ---"
"$PYTHON_EXE" -m rmr_count.aggregate --compare "$PRED_B5P" "$PRED_V3B" --name-a "B5-P_Deterministic" --name-b "V3-B_RW" --pred-col pred --output "runs/sha_a/comparison_b5p_vs_v3b.json"

# 3. V3-B vs V4 Candidate (Advancement from V3-B to V4)
echo -e "\n--- 3. Paired Comparison: V3-B (RW-RMR) vs V4 Candidate ---"
"$PYTHON_EXE" -m rmr_count.aggregate --compare "$PRED_V3B" "$PRED_V4" --name-a "V3-B_RW" --name-b "V4_Candidate" --pred-col pred --output "runs/sha_a/comparison_v3b_vs_v4_candidate.json"

echo -e "\n================================================================================"
echo "  COMPREHENSIVE 4-MODEL SUITE COMPLETED SUCCESSFULLY!"
echo -e "================================================================================\n"
