#!/usr/bin/env bash
# ==============================================================================
# RMR-V68 Grand Synthesis Campaign Execution Script
# Target Hardware: Remote Ubuntu Linux Workstation (RTX 5070 Ti 16GB GDDR7)
# Python: PyTorch 2.6+, CUDA 12.x, AMP FP16
# ==============================================================================

set -eo pipefail

export PYTHONUNBUFFERED=1
mkdir -p logs/v68

run_experiment() {
    local exp_name="$1"
    local config_file="$2"
    local log_file="logs/v68/${exp_name}_$(date +%Y%m%d_%H%M%S).log"

    echo "================================================================================"
    echo "  STARTING EXPERIMENT: ${exp_name}"
    echo "  Config: ${config_file}"
    echo "  Log:    ${log_file}"
    echo "  Time:   $(date)"
    echo "================================================================================"

    python train.py --config "${config_file}" 2>&1 | tee "${log_file}"

    echo "================================================================================"
    echo "  FINISHED EXPERIMENT: ${exp_name}"
    echo "  Time: $(date)"
    echo "================================================================================"
}

TARGET="${1:-help}"

case "${TARGET}" in
    exp0)
        run_experiment "v68_exp0_sha_a_flagship" "configs/rmr_v68/v68_exp0_sha_a_flagship_fed_pure2loss_1500ep.yaml"
        ;;
    exp1)
        run_experiment "v68_exp1_sha_a_pure_y0" "configs/rmr_v68/v68_exp1_sha_a_abl_pure_finehead_y0_1500ep.yaml"
        ;;
    exp2)
        run_experiment "v68_exp2_track_a_sha_a" "configs/rmr_v68/v68_exp2_track_a_sha_a_pure2loss_1500ep.yaml"
        ;;
    exp3)
        run_experiment "v68_exp3_shb_track_b" "configs/rmr_v68/v68_exp3_shb_track_b_universal_1500ep.yaml"
        ;;
    exp4)
        run_experiment "v68_exp4_shb_track_a" "configs/rmr_v68/v68_exp4_shb_track_a_universal_1500ep.yaml"
        ;;
    exp5)
        run_experiment "v68_exp5_qnrf_track_b" "configs/rmr_v68/v68_exp5_qnrf_track_b_universal_1500ep.yaml"
        ;;
    exp6)
        run_experiment "v68_exp6_nwpu_track_b" "configs/rmr_v68/v68_exp6_nwpu_track_b_universal_1500ep.yaml"
        ;;
    exp7)
        run_experiment "v68_exp7_ucf_cc_50" "configs/rmr_v68/v68_exp7_ucf_cc_50_universal_1500ep.yaml"
        ;;
    sha)
        echo "=== Running ShanghaiTech Part A Suite (Exp 0, 1, 2) ==="
        run_experiment "v68_exp0_sha_a_flagship" "configs/rmr_v68/v68_exp0_sha_a_flagship_fed_pure2loss_1500ep.yaml"
        run_experiment "v68_exp1_sha_a_pure_y0" "configs/rmr_v68/v68_exp1_sha_a_abl_pure_finehead_y0_1500ep.yaml"
        run_experiment "v68_exp2_track_a_sha_a" "configs/rmr_v68/v68_exp2_track_a_sha_a_pure2loss_1500ep.yaml"
        ;;
    shb)
        echo "=== Running ShanghaiTech Part B Universal Suite (Exp 3, 4) ==="
        run_experiment "v68_exp3_shb_track_b" "configs/rmr_v68/v68_exp3_shb_track_b_universal_1500ep.yaml"
        run_experiment "v68_exp4_shb_track_a" "configs/rmr_v68/v68_exp4_shb_track_a_universal_1500ep.yaml"
        ;;
    all)
        echo "=== Running Complete V68 Campaign Across All Canonical Datasets ==="
        run_experiment "v68_exp0_sha_a_flagship" "configs/rmr_v68/v68_exp0_sha_a_flagship_fed_pure2loss_1500ep.yaml"
        run_experiment "v68_exp1_sha_a_pure_y0" "configs/rmr_v68/v68_exp1_sha_a_abl_pure_finehead_y0_1500ep.yaml"
        run_experiment "v68_exp2_track_a_sha_a" "configs/rmr_v68/v68_exp2_track_a_sha_a_pure2loss_1500ep.yaml"
        run_experiment "v68_exp3_shb_track_b" "configs/rmr_v68/v68_exp3_shb_track_b_universal_1500ep.yaml"
        run_experiment "v68_exp4_shb_track_a" "configs/rmr_v68/v68_exp4_shb_track_a_universal_1500ep.yaml"
        run_experiment "v68_exp5_qnrf_track_b" "configs/rmr_v68/v68_exp5_qnrf_track_b_universal_1500ep.yaml"
        run_experiment "v68_exp6_nwpu_track_b" "configs/rmr_v68/v68_exp6_nwpu_track_b_universal_1500ep.yaml"
        run_experiment "v68_exp7_ucf_cc_50" "configs/rmr_v68/v68_exp7_ucf_cc_50_universal_1500ep.yaml"
        ;;
    *)
        echo "Usage: ./run_v68_campaign.sh [exp0|exp1|exp2|exp3|exp4|exp5|exp6|exp7|sha|shb|all]"
        echo ""
        echo "Experiments:"
        echo "  exp0 : SHA-A Flagship (Track B, FED-MCBPRO, Pure 2-Loss, Crowd-Aware Crop)"
        echo "  exp1 : SHA-A Ablation (Track B, Pure Fine Head Y0 without solver)"
        echo "  exp2 : SHA-A Track A  (Track A, Stride 4, <=104k params, Pure 2-Loss)"
        echo "  exp3 : SHB Track B    (Track B, Native Full-Image Inference, Dynamic k-NN [2,16]px)"
        echo "  exp4 : SHB Track A    (Track A, Native Full-Image Inference, Stride 4)"
        echo "  exp5 : UCF-QNRF       (Track B, Extreme Multiscale Gigapixel)"
        echo "  exp6 : NWPU-Crowd     (Track B, Large-Scale with Zero-Count Negative Images)"
        echo "  exp7 : UCF_CC_50      (Track B, Fold 1 High-Density)"
        echo "  sha  : Runs exp0, exp1, exp2 sequentially"
        echo "  shb  : Runs exp3, exp4 sequentially"
        echo "  all  : Runs all 8 experiments sequentially"
        exit 1
        ;;
esac
