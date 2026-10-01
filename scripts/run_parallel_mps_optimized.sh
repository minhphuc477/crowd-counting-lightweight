#!/usr/bin/env bash
# ==============================================================================
# High-Throughput Parallel 4-Model Training Runner for 1 GPU (16GB VRAM, 32GB RAM)
# Features:
#   1. NVIDIA MPS (Multi-Process Service) for Spatial Multitasking (zero context-switch latency)
#   2. CPU Core Affinity (taskset -c) & OMP_NUM_THREADS=2 (zero thread oversubscription)
#   3. Zero-Worker In-Memory Loading (--workers 0) (zero Copy-on-Write page duplication)
#   4. VRAM Virtual Memory Management (expandable_segments:True + 3.8GB memory limit per job)
#   5. Disabled cuDNN Benchmark (--no-cudnn-benchmark) to prevent lock contention
# ==============================================================================
set -euo pipefail

GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# 1. PyTorch Caching Allocator VMM to prevent virtual address fragmentation
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 2. Strict thread limits before initializing OpenMP / MKL
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

# 3. Setup and start NVIDIA MPS daemon
export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps_${GPU_ID}"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-log_${GPU_ID}"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

echo "================================================================================"
echo "  STARTING NVIDIA MULTI-PROCESS SERVICE (MPS) ON GPU $GPU_ID"
echo "================================================================================"

if pgrep -f "nvidia-cuda-mps-control" > /dev/null; then
    echo "[MPS] Daemon is already running."
else
    echo "[MPS] Launching nvidia-cuda-mps-control daemon..."
    nvidia-cuda-mps-control -d || echo "[MPS] Warning: MPS daemon could not start (non-Linux or permission error). Falling back to driver time-slicing."
    sleep 1
fi

# Cleanup trap to ensure MPS daemon and child jobs terminate cleanly
cleanup() {
    echo -e "\n[CLEANUP] Stopping background jobs and terminating MPS..."
    kill $(jobs -p) 2>/dev/null || true
    echo quit | nvidia-cuda-mps-control 2>/dev/null || true
    rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" 2>/dev/null || true
    echo "[CLEANUP] Done."
}
trap cleanup SIGINT SIGTERM EXIT

# 4. Partition GPU compute & memory per client (4 concurrent jobs on 16GB VRAM)
# 3800MB * 4 = 15.2GB / 16GB VRAM
export CUDA_MPS_PINNED_DEVICE_MEM_LIMIT="${GPU_ID}=3800M"
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25

PYTHON_EXE=".venv/bin/python"
if [ ! -f "$PYTHON_EXE" ]; then
    PYTHON_EXE="python3"
fi

LOG_DIR="runs/sha_a/parallel_mps_logs"
mkdir -p "$LOG_DIR"

echo "================================================================================"
echo "  LAUNCHING 4 JOBS IN PARALLEL (ISOLATED CPU CORES & ZERO-WORKER IN-MEMORY RAM)"
echo "================================================================================"

# Default core mapping (16 logical cores: 0-3, 4-7, 8-11, 12-15)
# If your machine has different core count, adjust the taskset ranges accordingly.
launch_job() {
    local job_idx="$1"
    local cores="$2"
    local run_id="$3"
    local config_file="$4"
    local log_file="$LOG_DIR/${run_id}.log"

    echo ">>> [Job $job_idx] Launching $run_id on Cores $cores (Log: $log_file)"
    taskset -c "$cores" "$PYTHON_EXE" -m rmr_v3.train \
        --config "$config_file" \
        --run-id "$run_id" \
        --workers 0 \
        --num-threads 2 \
        --no-cudnn-benchmark \
        --overwrite > "$log_file" 2>&1 &
    echo $!
}

# Example: 4 baseline experiments or ablated candidates
# Customize the configs as needed
PID0=$(launch_job 0 "0-3"   "job0_sub60_e5"          "configs/rmr_research/sub60_e5_pure_flat_dm16_no_cell.yaml")
PID1=$(launch_job 1 "4-7"   "job1_asym_morozov"      "configs/rmr_research/gen9/g9_asym_morozov_e5.yaml")
PID2=$(launch_job 2 "8-11"  "job2_dct_spectral"      "configs/rmr_research/gen9/g9_dct_spectral_fixed.yaml")
PID3=$(launch_job 3 "12-15" "job3_curvature_fixed"   "configs/rmr_research/gen9/g9_fix_curvature_init.yaml")

echo -e "\nAll 4 processes running concurrently under MPS:"
echo "  Job 0 (PID $PID0): tail -f $LOG_DIR/job0_sub60_e5.log"
echo "  Job 1 (PID $PID1): tail -f $LOG_DIR/job1_asym_morozov.log"
echo "  Job 2 (PID $PID2): tail -f $LOG_DIR/job2_dct_spectral.log"
echo "  Job 3 (PID $PID3): tail -f $LOG_DIR/job3_curvature_fixed.log"
echo -e "\nMonitoring GPU SMs: watch -n 1 nvidia-smi"

wait $PID0 $PID1 $PID2 $PID3

echo "================================================================================"
echo "  ALL 4 PARALLEL TRAINING JOBS COMPLETED SUCCESSFULLY!"
echo "================================================================================"
