# ==============================================================================
# RMR-v30 Hypothesis Ladder Benchmark Suite (Windows PowerShell)
# Target: Break the Sub-60 MAE barrier on ShanghaiTech Part A
# Gold Baseline: RMR-v19 Canonical Isotropic (72.61 TTA MAE, 72.84 Direct MAE)
# ==============================================================================
$ErrorActionPreference = "Stop"

$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$MODELS = @(
    @{ RunId = "rmr_v30_step0_v19_anchor";         Config = "configs/rmr_v30/rmr_v30_step0_v19_anchor.yaml" },
    @{ RunId = "rmr_v30_h1_anscombe_sirt";         Config = "configs/rmr_v30/rmr_v30_h1_anscombe_sirt.yaml" },
    @{ RunId = "rmr_v30_h2_dual_lattice_dcsr";     Config = "configs/rmr_v30/rmr_v30_h2_dual_lattice_dcsr.yaml" },
    @{ RunId = "rmr_v30_h3_anscombe_dual_lattice"; Config = "configs/rmr_v30/rmr_v30_h3_anscombe_dual_lattice.yaml" },
    @{ RunId = "rmr_v30_h4_deep_sirt_t8";          Config = "configs/rmr_v30/rmr_v30_h4_deep_sirt_t8.yaml" },
    @{ RunId = "rmr_v30_control_no_solver";        Config = "configs/rmr_v30/rmr_v30_control_no_solver.yaml" }
)

$LOG_DIR = "runs/sha_a/suite_logs_v30"
if (-not (Test-Path $LOG_DIR)) {
    New-Item -ItemType Directory -Path $LOG_DIR -Force | Out-Null
}

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  RMR-v30 HYPOTHESIS LADDER EXPERIMENT SUITE (PowerShell)" -ForegroundColor Cyan
Write-Host "  Gold Reference: RMR-v19 Canonical Isotropic (72.61 TTA MAE, 72.84 Direct MAE)" -ForegroundColor Cyan
Write-Host "  Target: Sub-60 MAE (< 60.0)" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

$PYTHON_BIN = "python"
if (Test-Path ".venv/Scripts/python.exe") {
    $PYTHON_BIN = ".venv/Scripts/python.exe"
}

foreach ($item in $MODELS) {
    $runId = $item.RunId
    $cfg = $item.Config
    $logFile = "$LOG_DIR/${runId}.log"

    Write-Host "`n>>> Starting training for: $runId ($cfg)" -ForegroundColor Green
    & $PYTHON_BIN -m rmr_v3.train --config "$cfg" --run-id "$runId" --overwrite 2>&1 | Tee-Object -FilePath "$logFile"
    Write-Host ">>> Completed: $runId" -ForegroundColor Green

    $ckptPath = "runs/sha_a/${runId}/best_val_mae.pt"
    if (Test-Path $ckptPath) {
        Write-Host ">>> Evaluating with Horizontal Flip TTA: $ckptPath" -ForegroundColor Yellow
        $evalLog = "$LOG_DIR/${runId}_eval_tta.log"
        & $PYTHON_BIN -m rmr_v3.eval --checkpoint "$ckptPath" --tta --no-tiling 2>&1 | Tee-Object -FilePath "$evalLog"
    }
}

Write-Host "`n>>> Summarizing RMR-v30 Suite Results:" -ForegroundColor Cyan
& $PYTHON_BIN scripts/summarize_rmr_v30_suite.py

Write-Host "`nAll RMR-v30 experiments completed!" -ForegroundColor Green
