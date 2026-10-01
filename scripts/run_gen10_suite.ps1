# PowerShell Runner for Gen 10 Anti-Regression Breakthrough Suite (Target: MAE < 60.0)
$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

if (-not $env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS = "2" }

$models = @(
    # 1. Structural Resolution: Stride 2 Sub-pixel Dirichlet-Multinomial (104,893 params)
    @{ RunId = "g10_subpixel_dm_stride2";     Config = "configs/rmr_research/gen10/g10_subpixel_dm_stride2.yaml" },

    # 2. Deadband Calibration: Clean Isolated Asymmetric Morozov (104,441 params)
    @{ RunId = "g10_clean_asym_morozov";     Config = "configs/rmr_research/gen10/g10_clean_asym_morozov.yaml" },

    # 3. Frequency Regularization: Calibrated ChfL lambda=0.35 (104,441 params)
    @{ RunId = "g10_chfl_calibrated_lam035"; Config = "configs/rmr_research/gen10/g10_chfl_calibrated_lam035.yaml" },

    # 4. Orthogonal Synthesis: Subpixel Stride 2 + Calibrated ChfL (104,893 params)
    @{ RunId = "g10_subpixel_stride2_chfl";  Config = "configs/rmr_research/gen10/g10_subpixel_stride2_chfl.yaml" }
)

$logDir = "runs/sha_a/suite_logs_gen10"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
}

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  RMR GEN 10 ANTI-REGRESSION BREAKTHROUGH SUITE (PowerShell)" -ForegroundColor Cyan
Write-Host "  4 Targeted Experiments | Target: Breakthrough Sub-60 MAE (< 60.0)" -ForegroundColor Cyan
Write-Host "  Baseline: sub60_e5 (71.51 MAE) | Strict Budget: <= 105k Params, Zero KD" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

$py = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }

foreach ($m in $models) {
    $runId = $m.RunId
    $cfg = $m.Config
    $log = "$logDir/$runId.log"

    Write-Host "`n>>> Starting training for: $runId ($cfg)" -ForegroundColor Green
    & $py -m rmr_v3.train `
        --config $cfg `
        --run-id $runId `
        --workers 0 `
        --num-threads 2 `
        --no-cudnn-benchmark `
        --overwrite 2>&1 | Tee-Object -FilePath $log
    Write-Host ">>> Completed training: $runId" -ForegroundColor Yellow

    $ckptPath = "runs/sha_a/$runId/best_val_mae.pt"
    if (Test-Path $ckptPath) {
        Write-Host ">>> Evaluating Standard Test: $ckptPath" -ForegroundColor Magenta
        & $py -m rmr_v3.evaluate `
            --checkpoint $ckptPath `
            --manifest data/sha_a_test.jsonl `
            --crop-size 512 `
            --tta $false `
            --output-json "runs/sha_a/$runId/eval_test_canonical.json"
    }
}

Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host "  GEN 10 EXPERIMENT SUITE COMPLETED!" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
