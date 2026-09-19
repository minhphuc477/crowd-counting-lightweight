# PowerShell Runner for RMR-v27 Comprehensive Experiment Suite
$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$models = @(
    @{ RunId = "rmr_v27_canonical_restored";    Config = "configs/rmr_v27/rmr_v27_canonical_restored.yaml" },
    @{ RunId = "rmr_v27_sota_push";             Config = "configs/rmr_v27/rmr_v27_sota_push.yaml" },
    @{ RunId = "rmr_v27_ablation_cyclic_bb";    Config = "configs/rmr_v27/rmr_v27_ablation_cyclic_bb.yaml" },
    @{ RunId = "rmr_v27_ablation_no_bb";        Config = "configs/rmr_v27/rmr_v27_ablation_no_bb.yaml" },
    @{ RunId = "rmr_v27_ablation_morozov05";    Config = "configs/rmr_v27/rmr_v27_ablation_morozov05.yaml" },
    @{ RunId = "rmr_v27_ablation_no_morozov";   Config = "configs/rmr_v27/rmr_v27_ablation_no_morozov.yaml" },
    @{ RunId = "rmr_v27_ablation_curv035";      Config = "configs/rmr_v27/rmr_v27_ablation_curv035.yaml" },
    @{ RunId = "rmr_v27_ablation_no_curvature"; Config = "configs/rmr_v27/rmr_v27_ablation_no_curvature.yaml" },
    @{ RunId = "rmr_v27_ablation_no_scale_align"; Config = "configs/rmr_v27/rmr_v27_ablation_no_scale_align.yaml" },
    @{ RunId = "rmr_v27_ablation_no_hard_bg";   Config = "configs/rmr_v27/rmr_v27_ablation_no_hard_bg.yaml" },
    @{ RunId = "rmr_v27_control_no_solver";     Config = "configs/rmr_v27/rmr_v27_control_no_solver.yaml" }
)

$logDir = "runs/sha_a/suite_logs_v27"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
}

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  RMR-v27 COMPREHENSIVE BENCHMARK & ABLATION EXPERIMENT SUITE (PowerShell)" -ForegroundColor Cyan
Write-Host "  Gold Reference: RMR-v19 Canonical Isotropic (72.61 TTA MAE, 72.84 Direct MAE)" -ForegroundColor Cyan
Write-Host "  Target: Sub-70 MAE (< 69.80)" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

$py = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }

foreach ($m in $models) {
    $runId = $m.RunId
    $cfg = $m.Config
    $log = "$logDir/$runId.log"

    Write-Host "`n>>> Starting training for: $runId ($cfg)" -ForegroundColor Green
    & $py -m rmr_v3.train --config $cfg --run-id $runId --overwrite 2>&1 | Tee-Object -FilePath $log
    Write-Host ">>> Completed: $runId" -ForegroundColor Yellow

    $ckptPath = "runs/sha_a/$runId/best_val_mae.pt"
    if (Test-Path $ckptPath) {
        Write-Host ">>> Evaluating with Horizontal Flip TTA: $ckptPath" -ForegroundColor Magenta
        $evalLog = "$logDir/${runId}_eval_tta.log"
        & $py -m rmr_v3.eval --checkpoint $ckptPath --tta --no-tiling 2>&1 | Tee-Object -FilePath $evalLog
    }
}

Write-Host "`n>>> Summarizing RMR-v27 Suite Results:" -ForegroundColor Cyan
& $py scripts/summarize_rmr_v27_suite.py

Write-Host "`nAll RMR-v27 experiments completed!" -ForegroundColor Cyan
