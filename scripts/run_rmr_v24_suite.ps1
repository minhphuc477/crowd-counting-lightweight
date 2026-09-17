# PowerShell Runner for RMR-v24 Sub-60 Experiment Suite
$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$models = @(
    @{ RunId = "rmr_v24_canonical";                             Config = "configs/rmr_v24/rmr_v24_canonical.yaml" },
    @{ RunId = "rmr_v24_ablation_no_abb";                        Config = "configs/rmr_v24/rmr_v24_ablation_no_abb.yaml" },
    @{ RunId = "rmr_v24_ablation_no_perspective";                Config = "configs/rmr_v24/rmr_v24_ablation_no_perspective.yaml" },
    @{ RunId = "rmr_v24_ablation_no_density_gated_diffusion";   Config = "configs/rmr_v24/rmr_v24_ablation_no_density_gated_diffusion.yaml" },
    @{ RunId = "rmr_v24_ablation_no_curvature";                 Config = "configs/rmr_v24/rmr_v24_ablation_no_curvature.yaml" },
    @{ RunId = "rmr_v24_control_no_solver";                      Config = "configs/rmr_v24/rmr_v24_control_no_solver.yaml" }
)

$logDir = "runs/sha_a/suite_logs_v24"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
}

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  RMR-v24 SUB-60 EXPERIMENT SUITE (PowerShell)" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

$py = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }

foreach ($m in $models) {
    $runId = $m.RunId
    $cfg = $m.Config
    $log = "$logDir/$runId.log"

    Write-Host "`n>>> Starting training for: $runId ($cfg)" -ForegroundColor Green
    & $py -m rmr_v3.train --config $cfg --run-id $runId 2>&1 | Tee-Object -FilePath $log
    Write-Host ">>> Completed: $runId" -ForegroundColor Yellow

    $ckptPath = "runs/sha_a/$runId/best_val_mae.pt"
    if (Test-Path $ckptPath) {
        Write-Host ">>> Evaluating with Horizontal Flip TTA: $ckptPath" -ForegroundColor Magenta
        $evalLog = "$logDir/${runId}_eval_tta.log"
        & $py -m rmr_v3.eval --checkpoint $ckptPath --tta --no-tiling 2>&1 | Tee-Object -FilePath $evalLog
    }
}

Write-Host "`n>>> Summarizing RMR-v24 Suite Results:" -ForegroundColor Cyan
& $py scripts/summarize_rmr_v24_suite.py

Write-Host "`nAll RMR-v24 experiments completed!" -ForegroundColor Cyan
