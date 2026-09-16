# PowerShell Runner for RMR-v21 Sub-60 Experiment Suite
$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$models = @(
    @{ RunId = "rmr_v21_canonical";                Config = "configs/rmr_v21/rmr_v21_canonical.yaml" },
    @{ RunId = "rmr_v21_ablation_no_hybrid_flux";   Config = "configs/rmr_v21/rmr_v21_ablation_no_hybrid_flux.yaml" },
    @{ RunId = "rmr_v21_ablation_no_bb_step";       Config = "configs/rmr_v21/rmr_v21_ablation_no_bb_step.yaml" },
    @{ RunId = "rmr_v21_ablation_no_sample_loss";   Config = "configs/rmr_v21/rmr_v21_ablation_no_sample_loss.yaml" },
    @{ RunId = "rmr_v21_ablation_no_curvature";     Config = "configs/rmr_v21/rmr_v21_ablation_no_curvature.yaml" },
    @{ RunId = "rmr_v21_control_no_solver";         Config = "configs/rmr_v21/rmr_v21_control_no_solver.yaml" }
)

$logDir = "runs/sha_a/suite_logs_v21"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
}

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  RMR-v21 SUB-60 EXPERIMENT SUITE (PowerShell)" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

foreach ($m in $models) {
    $runId = $m.RunId
    $cfg = $m.Config
    $log = "$logDir/$runId.log"

    Write-Host "`n>>> Starting training for: $runId ($cfg)" -ForegroundColor Green
    python -m rmr_v3.train --config $cfg --run-id $runId 2>&1 | Tee-Object -FilePath $log
    Write-Host ">>> Completed: $runId" -ForegroundColor Yellow

    $ckptPath = "runs/sha_a/$runId/best_val_mae.pt"
    if (Test-Path $ckptPath) {
        Write-Host ">>> Evaluating with Horizontal Flip TTA: $ckptPath" -ForegroundColor Magenta
        $evalLog = "$logDir/${runId}_eval_tta.log"
        python -m rmr_v3.eval --checkpoint $ckptPath --tta 2>&1 | Tee-Object -FilePath $evalLog
    }
}

Write-Host "`nAll RMR-v21 experiments completed!" -ForegroundColor Cyan
