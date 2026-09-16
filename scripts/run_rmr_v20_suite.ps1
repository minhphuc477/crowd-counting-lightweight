# PowerShell Runner for RMR-v20 Sub-60 Experiment Suite
$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$models = @(
    @{ RunId = "rmr_v20_canonical";                Config = "configs/rmr_v20/rmr_v20_canonical.yaml" },
    @{ RunId = "rmr_v20_ablation_no_coord_attn";    Config = "configs/rmr_v20/rmr_v20_ablation_no_coord_attn.yaml" },
    @{ RunId = "rmr_v20_ablation_no_nesterov";      Config = "configs/rmr_v20/rmr_v20_ablation_no_nesterov.yaml" },
    @{ RunId = "rmr_v20_ablation_no_adaptive_relax";Config = "configs/rmr_v20/rmr_v20_ablation_no_adaptive_relax.yaml" },
    @{ RunId = "rmr_v20_ablation_no_dense_loss";    Config = "configs/rmr_v20/rmr_v20_ablation_no_dense_loss.yaml" },
    @{ RunId = "rmr_v20_control_no_solver";         Config = "configs/rmr_v20/rmr_v20_control_no_solver.yaml" }
)

$logDir = "runs/sha_a/suite_logs_v20"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
}

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  RMR-v20 SUB-60 EXPERIMENT SUITE (PowerShell)" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

foreach ($m in $models) {
    $runId = $m.RunId
    $cfg = $m.Config
    $log = "$logDir/$runId.log"

    Write-Host "`n>>> Starting training for: $runId ($cfg)" -ForegroundColor Green
    python -m rmr_v3.train --config $cfg --run-id $runId 2>&1 | Tee-Object -FilePath $log
    Write-Host ">>> Completed: $runId" -ForegroundColor Yellow
}

Write-Host "`nAll RMR-v20 experiments completed!" -ForegroundColor Cyan
