# ==============================================================================
# RMR-v9 Comprehensive Multi-Model Experiment Suite for Windows (PowerShell)
# ==============================================================================
# Models in the suite (6 runs):
#   1. rmr_v9_canonical: Flagship canonical model (T=6, Additive SIRT, Isotropic Laplacian TV, Flat-DM16, tau=0.0)
#   2. rmr_v9_aq_rmr: Full Anisotropic & Quadratic model (+anisotropic [64,32] & [32,64] boxes, mean_std stats, tau=0.015)
#   3. rmr_v9_ablation_no_proximal: Ablation of proximal step (tau=0.0, with anisotropic boxes + mean_std)
#   4. rmr_v9_ablation_isotropic: Ablation of anisotropic boxes (square boxes [32,64,128], tau=0.015 + mean_std)
#   5. rmr_v9_ablation_mean_only: Ablation of variance statistics (mean only, with anisotropic boxes + tau=0.015)
#   6. rmr_v9_control_no_solver: Control baseline without SIRT solver (direct base density y0)
# ==============================================================================

param (
    [string]$Mode = "sequential",
    [int]$Epochs = 0
)

$ErrorActionPreference = "Stop"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:PYTHONUNBUFFERED = "1"

$PythonExe = ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  RMR-v9 MULTI-MODEL EXPERIMENT SUITE (Windows PowerShell)" -ForegroundColor Cyan
Write-Host "  Python: $PythonExe" -ForegroundColor Cyan
Write-Host "  Mode:   $Mode" -ForegroundColor Cyan
Write-Host "  Date:   $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

$LogDir = "runs\sha_a\suite_logs_v9"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

$RunConfigs = [ordered]@{
    "rmr_v9_canonical"             = "configs\rmr_v9\rmr_v9_canonical.yaml"
    "rmr_v9_aq_rmr"                = "configs\rmr_v9\rmr_v9_aq_rmr.yaml"
    "rmr_v9_ablation_no_proximal"  = "configs\rmr_v9\rmr_v9_ablation_no_proximal.yaml"
    "rmr_v9_ablation_isotropic"    = "configs\rmr_v9\rmr_v9_ablation_isotropic.yaml"
    "rmr_v9_ablation_mean_only"    = "configs\rmr_v9\rmr_v9_ablation_mean_only.yaml"
    "rmr_v9_control_no_solver"     = "configs\rmr_v9\rmr_v9_control_no_solver.yaml"
}

function Run-ModelForeground {
    param (
        [string]$RunId,
        [string]$ConfigFile
    )

    $OutDir = "runs\sha_a\$RunId"
    if (-not (Test-Path $OutDir)) {
        New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
    }

    $LastCkpt = Join-Path $OutDir "last.pt"
    $ArgsList = @("-m", "rmr_v3.train", "--config", $ConfigFile, "--run-id", $RunId)

    if (Test-Path $LastCkpt) {
        Write-Host "  [$RunId] Existing checkpoint found -> RESUMING" -ForegroundColor Yellow
        $ArgsList += @("--resume", $LastCkpt, "--allow-cross-commit-resume")
    } else {
        Write-Host "  [$RunId] Starting fresh run (config: $ConfigFile)" -ForegroundColor Green
        $ArgsList += @("--overwrite")
    }

    if ($Epochs -gt 0) {
        $ArgsList += @("--epochs", $Epochs)
    }

    & $PythonExe $ArgsList
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Run $RunId failed with exit code $LASTEXITCODE"
        return
    }

    # Automatically evaluate best checkpoint on test partition
    $BestCkpt = Join-Path $OutDir "best_val_mae.pt"
    $EvalDir = Join-Path $OutDir "eval_test"
    if (Test-Path $BestCkpt) {
        Write-Host "  [$RunId] Evaluating best checkpoint on test set..." -ForegroundColor Cyan
        & $PythonExe -m rmr_v3.eval --checkpoint $BestCkpt --manifest "data\sha_a_test.jsonl" --output-dir $EvalDir
    }
}

foreach ($entry in $RunConfigs.GetEnumerator()) {
    $RunId = $entry.Key
    $ConfigFile = $entry.Value

    Write-Host "`n================================================================================" -ForegroundColor Cyan
    Write-Host "  STARTING: $RunId ($ConfigFile)" -ForegroundColor Cyan
    Write-Host "================================================================================" -ForegroundColor Cyan

    Run-ModelForeground -RunId $RunId -ConfigFile $ConfigFile
}

Write-Host "`n================================================================================" -ForegroundColor Green
Write-Host "All RMR-v9 matrix runs completed." -ForegroundColor Green
Write-Host "Generating summary table..." -ForegroundColor Green
& $PythonExe scripts\summarize_rmr_v9_suite.py
Write-Host "================================================================================" -ForegroundColor Green
