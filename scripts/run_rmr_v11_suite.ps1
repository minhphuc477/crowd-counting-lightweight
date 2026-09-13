# ==============================================================================
# RMR-v11 Comprehensive Multi-Model Experiment Suite for Windows (PowerShell)
# ==============================================================================
# Models in the suite (6 runs):
#   1. rmr_v11_canonical_dsr: Flagship Canonical (DSR + FG-Gate + Curvature + Morozov TR + Hard-BG + Dual-Depth)
#   2. rmr_v11_ablation_no_curvature: Curvature ablation (lambda_curv = 0.0)
#   3. rmr_v11_ablation_no_trust_region: Morozov Trust-Region ablation (kappa = 0.0)
#   4. rmr_v11_ablation_no_hard_bg: Top-K Hard BG Mining ablation (lambda_hard_bg = 0.0)
#   5. rmr_v11_ablation_no_fg_gate: Foreground gate sub-head ablation (104,440 params)
#   6. rmr_v11_control_no_solver: Direct feedforward control (enable_solver = false)
# ==============================================================================

param (
    [string]$Mode = "sequential",
    [string]$Single = "",
    [int]$Epochs = 0,
    [switch]$CacheTeacher = $false
)

$ErrorActionPreference = "Stop"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:PYTHONUNBUFFERED = "1"

$PythonExe = ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  RMR-v11 MULTI-MODEL EXPERIMENT SUITE (Windows PowerShell)" -ForegroundColor Cyan
Write-Host "  Python: $PythonExe" -ForegroundColor Cyan
Write-Host "  Mode:   $Mode" -ForegroundColor Cyan
Write-Host "  Date:   $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

$LogDir = "runs\sha_a\suite_logs_v11"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

if ($CacheTeacher) {
    Write-Host "`n>>> [STEP 0] Pre-caching teacher density maps to disk..." -ForegroundColor Yellow
    & $PythonExe scripts/cache_teacher_density.py `
        --teacher-ckpt runs/sha_a/rmr_v10_dynamic_scale_routing/best_val_mae.pt `
        --manifest data/sha_a_train_all.jsonl `
        --output-dir data/teacher_density_cache
}

$RunConfigs = [ordered]@{
    "rmr_v11_canonical_dsr"            = "configs\rmr_v11\rmr_v11_canonical_dsr.yaml"
    "rmr_v11_ablation_no_curvature"     = "configs\rmr_v11\rmr_v11_ablation_no_curvature.yaml"
    "rmr_v11_ablation_no_trust_region"  = "configs\rmr_v11\rmr_v11_ablation_no_trust_region.yaml"
    "rmr_v11_ablation_no_hard_bg"       = "configs\rmr_v11\rmr_v11_ablation_no_hard_bg.yaml"
    "rmr_v11_ablation_no_fg_gate"       = "configs\rmr_v11\rmr_v11_ablation_no_fg_gate.yaml"
    "rmr_v11_control_no_solver"        = "configs\rmr_v11\rmr_v11_control_no_solver.yaml"
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

    $LogFile = "$LogDir\${RunId}.log"
    Write-Host "`n================================================================================" -ForegroundColor Green
    Write-Host "  STARTING: $RunId" -ForegroundColor Green
    Write-Host "  Config:   $ConfigFile" -ForegroundColor Green
    Write-Host "  Log:      $LogFile" -ForegroundColor Green
    Write-Host "================================================================================" -ForegroundColor Green

    $cmdArgs = @(
        "-m", "rmr_v3.train",
        "--config", $ConfigFile,
        "--run-id", $RunId
    )

    if ($Epochs -gt 0) {
        $cmdArgs += @("--epochs", "$Epochs")
    }

    $cmdLine = "$PythonExe " + ($cmdArgs -join " ")
    Write-Host "Executing: $cmdLine" -ForegroundColor DarkGray

    & $PythonExe @cmdArgs 2>&1 | Tee-Object -FilePath $LogFile

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Run $RunId failed with exit code $LASTEXITCODE." -ForegroundColor Red
        return $false
    }

    Write-Host "FINISHED: $RunId successfully." -ForegroundColor Green
    return $true
}

if ($Single -ne "") {
    if (-not $RunConfigs.Contains($Single)) {
        Write-Host "Unknown run ID: '$Single'. Available runs:" -ForegroundColor Red
        $RunConfigs.Keys | ForEach-Object { Write-Host "  - $_" -ForegroundColor Yellow }
        exit 1
    }
    Run-ModelForeground -RunId $Single -ConfigFile $RunConfigs[$Single]
    exit 0
}

$TotalRuns = $RunConfigs.Count
$CurrentRun = 0
$Results = [ordered]@{}

foreach ($kv in $RunConfigs.GetEnumerator()) {
    $CurrentRun++
    $runId = $kv.Key
    $cfgPath = $kv.Value

    Write-Host "`n[Run $CurrentRun / $TotalRuns] Launching $runId..." -ForegroundColor Yellow
    $success = Run-ModelForeground -RunId $runId -ConfigFile $cfgPath
    $Results[$runId] = if ($success) { "SUCCESS" } else { "FAILED" }
}

Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host "  ALL RMR-v11 SUITE RUNS FINISHED" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
foreach ($kv in $Results.GetEnumerator()) {
    $color = if ($kv.Value -eq "SUCCESS") { "Green" } else { "Red" }
    Write-Host "  $($kv.Key.PadRight(38)) : $($kv.Value)" -ForegroundColor $color
}
Write-Host "================================================================================" -ForegroundColor Cyan

# Automatically summarize results
& $PythonExe scripts/summarize_rmr_v11_suite.py
