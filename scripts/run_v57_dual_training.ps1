# ==============================================================================
# Dual Training Launcher for RMR V57 on Windows PowerShell
# Runs Wave 1 Proposed models (Track A + Track B) concurrently
# ==============================================================================
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$env:MALLOC_ARENA_MAX = "2"
$env:OMP_NUM_THREADS = "2"

$logDir = Join-Path $repoRoot "logs\rmr_v57"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " LAUNCHING RMR V57 PROPOSED DUAL EXPERIMENTS" -ForegroundColor Cyan
Write-Host " 1. Track A: v57_track_a_proposed_multistep_rn.yaml (103,959 params)" -ForegroundColor Green
Write-Host " 2. Track B: v57_track_b_proposed_deep_texture_rn.yaml (144,960 params)" -ForegroundColor Magenta
Write-Host "==========================================================" -ForegroundColor Cyan

$jobA = Start-Process -FilePath "python" `
    -ArgumentList "train.py --config configs/rmr_v57/v57_track_a_proposed_multistep_rn.yaml" `
    -RedirectStandardOutput (Join-Path $logDir "train_v57_track_a.log") `
    -RedirectStandardError (Join-Path $logDir "train_v57_track_a_err.log") `
    -PassThru -NoNewWindow

Start-Sleep -Seconds 3

$jobB = Start-Process -FilePath "python" `
    -ArgumentList "train.py --config configs/rmr_v57/v57_track_b_proposed_deep_texture_rn.yaml" `
    -RedirectStandardOutput (Join-Path $logDir "train_v57_track_b.log") `
    -RedirectStandardError (Join-Path $logDir "train_v57_track_b_err.log") `
    -PassThru -NoNewWindow

Write-Host "Both jobs launched successfully in background!" -ForegroundColor Green
Write-Host "Job Track A PID: $($jobA.Id)" -ForegroundColor Green
Write-Host "Job Track B PID: $($jobB.Id)" -ForegroundColor Magenta
Write-Host "Logs saved in: $logDir" -ForegroundColor Gray
Write-Host "Monitor with: Get-Content logs\rmr_v57\train_v57_track_a.log -Wait -Tail 20" -ForegroundColor Gray
