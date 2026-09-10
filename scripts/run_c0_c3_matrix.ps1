param (
    [string]$Run = "all",      # "C0", "C1", "C2", "C3", or "all"
    [switch]$Fresh = $false,
    [switch]$AllowDirty = $false
)

$ErrorActionPreference = "Stop"

$pythonExe = ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

$currentCommit = (git rev-parse HEAD).Trim()
$gitStatusRaw = git status --porcelain
$gitStatus = if ($null -ne $gitStatusRaw) { ("$gitStatusRaw").Trim() } else { "" }
$isDirty = [bool]$gitStatus

if ($isDirty -and (-not $AllowDirty)) {
    Write-Warning "Working tree is dirty at git commit $currentCommit. Running with -AllowDirty enabled for local matrix execution."
}

$matrix = @(
    @{
        ID = "C0"
        Config = "configs/rmr_v3/c0_observer_old_direct.yaml"
        OutDir = "runs/sha_a/c0_observer_old_direct_seed42"
        Desc = "Old Observer (Additive + Flat-DM16) + Solver OFF"
    },
    @{
        ID = "C1"
        Config = "configs/rmr_v3/c1_observer_old_rwsirt.yaml"
        OutDir = "runs/sha_a/c1_observer_old_rwsirt_seed42"
        Desc = "Old Observer (Additive + Flat-DM16) + Solver ON"
    },
    @{
        ID = "C2"
        Config = "configs/rmr_v3/c2_observer_new_direct.yaml"
        OutDir = "runs/sha_a/c2_observer_new_direct_seed42"
        Desc = "New Observer (RepWeighted + Bayesian Loss) + Solver OFF"
    },
    @{
        ID = "C3"
        Config = "configs/rmr_v3/c3_observer_new_rwsirt.yaml"
        OutDir = "runs/sha_a/c3_observer_new_rwsirt_seed42"
        Desc = "New Observer (RepWeighted + Bayesian Loss) + Solver ON"
    }
)

$targetRuns = @()
if ($Run -eq "all") {
    $targetRuns = $matrix
} else {
    $targetRuns = $matrix | Where-Object { $_.ID -eq $Run.ToUpper() }
    if ($targetRuns.Count -eq 0) {
        Write-Error "Unknown Run ID '$Run'. Valid choices: C0, C1, C2, C3, all."
        exit 1
    }
}

Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host "  RMR-V3/V4 OBSERVER-SOLVER 2x2 FACTORIAL MATRIX RUNNER" -ForegroundColor Cyan
Write-Host "  Targets: $($targetRuns.ID -join ', ')" -ForegroundColor Cyan
Write-Host "  Git HEAD: $currentCommit" -ForegroundColor Cyan
Write-Host "================================================================================`n" -ForegroundColor Cyan

foreach ($item in $targetRuns) {
    $id = $item.ID
    $cfg = $item.Config
    $outDir = $item.OutDir
    $desc = $item.Desc
    $bestCkpt = "$outDir/best_val_mae.pt"
    $lastCkpt = "$outDir/last.pt"
    $evalDir = "$outDir/eval_test"

    Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Yellow
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Starting Run $id : $desc" -ForegroundColor Yellow
    Write-Host "  Config: $cfg | OutDir: $outDir" -ForegroundColor Yellow
    Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Yellow

    if (-not $Fresh -and (Test-Path $lastCkpt)) {
        Write-Host "  [RESUME] Found existing checkpoint at $lastCkpt. Resuming..." -ForegroundColor Green
        $trainArgs = @("-m", "rmr_v3.train", "--config", $cfg, "--resume", $lastCkpt, "--allow-cross-commit-resume")
    } else {
        $trainArgs = @("-m", "rmr_v3.train", "--config", $cfg, "--overwrite")
    }

    & $pythonExe $trainArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Training failed for Run $id with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }

    # Evaluation
    Write-Host "  [EVAL] Evaluating on ShanghaiTech Part A Test Set (182 images)..." -ForegroundColor Cyan
    & $pythonExe -m rmr_v3.eval --checkpoint $bestCkpt --manifest "data/sha_a_test.jsonl" --output-dir $evalDir
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Evaluation failed for Run $id with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }

    Write-Host "  [DONE] Run $id completed successfully!`n" -ForegroundColor Green
}

Write-Host "================================================================================" -ForegroundColor Green
Write-Host "  ALL SELECTED MATRIX RUNS COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "================================================================================`n" -ForegroundColor Green
