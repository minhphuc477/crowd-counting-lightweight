param (
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
    Write-Warning "Working tree is dirty at git commit $currentCommit. Running with -AllowDirty for local RMR-v5 execution."
}

$cfg = "configs/rmr_v5/rmr_v5_canonical.yaml"
$outDir = "runs/sha_a/rmr_v5_canonical_seed42"
$bestCkpt = "$outDir/best_val_mae.pt"
$lastCkpt = "$outDir/last.pt"
$evalDir = "$outDir/eval_test"

Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host "  RMR-V5 CANONICAL BENCHMARK RUNNER" -ForegroundColor Cyan
Write-Host "  Architecture: RepWeightedFPN + Native Pyramid Pooling + Moment-2 Stats + RW-SIRT" -ForegroundColor Cyan
Write-Host "  Deploy Params: 103,785 (< 105K Budget)" -ForegroundColor Cyan
Write-Host "  Git HEAD: $currentCommit" -ForegroundColor Cyan
Write-Host "================================================================================`n" -ForegroundColor Cyan

# 1. Training Step
Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Yellow
Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Starting Training: RMR-v5 Canonical" -ForegroundColor Yellow
Write-Host "  Config: $cfg | OutDir: $outDir" -ForegroundColor Yellow
Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Yellow

if (-not $Fresh -and (Test-Path $lastCkpt)) {
    Write-Host "  [RESUME] Found valid resume checkpoint at $lastCkpt. Resuming..." -ForegroundColor Green
    $trainArgs = @("-m", "rmr_v3.train", "--config", $cfg, "--resume", $lastCkpt, "--allow-cross-commit-resume")
} else {
    $trainArgs = @("-m", "rmr_v3.train", "--config", $cfg, "--overwrite")
}

Write-Host "  [TRAIN] Executing: $pythonExe $($trainArgs -join ' ')"
& $pythonExe $trainArgs
if ($LASTEXITCODE -ne 0) {
    Write-Error "Training failed for RMR-v5 with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

# 2. Evaluation Step
Write-Host "`n--------------------------------------------------------------------------------" -ForegroundColor Yellow
Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Evaluating RMR-v5 on ShanghaiTech Part A Test Set (182 images)..." -ForegroundColor Yellow
Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Yellow

if (-not (Test-Path $bestCkpt)) {
    Write-Error "Best checkpoint not found at $bestCkpt"
    exit 1
}

& $pythonExe -m rmr_v3.eval --checkpoint $bestCkpt --manifest "data/sha_a_test.jsonl" --output-dir $evalDir
if ($LASTEXITCODE -ne 0) {
    Write-Error "Evaluation failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

# 3. Paired Statistical Comparisons
$predV5 = "$evalDir/predictions.csv"
$predV3B = "runs/sha_a/historical_commit_91c0b841/rmr_v3_rw_seed42/eval_test/predictions.csv"
$predC1 = "runs/sha_a/c1_observer_old_rwsirt_seed42/eval_test/predictions.csv"
$predV4N = "runs/sha_a/rmr_v4_native_pooling_seed42/eval_test/predictions.csv"
$predV4S = "runs/sha_a/rmr_v4_mean_std_seed42/eval_test/predictions.csv"

if (Test-Path $predV3B) {
    Write-Host "`n  [COMPARE] RMR-v5 vs V3-B (Historical Record 83.22)..." -ForegroundColor Cyan
    & $pythonExe -m rmr_count.aggregate --compare $predV3B $predV5 --name-a "V3-B_Record" --name-b "RMR_v5" --pred-col pred --output "runs/sha_a/comparison_v3b_vs_rmr_v5.json"
}

if (Test-Path $predC1) {
    Write-Host "`n  [COMPARE] RMR-v5 vs C1 (Observer Old + Solver 83.80)..." -ForegroundColor Cyan
    & $pythonExe -m rmr_count.aggregate --compare $predC1 $predV5 --name-a "C1_RWSIRT" --name-b "RMR_v5" --pred-col pred --output "runs/sha_a/comparison_c1_vs_rmr_v5.json"
}

if (Test-Path $predV4N) {
    Write-Host "`n  [COMPARE] RMR-v5 vs V4-N (Native Pooling 84.67)..." -ForegroundColor Cyan
    & $pythonExe -m rmr_count.aggregate --compare $predV4N $predV5 --name-a "V4_Native" --name-b "RMR_v5" --pred-col pred --output "runs/sha_a/comparison_v4n_vs_rmr_v5.json"
}

if (Test-Path $predV4S) {
    Write-Host "`n  [COMPARE] RMR-v5 vs V4-S (Mean+Std 88.76, RMSE 139.11)..." -ForegroundColor Cyan
    & $pythonExe -m rmr_count.aggregate --compare $predV4S $predV5 --name-a "V4_MeanStd" --name-b "RMR_v5" --pred-col pred --output "runs/sha_a/comparison_v4s_vs_rmr_v5.json"
}

Write-Host "`n================================================================================" -ForegroundColor Green
Write-Host "  RMR-V5 CANONICAL BENCHMARK EXPERIMENT COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "================================================================================`n" -ForegroundColor Green
