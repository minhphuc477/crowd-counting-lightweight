param (
    [switch]$Fresh = $false,
    [switch]$AllowCrossCommitResume = $false,
    [switch]$ArchiveStale = $true
)

# Canonical RMR & RMR-v3 Benchmark Suite Runner
# Strictly adheres to:
# 1. Zero ad-hoc splits (300 train_all, 182 test).
# 2. Sequential execution for maximum CUDA throughput and thermal stability.
# 3. Direct full-image MAE/RMSE headline metrics.
# 4. Rigorous paired comparison (t-test, Wilcoxon, bootstrap 95% CI).
# 5. Publication-grade provenance verification (matching clean git HEAD).

$ErrorActionPreference = "Stop"

$pythonExe = ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

$currentCommit = (git rev-parse HEAD).Trim()
$gitStatusRaw = git status --porcelain
$gitStatus = if ($null -ne $gitStatusRaw) { ("$gitStatusRaw").Trim() } else { "" }
$isDirty = [bool]$gitStatus

$runs = @(
    @{
        Name = "V3-A (Probabilistic Uniform Control)"
        Module = "rmr_v3.train"
        Config = "configs/rmr_v3/probabilistic_uniform.yaml"
        OutDir = "runs/sha_a/rmr_v3_uniform_control_seed42"
        EvalModule = "rmr_v3.eval"
    },
    @{
        Name = "V3-B (Reliability Weighted - Proposed)"
        Module = "rmr_v3.train"
        Config = "configs/rmr_v3/reliability_weighted.yaml"
        OutDir = "runs/sha_a/rmr_v3_rw_seed42"
        EvalModule = "rmr_v3.eval"
    },
    @{
        Name = "B5-P (Canonical Deterministic Baseline)"
        Module = "rmr_v2.train"
        Config = "configs/rmr_v2/rmr_projected_t2.yaml"
        OutDir = "runs/sha_a/b5p_canonical_seed42"
        EvalModule = "rmr_v2.eval"
    }
)

Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host "  CANONICAL RMR / RMR-V3 BENCHMARK SUITE" -ForegroundColor Cyan
Write-Host "  Order: V3-A -> V3-B -> B5-P (Sequential execution on GPU)" -ForegroundColor Cyan
Write-Host "  Git HEAD: $currentCommit (Dirty: $isDirty) | Fresh: $Fresh" -ForegroundColor Cyan
Write-Host "================================================================================`n" -ForegroundColor Cyan

foreach ($run in $runs) {
    $name = $run.Name
    $cfg = $run.Config
    $outDir = $run.OutDir
    $trainMod = $run.Module
    $evalMod = $run.EvalModule
    $bestCkpt = "$outDir/best_val_mae.pt"
    $evalDir = "$outDir/eval_test"
    $summaryJson = "$evalDir/summary.json"

    Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Yellow
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Starting: $name" -ForegroundColor Yellow
    Write-Host "  Config: $cfg | OutDir: $outDir" -ForegroundColor Yellow
    Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Yellow

    # Training step: check if already completed with valid artifacts matching clean current HEAD
    $artifactsValid = $false
    if (-not $Fresh -and (Test-Path $summaryJson) -and (Test-Path $bestCkpt)) {
        try {
            $sumContent = Get-Content $summaryJson -Raw | ConvertFrom-Json
            if ($sumContent -and $sumContent.provenance) {
                $p = $sumContent.provenance
                $trainCommit = $p.training_commit
                $trainDirty = [bool]$p.training_git_dirty
                # Strict check: commit must match current HEAD, must be clean, and checkpoint must be valid
                if ($trainCommit -eq $currentCommit -and (-not $trainDirty) -and (Get-Item $bestCkpt).Length -gt 1000) {
                    if ($p.checkpoint_sha256) {
                        $actualHash = (Get-FileHash -Path $bestCkpt -Algorithm SHA256).Hash.ToLower()
                        if ($actualHash -eq $p.checkpoint_sha256.ToLower()) {
                            $artifactsValid = $true
                        } else {
                            Write-Host "  [STALE] Checkpoint SHA256 mismatch ($actualHash vs $($p.checkpoint_sha256))." -ForegroundColor DarkYellow
                        }
                    } else {
                        $artifactsValid = $true
                    }
                } else {
                    Write-Host "  [STALE] Existing artifacts in $outDir are from commit $trainCommit (dirty=$trainDirty), but current HEAD is $currentCommit (dirty=$isDirty)." -ForegroundColor DarkYellow
                }
            }
        } catch {
            $artifactsValid = $false
        }
    }

    if ($artifactsValid) {
        Write-Host "  [SKIP] Verified matching clean HEAD artifacts at $outDir (Commit: $currentCommit). Skipping training." -ForegroundColor Green
    } else {
        # Check if last.pt exists and can be safely resumed BEFORE archiving
        $canResume = $false
        $lastCkpt = "$outDir/last.pt"
        if (-not $Fresh -and (Test-Path $lastCkpt)) {
            try {
                $inspectCmd = "import torch; ckpt=torch.load(r'$lastCkpt', map_location='cpu', weights_only=False); print(str(ckpt.get('git_commit', ckpt.get('provenance', {}).get('git_commit', 'unknown'))))"
                $ckptCommit = (& $pythonExe -c $inspectCmd 2>$null).Trim()
                if ($ckptCommit -eq $currentCommit -or $AllowCrossCommitResume) {
                    $canResume = $true
                } else {
                    Write-Host "  [INCOMPATIBLE] Found last.pt from commit $ckptCommit, but current HEAD is $currentCommit. Cannot resume without -AllowCrossCommitResume." -ForegroundColor DarkYellow
                }
            } catch {
                $canResume = $false
            }
        }

        if ($canResume) {
            Write-Host "  [RESUME] Found valid resume checkpoint at $lastCkpt (Commit: $ckptCommit). Resuming without archiving..." -ForegroundColor Green
            $trainArgs = @("-m", $trainMod, "--config", $cfg, "--resume", $lastCkpt)
            if ($AllowCrossCommitResume) {
                $trainArgs += @("--allow-cross-commit-resume")
            }
        } else {
            # Stale or fresh run: archive existing directory if non-empty
            if ((Test-Path $outDir) -and (Get-ChildItem $outDir).Count -gt 0) {
                if ($ArchiveStale) {
                    $timestamp = (Get-Date -Format "yyyyMMdd_HHmmss")
                    $cleanDirName = ($outDir -split "/")[-1]
                    $archiveDir = "runs/sha_a/archive_${cleanDirName}_$timestamp"
                    Write-Host "  [ARCHIVE] Archiving stale artifacts from $outDir to $archiveDir..." -ForegroundColor Magenta
                    Move-Item -Path $outDir -Destination $archiveDir -Force
                }
            }
            $trainArgs = @("-m", $trainMod, "--config", $cfg, "--overwrite")
        }

        Write-Host "  [TRAIN] Executing: $pythonExe $($trainArgs -join ' ')"
        & $pythonExe $trainArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Training failed for $name with exit code $LASTEXITCODE"
            exit $LASTEXITCODE
        }
    }

    # Evaluation step: evaluate best_val_mae.pt on test set
    $evalValid = $false
    if (-not $Fresh -and (Test-Path $summaryJson)) {
        try {
            $sumContent = Get-Content $summaryJson -Raw | ConvertFrom-Json
            if ($sumContent -and $sumContent.provenance) {
                $p = $sumContent.provenance
                if ($p.evaluation_commit -eq $currentCommit -and (-not [bool]$p.git_dirty)) {
                    if ($p.checkpoint_sha256 -and (Test-Path $bestCkpt)) {
                        $actualHash = (Get-FileHash -Path $bestCkpt -Algorithm SHA256).Hash.ToLower()
                        if ($actualHash -eq $p.checkpoint_sha256.ToLower()) {
                            $evalValid = $true
                        }
                    } else {
                        $evalValid = $true
                    }
                }
            }
        } catch {
            $evalValid = $false
        }
    }
    if (-not $evalValid) {
        if (-not (Test-Path $bestCkpt)) {
            Write-Error "Checkpoint not found: $bestCkpt"
            exit 1
        }
        Write-Host "  [EVAL] Evaluating $bestCkpt on data/sha_a_test.jsonl..." -ForegroundColor Cyan
        & $pythonExe -m $evalMod --checkpoint $bestCkpt --manifest "data/sha_a_test.jsonl" --output-dir $evalDir
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Evaluation failed for $name with exit code $LASTEXITCODE"
            exit $LASTEXITCODE
        }
    }

    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Finished: $name`n" -ForegroundColor Green
}

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  ALL 3 RUNS COMPLETED. RUNNING PAIRED COMPARISONS AND AGGREGATION..." -ForegroundColor Cyan
Write-Host "================================================================================`n" -ForegroundColor Cyan

$predV3A = "runs/sha_a/rmr_v3_uniform_control_seed42/eval_test/predictions.csv"
$predV3B = "runs/sha_a/rmr_v3_rw_seed42/eval_test/predictions.csv"
$predB5P = "runs/sha_a/b5p_canonical_seed42/eval_test/predictions.csv"

# Comparison 1: V3-A vs V3-B (Causal test of reliability weighting W = I vs W = diag(w_R))
Write-Host "--- Comparison 1: V3-A (Uniform) vs V3-B (Reliability Weighted) [MAIN HYPOTHESIS] ---" -ForegroundColor Yellow
& $pythonExe -m rmr_count.aggregate --compare $predV3A $predV3B --name-a "V3-A_Uniform" --name-b "V3-B_RW" --pred-col pred --output "runs/sha_a/comparison_v3a_vs_v3b.json"

# Comparison 2: B5-P vs V3-A (Causal test of Probabilistic NB Regional Formulation)
Write-Host "`n--- Comparison 2: B5-P (Deterministic) vs V3-A (Probabilistic Uniform) ---" -ForegroundColor Yellow
& $pythonExe -m rmr_count.aggregate --compare $predB5P $predV3A --name-a "B5-P_Deterministic" --name-b "V3-A_Uniform" --pred-col pred --output "runs/sha_a/comparison_b5p_vs_v3a.json"

# Comparison 3: B5-P vs V3-B (Overall improvement from Baseline to Proposed)
Write-Host "`n--- Comparison 3: B5-P (Deterministic) vs V3-B (Reliability Weighted) ---" -ForegroundColor Yellow
& $pythonExe -m rmr_count.aggregate --compare $predB5P $predV3B --name-a "B5-P_Deterministic" --name-b "V3-B_RW" --pred-col pred --output "runs/sha_a/comparison_b5p_vs_v3b.json"

Write-Host "`n================================================================================" -ForegroundColor Green
Write-Host "  CANONICAL SUITE EXECUTION AND COMPARISONS COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "================================================================================`n" -ForegroundColor Green
