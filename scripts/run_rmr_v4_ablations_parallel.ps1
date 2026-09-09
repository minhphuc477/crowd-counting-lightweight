# ==============================================================================
# RMR-v4 PARALLEL ABLATION EXPERIMENT RUNNER (Windows PowerShell)
# ==============================================================================

$pythonExe = ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

$logDir = "runs\sha_a\parallel_logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

$models = @(
    @{ Tag = "v4_s";  Name = "V4-S (Mean/Std Stats)";       Config = "configs/rmr_v4/mean_std_regions.yaml"; OutDir = "runs/sha_a/rmr_v4_mean_std_seed42" },
    @{ Tag = "v4_n";  Name = "V4-N (Native Pooling)";       Config = "configs/rmr_v4/native_pooling.yaml";   OutDir = "runs/sha_a/rmr_v4_native_pooling_seed42" },
    @{ Tag = "v4_ns"; Name = "V4-NS (Native + Mean/Std)";   Config = "configs/rmr_v4/native_meanstd.yaml";   OutDir = "runs/sha_a/rmr_v4_native_meanstd_seed42" },
    @{ Tag = "v4_dm"; Name = "V4-DM (Multi-Scale DM)";      Config = "configs/rmr_v4/multiscale_dm.yaml";    OutDir = "runs/sha_a/rmr_v4_multiscale_dm_seed42" }
)

Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host "  RMR-V4 DECISIVE 4-MODEL ABLATION SUITE (PARALLEL MODE)" -ForegroundColor Cyan
Write-Host "================================================================================`n" -ForegroundColor Cyan

$jobs = @()
foreach ($m in $models) {
    $logFile = "$logDir\$($m.Tag).log"
    $lastCkpt = "$($m.OutDir)/last.pt"
    $extraArgs = if (Test-Path $lastCkpt) { @("--resume", $lastCkpt, "--allow-cross-commit-resume") } else { @("--overwrite") }
    $allArgs = @("-m", "rmr_v3.train", "--config", $m.Config) + $extraArgs

    Write-Host "  [LAUNCH] $($m.Name) -> Logging to $logFile" -ForegroundColor Yellow
    $job = Start-Process -FilePath $pythonExe -ArgumentList $allArgs -RedirectStandardOutput $logFile -RedirectStandardError $logFile -PassThru
    $jobs += @{ Process = $job; Model = $m }
}

Write-Host "`nAll 4 processes launched in background. Waiting for completion..." -ForegroundColor Green
foreach ($j in $jobs) {
    $j.Process.WaitForExit()
    Write-Host "  Process $($j.Model.Name) exited with code $($j.Process.ExitCode)" -ForegroundColor Green
}

Write-Host "`nEvaluating best checkpoints on ShanghaiTech Part A Test Set..." -ForegroundColor Cyan
foreach ($m in $models) {
    $bestCkpt = "$($m.OutDir)/best_val_mae.pt"
    if (Test-Path $bestCkpt) {
        & $pythonExe -m rmr_v3.eval --checkpoint $bestCkpt --manifest "data/sha_a_test.jsonl" --output-dir "$($m.OutDir)/eval_test"
    }
}

Write-Host "`n================================================================================" -ForegroundColor Green
Write-Host "  ALL 4 RMR-V4 ABLATIONS COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "================================================================================`n" -ForegroundColor Green
