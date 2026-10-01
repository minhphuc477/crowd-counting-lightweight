# PowerShell High-Throughput Parallel 4-Model Training Runner for Windows (16GB VRAM, 32GB RAM)
$ErrorActionPreference = "Continue"

$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"

$logDir = "runs/sha_a/parallel_windows_logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
}

$py = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  PARALLEL 4-MODEL EXPERIMENT RUNNER FOR WINDOWS (16GB GPU / 32GB RAM)" -ForegroundColor Cyan
Write-Host "  Optimizations: In-Memory Zero-Worker, Thread Capping, No-cuDNN-Benchmark" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

$jobs = @(
    @{ RunId = "win_job0_sub60_e5";        Config = "configs/rmr_research/sub60_e5_pure_flat_dm16_no_cell.yaml"; Affinity = 0x000F }, # Cores 0-3
    @{ RunId = "win_job1_asym_morozov";    Config = "configs/rmr_research/gen9/g9_asym_morozov_e5.yaml";         Affinity = 0x00F0 }, # Cores 4-7
    @{ RunId = "win_job2_dct_spectral";    Config = "configs/rmr_research/gen9/g9_dct_spectral_fixed.yaml";     Affinity = 0x0F00 }, # Cores 8-11
    @{ RunId = "win_job3_curvature_fixed"; Config = "configs/rmr_research/gen9/g9_fix_curvature_init.yaml";    Affinity = 0xF000 }  # Cores 12-15
)

$processes = @()

foreach ($j in $jobs) {
    $runId = $j.RunId
    $cfg = $j.Config
    $log = "$logDir\$runId.log"
    $affinity = $j.Affinity

    Write-Host ">>> Launching $runId (Log: $log)" -ForegroundColor Green
    
    $pinfo = New-Object System.Diagnostics.ProcessStartInfo
    $pinfo.FileName = (Resolve-Path $py).Path
    $pinfo.Arguments = "-m rmr_v3.train --config `"$cfg`" --run-id `"$runId`" --workers 0 --num-threads 2 --no-cudnn-benchmark --overwrite"
    $pinfo.RedirectStandardOutput = $true
    $pinfo.RedirectStandardError = $true
    $pinfo.UseShellExecute = $false
    $pinfo.CreateNoWindow = $true

    $p = New-Object System.Diagnostics.Process
    $p.StartInfo = $pinfo
    $p.Start() | Out-Null
    
    try {
        # Set CPU core affinity bitmask
        $p.ProcessorAffinity = [IntPtr]$affinity
    } catch {
        # Fallback if host has fewer logical cores
    }

    # Redirect output asynchronously to log file
    Register-ObjectEvent -InputObject $p -EventName OutputDataReceived -Action {
        if ($EventArgs.Data) { Add-Content -Path $using:log -Value $EventArgs.Data }
    } | Out-Null
    Register-ObjectEvent -InputObject $p -EventName ErrorDataReceived -Action {
        if ($EventArgs.Data) { Add-Content -Path $using:log -Value $EventArgs.Data }
    } | Out-Null
    $p.BeginOutputReadLine()
    $p.BeginErrorReadLine()

    $processes += $p
}

Write-Host "`nAll 4 processes launched! Monitoring completion..." -ForegroundColor Yellow

foreach ($p in $processes) {
    $p.WaitForExit()
}

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  ALL 4 PARALLEL TRAINING JOBS HAVE COMPLETED!" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
