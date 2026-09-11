# ==============================================================================
# RMR-v8 Comprehensive Multi-Model Experiment Suite for Windows (PowerShell)
# ==============================================================================
# Models in the suite:
#   1. rmr_v8_canonical: Flagship Stage 2 (T=2, Mult-SIRT, Charbonnier TV, L1 count, Mass-weighted cell)
#   2. rmr_v8_t6_tv: Deep regularized solver (T=6, Mult-SIRT, Charbonnier TV, L1 count)
#   3. rmr_v8_no_mult_sirt: Ablation of Stage 2a (Additive SIRT)
#   4. rmr_v8_no_charbonnier: Ablation of Stage 2b (Isotropic Laplacian TV)
#   5. rmr_v8_no_l1_loss: Ablation of Stage 2c (NB count loss)
#   6. rmr_v8_no_mass_weight: Ablation of Stage 2d (Balanced cell loss)
#   7. rmr_v8_coord_attn: Stage 3 (+CoordinateAttention on P4, 784 params, GroupNorm)
#   8. rmr_v8_with_kd: Stage 3 (Knowledge Distillation)
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
Write-Host "  RMR-v8 MULTI-MODEL EXPERIMENT SUITE (Windows PowerShell)" -ForegroundColor Cyan
Write-Host "  Python: $PythonExe" -ForegroundColor Cyan
Write-Host "  Mode:   $Mode" -ForegroundColor Cyan
Write-Host "  Date:   $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

$LogDir = "runs\sha_a\suite_logs_v8"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

$RunConfigs = [ordered]@{
    "rmr_v8_canonical"      = "configs\rmr_v8\rmr_v8_canonical.yaml"
    "rmr_v8_t6_tv"          = "configs\rmr_v8\rmr_v8_t6_tv.yaml"
    "rmr_v8_no_mult_sirt"   = "configs\rmr_v8\rmr_v8_no_mult_sirt.yaml"
    "rmr_v8_no_charbonnier" = "configs\rmr_v8\rmr_v8_no_charbonnier.yaml"
    "rmr_v8_no_l1_loss"     = "configs\rmr_v8\rmr_v8_no_l1_loss.yaml"
    "rmr_v8_no_mass_weight" = "configs\rmr_v8\rmr_v8_no_mass_weight.yaml"
    "rmr_v8_coord_attn"     = "configs\rmr_v8\rmr_v8_coord_attn.yaml"
    "rmr_v8_with_kd"        = "configs\rmr_v8\rmr_v8_with_kd.yaml"
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

Write-Host "`nAll RMR-v8 matrix runs completed." -ForegroundColor Green
