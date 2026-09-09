# ==============================================================================
# download_shb.ps1: ShanghaiTech Part B Dataset Verification & Setup (PowerShell)
# ==============================================================================
$ErrorActionPreference = "Stop"

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$ROOT_DIR = Split-Path -Parent $SCRIPT_DIR
Set-Location $ROOT_DIR

$DATA_DIR = Join-Path $ROOT_DIR "data"
$SHB_DIR = Join-Path $DATA_DIR "part_B_final"
$TRAIN_IMG = Join-Path $SHB_DIR "train_data/images"
$TRAIN_GT = Join-Path $SHB_DIR "train_data/ground_truth"
$TEST_IMG = Join-Path $SHB_DIR "test_data/images"
$TEST_GT = Join-Path $SHB_DIR "test_data/ground_truth"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " ShanghaiTech Part B (SHB) Verification & Setup" -ForegroundColor Cyan
Write-Host " Working directory: $ROOT_DIR" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

function Test-SHBComplete {
    if ((Test-Path $TRAIN_IMG) -and (Test-Path $TRAIN_GT) -and (Test-Path $TEST_IMG) -and (Test-Path $TEST_GT)) {
        $nTrainImg = (Get-ChildItem -Path $TRAIN_IMG -Filter "*.jpg" | Measure-Object).Count
        $nTrainGt  = (Get-ChildItem -Path $TRAIN_GT  -Filter "*.mat" | Measure-Object).Count
        $nTestImg  = (Get-ChildItem -Path $TEST_IMG  -Filter "*.jpg" | Measure-Object).Count
        $nTestGt   = (Get-ChildItem -Path $TEST_GT   -Filter "*.mat" | Measure-Object).Count
        if ($nTrainImg -eq 400 -and $nTrainGt -eq 400 -and $nTestImg -eq 316 -and $nTestGt -eq 316) {
            return $true
        }
    }
    return $false
}

if (Test-SHBComplete) {
    Write-Host "[OK] ShanghaiTech Part B is already present and complete in data/part_B_final/:" -ForegroundColor Green
    Write-Host "     - Train images: 400 (.jpg), Ground Truth: 400 (.mat)"
    Write-Host "     - Test images:  316 (.jpg), Ground Truth: 316 (.mat)"
} else {
    Write-Host "[INFO] data/part_B_final is missing or incomplete. Checking Git..." -ForegroundColor Yellow
    git checkout HEAD -- data/part_B_final
    if (-not (Test-SHBComplete)) {
        Write-Host "[INFO] Pulling from origin/RMR..." -ForegroundColor Yellow
        git fetch origin RMR
        git checkout origin/RMR -- data/part_B_final
    }
}

$PY = if (Test-Path ".venv/Scripts/python.exe") { ".venv/Scripts/python.exe" } else { "python" }

$TRAIN_MANIFEST = Join-Path $DATA_DIR "shb_train_all.jsonl"
$TEST_MANIFEST = Join-Path $DATA_DIR "shb_test.jsonl"

if (-not (Test-Path $TRAIN_MANIFEST)) {
    Write-Host "[INFO] Generating data/shb_train_all.jsonl..." -ForegroundColor Cyan
    & $PY -m rmr_count.prepare_manifest --images $TRAIN_IMG --annotations $TRAIN_GT --dataset sha_b --out $TRAIN_MANIFEST
}

if (-not (Test-Path $TEST_MANIFEST)) {
    Write-Host "[INFO] Generating data/shb_test.jsonl..." -ForegroundColor Cyan
    & $PY -m rmr_count.prepare_manifest --images $TEST_IMG --annotations $TEST_GT --dataset sha_b --out $TEST_MANIFEST
}

$nTrainLines = (Get-Content $TRAIN_MANIFEST | Measure-Object -Line).Lines
$nTestLines = (Get-Content $TEST_MANIFEST | Measure-Object -Line).Lines
Write-Host "[DONE] SHB manifests verified: $nTrainLines train samples, $nTestLines test samples." -ForegroundColor Green
