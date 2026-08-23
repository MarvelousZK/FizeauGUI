$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:QT_QPA_PLATFORM = "offscreen"

$projectRoot = Split-Path -Parent $PSScriptRoot
$specFile = Join-Path $PSScriptRoot "FizeauInterferometer.spec"

Push-Location $projectRoot
try {
    Write-Host "[1/3] Installing build dependencies..."
    python -m pip install --disable-pip-version-check -r requirements.txt

    Write-Host "[2/3] Running regression tests..."
    $tests = @(
        "tests/test_core.py",
        "tests/test_luo_adapt2.py",
        "tests/test_masked_shift.py",
        "tests/test_qg_verify.py",
        "tests/test_takeda_ft.py",
        "tests/test_unwrap.py",
        "tests/test_wft2.py",
        "tests/test_gui_smoke.py"
    )
    foreach ($test in $tests) {
        python $test
        if ($LASTEXITCODE -ne 0) {
            throw "Regression test failed: $test"
        }
    }

    Write-Host "[3/3] Building Windows executable..."
    python -m PyInstaller --noconfirm --clean $specFile
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller build failed."
    }

    $exePath = Join-Path $projectRoot "dist/菲索干涉仪数据处理系统.exe"
    Write-Host "Build complete: $exePath"
}
finally {
    Pop-Location
}
