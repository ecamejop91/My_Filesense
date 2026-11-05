# Installs updated requirements into .venv, measures the PDF pipeline changes,
# and runs the test suite. Logs stay in the current account's private storage.
# Run from PowerShell:  powershell -ExecutionPolicy Bypass -File .\run_checks.ps1
$ErrorActionPreference = 'Continue'
$appRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $appRoot
$python = Join-Path $appRoot '.venv\Scripts\python.exe'
$log = & $python -c "from app_storage import prepare_storage; print(prepare_storage().logs / 'run_checks.log')"
if ($LASTEXITCODE -ne 0) { throw 'Could not create private log storage.' }
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:QT_QPA_PLATFORM = 'offscreen'
"FileSense checks - $(Get-Date -Format s)" | Set-Content -Path $log -Encoding utf8

function Run-Step([string]$title, [string[]]$arguments) {
    $header = "`n===== $title ====="
    Write-Host $header -ForegroundColor Cyan
    Add-Content -Path $log -Value $header -Encoding utf8
    $out = Join-Path (Split-Path -Parent $log) ([guid]::NewGuid().ToString() + '.out')
    $err = Join-Path (Split-Path -Parent $log) ([guid]::NewGuid().ToString() + '.err')
    $started = Get-Date
    $process = Start-Process -FilePath $python -ArgumentList $arguments -WindowStyle Hidden -Wait -PassThru `
        -RedirectStandardOutput $out -RedirectStandardError $err
    $text = (Get-Content $out -Raw -Encoding utf8) + (Get-Content $err -Raw -Encoding utf8)
    $footer = "[exit code $($process.ExitCode); $([int]((Get-Date) - $started).TotalSeconds) s]"
    Write-Host $text
    Write-Host $footer
    Add-Content -Path $log -Value $text -Encoding utf8
    Add-Content -Path $log -Value $footer -Encoding utf8
    Remove-Item -LiteralPath $out, $err -ErrorAction SilentlyContinue
}

Run-Step 'Install requirements' @('-m', 'pip', 'install', '-r', 'requirements.txt')
Run-Step 'Speed checks' @('tools\check_speedups.py')
Run-Step 'Test suite' @('-m', 'unittest', 'discover', '-s', 'tests', '-v')

Write-Host "`nDone. Full output saved to $log" -ForegroundColor Green
