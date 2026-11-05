$ErrorActionPreference = 'Stop'
$appRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $appRoot
& (Join-Path $appRoot '.venv\Scripts\python.exe') (Join-Path $appRoot 'document_query_visuals.py')
