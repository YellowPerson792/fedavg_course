param(
  [string]$Config = "configs\pi_server.yaml"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $root "src"
& "D:\MLLMs\.venv\Scripts\python.exe" -m fedavg.server --config (Join-Path $root $Config)
