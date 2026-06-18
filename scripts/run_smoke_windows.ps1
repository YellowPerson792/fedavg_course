$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $root "src"
& "D:\MLLMs\.venv\Scripts\python.exe" -m fedavg.simulate --config (Join-Path $root "configs\smoke_mnist.yaml") --clients 2
