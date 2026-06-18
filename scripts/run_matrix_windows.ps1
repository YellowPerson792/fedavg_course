param(
  [switch]$IncludeCifar
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $root "src"
$argsList = @("-m", "fedavg.experiment_matrix", "--base-config", (Join-Path $root "configs\mnist_iid_b16_e1.yaml"), "--clients", "2")
if ($IncludeCifar) {
  $argsList += "--include-cifar"
}
& "D:\MLLMs\.venv\Scripts\python.exe" @argsList
