param(
  [string[]]$Hosts = @("RaspberryPi_2", "RaspberryPi_3"),
  [string]$RemoteDir = "~/fedavg_course",
  [switch]$DryRun
)
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$items = @(
  "src",
  "configs",
  "scripts",
  "tests",
  "pyproject.toml",
  "requirements-pi.txt",
  "README.md"
)
$sshOptions = @("-o", "ClearAllForwardings=yes")

if ([string]::IsNullOrWhiteSpace($RemoteDir) -or $RemoteDir -in @("/", "~", ".")) {
  throw "RemoteDir looks unsafe: '$RemoteDir'"
}

function Format-Command {
  param([string]$Exe, [string[]]$Arguments)
  $quoted = $Arguments | ForEach-Object {
    if ($_ -match '\s') { "'$_'" } else { $_ }
  }
  return "$Exe $($quoted -join ' ')"
}

function Invoke-LoggedCommand {
  param([string]$Exe, [string[]]$Arguments)
  Write-Host (Format-Command $Exe $Arguments)
  if ($DryRun) {
    return
  }
  & $Exe @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "$Exe failed with exit code $LASTEXITCODE"
  }
}

foreach ($hostName in $Hosts) {
  Write-Host "`n==> Syncing code to ${hostName}:${RemoteDir}"
  Invoke-LoggedCommand "ssh" ($sshOptions + @($hostName, "mkdir -p $RemoteDir"))
  foreach ($item in $items) {
    $localPath = Join-Path $root $item
    if (-not (Test-Path -LiteralPath $localPath)) {
      throw "Missing local sync item: $localPath"
    }
    Invoke-LoggedCommand "ssh" ($sshOptions + @($hostName, "rm -rf $RemoteDir/$item"))
    Invoke-LoggedCommand "scp" ($sshOptions + @("-r", $localPath, "${hostName}:${RemoteDir}/"))
  }
}
