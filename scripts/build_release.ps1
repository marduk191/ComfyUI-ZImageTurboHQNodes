param(
  [string]$Version = "0.1.0"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$releaseDir = Join-Path $root "release"
if (Test-Path $releaseDir) { Remove-Item -Recurse -Force $releaseDir }

New-Item -ItemType Directory -Force (Join-Path $releaseDir "ComfyUI-ZImageTurboHQNodes") | Out-Null
Copy-Item -Force (Join-Path $root "__init__.py"), (Join-Path $root "zimage_hq_nodes.py"), (Join-Path $root "pyproject.toml"), (Join-Path $root "README.md"), (Join-Path $root "LICENSE"), (Join-Path $root "CHANGELOG.md"), (Join-Path $root "THIRD_PARTY_NOTICES.md") (Join-Path $releaseDir "ComfyUI-ZImageTurboHQNodes")
New-Item -ItemType Directory -Force (Join-Path $releaseDir "ComfyUI-ZImageTurboHQNodes\workflows") | Out-Null
Copy-Item -Force (Join-Path $root "workflows\*.json") (Join-Path $releaseDir "ComfyUI-ZImageTurboHQNodes\workflows")

$zipPath = Join-Path $root "ComfyUI-ZImageTurboHQNodes-v$Version.zip"
if (Test-Path $zipPath) { Remove-Item -Force $zipPath }
Compress-Archive -Path (Join-Path $releaseDir "ComfyUI-ZImageTurboHQNodes") -DestinationPath $zipPath
Write-Host "Created: $zipPath"
