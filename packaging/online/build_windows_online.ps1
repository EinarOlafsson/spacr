[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
if ($env:OS -ne "Windows_NT") {
    throw "The Windows online installer must be built on Windows."
}
if (-not (Get-Command makensis.exe -ErrorAction SilentlyContinue)) {
    throw "NSIS is required. Install it with: choco install nsis"
}

$Version = (python setup.py --version).Trim()
New-Item -ItemType Directory -Force -Path "dist\online" | Out-Null

Push-Location "packaging\online"
try {
    makensis.exe "/DVERSION=$Version" "spacr_online_installer.nsi"
    if ($LASTEXITCODE -ne 0) {
        throw "makensis failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

Write-Host "Built dist\online\SpaCR-$Version-Windows-Online-Setup.exe" -ForegroundColor Green
