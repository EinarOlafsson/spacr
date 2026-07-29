[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
if ($env:OS -ne "Windows_NT") {
    throw "The Windows online installer must be built on Windows."
}
$MakeNsis = Get-Command makensis.exe -ErrorAction SilentlyContinue
if (-not $MakeNsis) {
    $NsisCandidates = @(
        (Join-Path ${env:ProgramFiles(x86)} "NSIS\makensis.exe"),
        (Join-Path $env:ProgramFiles "NSIS\makensis.exe"),
        (Join-Path $env:ChocolateyInstall "bin\makensis.exe")
    )
    $MakeNsis = $NsisCandidates |
        Where-Object { $_ -and (Test-Path $_) } |
        Select-Object -First 1
}
if (-not $MakeNsis) {
    throw "NSIS is required. Install it with: choco install nsis"
}

$VersionMatch = Select-String -Path "setup.py" -Pattern '^VERSION\s*=\s*["'']([^"'']+)'
if (-not $VersionMatch) {
    throw "Could not read VERSION from setup.py"
}
$Version = $VersionMatch.Matches[0].Groups[1].Value
New-Item -ItemType Directory -Force -Path "dist\online" | Out-Null

Push-Location "packaging\online"
try {
    & $MakeNsis "/DVERSION=$Version" "spacr_online_installer.nsi"
    if ($LASTEXITCODE -ne 0) {
        throw "makensis failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

Write-Host "Built dist\online\SpaCR-$Version-Windows-Online-Setup.exe" -ForegroundColor Green
