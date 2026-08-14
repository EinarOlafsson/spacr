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
python packaging\i18n\render.py
if ($LASTEXITCODE -ne 0) {
    throw "Installer locale generation failed with exit code $LASTEXITCODE"
}
# Windows PowerShell 5.1 interprets scripts without a byte-order mark using
# the active ANSI code page. Preserve every translated message by making the
# generated catalog's UTF-8 encoding explicit before parsing or bundling it.
$CatalogPath = (Resolve-Path "packaging\online\generated\installer_messages.ps1").Path
$CatalogText = [System.IO.File]::ReadAllText(
    $CatalogPath,
    [System.Text.Encoding]::UTF8
)
[System.IO.File]::WriteAllText(
    $CatalogPath,
    $CatalogText,
    (New-Object System.Text.UTF8Encoding($true))
)

Push-Location "packaging\online"
try {
    & $MakeNsis "/DVERSION=$Version" "spacr_online_installer.nsi"
    if ($LASTEXITCODE -ne 0) {
        throw "makensis failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

Write-Host "Built dist\online\spaCR-$Version-Windows-Online-Setup.exe" -ForegroundColor Green
