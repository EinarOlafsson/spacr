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

$Version = (
    python -c 'import ast,pathlib; t=ast.parse(pathlib.Path("setup.py").read_text()); print(next(ast.literal_eval(n.value) for n in t.body if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "VERSION" for x in n.targets)))'
).Trim()
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
