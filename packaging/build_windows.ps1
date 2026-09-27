# Build a portable onedir archive and, when NSIS is available, a native installer.
param(
    [switch]$SkipDependencyInstall,
    [switch]$RequireInstaller
)
$ErrorActionPreference = "Stop"
if ($env:OS -ne "Windows_NT") { throw "Run this builder on Windows." }

function Invoke-CheckedNative {
    param([string]$File, [string[]]$Arguments)
    & $File @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$File failed with exit code $LASTEXITCODE"
    }
}

$Root = Split-Path $PSScriptRoot -Parent
Push-Location $Root
try {
    $Match = Select-String -Path "setup.py" -Pattern '^VERSION\s*=\s*["'']([^"'']+)'
    if (-not $Match) { throw "setup.py has no VERSION assignment" }
    $Version = $Match.Matches[0].Groups[1].Value
    foreach ($Directory in @("build", "dist")) {
        if (Test-Path $Directory) { Remove-Item -Recurse -Force $Directory }
    }
    if (-not $SkipDependencyInstall) {
        Invoke-CheckedNative "python" @("-m", "pip", "install", "--upgrade", "pip")
        Invoke-CheckedNative "python" @("-m", "pip", "install", "pyinstaller>=6.10,<7")
        Invoke-CheckedNative "python" @("-m", "pip", "install", ".")
    }
    Invoke-CheckedNative "python" @("-m", "PyInstaller", "--noconfirm", "--clean", "packaging\spacr.spec")
    $Source = Join-Path $Root "dist\spacr"
    if (-not (Test-Path (Join-Path $Source "spacr.exe"))) {
        throw "PyInstaller did not produce the onedir executable"
    }
    Compress-Archive -Path "$Source\*" -DestinationPath "dist\spaCR-$Version-windows.zip" -Force
    $Nsis = Get-Command makensis.exe -ErrorAction SilentlyContinue
    $NsisPath = if ($Nsis) { $Nsis.Source } else { $null }
    if (-not $Nsis) {
        $Candidate = Join-Path ${env:ProgramFiles(x86)} "NSIS\makensis.exe"
        if (Test-Path $Candidate) { $NsisPath = $Candidate }
    }
    if (-not $NsisPath) {
        if ($RequireInstaller) { throw "NSIS is required for installer acceptance" }
        Write-Warning "NSIS unavailable: only the portable onedir archive was built"
        return
    }
    $Installer = Join-Path $Root "dist\spaCR-$Version-setup.exe"
    $DeleteFiles = @(Get-ChildItem $Source -File -Recurse | ForEach-Object {
        $Relative = $_.FullName.Substring($Source.Length).TrimStart([char]'\')
        '  Delete "$INSTDIR\' + $Relative.Replace('$', '$$') + '"'
    }) -join "`n"
    $DeleteDirectories = @(Get-ChildItem $Source -Directory -Recurse |
        Sort-Object { $_.FullName.Length } -Descending | ForEach-Object {
        $Relative = $_.FullName.Substring($Source.Length).TrimStart([char]'\')
        '  RMDir "$INSTDIR\' + $Relative.Replace('$', '$$') + '"'
    }) -join "`n"
    $Template = @'
!include "MUI2.nsh"
Name "spaCR"
OutFile "@INSTALLER@"
InstallDir "$PROGRAMFILES64\spaCR"
RequestExecutionLevel admin
Page directory
Page instfiles
UninstPage instfiles
Section
  SetShellVarContext all
  SetOutPath "$INSTDIR"
  File /r "@SOURCE@\*"
  WriteUninstaller "$INSTDIR\Uninstall.exe"
  CreateShortcut "$SMPROGRAMS\spaCR.lnk" "$INSTDIR\spacr.exe"
SectionEnd
Section "Uninstall"
  SetShellVarContext all
  Delete "$SMPROGRAMS\spaCR.lnk"
@DELETE_FILES@
@DELETE_DIRECTORIES@
  Delete "$INSTDIR\Uninstall.exe"
  RMDir "$INSTDIR"
SectionEnd
'@
    $Script = Join-Path $Root "build\spacr_installer.nsi"
    $Template.Replace("@INSTALLER@", $Installer.Replace('$', '$$')).Replace("@SOURCE@", $Source.Replace('$', '$$')).Replace("@DELETE_FILES@", $DeleteFiles).Replace("@DELETE_DIRECTORIES@", $DeleteDirectories) |
        Set-Content -Encoding UTF8 $Script
    Invoke-CheckedNative $NsisPath @($Script)
    if (-not (Test-Path $Installer)) { throw "NSIS returned without an installer" }
    Write-Host "Built $Installer and the portable onedir archive"
} finally {
    Pop-Location
}
