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
    $ManifestBuilder = @'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve(strict=True)
manifest = root / "spacr-payload.json"
if manifest.exists() or manifest.is_symlink():
    raise ValueError("The payload ownership record already exists")
for required in ("spacr.exe", "_internal/spacr-update-helper.exe"):
    target = root / required
    if not target.is_file() or target.is_symlink():
        raise ValueError("The frozen executable and independent helper are required")
files = {}
for path in sorted(root.rglob("*")):
    if path.is_symlink():
        raise ValueError("A frozen payload cannot contain symbolic links")
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        files[path.relative_to(root).as_posix()] = digest.hexdigest()
    elif not path.is_dir():
        raise ValueError("A frozen payload contains a nonregular entry")
with manifest.open("x", encoding="utf-8") as stream:
    json.dump({"schema": 1, "version": sys.argv[2], "files": files},
              stream, indent=2, ensure_ascii=False)
    stream.write("\n")
'@
    $ManifestScript = Join-Path $Root "build\spacr_payload_manifest.py"
    [System.IO.File]::WriteAllText($ManifestScript, $ManifestBuilder,
        [System.Text.UTF8Encoding]::new($false))
    Invoke-CheckedNative "python" @("-I", $ManifestScript, $Source, $Version)
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
    $Updater = Join-Path $Root "dist\spaCR-$Version-update.exe"
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
!include "LogicLib.nsh"
!include "FileFunc.nsh"
Name "spaCR"
!ifdef SPACR_UPDATE_ONLY
OutFile "@UPDATER@"
SilentInstall silent
!else
OutFile "@INSTALLER@"
!endif
InstallDir "$PROGRAMFILES64\spaCR"
RequestExecutionLevel admin
Page directory
Page instfiles
UninstPage instfiles
Section
  SetShellVarContext all
  InitPluginsDir
  System::Call 'advapi32::ConvertStringSecurityDescriptorToSecurityDescriptorW(w "O:BAG:BAD:P(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)", i 1, *p .r5, p 0) i .r6'
  ${If} $6 == 0
    Goto frozen_install_failed
  ${EndIf}
  System::Call 'advapi32::SetFileSecurityW(w "$PLUGINSDIR", i 0x80000005, p r5) i .r6'
  System::Call 'kernel32::LocalFree(p r5)'
  ${If} $6 == 0
    Goto frozen_install_failed
  ${EndIf}
  ClearErrors
  SetOutPath "$PLUGINSDIR\spaCR-payload"
  File /r "@SOURCE@\*"
  IfErrors frozen_install_failed
  WriteUninstaller "$PLUGINSDIR\spaCR-payload\Uninstall.exe"
  IfErrors frozen_install_failed
  CopyFiles /SILENT "$PLUGINSDIR\spaCR-payload\_internal\spacr-update-helper.exe" "$PLUGINSDIR\spacr-update-helper.exe"
  IfErrors frozen_install_failed
  SetOutPath "$PLUGINSDIR"
  System::Call 'kernel32::SetEnvironmentVariableW(w "PYINSTALLER_RESET_ENVIRONMENT", w "1") i .r1'
  ${If} $1 == 0
    Goto frozen_install_failed
  ${EndIf}
!ifdef SPACR_UPDATE_ONLY
  ${GetParameters} $R0
  ClearErrors
  ${GetOptions} $R0 "/SPACRUPDATE=" $R1
  IfErrors frozen_install_failed
  ${GetOptions} $R0 "/SPACRPID=" $R2
  IfErrors frozen_install_failed
  ${If} $R1 == ""
  ${OrIf} $R2 == ""
    Goto frozen_install_failed
  ${EndIf}
  nsExec::ExecToLog '"$PLUGINSDIR\spacr-update-helper.exe" install-frozen-windows --source "$PLUGINSDIR\spaCR-payload" --destination "$INSTDIR" --version "@VERSION@" --phase-channel "$R1" --desktop-pid "$R2"'
!else
  nsExec::ExecToLog '"$PLUGINSDIR\spacr-update-helper.exe" install-frozen-windows --source "$PLUGINSDIR\spaCR-payload" --destination "$INSTDIR" --version "@VERSION@"'
!endif
  Pop $0
  ${If} $0 != 0
    Goto frozen_install_failed
  ${EndIf}
!ifndef SPACR_UPDATE_ONLY
  SetOutPath "$INSTDIR"
  CreateShortcut "$SMPROGRAMS\spaCR.lnk" "$INSTDIR\spacr.exe"
!endif
  Goto frozen_install_finished
frozen_install_failed:
  SetErrorLevel 1
  Abort
frozen_install_finished:
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
    $Template.Replace("@VERSION@", $Version).Replace("@INSTALLER@", $Installer.Replace('$', '$$')).Replace("@UPDATER@", $Updater.Replace('$', '$$')).Replace("@SOURCE@", $Source.Replace('$', '$$')).Replace("@DELETE_FILES@", $DeleteFiles).Replace("@DELETE_DIRECTORIES@", $DeleteDirectories) |
        Set-Content -Encoding UTF8 $Script
    Invoke-CheckedNative $NsisPath @($Script)
    if (-not (Test-Path $Installer)) { throw "NSIS returned without an installer" }
    Invoke-CheckedNative $NsisPath @("/DSPACR_UPDATE_ONLY", $Script)
    if (-not (Test-Path $Updater)) { throw "NSIS returned without the phased updater" }
    Write-Host "Built $Installer, $Updater and the portable onedir archive"
} finally {
    Pop-Location
}
