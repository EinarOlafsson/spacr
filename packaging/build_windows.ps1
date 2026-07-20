# build_windows.ps1 — produce dist/SpaCR-<version>.exe on Windows 10+
#
# Run from the spacr repo root in a "Developer PowerShell" (or plain PS)
# with a Python 3.10+ interpreter on PATH:
#
#     .\packaging\build_windows.ps1
#
# Prerequisites (checked before build):
#   * Windows 10 or newer (x64)
#   * python.exe on PATH, version >= 3.10
#   * spacr installed in a venv (or global)
#   * pyinstaller >= 6.0
#
# Output: dist/SpaCR-<version>.exe (a single-file windowed executable).

$ErrorActionPreference = "Stop"

if ($env:OS -ne "Windows_NT") {
    Write-Error "This script must run on Windows. For macOS use build_macos.sh; for Debian use build_debian.sh."
    exit 1
}

Write-Host "==> spacr Windows installer build" -ForegroundColor Cyan

# --- version ---
$version = (python -c "import re,pathlib; s=pathlib.Path('setup.py').read_text(); m=re.search(r'VERSION\s*=\s*[\"\']([^\"\']+)', s); print(m.group(1))").Trim()
Write-Host "    version: $version"

# --- clean previous build outputs so we don't ship stale binaries ---
if (Test-Path .\build) { Remove-Item -Recurse -Force .\build }
if (Test-Path .\dist)  { Remove-Item -Recurse -Force .\dist  }

# --- deps ---
Write-Host "==> installing build deps (pip)" -ForegroundColor Cyan
python -m pip install --upgrade pip
python -m pip install --upgrade pyinstaller
python -m pip install -e .

# --- run PyInstaller against the shared spec ---
Write-Host "==> running PyInstaller" -ForegroundColor Cyan
pyinstaller --noconfirm --clean packaging\spacr.spec

# --- rename the collected folder to a versioned single-file drop ---
$src = ".\dist\spacr"
if (-not (Test-Path $src)) { Write-Error "PyInstaller output not found at $src" }

# Compress the folder into a versioned zip (for direct download) AND
# also emit a single-file portable exe when possible.
$zip = ".\dist\SpaCR-$version-windows.zip"
Compress-Archive -Path "$src\*" -DestinationPath $zip -Force
Write-Host "==> wrote $zip" -ForegroundColor Green

# Optional NSIS installer: skip if makensis isn't present.
$nsis = Get-Command makensis -ErrorAction SilentlyContinue
if ($nsis) {
    Write-Host "==> building NSIS installer" -ForegroundColor Cyan
    $nsisScript = @"
!include "MUI2.nsh"
Name "SpaCR"
OutFile "dist\SpaCR-$version-setup.exe"
InstallDir "\$PROGRAMFILES64\SpaCR"
RequestExecutionLevel admin

Page directory
Page instfiles
UninstPage instfiles

Section
  SetOutPath "\$INSTDIR"
  File /r "dist\spacr\*"
  WriteUninstaller "\$INSTDIR\Uninstall.exe"
  CreateShortcut "\$SMPROGRAMS\SpaCR.lnk" "\$INSTDIR\spacr.exe"
SectionEnd

Section "Uninstall"
  Delete "\$SMPROGRAMS\SpaCR.lnk"
  RMDir /r "\$INSTDIR"
SectionEnd
"@
    $nsisScript | Set-Content -Encoding ASCII .\packaging\spacr_installer.nsi
    makensis .\packaging\spacr_installer.nsi
    Write-Host "==> wrote dist\SpaCR-$version-setup.exe" -ForegroundColor Green
} else {
    Write-Host "    (NSIS not installed; skipping .exe installer)" -ForegroundColor Yellow
}

Write-Host "==> done" -ForegroundColor Cyan
