# Small per-user online installer for Windows 10/11.
#
# No Python runtime or scientific packages are bundled. A pinned uv bootstrap
# downloads a private CPython 3.12 runtime and installs spaCR atomically.

[CmdletBinding()]
param(
    [string]$InstallRoot = (Join-Path $env:LOCALAPPDATA "SpaCR"),
    [string]$Version = "",
    [string]$PackageSpec = "",
    [string]$TorchBackend = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$UvVersion = "0.11.32"
$PythonVersion = "3.12"
$DefaultExtras = "qt"
$UvInstallUrl = "https://astral.sh/uv/$UvVersion/install.ps1"
# SHAP 0.52 leaves these dependencies unbounded. Without explicit floors uv
# may choose numba 0.53.1 and llvmlite 0.36.0, whose metadata admits Python
# 3.12 even though their build scripts reject it.
$ResolverGuards = @("numba>=0.60,<1.0", "llvmlite>=0.43,<1.0")

if ([string]::IsNullOrWhiteSpace($TorchBackend)) {
    $TorchBackend = $env:SPACR_TORCH_BACKEND
}
if ([string]::IsNullOrWhiteSpace($TorchBackend)) {
    $TorchBackend = "cpu"
}
if ($TorchBackend -notmatch '^[a-z0-9]+$') {
    throw "Invalid PyTorch backend '$TorchBackend'."
}

if ([string]::IsNullOrWhiteSpace($PackageSpec)) {
    if ([string]::IsNullOrWhiteSpace($Version)) {
        $PackageSpec = "spacr[$DefaultExtras]"
    } else {
        $PackageSpec = "spacr[$DefaultExtras]==$Version"
    }
}

$FullInstallRoot = [System.IO.Path]::GetFullPath($InstallRoot).TrimEnd("\")
$unsafeRoots = @(
    [System.IO.Path]::GetPathRoot($FullInstallRoot).TrimEnd("\"),
    $env:USERPROFILE.TrimEnd("\"),
    $env:LOCALAPPDATA.TrimEnd("\")
)
if ($unsafeRoots -contains $FullInstallRoot) {
    throw "Refusing unsafe install root '$InstallRoot'. Choose a dedicated SpaCR directory."
}
$InstallRoot = $FullInstallRoot

$BootstrapDir = Join-Path $InstallRoot "bootstrap"
$PythonDir = Join-Path $InstallRoot "python"
$VenvDir = Join-Path $InstallRoot "venv"
$CacheDir = Join-Path $InstallRoot "cache"
$UvExe = Join-Path $BootstrapDir "uv.exe"
$StageVenv = Join-Path $InstallRoot (".venv-staging-" + $PID)
$StagePython = Join-Path $StageVenv "Scripts\python.exe"
$Launcher = Join-Path $InstallRoot "launch_spacr.pyw"
$CliLauncher = Join-Path $InstallRoot "spacr.cmd"

Write-Host "spaCR lightweight online installer" -ForegroundColor Cyan
Write-Host "  application:    $PackageSpec"
Write-Host "  private Python: $PythonVersion"
Write-Host "  install root:   $InstallRoot"
Write-Host "  PyTorch backend: $TorchBackend"
Write-Host "  resolver guards: $($ResolverGuards -join ', ')"

if ($DryRun -or $env:SPACR_INSTALL_DRY_RUN -eq "1") {
    Write-Host "DRY RUN: would download $UvInstallUrl"
    Write-Host "DRY RUN: would create and validate $VenvDir"
    Write-Host "DRY RUN: would create $Launcher"
    exit 0
}

$driveName = [System.IO.Path]::GetPathRoot($InstallRoot)
$drive = [System.IO.DriveInfo]::new($driveName)
if ($drive.AvailableFreeSpace -lt 5GB) {
    $available = [math]::Round($drive.AvailableFreeSpace / 1GB, 1)
    throw "spaCR needs at least 5 GB free while dependencies install; only $available GB is available."
}

New-Item -ItemType Directory -Force -Path $BootstrapDir, $PythonDir, $CacheDir | Out-Null
$InstallerScript = Join-Path $env:TEMP ("spacr-uv-installer-" + $PID + ".ps1")
$LogPath = Join-Path $InstallRoot "install.log"
Start-Transcript -Path $LogPath -Append | Out-Null
Write-Host "Detailed installation log: $LogPath"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments
    )
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Command failed with exit code $LASTEXITCODE"
    }
}

try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Write-Host "Downloading the pinned uv bootstrap..." -ForegroundColor Cyan
    Invoke-WebRequest -UseBasicParsing -Uri $UvInstallUrl -OutFile $InstallerScript

    $env:UV_UNMANAGED_INSTALL = $BootstrapDir
    $env:UV_NO_MODIFY_PATH = "1"
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $InstallerScript
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $UvExe)) {
        throw "uv did not install at the expected path: $UvExe"
    }

    $env:UV_PYTHON_INSTALL_DIR = $PythonDir
    $env:UV_CACHE_DIR = $CacheDir
    $env:UV_SYSTEM_CERTS = "true"

    Write-Host "Downloading private Python $PythonVersion..." -ForegroundColor Cyan
    Invoke-Checked $UvExe python install $PythonVersion --managed-python --no-bin --no-registry

    if (Test-Path $StageVenv) {
        Remove-Item -Recurse -Force $StageVenv
    }
    Write-Host "Creating an isolated spaCR environment..." -ForegroundColor Cyan
    Invoke-Checked $UvExe venv $StageVenv --python $PythonVersion --managed-python --relocatable

    Write-Host "Downloading spaCR, Qt, PyTorch and scientific dependencies..." -ForegroundColor Cyan
    Invoke-Checked $UvExe pip install --python $StagePython --torch-backend $TorchBackend $PackageSpec @ResolverGuards

    Write-Host "Validating the installation before activating it..." -ForegroundColor Cyan
    Invoke-Checked $UvExe pip check --python $StagePython
    $env:QT_QPA_PLATFORM = "offscreen"
    Invoke-Checked -Command $StagePython -Arguments @(
        "-I",
        "-c",
        "import spacr, PySide6, torch; print('spaCR', spacr.__version__, '| torch', torch.__version__)"
    )

    $OldVenv = Join-Path $InstallRoot ".venv-previous"
    if (Test-Path $OldVenv) {
        Remove-Item -Recurse -Force $OldVenv
    }
    if (Test-Path $VenvDir) {
        Move-Item $VenvDir $OldVenv
    }
    Move-Item $StageVenv $VenvDir
    if (Test-Path $OldVenv) {
        Remove-Item -Recurse -Force $OldVenv
    }

    @"
from spacr.qt import run

raise SystemExit(run())
"@ | Set-Content -Encoding UTF8 $Launcher

    $InstalledPython = Join-Path $VenvDir "Scripts\python.exe"
    "@echo off`r`n`"$InstalledPython`" -m spacr.qt %*`r`n" |
        Set-Content -Encoding ASCII $CliLauncher

    Write-Host ""
    Write-Host "spaCR installed successfully." -ForegroundColor Green
    Write-Host "Launcher: $Launcher"
} catch {
    if (Test-Path $StageVenv) {
        Remove-Item -Recurse -Force $StageVenv
    }
    throw
} finally {
    Remove-Item -Force -ErrorAction SilentlyContinue $InstallerScript
    Stop-Transcript -ErrorAction SilentlyContinue | Out-Null
}
