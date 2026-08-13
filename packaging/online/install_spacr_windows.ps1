# Small per-user online installer for Windows 10/11.
#
# No Python runtime or scientific packages are bundled. A pinned uv bootstrap
# downloads a private CPython 3.12 runtime and installs spaCR atomically.

[CmdletBinding()]
param(
    [string]$InstallRoot = (Join-Path $env:LOCALAPPDATA "spaCR"),
    [string]$Version = "",
    [string]$PackageSpec = "",
    [string]$TorchBackend = "",
    [string]$Language = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
if (-not [string]::IsNullOrWhiteSpace($Language)) {
    $env:SPACR_INSTALL_LANGUAGE = $Language
}
. (Join-Path $PSScriptRoot "generated\installer_messages.ps1")
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
    throw (Get-SpacrInstallerMessage "invalid_backend" @($TorchBackend))
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
    throw ((Get-SpacrInstallerMessage "unsafe_root" @($InstallRoot)) + " " +
        (Get-SpacrInstallerMessage "choose_directory"))
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

Write-Host (Get-SpacrInstallerMessage "installer_title") -ForegroundColor Cyan
Write-Host "  $(Get-SpacrInstallerMessage 'application'):    $PackageSpec"
Write-Host "  $(Get-SpacrInstallerMessage 'private_python'): $PythonVersion"
Write-Host "  $(Get-SpacrInstallerMessage 'install_root'):   $InstallRoot"
Write-Host "  $(Get-SpacrInstallerMessage 'pytorch_backend'): $TorchBackend"
Write-Host "  $(Get-SpacrInstallerMessage 'resolver_guards'): $($ResolverGuards -join ', ')"

if ($DryRun -or $env:SPACR_INSTALL_DRY_RUN -eq "1") {
    Write-Host (Get-SpacrInstallerMessage "dry_download" @($UvInstallUrl))
    Write-Host (Get-SpacrInstallerMessage "dry_create" @($VenvDir))
    Write-Host (Get-SpacrInstallerMessage "dry_launcher" @($Launcher))
    exit 0
}

$driveName = [System.IO.Path]::GetPathRoot($InstallRoot)
$drive = [System.IO.DriveInfo]::new($driveName)
if ($drive.AvailableFreeSpace -lt 5GB) {
    $available = [math]::Round($drive.AvailableFreeSpace / 1GB, 1)
    throw ((Get-SpacrInstallerMessage "needs_free_space") + " " +
        (Get-SpacrInstallerMessage "available_space" @($available, $InstallRoot)))
}

New-Item -ItemType Directory -Force -Path $BootstrapDir, $PythonDir, $CacheDir | Out-Null
$InstallerScript = Join-Path $env:TEMP ("spacr-uv-installer-" + $PID + ".ps1")
$LogPath = Join-Path $InstallRoot "install.log"
Start-Transcript -Path $LogPath -Append | Out-Null
Write-Host (Get-SpacrInstallerMessage "detailed_log" @($LogPath))

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments
    )
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw (Get-SpacrInstallerMessage "command_failed" @($Command, $LASTEXITCODE))
    }
}

try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Write-Host (Get-SpacrInstallerMessage "downloading_uv") -ForegroundColor Cyan
    Invoke-WebRequest -UseBasicParsing -Uri $UvInstallUrl -OutFile $InstallerScript

    $env:UV_UNMANAGED_INSTALL = $BootstrapDir
    $env:UV_NO_MODIFY_PATH = "1"
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $InstallerScript
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $UvExe)) {
        throw (Get-SpacrInstallerMessage "uv_missing" @($UvExe))
    }

    $env:UV_PYTHON_INSTALL_DIR = $PythonDir
    $env:UV_CACHE_DIR = $CacheDir
    $env:UV_SYSTEM_CERTS = "true"

    Write-Host (Get-SpacrInstallerMessage "downloading_python" @($PythonVersion)) -ForegroundColor Cyan
    Invoke-Checked $UvExe python install $PythonVersion --managed-python --no-bin --no-registry

    if (Test-Path $StageVenv) {
        Remove-Item -Recurse -Force $StageVenv
    }
    Write-Host (Get-SpacrInstallerMessage "creating_environment") -ForegroundColor Cyan
    Invoke-Checked $UvExe venv $StageVenv --python $PythonVersion --managed-python --relocatable

    Write-Host (Get-SpacrInstallerMessage "downloading_dependencies") -ForegroundColor Cyan
    Invoke-Checked $UvExe pip install --python $StagePython --torch-backend $TorchBackend $PackageSpec @ResolverGuards

    Write-Host (Get-SpacrInstallerMessage "validating_install") -ForegroundColor Cyan
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
    Write-Host (Get-SpacrInstallerMessage "installed") -ForegroundColor Green
    Write-Host (Get-SpacrInstallerMessage "launcher" @($Launcher))
} catch {
    if (Test-Path $StageVenv) {
        Remove-Item -Recurse -Force $StageVenv
    }
    throw
} finally {
    Remove-Item -Force -ErrorAction SilentlyContinue $InstallerScript
    Stop-Transcript -ErrorAction SilentlyContinue | Out-Null
}
