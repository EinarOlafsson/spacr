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
    [ValidateSet(0, 1)][int]$ConsentCollected = 0,
    [ValidateSet(0, 1)][int]$ShareDiagnostics = 0,
    [ValidateSet(0, 1)][int]$ReportIssues = 0,
    [string]$OfflineBundle = $env:SPACR_OFFLINE_BUNDLE,
    [switch]$CheckMask,
    [switch]$WithoutWgpu,
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
$UvInstallUrl = "https://astral.sh/uv/$UvVersion/install.ps1"
# SHAP 0.52 leaves these dependencies unbounded. Without explicit floors uv
# may choose numba 0.53.1 and llvmlite 0.36.0, whose metadata admits Python
# 3.12 even though their build scripts reject it.
$ResolverGuards = @("numba>=0.60,<1.0", "llvmlite>=0.43,<1.0")

if ([string]::IsNullOrWhiteSpace($TorchBackend)) {
    $TorchBackend = $env:SPACR_TORCH_BACKEND
}
# wgpu draws the True 3D view through Vulkan or Direct3D 12. Without it spaCR
# draws 3D through OpenGL (VisPy) or the CPU. Off by default.
if ($env:SPACR_WITHOUT_WGPU -eq "1") {
    $WithoutWgpu = [switch]$true
}
$DetectedAccelerator = "none"
if (Get-Command "nvidia-smi.exe" -ErrorAction SilentlyContinue) {
    & nvidia-smi.exe -L *> $null
    if ($LASTEXITCODE -eq 0) {
        $DetectedAccelerator = "nvidia"
    }
}
if ([string]::IsNullOrWhiteSpace($TorchBackend)) {
    # uv's automatic backend selects CUDA on compatible NVIDIA systems and
    # safely falls back elsewhere. Users can still request "cpu" explicitly.
    $TorchBackend = "auto"
}
if ($TorchBackend -notmatch '^[a-z0-9]+$') {
    throw (Get-SpacrInstallerMessage "invalid_backend" @($TorchBackend))
}

if ([string]::IsNullOrWhiteSpace($PackageSpec)) {
    $PackageSpec = $env:SPACR_PACKAGE_SPEC
}
if ([string]::IsNullOrWhiteSpace($PackageSpec)) {
    if ([string]::IsNullOrWhiteSpace($Version)) {
        $PackageSpec = "spacr"
    } else {
        $PackageSpec = "spacr==$Version"
    }
}

# An offline bundle (packaging/offline/build_offline_bundle.py) carries uv,
# the private Python, every wheel, Cellpose weights and Mask test data. It was
# locked for one PyTorch wheel line when it was built, so that comes from it.
if (-not [string]::IsNullOrWhiteSpace($OfflineBundle)) {
    $OfflineBundle = (Resolve-Path $OfflineBundle).Path
    $BundleInfo = Get-Content -Raw (Join-Path $OfflineBundle "bundle.json") | ConvertFrom-Json
    $TorchBackend = $BundleInfo.torch_backend
    $PackageSpec = $BundleInfo.package_spec
    Write-Host "  Offline bundle: $OfflineBundle"
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
$StageProfile = Join-Path $InstallRoot (".install-profile-staging-" + $PID + ".json")
$Launcher = Join-Path $InstallRoot "launch_spacr.pyw"
$CliLauncher = Join-Path $InstallRoot "spacr.cmd"

Write-Host (Get-SpacrInstallerMessage "installer_title") -ForegroundColor Cyan
Write-Host "  $(Get-SpacrInstallerMessage 'application'):    $PackageSpec"
Write-Host "  $(Get-SpacrInstallerMessage 'private_python'): $PythonVersion"
Write-Host "  $(Get-SpacrInstallerMessage 'install_root'):   $InstallRoot"
Write-Host "  $(Get-SpacrInstallerMessage 'pytorch_backend'): $TorchBackend"
Write-Host "  GPU benchmark: RTX 3090 measured 13x faster Cellpose segmentation and 20x faster ResNet classification than CPU; hardware varies."
Write-Host "  $(Get-SpacrInstallerMessage 'resolver_guards'): $($ResolverGuards -join ', ')"
if ($WithoutWgpu) {
    Write-Host "  $(Get-SpacrInstallerMessage 'wgpu_plan'): $(Get-SpacrInstallerMessage 'wgpu_plan_skip')"
} else {
    Write-Host "  $(Get-SpacrInstallerMessage 'wgpu_plan'): $(Get-SpacrInstallerMessage 'wgpu_plan_install')"
}

if ($DryRun -or $env:SPACR_INSTALL_DRY_RUN -eq "1") {
    Write-Host (Get-SpacrInstallerMessage "dry_remove_old")
    Write-Host (Get-SpacrInstallerMessage "dry_download" @($UvInstallUrl))
    Write-Host (Get-SpacrInstallerMessage "dry_create" @($VenvDir))
    Write-Host (Get-SpacrInstallerMessage "dry_launcher" @($Launcher))
    exit 0
}

function Get-SpacrAvailableDiskSpace {
    param([Parameter(Mandatory = $true)][string]$Path)
    $Directory = [System.IO.Path]::GetFullPath($Path)
    while (-not [System.IO.Directory]::Exists($Directory)) {
        $Parent = [System.IO.Path]::GetDirectoryName($Directory)
        if ([string]::IsNullOrEmpty($Parent) -or $Parent -eq $Directory) {
            break
        }
        $Directory = $Parent
    }
    if (-not ("SpacrInstaller.DiskSpace" -as [type])) {
        Add-Type -Namespace SpacrInstaller -Name DiskSpace -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("kernel32.dll", CharSet = System.Runtime.InteropServices.CharSet.Unicode, SetLastError = true)]
public static extern bool GetDiskFreeSpaceExW(string directory, out ulong available, out ulong total, out ulong free);
'@
    }
    [UInt64]$Available = 0
    [UInt64]$Total = 0
    [UInt64]$Free = 0
    if (-not [SpacrInstaller.DiskSpace]::GetDiskFreeSpaceExW($Directory, [ref]$Available, [ref]$Total, [ref]$Free)) {
        throw [System.ComponentModel.Win32Exception]::new([System.Runtime.InteropServices.Marshal]::GetLastWin32Error())
    }
    return $Available
}

$AvailableSpace = Get-SpacrAvailableDiskSpace $InstallRoot
if ($AvailableSpace -lt 5GB) {
    $available = [math]::Round($AvailableSpace / 1GB, 1)
    throw ((Get-SpacrInstallerMessage "needs_free_space") + " " +
        (Get-SpacrInstallerMessage "available_space" @($available, $InstallRoot)))
}

$InstallerLeaseHandles = @()
$InstallerStarted = $false
$TranscriptStarted = $false
$WorkDir = Join-Path $env:TEMP ("spacr-install-" + $PID)
$WorkUv = Join-Path $WorkDir "bootstrap\uv.exe"
$InstallerScript = Join-Path $WorkDir "uv-installer.ps1"
$LogPath = Join-Path $InstallRoot "install.log"

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

# Each lease file records its owner (pid, start time, host). A later
# installer takes a lease over only when no process holds it open, it carries
# no rollback line, and its recorded owner on this computer is provably gone
# (no such process, or a process started at another time). A lease from a
# version that recorded no owner is never taken over.
#
# Children of a killed installer (uv, pip, Python) would keep working in the
# root after their parent is gone. The installer runs in a Job Object that
# Windows closes, killing every member, when the installer dies; its lease
# records tree=1, and stays live while any process the owner started (parent
# id equal to the owner's, created at or after it) still runs. A process
# table that cannot be read refuses.
function Enable-SpacrInstallerJob {
    try {
        Add-Type -Namespace SpacrInstaller -Name Job -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("kernel32.dll", CharSet = System.Runtime.InteropServices.CharSet.Unicode, SetLastError = true)]
public static extern System.IntPtr CreateJobObject(System.IntPtr attributes, string name);
[System.Runtime.InteropServices.DllImport("kernel32.dll", SetLastError = true)]
public static extern bool SetInformationJobObject(System.IntPtr job, int infoClass, byte[] info, int length);
[System.Runtime.InteropServices.DllImport("kernel32.dll", SetLastError = true)]
public static extern bool AssignProcessToJobObject(System.IntPtr job, System.IntPtr process);
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern System.IntPtr GetCurrentProcess();
'@
        $Job = [SpacrInstaller.Job]::CreateJobObject([System.IntPtr]::Zero, $null)
        if ($Job -eq [System.IntPtr]::Zero) { return $false }
        $Limits = New-Object byte[] ($(if ([System.IntPtr]::Size -eq 8) { 144 } else { 112 }))
        [System.BitConverter]::GetBytes([int]0x2000).CopyTo($Limits, 16)
        if (-not [SpacrInstaller.Job]::SetInformationJobObject($Job, 9, $Limits, $Limits.Length)) { return $false }
        if (-not [SpacrInstaller.Job]::AssignProcessToJobObject($Job, [SpacrInstaller.Job]::GetCurrentProcess())) { return $false }
        $script:SpacrInstallerJob = $Job
        return $true
    } catch { return $false }
}
function Test-SpacrOwnerChildrenLive {
    param([int]$OwnerId, [string]$Started)
    if ($Started -notmatch '^filetime:([0-9]+)$') { return $null }
    $Since = [long]$Matches[1]
    try {
        $Children = @(Get-CimInstance -ClassName Win32_Process -Filter "ParentProcessId = $OwnerId" -ErrorAction Stop)
    } catch { return $null }
    foreach ($Child in $Children) {
        if ($Child.ProcessId -eq $OwnerId) { continue }
        if ($null -eq $Child.CreationDate) { return $null }
        if ($Child.CreationDate.ToFileTimeUtc() -ge $Since) { return $true }
    }
    return $false
}
function Get-SpacrProcessStarted {
    param([int]$ProcessId)
    try { $Process = [System.Diagnostics.Process]::GetProcessById($ProcessId) } catch [System.ArgumentException] { return "" }
    try { return "filetime:" + $Process.StartTime.ToFileTimeUtc() } catch { return $null } finally { $Process.Dispose() }
}
function Test-SpacrLeaseOwnerDead {
    param([Parameter(Mandatory = $true)][string]$LeasePath)
    try {
        $Probe = [System.IO.FileStream]::new($LeasePath, [System.IO.FileMode]::Open,
            [System.IO.FileAccess]::Read, [System.IO.FileShare]::None)
    } catch { return $false }
    try {
        $Text = [System.IO.StreamReader]::new($Probe).ReadToEnd()
    } finally { $Probe.Dispose() }
    $Fields = @{}
    foreach ($Line in ($Text -split "`n")) {
        $Pair = $Line.TrimEnd("`r") -split "=", 2
        if ($Pair.Count -eq 2) { $Fields[$Pair[0]] = $Pair[1] }
    }
    $Tree = $Fields["tree"]
    if ($Fields.ContainsKey("tree")) { $Fields.Remove("tree") }
    if ($Fields.Count -ne 3 -or -not $Fields.ContainsKey("pid") -or -not $Fields.ContainsKey("started") -or
            $Fields["host"] -ne $env:COMPUTERNAME -or -not $Fields["started"]) { return $false }
    $OwnerId = 0
    if (-not [int]::TryParse($Fields["pid"], [ref]$OwnerId) -or $OwnerId -le 0 -or $OwnerId -eq $PID) { return $false }
    $Now = Get-SpacrProcessStarted $OwnerId
    if (-not (($null -ne $Now) -and (($Now -eq "") -or ($Now -ne $Fields["started"])))) { return $false }
    if ($null -eq $Tree) { return $true }
    if ($Tree -ne "1") { return $false }
    return ($false -eq (Test-SpacrOwnerChildrenLive $OwnerId $Fields["started"]))
}
function Get-SpacrLeaseOwner {
    param([string]$LeasePath)
    try {
        $Reader = [System.IO.StreamReader]::new([System.IO.FileStream]::new($LeasePath, [System.IO.FileMode]::Open,
            [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite))
        try {
            foreach ($Line in ($Reader.ReadToEnd() -split "`n")) {
                if ($Line -like "pid=*") { return $Line.Substring(4).TrimEnd("`r") }
            }
        } finally { $Reader.Dispose() }
    } catch { }
    return ""
}
# A lease written by a later installer may carry fields unknown here. It is
# never taken over, and when its pid is gone it is reported as a lease this
# version cannot check, not as held by that dead pid.
function Test-SpacrLeaseHasUnknownFields {
    param([string]$LeasePath)
    try {
        $Reader = [System.IO.StreamReader]::new([System.IO.FileStream]::new($LeasePath, [System.IO.FileMode]::Open,
            [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite))
        try {
            foreach ($Line in ($Reader.ReadToEnd() -split "`n")) {
                $Line = $Line.TrimEnd("`r")
                if ($Line -and $Line -notmatch '^(pid|started|host|tree|rollback)=') { return $true }
            }
        } finally { $Reader.Dispose() }
    } catch { }
    return $false
}
# A killed installer leaves its staging runtime, staging profile and temporary
# work folder behind. They are removed only for an owner whose lease this
# installer just took over, only once this installer holds every lease, and
# only while that owner is still provably gone. A link or junction is never
# followed or removed, and nothing outside the install root and TEMP is named.
$SpacrAbandonedOwners = @()
function Add-SpacrAbandonedOwner {
    param([string]$LeasePath)
    $Fields = @{}
    try {
        $Reader = [System.IO.StreamReader]::new([System.IO.FileStream]::new($LeasePath, [System.IO.FileMode]::Open,
            [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite))
        try {
            foreach ($Line in ($Reader.ReadToEnd() -split "`n")) {
                $Pair = $Line.TrimEnd("`r") -split "=", 2
                if ($Pair.Count -eq 2) { $Fields[$Pair[0]] = $Pair[1] }
            }
        } finally { $Reader.Dispose() }
    } catch { return }
    if ($Fields["pid"] -notmatch '^[1-9][0-9]*$' -or -not $Fields["started"]) { return }
    foreach ($Known in $script:SpacrAbandonedOwners) {
        if ($Known.Id -eq $Fields["pid"]) { return }
    }
    $script:SpacrAbandonedOwners += @{ Id = $Fields["pid"]; Started = $Fields["started"] }
}
# Directory.Delete(path, $true) on .NET Framework treats a junction inside the
# tree as a volume mount point and fails, and uv's Python download contains
# one. A link is removed as a link, never entered; read-only files are cleared.
function Remove-SpacrTree {
    param([string]$Folder)
    foreach ($Entry in ([System.IO.DirectoryInfo]::new($Folder)).GetFileSystemInfos()) {
        if ($Entry.Attributes -band [System.IO.FileAttributes]::ReadOnly) {
            $Entry.Attributes = $Entry.Attributes -band (-bnot [System.IO.FileAttributes]::ReadOnly)
        }
        if ($Entry -is [System.IO.DirectoryInfo]) {
            if ($Entry.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
                [System.IO.Directory]::Delete($Entry.FullName, $false)
            } else {
                Remove-SpacrTree $Entry.FullName
            }
        } else {
            [System.IO.File]::Delete($Entry.FullName)
        }
    }
    [System.IO.Directory]::Delete($Folder, $false)
}
function Remove-SpacrAbandonedPath {
    param([string]$Parent, [string]$Name)
    if ([string]::IsNullOrWhiteSpace($Parent)) { return }
    $Base = [System.IO.Path]::GetFullPath($Parent).TrimEnd("\")
    $Path = [System.IO.Path]::GetFullPath((Join-Path $Base $Name))
    if ([System.IO.Path]::GetDirectoryName($Path).TrimEnd("\") -ne $Base) { return }
    $Item = Get-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue
    if ($null -eq $Item) { return }
    if ($Item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) { return }
    try {
        if ($Item.PSIsContainer) {
            Remove-SpacrTree $Path
        } else {
            [System.IO.File]::Delete($Path)
        }
        Write-Host (Get-SpacrInstallerMessage "staging_removed" @($Path))
    } catch { }
}
function Remove-SpacrAbandonedStaging {
    foreach ($Owner in $script:SpacrAbandonedOwners) {
        $OwnerId = [int]$Owner.Id
        if ($OwnerId -eq $PID) { continue }
        $Now = Get-SpacrProcessStarted $OwnerId
        if ($null -eq $Now -or ($Now -ne "" -and $Now -eq $Owner.Started)) { continue }
        Remove-SpacrAbandonedPath $InstallRoot (".venv-staging-" + $OwnerId)
        Remove-SpacrAbandonedPath $InstallRoot (".install-profile-staging-" + $OwnerId + ".json")
        Remove-SpacrAbandonedPath $env:TEMP ("spacr-install-" + $OwnerId)
    }
}
function New-SpacrInstallerLeaseFile {
    param([string]$LeasePath)
    $Stream = [System.IO.FileStream]::new($LeasePath, [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::Read)
    $Record = "pid=$PID`nstarted=$(Get-SpacrProcessStarted $PID)`nhost=$env:COMPUTERNAME`ntree=1`n"
    $Bytes = [System.Text.Encoding]::UTF8.GetBytes($Record)
    $Stream.Write($Bytes, 0, $Bytes.Length)
    $Stream.Flush()
    return $Stream
}
function Open-SpacrInstallerLease {
    param([Parameter(Mandatory = $true)][string]$LeasePath)
    try {
        return New-SpacrInstallerLeaseFile $LeasePath
    } catch [System.IO.IOException] {
        if (Test-Path -LiteralPath $LeasePath) {
            if (Test-SpacrLeaseOwnerDead $LeasePath) {
                Add-SpacrAbandonedOwner $LeasePath
                Remove-Item -LiteralPath $LeasePath -Force
                Write-Host (Get-SpacrInstallerMessage "lease_cleared" @($LeasePath))
                return New-SpacrInstallerLeaseFile $LeasePath
            }
            $Owner = Get-SpacrLeaseOwner $LeasePath
            $OwnerId = 0
            if ($Owner -and [int]::TryParse($Owner, [ref]$OwnerId) -and ((Get-SpacrProcessStarted $OwnerId) -eq "") -and
                    (Test-SpacrLeaseHasUnknownFields $LeasePath)) { $Owner = "" }
            if ($Owner) {
                throw ((Get-SpacrInstallerMessage "already_running") + " " + (Get-SpacrInstallerMessage "lease_held" @($LeasePath, $Owner)))
            }
            throw ((Get-SpacrInstallerMessage "already_running") + " " + (Get-SpacrInstallerMessage "lease_held_unknown" @($LeasePath)))
        }
        throw
    }
}

$MetadataBackup = Join-Path $WorkDir "previous-metadata"
$MetadataTouched = $false
$RetainWorkDir = $false
$Committed = $false
$null = Enable-SpacrInstallerJob
try {
    $InstallParent = [System.IO.Path]::GetDirectoryName($InstallRoot)
    New-Item -ItemType Directory -Force -Path $InstallParent | Out-Null
    $OuterLease = Join-Path $InstallParent (("." + [System.IO.Path]::GetFileName($InstallRoot)) + ".spacr-installer-lock")
    $InstallerLeaseHandles += @{ Path = $OuterLease; Handle = (Open-SpacrInstallerLease $OuterLease) }
    New-Item -ItemType Directory -Force -Path $InstallRoot | Out-Null
    $RuntimeLease = Join-Path $InstallRoot ".venv.spacr-update-lock"
    $InstallerLeaseHandles += @{ Path = $RuntimeLease; Handle = (Open-SpacrInstallerLease $RuntimeLease) }
    Remove-SpacrAbandonedStaging
    $InstallerStarted = $true
    Start-Transcript -Path $LogPath -Append | Out-Null
    $TranscriptStarted = $true
    Write-Host (Get-SpacrInstallerMessage "detailed_log" @($LogPath))
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null

    # find old spaCR files --> delete old spaCR files --> install new spaCR.
    #
    # uv and a private Python are fetched into a temporary folder first,
    # outside every installation: a failed download stops the install here,
    # before anything old is deleted, and the Python that runs the removal is
    # not one of the files it removes. install_cleanup.py is the finder and
    # remover the in-app update uses too, shipped beside this script. It never
    # runs an old version's own Uninstall.exe, keeps the settings stored under
    # the same registry key, and lists environments the user made without
    # touching them.
    if ($OfflineBundle) {
        # A bundle carried on a USB stick has no TLS to vouch for it: every
        # file is checked against its SHA256SUMS before any of it is run.
        Write-Host "Verifying the offline bundle..." -ForegroundColor Cyan
        foreach ($line in Get-Content (Join-Path $OfflineBundle "SHA256SUMS")) {
            $expected, $relative = $line -split "  ", 2
            $actual = (Get-FileHash -Algorithm SHA256 (Join-Path $OfflineBundle $relative)).Hash
            if ($actual -ne $expected.ToUpperInvariant()) {
                throw "Offline bundle file does not match SHA256SUMS: $relative"
            }
        }
        New-Item -ItemType Directory -Force -Path (Split-Path $WorkUv) | Out-Null
        Copy-Item (Join-Path $OfflineBundle "uv\uv.exe") $WorkUv
        $global:LASTEXITCODE = 0
        $env:UV_PYTHON_INSTALL_MIRROR = "file:///" + ($OfflineBundle -replace "\\", "/") + "/python"
    } else {
        Write-Host (Get-SpacrInstallerMessage "downloading_uv") -ForegroundColor Cyan
        Invoke-WebRequest -UseBasicParsing -Uri $UvInstallUrl -OutFile $InstallerScript

        $env:UV_UNMANAGED_INSTALL = Join-Path $WorkDir "bootstrap"
        $env:UV_NO_MODIFY_PATH = "1"
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $InstallerScript
    }
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $WorkUv)) {
        throw (Get-SpacrInstallerMessage "uv_missing" @($WorkUv))
    }

    $env:UV_SYSTEM_CERTS = "true"
    $env:UV_PYTHON_CACHE_DIR = Join-Path $WorkDir "python-downloads"
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $WorkDir "python"
    $env:UV_CACHE_DIR = Join-Path $WorkDir "cache"
    Write-Host (Get-SpacrInstallerMessage "downloading_python" @($PythonVersion)) -ForegroundColor Cyan
    Invoke-Checked $WorkUv python install $PythonVersion --managed-python --no-bin --no-registry
    $CleanupPython = & $WorkUv python find $PythonVersion --managed-python | Select-Object -Last 1
    if ($LASTEXITCODE -ne 0 -or -not $CleanupPython -or -not (Test-Path $CleanupPython.Trim())) {
        throw (Get-SpacrInstallerMessage "command_failed" @($WorkUv, $LASTEXITCODE))
    }
    $CleanupPython = $CleanupPython.Trim()
    $CleanupModule = Join-Path $PSScriptRoot "install_cleanup.py"
    if (-not (Test-Path $CleanupModule)) {
        $CleanupModule = Join-Path $PSScriptRoot "..\..\spacr\install_cleanup.py"
    }
    $ProfileModule = Join-Path $PSScriptRoot "install_profile.py"
    if (-not (Test-Path $ProfileModule)) {
        $ProfileModule = Join-Path $PSScriptRoot "..\..\spacr\install_profile.py"
    }
    # The log of this install and the files the NSIS wrapper has already
    # written for it are not old files.


    New-Item -ItemType Directory -Force -Path $BootstrapDir, $PythonDir, $CacheDir | Out-Null
    Move-Item -Force $WorkUv $UvExe
    $env:UV_PYTHON_INSTALL_DIR = $PythonDir
    $env:UV_CACHE_DIR = $CacheDir
    Invoke-Checked $UvExe python install $PythonVersion --managed-python --no-bin --no-registry

    if (Test-Path $StageVenv) {
        Remove-Item -Recurse -Force $StageVenv
    }
    Write-Host (Get-SpacrInstallerMessage "creating_environment") -ForegroundColor Cyan
    Invoke-Checked $UvExe venv $StageVenv --python $PythonVersion --managed-python --relocatable

    # Skipping wgpu at resolve time: an override whose marker never holds
    # removes wgpu (and rendercanvas, which only wgpu pulls) from the
    # resolution instead of installing and then uninstalling it.
    $OverrideArgs = @()
    if ($WithoutWgpu) {
        $WgpuOverrides = Join-Path $WorkDir "without-wgpu-overrides.txt"
        [System.IO.File]::WriteAllText($WgpuOverrides,
            "wgpu; sys_platform == `"never`"`n", [System.Text.UTF8Encoding]::new($false))
        $OverrideArgs = @("--override", $WgpuOverrides)
    }

    Write-Host (Get-SpacrInstallerMessage "downloading_dependencies") -ForegroundColor Cyan
    if ($OfflineBundle) {
        Invoke-Checked $UvExe pip install --python $StagePython --compile-bytecode --offline --no-index `
            --find-links (Join-Path $OfflineBundle "wheels") @OverrideArgs `
            -r (Join-Path $OfflineBundle "requirements.txt")
    } else {
        Invoke-Checked $UvExe pip install --python $StagePython --compile-bytecode --torch-backend $TorchBackend @OverrideArgs $PackageSpec @ResolverGuards
    }

    Write-Host (Get-SpacrInstallerMessage "validating_install") -ForegroundColor Cyan
    # An install without wgpu leaves one declared dependency out on purpose, so
    # uv's sole "spacr requires wgpu" diagnostic is accepted then and nothing
    # else is (install_cleanup._only_saved_wgpu_omission, as the updater does).
    # Start-Process keeps uv's own UTF-8 bytes; PowerShell redirection would
    # re-encode them and wrap stderr lines as error records.
    $PipCheckOut = Join-Path $WorkDir "pip-check-out.txt"
    $PipCheckErr = Join-Path $WorkDir "pip-check-err.txt"
    $PipCheckLog = Join-Path $WorkDir "pip-check.txt"
    $PipCheck = Start-Process -FilePath $UvExe -ArgumentList @("pip", "check", "--python", "`"$StagePython`"") `
        -NoNewWindow -Wait -PassThru -RedirectStandardOutput $PipCheckOut -RedirectStandardError $PipCheckErr
    $PipCheckCode = $PipCheck.ExitCode
    [System.IO.File]::WriteAllText($PipCheckLog,
        [System.IO.File]::ReadAllText($PipCheckErr) + [System.IO.File]::ReadAllText($PipCheckOut),
        [System.Text.UTF8Encoding]::new($false))
    Write-Host ([System.IO.File]::ReadAllText($PipCheckLog))
    if ($PipCheckCode -ne 0) {
        $Accepted = $false
        if ($WithoutWgpu) {
            & $StagePython -I -c "import runpy, sys; only=runpy.run_path(sys.argv[1])['_only_saved_wgpu_omission']; sys.exit(0 if only(int(sys.argv[2]), open(sys.argv[3], encoding='utf-8', errors='replace').read()) else 1)" $CleanupModule "$PipCheckCode" $PipCheckLog
            $Accepted = ($LASTEXITCODE -eq 0)
        }
        if (-not $Accepted) {
            throw (Get-SpacrInstallerMessage "command_failed" @($UvExe, $PipCheckCode))
        }
    }
    $env:QT_QPA_PLATFORM = "offscreen"
    Invoke-Checked -Command $StagePython -Arguments @(
        "-I",
        "-c",
        "import spacr, PySide6, torch; print('spaCR', spacr.__version__, '| torch', torch.__version__)"
    )
    if ($TorchBackend -ne "cpu" -and $DetectedAccelerator -eq "nvidia") {
        Invoke-Checked -Command $StagePython -Arguments @(
            "-I",
            "-c",
            "import torch; assert torch.cuda.is_available(), 'GPU install selected but CUDA is unavailable'"
        )
    }

    Invoke-Checked -Command $StagePython -Arguments @(
        "-I",
        $ProfileModule,
        "--path",
        $StageProfile,
        "--requested",
        $TorchBackend,
        "--detected",
        $DetectedAccelerator,
        "--consent-collected",
        [string]$ConsentCollected,
        "--share-diagnostics",
        [string]$ShareDiagnostics,
        "--report-issues",
        [string]$ReportIssues,
        "--without-wgpu",
        $(if ($WithoutWgpu) { "1" } else { "0" })
    )

    $IdentityArgs = @("-I", $CleanupModule, "verify-runtime", "--python", $StagePython, "--retiring-prefix", $VenvDir)
    if (-not [string]::IsNullOrWhiteSpace($Version)) { $IdentityArgs += @("--version", $Version) }
    Invoke-Checked -Command $CleanupPython -Arguments $IdentityArgs
    & $CleanupPython -I $CleanupModule remove `
        --keep $InstallRoot `
        --keep $LogPath `
        --keep (Join-Path $InstallRoot "spacr.ico") `
        --keep (Join-Path $InstallRoot "nsis-bootstrap-status.txt")
    if ($LASTEXITCODE -ne 0) {
        throw (Get-SpacrInstallerMessage "old_copy_not_removed")
    }
    $MetadataPaths = @((Join-Path $InstallRoot "runtime-identity.json"),
        (Join-Path $InstallRoot "install-profile.json"), $Launcher, $CliLauncher)
    if ($OfflineBundle) {
        $MetadataPaths += @((Join-Path $InstallRoot "offline_mask_check.py"),
            (Join-Path $InstallRoot "offline-bundle.json"))
    }
    Invoke-Checked -Command $CleanupPython -Arguments (@("-I", $CleanupModule,
        "installer-metadata", "snapshot", "--folder", $MetadataBackup, "--paths") + $MetadataPaths)
    $OldVenv = Join-Path $InstallRoot (".venv-previous-" + $PID)
    if (Test-Path $VenvDir) {
        Move-Item $VenvDir $OldVenv
    }
    Move-Item $StageVenv $VenvDir
    $IdentityArgs[[Array]::IndexOf($IdentityArgs, "--python") + 1] = Join-Path $VenvDir "Scripts\python.exe"
    try {
        $FinalIdentity = Invoke-Checked -Command $CleanupPython -Arguments $IdentityArgs
        $Receipt = $FinalIdentity | ConvertFrom-Json
        if (-not $Receipt.verified) { throw "runtime_identity_unverified" }
        $StageIdentity = Join-Path $WorkDir "runtime-identity.new.json"
        [System.IO.File]::WriteAllText($StageIdentity,
            [string]::Join([Environment]::NewLine, @($FinalIdentity)),
            [System.Text.UTF8Encoding]::new($false))
    } catch {
        Remove-Item -Recurse -Force $VenvDir
        if (Test-Path $OldVenv) { Move-Item $OldVenv $VenvDir }
        throw
    }
    $InstalledPython = Join-Path $VenvDir "Scripts\python.exe"
    $MetadataTouched = $true
    Move-Item -Force $StageIdentity (Join-Path $InstallRoot "runtime-identity.json")
    Move-Item -Force $StageProfile (Join-Path $InstallRoot "install-profile.json")

    if ($OfflineBundle) {
        # Cellpose looks for its weights here before it would download them.
        # Weights already present are left as they are.
        $CellposeDir = $env:CELLPOSE_LOCAL_MODELS_PATH
        if (-not $CellposeDir) {
            $CellposeDir = Join-Path $env:USERPROFILE ".cellpose\models"
        }
        $BundledModels = Join-Path $OfflineBundle "models\cellpose"
        if (Test-Path $BundledModels) {
            New-Item -ItemType Directory -Force -Path $CellposeDir | Out-Null
            foreach ($weights in Get-ChildItem $BundledModels) {
                $target = Join-Path $CellposeDir $weights.Name
                if (-not (Test-Path $target)) {
                    Copy-Item $weights.FullName $target
                }
            }
        }
        $TestData = Join-Path $InstallRoot "test_data"
        if ((Test-Path (Join-Path $OfflineBundle "test_data")) -and -not (Test-Path $TestData)) {
            Copy-Item -Recurse (Join-Path $OfflineBundle "test_data") $TestData
        }
        $MaskCheck = Join-Path $InstallRoot "offline_mask_check.py"
        Copy-Item (Join-Path $OfflineBundle "offline_mask_check.py") $MaskCheck
        Copy-Item (Join-Path $OfflineBundle "bundle.json") (Join-Path $InstallRoot "offline-bundle.json")
        if ($CheckMask) {
            Invoke-Checked -Command $InstalledPython -Arguments @(
                "-I", $MaskCheck, "--data", (Join-Path $TestData "plate1"))
        }
    }

    @"
from spacr.qt import run

raise SystemExit(run())
"@ | Set-Content -Encoding UTF8 $Launcher

    "@echo off`r`n`"$InstalledPython`" -I -m spacr.qt %*`r`n" |
        Set-Content -Encoding ASCII $CliLauncher

    $Committed = $true
    $MetadataTouched = $false
    if (Test-Path $OldVenv) { Remove-Item -Recurse -Force $OldVenv }
    Write-Host ""
    Write-Host (Get-SpacrInstallerMessage "installed") -ForegroundColor Green
    Write-Host (Get-SpacrInstallerMessage "launcher" @($Launcher))
} catch {
    $OriginalError = $_
    if (-not $InstallerStarted) { throw $OriginalError }
    $RuntimeRestored = $true
    if (-not $Committed -and $OldVenv -and (Test-Path $OldVenv)) {
        try {
            if (Test-Path -LiteralPath $VenvDir) {
                Remove-Item -LiteralPath $VenvDir -Recurse -Force -ErrorAction Stop
            }
            Move-Item -LiteralPath $OldVenv -Destination $VenvDir
        } catch {
            $RuntimeRestored = $false
            $RetainWorkDir = $true
            Write-Warning (Get-SpacrInstallerMessage "gui_failed")
            Write-Host $OldVenv
        }
    }
    if ($MetadataTouched -and $RuntimeRestored) {
        try {
            Invoke-Checked -Command $CleanupPython -Arguments @("-I", $CleanupModule,
                "installer-metadata", "restore", "--folder", $MetadataBackup)
        } catch {
            $RetainWorkDir = $true
            Write-Warning (Get-SpacrInstallerMessage "gui_failed")
            Write-Host $MetadataBackup
        }
    }
    if (Test-Path $StageVenv) {
        Remove-Item -Recurse -Force $StageVenv
    }
    Remove-Item -Force -ErrorAction SilentlyContinue $StageProfile
    throw $OriginalError
} finally {
    if ($InstallerStarted -and -not $RetainWorkDir) {
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $WorkDir
    }
    if ($TranscriptStarted) { Stop-Transcript -ErrorAction SilentlyContinue | Out-Null }
    for ($LeaseIndex = $InstallerLeaseHandles.Count - 1; $LeaseIndex -ge 0; $LeaseIndex--) {
        $Lease = $InstallerLeaseHandles[$LeaseIndex]
        if ($RetainWorkDir) {
            $Marker = [System.Text.Encoding]::UTF8.GetBytes("rollback=$WorkDir`n")
            $Lease.Handle.Seek(0, [System.IO.SeekOrigin]::End) | Out-Null
            $Lease.Handle.Write($Marker, 0, $Marker.Length)
            $Lease.Handle.Flush()
        }
        $Lease.Handle.Dispose()
        if (-not $RetainWorkDir) {
            Remove-Item -LiteralPath $Lease.Path -ErrorAction SilentlyContinue
        }
    }
}
