param(
    [Parameter(Mandatory=$true)][string]$OldRoot,
    [Parameter(Mandatory=$true)][string]$NewRoot,
    [Parameter(Mandatory=$true)][string]$Output,
    [int]$InterruptAfterSeconds = 0
)
$ErrorActionPreference = 'Stop'
if ($env:OS -ne 'Windows_NT') { throw 'Windows acceptance requires Windows.' }
New-Item -ItemType Directory -Force $Output | Out-Null
$Output = (Resolve-Path $Output).Path

function Get-VerifiedInstaller([string]$Root, [string]$Version) {
    $Root = (Resolve-Path $Root).Path
    $Checksums = Join-Path $Root 'acceptance/SHA256SUMS'
    if (-not (Test-Path $Checksums -PathType Leaf)) { throw "Missing $Checksums" }
    foreach ($Line in Get-Content $Checksums) {
        if ($Line -notmatch '^([0-9a-fA-F]{64})  (.+)$') { throw 'Malformed artifact checksum row.' }
        $Path = Join-Path (Join-Path $Root 'dist') $Matches[2]
        if (-not (Test-Path $Path -PathType Leaf)) { throw "Missing artifact $Path" }
        if ((Get-FileHash $Path -Algorithm SHA256).Hash -ne $Matches[1]) { throw "Artifact checksum mismatch: $Path" }
    }
    $Installers = @(Get-ChildItem (Join-Path $Root 'dist') -Filter "spaCR-$Version-setup.exe" -File)
    if ($Installers.Count -ne 1) { throw "Expected exactly one spaCR $Version installer." }
    if ($Version -eq '*' -and $Installers[0].Name -eq 'spaCR-1.5.1.0-setup.exe') { throw 'The new root holds the old release.' }
    return $Installers[0]
}

function Invoke-Installer([string]$Path, [string]$Destination) {
    $Process = Start-Process $Path -ArgumentList @('/S', "/D=$Destination") -Wait -PassThru
    if ($Process.ExitCode -ne 0) { throw "Installer exited $($Process.ExitCode): $Path" }
    if (-not (Test-Path (Join-Path $Destination 'spacr.exe') -PathType Leaf)) { throw 'Installer wrote no executable.' }
}

function Invoke-FrozenSmoke([string]$Destination, [string]$Label, [string]$Commit, [string]$Receipt) {
    $env:SPACR_BENCHMARK_JSON = Join-Path $Receipt "$Label-smoke.json"
    $env:SPACR_ACCEPTANCE_SOURCE_COMMIT = $Commit
    $env:APPDATA = Join-Path $Receipt 'profile/roaming'
    $env:LOCALAPPDATA = Join-Path $Receipt 'profile/local'
    New-Item -ItemType Directory -Force $env:APPDATA,$env:LOCALAPPDATA | Out-Null
    $env:PYTHONHOME = ''
    $env:PATH = "$env:SystemRoot\system32;$env:SystemRoot"
    $Clean = Join-Path $Receipt 'empty-cwd'
    New-Item -ItemType Directory -Force $Clean | Out-Null
    $Process = Start-Process (Join-Path $Destination 'spacr.exe') -WorkingDirectory $Clean -ArgumentList '--no-setup' -PassThru
    if (-not $Process.WaitForExit(660000)) {
        & "$env:SystemRoot\system32\taskkill.exe" /PID $Process.Id /T /F | Out-Null
        throw "$Label frozen application timed out."
    }
    if ($Process.ExitCode -ne 0) { throw "$Label frozen application exited $($Process.ExitCode)." }
    $Result = Get-Content $env:SPACR_BENCHMARK_JSON -Raw | ConvertFrom-Json
    if ($Result.status -ne 'passed' -or -not $Result.frozen) { throw "$Label frozen Measure smoke failed." }
    return $Result
}

$Old = Get-VerifiedInstaller $OldRoot '1.5.1.0'
$New = Get-VerifiedInstaller $NewRoot '*'
$OldCommit = (Get-Content (Join-Path (Resolve-Path $OldRoot).Path 'acceptance/source-commit.txt') -Raw).Trim()
$NewCommit = (Get-Content (Join-Path (Resolve-Path $NewRoot).Path 'acceptance/source-commit.txt') -Raw).Trim()
$Install = Join-Path $env:RUNNER_TEMP 'frozen-in-place-upgrade'
Invoke-Installer $Old.FullName $Install
$Before = Invoke-FrozenSmoke $Install 'before' $OldCommit $Output
$Marker = Join-Path $Install 'preserve-user-note.txt'
Set-Content $Marker 'not installed by NSIS'
$DatabaseHash = (Get-FileHash $Before.database -Algorithm SHA256).Hash
$OldRuntime = @(Get-ChildItem (Join-Path $Install '_internal') -Filter 'imageio-*.dist-info' -Directory | Select-Object -ExpandProperty Name)
if ($OldRuntime.Count -ne 1) { throw 'Old install has an unexpected imageio runtime.' }
Invoke-Installer $New.FullName $Install
if ((Get-Content $Marker -Raw).Trim() -ne 'not installed by NSIS') { throw 'Upgrade changed an unrelated file.' }
if ((Get-FileHash $Before.database -Algorithm SHA256).Hash -ne $DatabaseHash) { throw 'Upgrade changed prior analysis.' }
$NewRuntime = @(Get-ChildItem (Join-Path $Install '_internal') -Filter 'imageio-*.dist-info' -Directory | Select-Object -ExpandProperty Name)
$StaleRuntime = ($NewRuntime.Count -ne 1 -or $NewRuntime -contains $OldRuntime[0])
$After = Invoke-FrozenSmoke $Install 'after' $NewCommit (Join-Path $Output 'after')
$Receipt = [ordered]@{
    schema = 1
    previous_run = $env:PREVIOUS_RUN
    current_run = $env:CURRENT_RUN
    previous_commit = $OldCommit
    current_commit = $NewCommit
    windows = [Environment]::OSVersion.VersionString
    old_installer_sha256 = (Get-FileHash $Old.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    new_installer_sha256 = (Get-FileHash $New.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    old_imageio = $OldRuntime[0]
    new_installer = $New.Name
    new_imageio = ($NewRuntime -join ',')
    stale_runtime_after_upgrade = $StaleRuntime
    prior_analysis_unchanged_on_upgrade = $true
    unrelated_file_preserved = $true
    before_smoke = $Before.status
    after_smoke = $After.status
}
if ($InterruptAfterSeconds -gt 0) {
    # Interruption: kill the 1.5.1.1 installer mid-copy over 1.5.1.0, try a launch, rerun, relaunch.
    $Interrupted = Join-Path $env:RUNNER_TEMP 'frozen-interrupted-upgrade'
    Invoke-Installer $Old.FullName $Interrupted
    $Killed = Start-Process $New.FullName -ArgumentList @('/S', "/D=$Interrupted") -PassThru
    Start-Sleep -Seconds $InterruptAfterSeconds
    $StillRunning = -not $Killed.HasExited
    & "$env:SystemRoot\system32\taskkill.exe" /PID $Killed.Id /T /F | Out-Null
    Start-Sleep -Seconds 3
    $Partial = @(Get-ChildItem (Join-Path $Interrupted '_internal') -Filter 'imageio-*.dist-info' -Directory -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name)
    $Midway = 'not-attempted'
    if (Test-Path (Join-Path $Interrupted 'spacr.exe')) {
        try { $Midway = (Invoke-FrozenSmoke $Interrupted 'interrupted' $NewCommit (Join-Path $Output 'interrupted')).status }
        catch { $Midway = "failed: $($_.Exception.Message)" }
    } else { $Midway = 'no-executable' }
    Invoke-Installer $New.FullName $Interrupted
    $Recovered = Invoke-FrozenSmoke $Interrupted 'recovered' $NewCommit (Join-Path $Output 'recovered')
    $RecoveredRuntime = @(Get-ChildItem (Join-Path $Interrupted '_internal') -Filter 'imageio-*.dist-info' -Directory | Select-Object -ExpandProperty Name)
    $Receipt.interrupt_after_seconds = $InterruptAfterSeconds
    $Receipt.installer_running_when_killed = $StillRunning
    $Receipt.imageio_after_kill = ($Partial -join ',')
    $Receipt.launch_after_kill = $Midway
    $Receipt.rerun_imageio = ($RecoveredRuntime -join ',')
    $Receipt.recovered_smoke = $Recovered.status
}
$Receipt | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $Output 'upgrade.json') -Encoding utf8
if ($StaleRuntime) { throw "Old imageio runtime survived the in-place upgrade: $($NewRuntime -join ',')" }
