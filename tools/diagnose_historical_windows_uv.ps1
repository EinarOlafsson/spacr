<#
.SYNOPSIS
Replay only the retained uv child after the authentic historical installer failed.
.PARAMETER CapturePath
Passive original-child receipt beside its retained uv and bootstrap scripts.
.PARAMETER EvidenceDirectory
New directory for diagnostic output and receipt; never an acceptance result.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$CapturePath,
    [Parameter(Mandatory=$true)][string]$EvidenceDirectory
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
if (Test-Path -LiteralPath $EvidenceDirectory) { throw 'Diagnostic evidence already exists.' }
New-Item -ItemType Directory -Path $EvidenceDirectory | Out-Null
$receipt = [ordered]@{purpose='diagnostic only; original installer acceptance remains failed'; status='preparing'; started_utc=[DateTime]::UtcNow.ToString('o')}
$child = $null
$oldUnmanaged = $env:UV_UNMANAGED_INSTALL
$oldNoModify = $env:UV_NO_MODIFY_PATH
try {
    $capture = Get-Content -Raw -LiteralPath $CapturePath | ConvertFrom-Json
    if (-not $capture.available -or -not $capture.original_child_exited) { throw 'Original child identity and exit were not captured.' }
    $originalEvidence = Split-Path -Parent $CapturePath
    $originalBootstrap = Join-Path $originalEvidence 'original-bootstrap.ps1'
    $retainedUv = Join-Path $originalEvidence 'original-uv-installer.ps1'
    $bootstrapHash = (Get-FileHash -LiteralPath $originalBootstrap -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($bootstrapHash -ne '94f05a5b8150f7491c7319eb3beeb62e8405b0fdf6a2d498f58313103ac7b28e') { throw 'Extracted historical bootstrap bytes differ.' }
    $uvScriptHash = (Get-FileHash -LiteralPath $retainedUv -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($uvScriptHash -ne $capture.script_sha256) { throw 'Retained original uv script bytes changed.' }
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    if ($identity.User.Value -ne $capture.owner_sid) { throw 'Diagnostic account differs from the original child.' }
    $engine = [Diagnostics.Process]::GetCurrentProcess().MainModule.FileName
    if ($PSVersionTable.PSEdition -ne 'Desktop' -or $PSVersionTable.PSVersion.Major -ne 5 -or $engine -ine $capture.executable_path) { throw 'Diagnostic engine differs from the original Windows PowerShell child.' }
    $engineHash = (Get-FileHash -LiteralPath $engine -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($engineHash -ne $capture.engine_sha256) { throw 'Original child engine bytes changed.' }
    $receipt.account_sid = $identity.User.Value
    $receipt.engine = $engine
    $receipt.engine_sha256 = $engineHash
    $receipt.is_64_bit_process = [Environment]::Is64BitProcess
    $receipt.ps_version = $PSVersionTable.PSVersion.ToString()
    $receipt.original_child_exit_code = $capture.original_child_exit_code
    $receipt.original_expected_uv_present = $capture.expected_uv_present_before_diagnostic
    $receipt.bootstrap_sha256 = $bootstrapHash
    $receipt.script_sha256 = $uvScriptHash
    $receipt.script_provenance = 'Exact retained original-attempt bytes; no download or script modification.'
    $installRoot = Join-Path $env:LOCALAPPDATA 'SpaCR'
    $bootstrapDir = Join-Path $installRoot 'bootstrap'
    $uvExe = Join-Path $bootstrapDir 'uv.exe'
    if ((Test-Path -LiteralPath $uvExe) -or (Test-Path -LiteralPath (Join-Path $installRoot 'venv'))) { throw 'Refusing to overwrite an installed uv or private environment.' }
    $installerScript = [string]$capture.original_script_path
    $expectedScript = Join-Path $env:TEMP ('spacr-uv-installer-' + $capture.bootstrap_pid + '.ps1')
    $workingDirectory = Join-Path $env:TEMP 'spaCR-online-installer'
    if ($installerScript -ine $expectedScript -or $capture.working_directory -ine $workingDirectory) { throw 'Original temporary path context changed.' }
    if (Test-Path -LiteralPath $installerScript) { throw 'Original temporary uv script is still present; do not overwrite it.' }
    $env:UV_UNMANAGED_INSTALL = $bootstrapDir
    $env:UV_NO_MODIFY_PATH = '1'
    $allowed = @('PATH','PATHEXT','TEMP','TMP','USERPROFILE','LOCALAPPDATA','UV_INSTALL_DIR','CARGO_DIST_FORCE_INSTALL_DIR','CARGO_HOME','XDG_BIN_HOME','XDG_DATA_HOME','XDG_CONFIG_HOME','UV_UNMANAGED_INSTALL','UV_NO_MODIFY_PATH')
    $properties = @($capture.environment.PSObject.Properties)
    if (@(Compare-Object ($allowed | Sort-Object) ($properties.Name | Sort-Object)).Count -ne 0) { throw 'Original environment receipt has an unexpected field set.' }
    $receipt.environment_provenance = $capture.environment_provenance
    $receipt.environment = [ordered]@{}
    $mismatches = @()
    foreach ($name in $allowed) {
        $value = [Environment]::GetEnvironmentVariable($name)
        $receipt.environment[$name] = $value
        if ($value -cne $capture.environment.$name) { $mismatches += $name }
    }
    if ($mismatches.Count) { throw ('Original inherited environment changed: ' + ($mismatches -join ', ')) }
    New-Item -ItemType Directory -Force -Path $bootstrapDir,$workingDirectory | Out-Null
    Copy-Item -LiteralPath $retainedUv -Destination $installerScript
    $receipt.script_path = $installerScript
    $receipt.cwd = $workingDirectory
    $info = New-Object Diagnostics.ProcessStartInfo
    $info.FileName = $engine
    $info.Arguments = '-NoProfile -ExecutionPolicy Bypass -File "' + $installerScript + '"'
    $info.WorkingDirectory = $workingDirectory
    $info.UseShellExecute = $false
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $receipt.child_arguments = $info.Arguments
    $child = New-Object Diagnostics.Process
    $child.StartInfo = $info
    if (-not $child.Start()) { throw 'Could not launch the original child engine.' }
    $receipt.child_pid = $child.Id
    $stdout = $child.StandardOutput.ReadToEndAsync()
    $stderr = $child.StandardError.ReadToEndAsync()
    $receipt.timed_out = -not $child.WaitForExit(120000)
    if ($receipt.timed_out) {
        & "$env:WINDIR\System32\taskkill.exe" /PID $child.Id /T /F 2>&1 |
            Out-File -LiteralPath (Join-Path $EvidenceDirectory 'termination.txt') -Encoding utf8
        if (-not $child.WaitForExit(10000)) { throw 'Diagnostic child did not exit after bounded tree termination.' }
    }
    $receipt.child_exit_code = $child.ExitCode
    $receipt.stdout_complete = $stdout.Wait(5000)
    $receipt.stderr_complete = $stderr.Wait(5000)
    if ($receipt.stdout_complete) { [IO.File]::WriteAllText((Join-Path $EvidenceDirectory 'stdout.txt'), $stdout.Result) }
    if ($receipt.stderr_complete) { [IO.File]::WriteAllText((Join-Path $EvidenceDirectory 'stderr.txt'), $stderr.Result) }
    if (-not $receipt.stdout_complete -or -not $receipt.stderr_complete) { throw 'Child stream capture did not finish within its deadline.' }
    $receipt.expected_uv_present = Test-Path -LiteralPath $uvExe
    if ($receipt.expected_uv_present) { $receipt.uv_sha256 = (Get-FileHash -LiteralPath $uvExe -Algorithm SHA256).Hash.ToLowerInvariant() }
    $receipt.status = 'captured'
} catch {
    $receipt.status = 'diagnostic_error'
    $receipt.error = $_.Exception.ToString()
    throw
} finally {
    $env:UV_UNMANAGED_INSTALL = $oldUnmanaged
    $env:UV_NO_MODIFY_PATH = $oldNoModify
    $receipt.completed_utc = [DateTime]::UtcNow.ToString('o')
    $receipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $EvidenceDirectory 'receipt.json') -Encoding utf8
    if ($child) { $child.Dispose() }
}
