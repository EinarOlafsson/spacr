<#
.SYNOPSIS
Recover the original private runtime after the preserved released wrapper failed.
.PARAMETER OriginalEvidence
Existing original-failure and uv-replay evidence; never overwritten by recovery.
#>
[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$OriginalEvidence)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$recovery = Join-Path $OriginalEvidence 'bootstrap-recovery'
if (Test-Path -LiteralPath $recovery) { throw 'Bootstrap recovery evidence already exists.' }
New-Item -ItemType Directory -Path $recovery | Out-Null
$receipt = [ordered]@{
    status='failed'; purpose='Explicit external bootstrap recovery; original released wrapper remains failed; GUI updater acceptance is separate'
    original_wrapper_accepted=$false; gui_update_accepted=$false; frozen_update_accepted=$false
    started_utc=[DateTime]::UtcNow.ToString('o')
}
$child = $null
try {
    $observationPath = Join-Path $OriginalEvidence 'installer-observation.json'
    $capturePath = Join-Path $OriginalEvidence 'original-uv-child.json'
    $replayPath = Join-Path $OriginalEvidence 'uv-child-diagnostic\receipt.json'
    $observation = Get-Content -Raw -LiteralPath $observationPath | ConvertFrom-Json
    $capture = Get-Content -Raw -LiteralPath $capturePath | ConvertFrom-Json
    $replay = Get-Content -Raw -LiteralPath $replayPath | ConvertFrom-Json
    if ($observation.status -ne 'failed' -or -not $observation.observed_tree_exited -or
        -not $observation.bootstrap_gone -or -not $observation.transcript_ended -or
        $observation.failure_dialog.text -cnotmatch '^spaCR installation failed with exit code [1-9][0-9]*\. The existing installation, if any, was preserved\.$') {
        throw 'Explicit recovery requires the captured original failure and proven tree exit.'
    }
    if (-not $capture.available -or -not $capture.original_child_exited -or
        $null -eq $capture.original_child_exit_code -or $capture.original_child_exit_code -eq 0 -or $capture.expected_uv_present_before_diagnostic) {
        throw 'Original uv child failure is not established.'
    }
    if ($replay.status -ne 'captured' -or $replay.child_exit_code -ne 0 -or $replay.timed_out -or
        -not $replay.stdout_complete -or -not $replay.stderr_complete -or -not $replay.expected_uv_present) {
        throw 'The separate real uv replay did not complete successfully.'
    }
    $artifact = Join-Path $OriginalEvidence 'SpaCR-1.5.0.1-Windows-Online-Setup.exe'
    $bootstrap = Join-Path $OriginalEvidence 'original-bootstrap.ps1'
    $artifactHash = (Get-FileHash -LiteralPath $artifact -Algorithm SHA256).Hash.ToLowerInvariant()
    $bootstrapHash = (Get-FileHash -LiteralPath $bootstrap -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($artifactHash -ne '2b6b07ad12926288f693c01945f2fc4555b0a37022fd73dce932e2d8b6e4ee76' -or
        $observation.artifact_sha256 -ne $artifactHash -or
        $bootstrapHash -ne '94f05a5b8150f7491c7319eb3beeb62e8405b0fdf6a2d498f58313103ac7b28e' -or
        $capture.bootstrap_sha256 -ne $bootstrapHash) { throw 'Original artifact/bootstrap identity changed.' }
    $engine = [Diagnostics.Process]::GetCurrentProcess().MainModule.FileName
    $engineHash = (Get-FileHash -LiteralPath $engine -Algorithm SHA256).Hash.ToLowerInvariant()
    $account = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    if ($PSVersionTable.PSEdition -ne 'Desktop' -or $PSVersionTable.PSVersion.Major -ne 5 -or
        $engine -ine $capture.executable_path -or $engineHash -ne $capture.engine_sha256 -or
        $account -ne $capture.owner_sid) { throw 'Recovery requires the captured native engine and account.' }
    if ($env:SPACR_PACKAGE_SPEC) { throw 'Checkout/package overrides are forbidden for released recovery.' }
    if ($env:LOCALAPPDATA -cne $capture.environment.LOCALAPPDATA -or $env:TEMP -cne $capture.environment.TEMP) {
        throw 'Original private-root or temporary-directory context changed.'
    }
    $installRoot = Join-Path $env:LOCALAPPDATA 'SpaCR'
    $python = Join-Path $installRoot 'venv\Scripts\python.exe'
    $uv = Join-Path $installRoot 'bootstrap\uv.exe'
    if (Test-Path -LiteralPath (Join-Path $installRoot 'venv')) { throw 'Refusing to replace a private environment.' }
    if ((Get-FileHash -LiteralPath $uv -Algorithm SHA256).Hash.ToLowerInvariant() -ne $replay.uv_sha256) {
        throw 'The actual replay-installed uv changed.'
    }
    $cwd = Join-Path $env:TEMP 'spaCR-online-installer'
    if ($cwd -ine $capture.working_directory) { throw 'Original working-directory context changed.' }
    $receipt.original_artifact_sha256 = $artifactHash
    $receipt.bootstrap_sha256 = $bootstrapHash
    $receipt.original_failure_receipt_sha256 = (Get-FileHash -LiteralPath $observationPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $receipt.original_child_receipt_sha256 = (Get-FileHash -LiteralPath $capturePath -Algorithm SHA256).Hash.ToLowerInvariant()
    $receipt.uv_replay_receipt_sha256 = (Get-FileHash -LiteralPath $replayPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $receipt.driver_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $receipt.engine_sha256 = $engineHash
    $receipt.account_sid = $account
    $receipt.private_prefix = Join-Path $installRoot 'venv'
    $receipt.private_python = $python
    $receipt.cwd = $cwd
    $receipt.bootstrap_argv = @($capture.executable_path,'-NoProfile','-ExecutionPolicy','Bypass','-File',$bootstrap,'-InstallRoot',$installRoot,'-Version','1.5.0.1','-TorchBackend','cpu')
    New-Item -ItemType Directory -Force -Path $cwd | Out-Null
    $info = New-Object Diagnostics.ProcessStartInfo
    $info.FileName = $capture.executable_path
    $info.Arguments = '-NoProfile -ExecutionPolicy Bypass -File "' + $bootstrap + '" -InstallRoot "' + $installRoot + '" -Version 1.5.0.1 -TorchBackend cpu'
    $info.WorkingDirectory = $cwd
    $info.UseShellExecute = $false
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $child = New-Object Diagnostics.Process
    $child.StartInfo = $info
    if (-not $child.Start()) { throw 'Could not launch the unchanged original bootstrap.' }
    $receipt.child_pid = $child.Id
    $stdout = $child.StandardOutput.ReadToEndAsync()
    $stderr = $child.StandardError.ReadToEndAsync()
    $receipt.timed_out = -not $child.WaitForExit(2400000)
    if ($receipt.timed_out) {
        & "$env:WINDIR\System32\taskkill.exe" /PID $child.Id /T /F 2>&1 |
            Out-File -LiteralPath (Join-Path $recovery 'termination.txt') -Encoding utf8
        if (-not $child.WaitForExit(10000)) { throw 'Bootstrap tree did not exit after bounded termination.' }
    }
    $receipt.child_exit_code = $child.ExitCode
    $receipt.stdout_complete = $stdout.Wait(10000)
    $receipt.stderr_complete = $stderr.Wait(10000)
    if ($receipt.stdout_complete) { [IO.File]::WriteAllText((Join-Path $recovery 'stdout.txt'), $stdout.Result) }
    if ($receipt.stderr_complete) { [IO.File]::WriteAllText((Join-Path $recovery 'stderr.txt'), $stderr.Result) }
    if ($receipt.timed_out -or $child.ExitCode -ne 0 -or -not $receipt.stdout_complete -or -not $receipt.stderr_complete) {
        throw 'Actual original-bootstrap recovery did not complete.'
    }
    if (-not (Test-Path -LiteralPath $python) -or -not (Test-Path -LiteralPath (Join-Path $installRoot 'launch_spacr.pyw'))) {
        throw 'Recovered private runtime or original launcher is missing.'
    }
    $receipt.uv_sha256 = (Get-FileHash -LiteralPath $uv -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($receipt.uv_sha256 -ne $replay.uv_sha256) { throw 'Recovery replaced the replay-verified uv with different bytes.' }
    if ((Get-FileHash -LiteralPath $bootstrap -Algorithm SHA256).Hash.ToLowerInvariant() -ne $bootstrapHash) {
        throw 'Original bootstrap bytes changed during recovery.'
    }
    $receipt.status = 'passed'
} catch {
    $receipt.error = $_.Exception.ToString()
    throw
} finally {
    $installLog = Join-Path (Join-Path $env:LOCALAPPDATA 'SpaCR') 'install.log'
    if (Test-Path -LiteralPath $installLog) {
        try { Copy-Item -LiteralPath $installLog -Destination (Join-Path $recovery 'install.log') }
        catch { $receipt.install_log_copy_error = $_.Exception.Message }
    }
    $receipt.completed_utc = [DateTime]::UtcNow.ToString('o')
    $receipt | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath (Join-Path $recovery 'receipt.json') -Encoding utf8
    if ($child) { $child.Dispose() }
}
