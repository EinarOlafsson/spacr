param(
    [Parameter(Mandatory=$true)][string]$NativeReceipt,
    [Parameter(Mandatory=$true)][ValidateSet('cmd','powershell51')][string]$Shell,
    [Parameter(Mandatory=$true)][string]$Output
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Output = [IO.Path]::GetFullPath($Output)
New-Item -ItemType Directory -Force $Output | Out-Null
$Proof = Get-Content $NativeReceipt -Raw | ConvertFrom-Json
if ($Proof.status -ne 'passed' -or -not $Proof.frozen -or $Proof.qt_platform -ne 'windows') {
    throw 'A passed real Windows artifact receipt is required'
}
if ($Proof.source_commit -ne $env:GITHUB_SHA) { throw 'Artifact receipt belongs to another commit' }
$Hint = [string]$Proof.claude_install_hint
$PathSetup = @'
try{if(-not([IO.File]::Exists([IO.Path]::Combine([Environment]::GetFolderPath('UserProfile'),'.local\bin\claude.exe')))){throw('Claude_native_executable_missing')}if(-not([Environment]::ExpandEnvironmentVariables([string][Environment]::GetEnvironmentVariable('Path','User')).ToLowerInvariant().Split(';').Contains([Environment]::GetFolderPath('UserProfile').ToLowerInvariant()+'\.local\bin'))){[Environment]::SetEnvironmentVariable('Path',([string][Environment]::GetEnvironmentVariable('Path','User'))+';'+[Environment]::GetFolderPath('UserProfile')+'\.local\bin','User')}}catch{[Console]::Error.WriteLine('Claude_PATH_registration_failed');exit(1)}
'@
$Expected = 'cmd /c "curl -fsSL https://claude.ai/install.cmd -o install.cmd && install.cmd && del install.cmd && powershell.exe -NoProfile -NonInteractive -Command ' + $PathSetup.Replace('(', '^(').Replace(')', '^)') + '"'
if ($Hint -cne $Expected) { throw 'Displayed installer changed; review current vendor documentation before execution' }
$Report = [ordered]@{
    status = 'running'; shell = $Shell; command = $Hint
    source_commit = $Proof.source_commit
    native_receipt_sha256 = (Get-FileHash $NativeReceipt -Algorithm SHA256).Hash.ToLowerInvariant()
    vendor_documentation = 'https://code.claude.com/docs/en/setup'
    path_documentation = 'https://code.claude.com/docs/en/troubleshoot-install#verify-your-path'
    path_registration_scope = 'The artifact-provided command registers the native directory in User PATH; this verifier never repairs PATH.'
    node_scope = 'No node or npm command available to the installer or new terminal; hosted runner tooling outside this PATH is not asserted absent.'
    authenticated = $false
}
$InstallProfile = [Environment]::GetFolderPath('UserProfile')
if (-not $InstallProfile -or -not [IO.Path]::GetFullPath($env:USERPROFILE).Equals(
    [IO.Path]::GetFullPath($InstallProfile), [StringComparison]::OrdinalIgnoreCase)) {
    throw 'The runner USERPROFILE must match its actual Windows profile'
}
$env:HOME = $InstallProfile
$Report.profile = $InstallProfile
$Report.profile_scope = 'Actual OS profile on a fresh hosted VM; no USERPROFILE-only redirection.'
$SystemPath = "$env:SystemRoot\system32;$env:SystemRoot;$env:SystemRoot\System32\WindowsPowerShell\v1.0"
$env:PATH = $SystemPath
$env:NODE_OPTIONS = ''
$env:NPM_CONFIG_PREFIX = ''
Remove-Item Env:ANTHROPIC_API_KEY -ErrorAction SilentlyContinue
Remove-Item Env:CLAUDE_CODE_OAUTH_TOKEN -ErrorAction SilentlyContinue
$WindowsPowerShell = Join-Path $env:SystemRoot 'System32/WindowsPowerShell/v1.0/powershell.exe'
$Cmd = Join-Path $env:SystemRoot 'System32/cmd.exe'
$BeforeUserPath = [Environment]::GetEnvironmentVariable('Path','User')
try {
    foreach ($Existing in @('.local/bin/claude.exe', '.local/share/claude', '.claude/downloads')) {
        if (Test-Path (Join-Path $InstallProfile $Existing)) {
            throw "Fresh-host precondition failed: existing Claude installation data at $Existing"
        }
    }
    if ((Get-Command node,npm,claude -ErrorAction SilentlyContinue)) {
        throw 'Node, npm or an existing Claude executable is visible before installation'
    }
    $Report.curl = (Get-Command curl.exe -CommandType Application).Source
    $Report.node_unavailable_before = $true
    if ($Shell -eq 'cmd') {
        $Script = Join-Path $Output 'pasted-command.cmd'
        [IO.File]::WriteAllText($Script, $Hint + "`r`n", [Text.Encoding]::ASCII)
        & $Cmd /d /c $Script 2>&1 | Tee-Object (Join-Path $Output 'install.log')
    } else {
        $Version = & $WindowsPowerShell -NoProfile -Command '$PSVersionTable.PSVersion.ToString()'
        if ($Version -notlike '5.1.*') { throw 'Expected actual Windows PowerShell 5.1' }
        $Report.powershell_version = [string]$Version
        $Script = Join-Path $Output 'pasted-command.ps1'
        [IO.File]::WriteAllText($Script, $Hint + "`r`nexit `$LASTEXITCODE`r`n", [Text.Encoding]::ASCII)
        & $WindowsPowerShell -NoProfile -ExecutionPolicy Bypass -File $Script 2>&1 | Tee-Object (Join-Path $Output 'install.log')
    }
    $Report.install_exit_code = $LASTEXITCODE
    if ($LASTEXITCODE -ne 0) { throw 'Exact displayed command failed in the selected shell' }
    if (Test-Path (Join-Path (Get-Location) 'install.cmd')) { throw 'The pasted command did not remove its downloaded script' }
    $FreshPaths = @($SystemPath)
    foreach ($Scope in @('Machine','User')) {
        foreach ($Entry in ([Environment]::GetEnvironmentVariable('Path',$Scope) -split ';')) {
            if (-not $Entry) { continue }
            $Expanded = [Environment]::ExpandEnvironmentVariables($Entry)
            if ((Test-Path (Join-Path $Expanded 'node.exe')) -or
                (Test-Path (Join-Path $Expanded 'npm.cmd')) -or
                (Test-Path (Join-Path $Expanded 'npm.ps1'))) { continue }
            $FreshPaths += $Expanded
        }
    }
    $env:PATH = ($FreshPaths | Select-Object -Unique) -join ';'
    $Report.user_path_changed_by_installer = ([Environment]::GetEnvironmentVariable('Path','User') -ne $BeforeUserPath)
    if (Get-Command node,npm -ErrorAction SilentlyContinue) { throw 'Node/npm became available to the fresh terminal' }
    $NativeExecutable = Join-Path $InstallProfile '.local/bin/claude.exe'
    $Report.native_executable_exists = Test-Path -LiteralPath $NativeExecutable -PathType Leaf
    if (-not $Report.native_executable_exists) { throw 'The native Claude executable was not installed in the actual OS profile' }
    $Report.registered_user_path = [Environment]::GetEnvironmentVariable('Path','User')
    $Executable = (Get-Command claude -CommandType Application -ErrorAction Stop).Source
    if (-not [IO.Path]::GetFullPath($Executable).Equals([IO.Path]::GetFullPath($NativeExecutable), [StringComparison]::OrdinalIgnoreCase)) {
        throw 'New terminal did not resolve the exact native Claude executable'
    }
    if ($Shell -eq 'cmd') {
        $Answer = & $Cmd /d /c 'claude --version' 2>&1
    } else {
        $Answer = & $WindowsPowerShell -NoProfile -Command 'claude --version; exit $LASTEXITCODE' 2>&1
    }
    $Report.version_exit_code = $LASTEXITCODE
    $Report.version_output = ($Answer | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $Report.version_output -notmatch '\d+\.\d+\.\d+') {
        throw 'Claude did not report a version in a new terminal'
    }
    $Report.executable = $Executable
    $Report.executable_sha256 = (Get-FileHash $Executable -Algorithm SHA256).Hash.ToLowerInvariant()
    $Report.node_unavailable_after = $true
    $Report.status = 'passed'
} catch {
    $Report.status = 'failed'
    $Report.error = $_.Exception.Message
    throw
} finally {
    $Report | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $Output 'acceptance.json') -Encoding utf8
    [Environment]::SetEnvironmentVariable('Path',$BeforeUserPath,'User')
}
