param(
    [Parameter(Position=0)][string]$Label,
    [Parameter(Position=1, ValueFromRemainingArguments=$true)][string[]]$Command
)

$ErrorActionPreference = 'Stop'

function Test-SpacrGpuGraphicsCommand {
    param([string]$CommandLine)
    if (-not $CommandLine.Contains('--type=gpu-process')) { return $false }
    if (-not ('SpacrGpuCommandLine' -as [type])) {
        Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class SpacrGpuCommandLine {
    [DllImport("shell32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    public static extern IntPtr CommandLineToArgvW(string line, out int count);
    [DllImport("kernel32.dll")]
    public static extern IntPtr LocalFree(IntPtr memory);
}
'@
    }
    $count = 0
    $arguments = [SpacrGpuCommandLine]::CommandLineToArgvW($CommandLine, [ref]$count)
    if ($arguments -eq [IntPtr]::Zero) { throw 'GPU process arguments cannot be decoded' }
    try {
        for ($index = 1; $index -lt $count; $index++) {
            $pointer = [Runtime.InteropServices.Marshal]::ReadIntPtr($arguments, $index * [IntPtr]::Size)
            if ([Runtime.InteropServices.Marshal]::PtrToStringUni($pointer) -ceq '--type=gpu-process') { return $true }
        }
        return $false
    }
    finally { [void][SpacrGpuCommandLine]::LocalFree($arguments) }
}

function Select-SpacrGpuActivity {
    param([object[]]$Samples, [object[]]$Processes)
    if (-not $Samples -or -not $Processes) { throw 'GPU activity/process enumeration returned no records' }
    $processesById = @{}
    $excluded = [Collections.Generic.HashSet[int]]::new()
    $browserTree = [Collections.Generic.HashSet[int]]::new()
    $browserNames = @('chrome.exe', 'chromium.exe', 'chromium-browser.exe', 'google-chrome.exe')
    $desktopNames = @('dwm.exe', 'csrss.exe', 'explorer.exe', 'ApplicationFrameHost.exe',
        'ShellExperienceHost.exe', 'StartMenuExperienceHost.exe', 'SearchHost.exe', 'TextInputHost.exe')
    $windowsRoot = [Environment]::GetFolderPath('Windows').TrimEnd('\') + '\'
    foreach ($process in $Processes) {
        $processesById[[int]$process.ProcessId] = $process
        $browser = $browserNames -contains $process.Name
        $graphics = Test-SpacrGpuGraphicsCommand ([string]$process.CommandLine)
        $desktop = $false
        if (($desktopNames -contains $process.Name) -and $process.ExecutablePath) {
            $executable = [IO.Path]::GetFullPath($process.ExecutablePath)
            $desktop = $executable.Equals(($windowsRoot + 'System32\' + $process.Name), [StringComparison]::OrdinalIgnoreCase) -or
                ($process.Name -eq 'explorer.exe' -and $executable.Equals(($windowsRoot + 'explorer.exe'), [StringComparison]::OrdinalIgnoreCase)) -or
                $executable.StartsWith(($windowsRoot + 'SystemApps\'), [StringComparison]::OrdinalIgnoreCase)
        }
        if ($browser -or $graphics -or $desktop) { [void]$excluded.Add([int]$process.ProcessId) }
        if ($browser -or $graphics) { [void]$browserTree.Add([int]$process.ProcessId) }
    }
    do {
        $changed = $false
        foreach ($process in $Processes) {
            $parent = $processesById[[int]$process.ParentProcessId]
            if ($parent -and $browserTree.Contains([int]$parent.ProcessId) -and
                    $parent.CreationDate -and $process.CreationDate -and
                    $parent.CreationDate -le $process.CreationDate) {
                [void]$excluded.Add([int]$process.ProcessId)
                if ($browserTree.Add([int]$process.ProcessId)) { $changed = $true }
            }
        }
    } while ($changed)
    $busy = [Collections.Generic.HashSet[int]]::new()
    foreach ($sample in $Samples) {
        $value = [double]$sample.CookedValue
        if ($sample.Status -ne 0 -or [double]::IsNaN($value) -or [double]::IsInfinity($value) -or $value -lt 0) {
            throw 'GPU engine counter returned an invalid sample'
        }
        if ([string]$sample.InstanceName -notmatch '^pid_(\d+)_') { throw 'GPU engine counter has no process identity' }
        $identity = [int]$Matches[1]
        if ($value -gt 0 -and -not $excluded.Contains($identity)) { [void]$busy.Add($identity) }
    }
    [pscustomobject]@{ BusyPids = @($busy | Sort-Object); ExcludedPids = @($excluded | Sort-Object) }
}

function Get-SpacrGpuActivity {
    $readings = @(Get-Counter -Counter '\GPU Engine(*)\Utilization Percentage' -SampleInterval 1 -MaxSamples 2 -ErrorAction Stop)
    $samples = @($readings | ForEach-Object { $_.CounterSamples })
    $processes = @(Get-CimInstance Win32_Process -ErrorAction Stop)
    Select-SpacrGpuActivity -Samples $samples -Processes $processes
}

function Get-SpacrGpuMutexName {
    param([string]$Directory)
    $canonical = [IO.Path]::GetFullPath($Directory).TrimEnd('\').ToLowerInvariant()
    $hash = [Security.Cryptography.SHA256]::Create()
    try { $digest = [BitConverter]::ToString($hash.ComputeHash([Text.Encoding]::UTF8.GetBytes($canonical))).Replace('-', '') }
    finally { $hash.Dispose() }
    'Global\spaCR-GPU-' + $digest
}

function Test-SpacrGpuAdmission {
    param([int[]]$BusyPids, [double]$IdleElapsed, [double]$GapElapsed,
        [int]$IdleSeconds = 360, [int]$GapSeconds = 600)
    -not $BusyPids.Count -and $IdleElapsed -ge $IdleSeconds -and $GapElapsed -ge $GapSeconds
}

function Invoke-SpacrGpuTurn {
    param([string]$TurnLabel, [string[]]$TurnCommand,
        [string]$Directory = (Join-Path $env:ProgramData 'spaCR\gpu'),
        [int]$IdleSeconds = 360, [int]$GapSeconds = 600, [int]$PollSeconds = 15)
    if (-not $TurnLabel -or -not $TurnCommand.Count) { throw 'usage: gpu_turn.ps1 <label> <command...>' }
    if ($IdleSeconds -lt 0 -or $GapSeconds -lt 0 -or $PollSeconds -lt 1) { throw 'Invalid GPU queue timing' }
    [void][IO.Directory]::CreateDirectory($Directory)
    $mutex = [Threading.Mutex]::new($false, (Get-SpacrGpuMutexName $Directory))
    $held = $false
    $started = $false
    $release = Join-Path $Directory 'last_release'
    $holder = Join-Path $Directory 'holder'
    $log = Join-Path $Directory 'log'
    function Write-TurnLog([string]$Message) {
        $line = '{0:o} [{1}] {2}' -f [DateTime]::UtcNow, $TurnLabel, $Message
        [IO.File]::AppendAllText($log, $line + [Environment]::NewLine)
        [Console]::Error.WriteLine($line)
    }
    try {
        Write-TurnLog "queued (pid $PID, $(Get-Location))"
        try { $held = $mutex.WaitOne() }
        catch [Threading.AbandonedMutexException] {
            $held = $true
            [IO.File]::WriteAllText($release, [DateTimeOffset]::UtcNow.ToUnixTimeSeconds().ToString())
        }
        Write-TurnLog "holds the queue; waiting for ${IdleSeconds}s idle and ${GapSeconds}s since the last turn"
        $idle = [Diagnostics.Stopwatch]::StartNew()
        while ($true) {
            $activity = Get-SpacrGpuActivity
            if ($activity.BusyPids.Count) { $idle.Restart() }
            $last = 0L
            if ([IO.File]::Exists($release)) {
                if (-not [long]::TryParse([IO.File]::ReadAllText($release).Trim(), [ref]$last)) { throw 'GPU queue release time is unreadable' }
            }
            $gap = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds() - $last
            if (Test-SpacrGpuAdmission $activity.BusyPids $idle.Elapsed.TotalSeconds $gap $IdleSeconds $GapSeconds) { break }
            Start-Sleep -Seconds $PollSeconds
        }
        [IO.File]::WriteAllText($holder, "label=$TurnLabel pid=$PID start=$([DateTime]::UtcNow.ToString('o')) checkout=$(Get-Location)")
        $started = $true
        Write-TurnLog ('START: ' + ($TurnCommand -join ' '))
        $arguments = @($TurnCommand | Select-Object -Skip 1)
        $global:LASTEXITCODE = 0
        & $TurnCommand[0] @arguments | Out-Host
        $succeeded = $?
        $result = $LASTEXITCODE
        if (-not $succeeded -and $result -eq 0) { $result = 1 }
        Write-TurnLog "FINISH rc=$result"
        return $result
    }
    finally {
        if ($held) {
            if ($started) { [IO.File]::WriteAllText($release, [DateTimeOffset]::UtcNow.ToUnixTimeSeconds().ToString()) }
            [IO.File]::WriteAllText($holder, '')
            $mutex.ReleaseMutex()
        }
        $mutex.Dispose()
    }
}

if ($MyInvocation.InvocationName -ne '.') {
    try {
        $idleSeconds = if ($env:GPU_IDLE_S) { [int]$env:GPU_IDLE_S } else { 360 }
        $gapSeconds = if ($env:GPU_GAP_S) { [int]$env:GPU_GAP_S } else { 600 }
        $pollSeconds = if ($env:GPU_POLL_S) { [int]$env:GPU_POLL_S } else { 15 }
        exit (Invoke-SpacrGpuTurn -TurnLabel $Label -TurnCommand $Command -IdleSeconds $idleSeconds -GapSeconds $gapSeconds -PollSeconds $pollSeconds)
    }
    catch { [Console]::Error.WriteLine($_.ToString()); exit 1 }
}
