param()

# Concurrent jobs (maintainer, 2026-11-18): `gpu_turn.ps1 --vram-gb N <label> <command...>`
# skips the exclusive queue, the idle wait and the gap, and starts once free VRAM
# minus the needs other declared jobs announced in the last GPU_RESERVE_S seconds
# (120) is at least N + GPU_VRAM_MARGIN_GB (2) GB, the GPU is below GPU_MAX_TEMP_C
# (80) C and there is no ram_guard ALERT; it refuses (exit 3) unless power.limit is
# GPU_POWER_LIMIT_W (280) W. Without --vram-gb the exclusive turn is unchanged.
$VramGb = $null
$rest = @($args)
if ($rest.Count -ge 2 -and ([string]$rest[0] -eq '--vram-gb' -or [string]$rest[0] -eq '-VramGb')) {
    $parsed = 0.0
    if (-not [double]::TryParse([string]$rest[1], [Globalization.NumberStyles]::Float,
            [Globalization.CultureInfo]::InvariantCulture, [ref]$parsed) -or $parsed -lt 0) {
        [Console]::Error.WriteLine('usage: gpu_turn.ps1 [--vram-gb N] <label> <command...> (N in GB)'); exit 2
    }
    $VramGb = $parsed
    $rest = @($rest | Select-Object -Skip 2)
}
$Label = if ($rest.Count) { [string]$rest[0] } else { '' }
[string[]]$Command = @($rest | Select-Object -Skip 1)

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

# A GPU engine instance can vanish between the two samples when its process
# exits, and Get-Counter then reports invalid data for that sample. That is
# not a reading of idleness, so the waiting turn counts it as activity and
# asks again; any other failure, or a run of these, still refuses the turn.
function Test-SpacrGpuTransientSample {
    param($ErrorRecord)
    $text = [string]$ErrorRecord
    $text.Contains('performance counter samples is not valid') -or $text.Contains('GPU engine counter returned an invalid sample')
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
        [int]$IdleSeconds = 360, [int]$GapSeconds = 600, [int]$PollSeconds = 15,
        [int]$TransientLimit = 40)
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
        $unreadable = 0
        while ($true) {
            try {
                $activity = Get-SpacrGpuActivity
                $unreadable = 0
            }
            catch {
                if (-not (Test-SpacrGpuTransientSample $_) -or ++$unreadable -gt $TransientLimit) { throw }
                if ($unreadable -eq 1) { Write-TurnLog 'a GPU counter sample was unreadable; counting it as activity and waiting' }
                $idle.Restart()
                Start-Sleep -Seconds $PollSeconds
                continue
            }
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

function Get-SpacrGpuReading {
    param([string]$Query)
    $text = & nvidia-smi "--query-gpu=$Query" '--format=csv,noheader,nounits' 2>$null | Select-Object -First 1
    $value = 0.0
    if ($null -ne $text -and [double]::TryParse(([string]$text).Trim(), [Globalization.NumberStyles]::Float,
            [Globalization.CultureInfo]::InvariantCulture, [ref]$value)) { return $value }
    return $null
}

function Test-SpacrGpuFits {
    param([double]$NeedGb, $FreeMib, $TemperatureC, [double]$ReservedGb, [bool]$Alert,
        [double]$MarginGb = 2, [double]$MaxTemperatureC = 80)
    if ($Alert -or $null -eq $FreeMib -or $null -eq $TemperatureC) { return $false }
    (([double]$FreeMib / 1024 - $ReservedGb) -ge ($NeedGb + $MarginGb)) -and ([double]$TemperatureC -lt $MaxTemperatureC)
}

function Invoke-SpacrGpuConcurrent {
    param([string]$TurnLabel, [string[]]$TurnCommand, [double]$NeedGb,
        [string]$Directory = (Join-Path $env:ProgramData 'spaCR\gpu'), [int]$PollSeconds = 15)
    if (-not $TurnLabel -or -not $TurnCommand.Count) { throw 'usage: gpu_turn.ps1 [--vram-gb N] <label> <command...>' }
    $limit = if ($env:GPU_POWER_LIMIT_W) { [double]$env:GPU_POWER_LIMIT_W } else { 280 }
    $margin = if ($env:GPU_VRAM_MARGIN_GB) { [double]$env:GPU_VRAM_MARGIN_GB } else { 2 }
    $maxTemp = if ($env:GPU_MAX_TEMP_C) { [double]$env:GPU_MAX_TEMP_C } else { 80 }
    $reserveS = if ($env:GPU_RESERVE_S) { [int]$env:GPU_RESERVE_S } else { 120 }
    $alert = if ($env:GPU_RAM_ALERT) { $env:GPU_RAM_ALERT } else { Join-Path $env:USERPROFILE '.claude\ram_guard\ALERT' }
    [void][IO.Directory]::CreateDirectory($Directory)
    $reserve = Join-Path $Directory 'reserve'
    [void][IO.Directory]::CreateDirectory($reserve)
    $log = Join-Path $Directory 'log'
    function Write-TurnLog([string]$Message) {
        $line = '{0:o} [{1}] {2}' -f [DateTime]::UtcNow, $TurnLabel, $Message
        [IO.File]::AppendAllText($log, $line + [Environment]::NewLine)
        [Console]::Error.WriteLine($line)
    }
    $power = Get-SpacrGpuReading 'power.limit'
    if ($null -eq $power -or [double]$power -ne $limit) {
        Write-TurnLog "REFUSED: --vram-gb needs the GPU power limit at $limit W; nvidia-smi reports '$power' W. Set it (nvidia-smi -pl $limit, as administrator) and try again."
        return 3
    }
    $mutex = [Threading.Mutex]::new($false, ((Get-SpacrGpuMutexName $Directory) + '-vram'))
    $mine = Join-Path $reserve ([string]$PID)
    Write-TurnLog "queued concurrently (pid $PID, $(Get-Location)); needs $NeedGb GB + $margin GB margin"
    $lastWhy = ''
    try {
        while ($true) {
            try { [void]$mutex.WaitOne() } catch [Threading.AbandonedMutexException] { }
            $now = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
            $reserved = 0.0
            foreach ($file in [IO.Directory]::GetFiles($reserve)) {
                $parts = ([IO.File]::ReadAllText($file).Trim() -split '\s+')
                if ($parts.Count -lt 3) { continue }
                if (-not (Get-Process -Id ([int]$parts[0]) -ErrorAction SilentlyContinue)) { [IO.File]::Delete($file); continue }
                if ($now - [long]$parts[1] -lt $reserveS) { $reserved += [double]$parts[2] }
            }
            $free = Get-SpacrGpuReading 'memory.free'
            $temp = Get-SpacrGpuReading 'temperature.gpu'
            $hasAlert = Test-Path -LiteralPath $alert
            if (Test-SpacrGpuFits $NeedGb $free $temp $reserved $hasAlert $margin $maxTemp) {
                [IO.File]::WriteAllText($mine, "$PID $now $NeedGb")
                $apps = (& nvidia-smi '--query-compute-apps=pid,used_memory' '--format=csv,noheader,nounits' 2>$null) -join ' '
                $spare = '{0:N1}' -f ([double]$free / 1024 - $reserved)
                Write-TurnLog "START concurrent: needs $NeedGb GB, free $spare GB (after reservations), $temp C, power $power W; compute=[$apps]: $($TurnCommand -join ' ')"
                $mutex.ReleaseMutex()
                break
            }
            $why = "free $('{0:N1}' -f ([double]$free / 1024 - $reserved)) GB, $temp C" + $(if ($hasAlert) { ', ram_guard ALERT' } else { '' })
            if ($why -ne $lastWhy) { Write-TurnLog "waits: needs $NeedGb + $margin GB below $maxTemp C; now $why"; $lastWhy = $why }
            $mutex.ReleaseMutex()
            Start-Sleep -Seconds $PollSeconds
        }
        $arguments = @($TurnCommand | Select-Object -Skip 1)
        $global:LASTEXITCODE = 0
        & $TurnCommand[0] @arguments | Out-Host
        $succeeded = $?
        $result = $LASTEXITCODE
        if (-not $succeeded -and $result -eq 0) { $result = 1 }
        Write-TurnLog "FINISH concurrent rc=$result"
        return $result
    }
    finally {
        if ([IO.File]::Exists($mine)) { [IO.File]::Delete($mine) }
        $mutex.Dispose()
    }
}

if ($MyInvocation.InvocationName -ne '.') {
    try {
        $idleSeconds = if ($env:GPU_IDLE_S) { [int]$env:GPU_IDLE_S } else { 360 }
        $gapSeconds = if ($env:GPU_GAP_S) { [int]$env:GPU_GAP_S } else { 600 }
        $pollSeconds = if ($env:GPU_POLL_S) { [int]$env:GPU_POLL_S } else { 15 }
        if ($null -ne $VramGb) {
            exit (Invoke-SpacrGpuConcurrent -TurnLabel $Label -TurnCommand $Command -NeedGb $VramGb -PollSeconds $pollSeconds)
        }
        exit (Invoke-SpacrGpuTurn -TurnLabel $Label -TurnCommand $Command -IdleSeconds $idleSeconds -GapSeconds $gapSeconds -PollSeconds $pollSeconds)
    }
    catch { [Console]::Error.WriteLine($_.ToString()); exit 1 }
}
