"""The native Windows queue excludes browsers and fails closed before admission."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.skipif(sys.platform != 'win32', reason='Native Windows mutex and GPU counters')
ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'tools' / 'gpu_turn.ps1'
POWERSHELL = str(Path(os.environ.get('SYSTEMROOT', 'C:/Windows')) / 'System32/WindowsPowerShell/v1.0/powershell.exe')


def _quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def _program(tmp_path, body, name='probe.ps1'):
    script = tmp_path / name
    script.write_text("$ErrorActionPreference = 'Stop'\n. " + _quote(SCRIPT) + '\n' + body, encoding='utf-8-sig')
    return [POWERSHELL, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(script)]


def _run(tmp_path, body):
    result = subprocess.run(_program(tmp_path, body), capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def test_windows_dispatch_reaches_native_usage_without_starting_a_turn(tmp_path):
    bash = shutil.which('bash')
    if not bash:
        candidate = Path(sys.executable).parent.parent / 'tools/Library/usr/bin/bash.exe'
        bash = str(candidate) if candidate.exists() else None
    if not bash:
        pytest.skip('MSYS bash is not installed')
    result = subprocess.run([bash, '--noprofile', '--norc', str(ROOT / 'tools/gpu_turn.sh')],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert 'usage: gpu_turn.ps1' in result.stderr
    assert 'flock' not in result.stderr


def test_browser_descendants_and_exact_graphics_argument_are_excluded(tmp_path):
    result = _run(tmp_path, r'''
$date = [datetime]'2026-10-21T00:00:00Z'
$processes = @(
    [pscustomobject]@{ProcessId=10;ParentProcessId=0;Name='CHROME.EXE';CreationDate=$date},
    [pscustomobject]@{ProcessId=11;ParentProcessId=10;Name='helper.exe';CreationDate=$date.AddSeconds(1)},
    [pscustomobject]@{ProcessId=12;ParentProcessId=11;Name='nested.exe';CreationDate=$date.AddSeconds(2)},
    [pscustomobject]@{ProcessId=13;ParentProcessId=0;Name='helper.exe';CommandLine='helper.exe "--type=gpu-process"'},
    [pscustomobject]@{ProcessId=14;ParentProcessId=0;Name='python.exe';CommandLine='python.exe --type=gpu-process-extra'},
    [pscustomobject]@{ProcessId=15;ParentProcessId=10;Name='older.exe';CreationDate=$date.AddSeconds(-1)},
    [pscustomobject]@{ProcessId=16;ParentProcessId=0;Name='python.exe';CommandLine='python.exe prefix--type=gpu-process'},
    [pscustomobject]@{ProcessId=17;ParentProcessId=0;Name='dwm.exe';CreationDate=$date;ExecutablePath=($env:SystemRoot + '\System32\dwm.exe')},
    [pscustomobject]@{ProcessId=18;ParentProcessId=0;Name='dwm.exe';ExecutablePath='C:\owned\dwm.exe'},
    [pscustomobject]@{ProcessId=20;ParentProcessId=0;Name='python.exe';CommandLine='python.exe "a --type=gpu-process"'},
    [pscustomobject]@{ProcessId=21;ParentProcessId=17;Name='python.exe';CreationDate=$date.AddSeconds(1)}
)
$samples = @(10..21 | ForEach-Object { [pscustomobject]@{InstanceName="pid_$($_)_engine";CookedValue=20;Status=0} })
Select-SpacrGpuActivity $samples $processes | ConvertTo-Json -Compress
''')
    assert result['ExcludedPids'] == [10, 11, 12, 13, 17]
    assert result['BusyPids'] == [14, 15, 16, 18, 19, 20, 21]


@pytest.mark.parametrize('sample', [
    "@{InstanceName='pid_99_engine';CookedValue=[double]::NaN;Status=0}",
    "@{InstanceName='pid_99_engine';CookedValue=1;Status=5}",
    "@{InstanceName='unknown';CookedValue=0;Status=0}",
])
def test_invalid_activity_fails_closed(tmp_path, sample):
    result = _run(tmp_path, r'''
$refused = $false
try { Select-SpacrGpuActivity @([pscustomobject]''' + sample + r''') @([pscustomobject]@{ProcessId=1;ParentProcessId=0;Name='python.exe'}) | Out-Null }
catch { $refused = $true }
@{refused=$refused} | ConvertTo-Json -Compress
''')
    assert result['refused'] is True


def test_idle_gap_and_unknown_compute_activity_keep_the_admission_gates(tmp_path):
    result = _run(tmp_path, r'''
@{
    idle_short=(Test-SpacrGpuAdmission @() 359 600)
    gap_short=(Test-SpacrGpuAdmission @() 360 599)
    busy=(Test-SpacrGpuAdmission @(99) 360 600)
    ready=(Test-SpacrGpuAdmission @() 360 600)
} | ConvertTo-Json -Compress
''')
    assert result == {'idle_short': False, 'gap_short': False, 'busy': False, 'ready': True}


def test_query_failure_never_runs_the_requested_command(tmp_path):
    marker = tmp_path / 'must-not-start'
    command = tmp_path / 'payload.ps1'
    command.write_text('[IO.File]::WriteAllText(' + _quote(marker) + ", 'unsafe')", encoding='utf-8-sig')
    lock_probe = _program(tmp_path, r'''
$mutex = [Threading.Mutex]::new($false, (Get-SpacrGpuMutexName ''' + _quote(tmp_path / 'queue') + r'''))
$free = $mutex.WaitOne(0)
if ($free) { $mutex.ReleaseMutex() }
$mutex.Dispose()
if (-not $free) { exit 9 }
''', 'released-lock.ps1')
    result = _run(tmp_path, r'''
function Get-Counter { throw 'actual counter provider unavailable' }
$refused = $false
try { Invoke-SpacrGpuTurn -TurnLabel failure -Directory ''' + _quote(tmp_path / 'queue') + r''' -IdleSeconds 0 -GapSeconds 0 -TurnCommand @(''' + _quote(command) + r''') | Out-Null }
catch { $refused = $_.ToString().Contains('counter provider unavailable') }
& ''' + ' '.join(map(_quote, lock_probe)) + r''' | Out-Null
$free = $LASTEXITCODE -eq 0
@{refused=$refused;released=$free} | ConvertTo-Json -Compress
''')
    assert result['refused'] is True
    assert result['released'] is True
    assert not marker.exists()


def test_native_browser_identity_and_actual_child_are_excluded(tmp_path):
    psutil = pytest.importorskip('psutil')
    candidates = [Path(os.environ.get('PROGRAMFILES', 'C:/Program Files')) / 'Google/Chrome/Application/chrome.exe',
                  Path(os.environ.get('LOCALAPPDATA', '')) / 'Google/Chrome/Application/chrome.exe']
    browser = next((path for path in candidates if path.is_file()), None)
    if browser is None:
        pytest.skip('Chrome is not installed; classifier cases remain required')
    process = subprocess.Popen([str(browser), '--headless=new', '--disable-gpu', '--disable-software-rasterizer',
                                '--disable-background-networking', '--no-first-run', '--no-default-browser-check',
                                '--user-data-dir=' + str(tmp_path / 'browser-profile'), 'about:blank'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        result = _run(tmp_path, r'''
$until = [datetime]::UtcNow.AddSeconds(5)
do {
    $processes = @(Get-CimInstance Win32_Process -ErrorAction Stop)
    $children = @($processes | Where-Object { $_.ParentProcessId -eq ''' + str(process.pid) + r''' })
    if (-not $children.Count) { Start-Sleep -Milliseconds 50 }
} while (-not $children.Count -and [datetime]::UtcNow -lt $until)
if (-not $children.Count) { throw 'owned browser child not found' }
$identities = @(''' + str(process.pid) + r''') + @($children.ProcessId)
$samples = @($identities | ForEach-Object { [pscustomobject]@{InstanceName="pid_$($_)_engine";CookedValue=90;Status=0} })
$activity = Select-SpacrGpuActivity $samples $processes
@{busy=@($activity.BusyPids);missing=@($identities | Where-Object { $_ -notin $activity.ExcludedPids })} | ConvertTo-Json -Compress
''')
        assert result == {'busy': [], 'missing': []}
    finally:
        if process.poll() is None:
            owner = psutil.Process(process.pid)
            children = owner.children(recursive=True)
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            process.terminate()
            process.wait(timeout=10)
            _, alive = psutil.wait_procs(children, timeout=5)
            for child in alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(alive, timeout=5)


def test_admitted_owned_command_preserves_arguments_exit_code_and_release(tmp_path):
    marker = tmp_path / 'literal arguments'
    queue = tmp_path / 'queue'
    code = 'import sys; from pathlib import Path; Path(sys.argv[1]).write_text(sys.argv[2]); sys.exit(7)'
    result = _run(tmp_path, r'''
function Get-SpacrGpuActivity { [pscustomobject]@{BusyPids=@();ExcludedPids=@()} }
$result = Invoke-SpacrGpuTurn -TurnLabel owned-cpu -Directory ''' + _quote(queue) + r''' -IdleSeconds 0 -GapSeconds 0 -TurnCommand @(''' + ','.join(map(_quote, [sys.executable, '-c', code, marker, 'literal $HOME'])) + r''')
@{exit=$result;released=[IO.File]::ReadAllText(''' + _quote(queue / 'last_release') + r''');holder=[IO.File]::ReadAllText(''' + _quote(queue / 'holder') + r''')} | ConvertTo-Json -Compress
''')
    assert result['exit'] == 7
    assert int(result['released']) > 0 and result['holder'] == ''
    assert marker.read_text() == 'literal $HOME'


def test_shared_native_mutex_serializes_owned_processes(tmp_path):
    queue = tmp_path / 'queue'
    ready, release, second = (tmp_path / name for name in ('ready', 'release', 'second'))
    common = '$mutex = [Threading.Mutex]::new($false, (Get-SpacrGpuMutexName ' + _quote(queue) + '))\n'
    owner_body = common + r'''
try {
    [void]$mutex.WaitOne()
    [IO.File]::WriteAllText(''' + _quote(ready) + r''', 'held')
    $until = [datetime]::UtcNow.AddSeconds(15)
    while (-not [IO.File]::Exists(''' + _quote(release) + r''')) {
        if ([datetime]::UtcNow -gt $until) { throw 'test owner release deadline' }
        Start-Sleep -Milliseconds 20
    }
} finally { $mutex.ReleaseMutex(); $mutex.Dispose() }
'''
    waiter_body = common + r'''
try {
    [void]$mutex.WaitOne()
    [IO.File]::WriteAllText(''' + _quote(second) + r''', 'held')
} finally { $mutex.ReleaseMutex(); $mutex.Dispose() }
'''
    owner = subprocess.Popen(_program(tmp_path, owner_body, 'owner.ps1'), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    waiter = None
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            assert owner.poll() is None
            time.sleep(.02)
        assert ready.exists()
        waiter = subprocess.Popen(_program(tmp_path, waiter_body, 'waiter.ps1'), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(.5)
        assert waiter.poll() is None and not second.exists()
        release.write_text('release')
        owner.communicate(timeout=10)
        output, error = waiter.communicate(timeout=10)
        assert owner.returncode == waiter.returncode == 0, output + error
        assert second.read_text() == 'held'
    finally:
        for process in (owner, waiter):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=5)
