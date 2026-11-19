"""A concurrent GPU job checks the power cap again after waiting for memory."""

import json
import os
import shutil
import subprocess
import sys

import pytest

from tests.test_gpu_turn_concurrent import ROOT, SCRIPT, _env


@pytest.mark.skipif(sys.platform == "win32" or shutil.which("flock") is None,
                    reason="the bash queue needs flock(1)")
@pytest.mark.parametrize("later_power", ["350", ""], ids=["changed", "unreadable"])
def test_bash_refuses_a_changed_or_unreadable_cap_after_waiting(tmp_path, later_power):
    """Free memory must not admit a job after its power cap becomes unsuitable."""
    env = _env(tmp_path, GPU_PRIORITY_PREFIXES="")
    fake = tmp_path / "bin" / "nvidia-smi"
    fake.write_text('''#!/bin/sh
case "$*" in
 *power.limit*)
   if [ -e "$FAKE_STATE" ]; then printf '%s\\n' "$FAKE_LATER_POWER"; else echo 280; fi ;;
 *memory.free*) if [ -e "$FAKE_STATE" ]; then echo 20000; else echo 1000; fi ;;
 *temperature.gpu*) touch "$FAKE_STATE"; echo 40 ;;
 *query-compute-apps*) echo '' ;;
esac
''')
    env.update(FAKE_STATE=str(tmp_path / "sampled"), FAKE_LATER_POWER=later_power)
    marker = tmp_path / "ran"
    result = subprocess.run(
        ["bash", str(SCRIPT), "--vram-gb", "6", "cap-change", "touch", str(marker)],
        env=env, capture_output=True, text=True, timeout=10,
    )
    assert "waits:" in result.stderr
    assert result.returncode == 3, result.stderr
    assert not marker.exists()
    assert "REFUSED" in result.stderr and "START concurrent" not in result.stderr
    assert not list((tmp_path / "home" / ".spacr" / "gpu" / "reserve").iterdir())


@pytest.mark.skipif(sys.platform != "win32", reason="the native queue runs on Windows")
@pytest.mark.parametrize("later_power", ["350", ""], ids=["changed", "unreadable"])
def test_windows_refuses_a_changed_or_unreadable_cap_after_waiting(tmp_path, later_power):
    """The Windows queue checks the current cap before reserving or running a job."""
    probe = tmp_path / "probe.ps1"
    probe.write_text('''
. $env:SPACR_GPU_TEST_SCRIPT
$script:PowerSamples = 0
$script:MemorySamples = 0
function Get-SpacrGpuReading {
    <# .SYNOPSIS Return controlled telemetry for successive admission attempts. #>
    param([string]$Query)
    switch ($Query) {
        'power.limit' {
            $script:PowerSamples++
            if ($script:PowerSamples -eq 1) { return 280 }
            if ($env:SPACR_GPU_TEST_POWER -eq '') { return $null }
            return [double]$env:SPACR_GPU_TEST_POWER
        }
        'memory.free' {
            $script:MemorySamples++
            if ($script:MemorySamples -eq 1) { return 1000 }
            return 20000
        }
        'temperature.gpu' { return 40 }
    }
}
function Start-Sleep {
    <# .SYNOPSIS Advance the queue without a real polling delay. #>
    param([int]$Seconds)
}
function nvidia-smi {
    <# .SYNOPSIS Keep command logging independent of real GPU hardware. #>
    return ''
}
$marker = Join-Path $env:SPACR_GPU_TEST_ROOT 'ran'
$queue = Join-Path $env:SPACR_GPU_TEST_ROOT 'queue'
$code = Invoke-SpacrGpuConcurrent -TurnLabel 'cap-change' -NeedGb 6 -Directory $queue `
    -TurnCommand @('New-Item', '-ItemType', 'File', '-Path', $marker)
@{ code = $code; ran = (Test-Path $marker); log = (Get-Content (Join-Path $queue 'log') -Raw) } |
    ConvertTo-Json -Compress
''', encoding="utf-8")
    env = dict(os.environ, SPACR_GPU_TEST_SCRIPT=str(ROOT / "tools" / "gpu_turn.ps1"),
               SPACR_GPU_TEST_ROOT=str(tmp_path), SPACR_GPU_TEST_POWER=later_power,
               GPU_RAM_ALERT=str(tmp_path / "ALERT"), GPU_POWER_LIMIT_W="280",
               GPU_VRAM_MARGIN_GB="2", GPU_MAX_TEMP_C="80")
    powershell = shutil.which("powershell.exe") or shutil.which("pwsh")
    assert powershell is not None
    result = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(probe)], env=env, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout.strip().splitlines()[-1])
    assert "waits:" in record["log"]
    assert record["code"] == 3 and not record["ran"], record
    assert "REFUSED" in record["log"] and "START concurrent" not in record["log"]
    assert not list((tmp_path / "queue" / "reserve").iterdir())
