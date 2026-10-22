"""Native disk admission for Windows installs on restricted volumes."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / 'packaging/online/install_spacr_windows.ps1'
pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Windows disk API contract')


def _source():
    source = INSTALLER.read_text(encoding='utf-8')
    function, guard = source.split('function Get-SpacrAvailableDiskSpace {', 1)[1].split(
        '\n$AvailableSpace = Get-SpacrAvailableDiskSpace $InstallRoot', 1)
    return ('function Get-SpacrAvailableDiskSpace {' + function,
            '$AvailableSpace = Get-SpacrAvailableDiskSpace $InstallRoot' + guard.split(
                '\n$InstallerLeaseHandles', 1)[0])


def _powershell(tmp_path, source):
    script = tmp_path / 'disk-admission.ps1'
    script.write_text('$ErrorActionPreference = "Stop"\n' + source, encoding='utf-8')
    executable = shutil.which('powershell.exe') or str(
        Path(os.environ['WINDIR']) / 'System32/WindowsPowerShell/v1.0/powershell.exe')
    return subprocess.run([executable, '-NoLogo', '-NoProfile', '-NonInteractive',
                           '-File', str(script)], capture_output=True, text=True, timeout=30)


def test_native_disk_space_uses_existing_target_parent(tmp_path):
    function, _guard = _source()
    result = _powershell(tmp_path, function + r'''
$Existing = Get-SpacrAvailableDiskSpace $PSScriptRoot
$Future = Get-SpacrAvailableDiskSpace (Join-Path $PSScriptRoot 'missing-parent\target')
if ($Existing -le 0 -or $Future -le 0) { throw 'No available disk space returned' }
if ([math]::Abs([double]$Existing - [double]$Future) -gt 64MB) { throw 'Different target volume' }
if (Test-Path (Join-Path $PSScriptRoot 'missing-parent')) { throw 'Query created target paths' }
Write-Output 'Existing and future target parents verified'
''')
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Existing and future target parents verified' in result.stdout


def test_unknown_volume_refuses_without_a_free_space_value(tmp_path):
    function, _guard = _source()
    result = _powershell(tmp_path, function + r'''
$Missing = 90..68 | ForEach-Object { ([char]$_).ToString() + ':\' } |
    Where-Object { -not [System.IO.Directory]::Exists($_) } | Select-Object -First 1
if (-not $Missing) { throw 'Test requires an unmounted drive letter' }
try {
    $Unexpected = Get-SpacrAvailableDiskSpace $Missing
    throw 'Unknown volume admitted'
} catch [System.ComponentModel.Win32Exception] {
    if ($_.Exception.NativeErrorCode -ne 3) { throw }
    Write-Output 'Unknown volume refused with native path error'
}
''')
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Unknown volume refused with native path error' in result.stdout


@pytest.mark.parametrize('available,admitted', [(5 * 1024**3 - 1, False),
                                              (5 * 1024**3, True),
                                              (5 * 1024**3 + 1, True)])
def test_disk_admission_preserves_the_five_gib_boundary(tmp_path, available, admitted):
    _function, guard = _source()
    result = _powershell(tmp_path, f'''
function Get-SpacrAvailableDiskSpace {{ return [UInt64]{available} }}
function Get-SpacrInstallerMessage {{ param($Name, $Values) return $Name }}
$InstallRoot = Join-Path $PSScriptRoot 'target'
{guard}
Write-Output 'Disk admitted'
''')
    assert (result.returncode == 0) is admitted, result.stdout + result.stderr
    if admitted:
        assert 'Disk admitted' in result.stdout
    else:
        assert 'needs_free_space' in result.stderr
        assert 'Disk admitted' not in result.stdout
