"""Native Windows argv, stream, exit and owned-cleanup boundaries."""

import base64
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / 'packaging/online/install_spacr_windows.ps1'
pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Windows native installer')


def _functions():
    source = INSTALLER.read_text(encoding='utf-8')
    return ('function ConvertTo-SpacrNativeArgument {' + source.split(
        'function ConvertTo-SpacrNativeArgument {', 1)[1].split('# Each lease file', 1)[0])


def _ps(value):
    return "'" + str(value).replace("'", "''") + "'"


def _run(tmp_path, body, timeout=35):
    script = tmp_path / 'native-contract.ps1'
    script.write_text('$ErrorActionPreference="Stop"\n' + _functions() + '\n' + body,
                      encoding='utf-8-sig')
    executable = Path(os.environ['WINDIR']) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
    return subprocess.run([str(executable), '-NoLogo', '-NoProfile', '-NonInteractive',
                           '-File', str(script)], capture_output=True, timeout=timeout)


@pytest.fixture(scope='module')
def native_child(tmp_path_factory):
    folder = tmp_path_factory.mktemp('native-child')
    source = folder / 'Child.cs'
    source.write_text(r'''
using System;
using System.IO;
using System.Text;
using System.Threading;
using System.Runtime.InteropServices;
class Child {
 [DllImport("kernel32.dll")] static extern IntPtr GetCurrentProcess();
 [DllImport("kernel32.dll", SetLastError=true)] static extern bool IsProcessInJob(IntPtr process, IntPtr job, out bool member);
 static int Main(string[] args) {
  Console.OutputEncoding = new UTF8Encoding(false);
  if (args[0] == "argv") {
   for (int i=1; i<args.Length; i++) Console.WriteLine(Convert.ToBase64String(Encoding.UTF8.GetBytes(args[i])));
   bool member;
   if (!IsProcessInJob(GetCurrentProcess(), IntPtr.Zero, out member) || !member) return 9;
   return 0;
  }
  if (args[0] == "quiet") return 7;
  if (args[0] == "transcript") {
   if (args[1] == "success") { Console.Error.WriteLine("successful native progress"); return 0; }
   Console.Error.WriteLine("first failure line outside retained tail");
   for (int i=0; i<128; i++) Console.Error.WriteLine("P" + new string('z',4096));
   Console.Error.WriteLine("final traceback: " + (char)248 + (char)26085 + (char)26412);
   return 7;
  }
  if (args[0] == "stream") {
   Console.WriteLine("early");
   for (int i=0; i<100 && !File.Exists(args[1]); i++) Thread.Sleep(50);
   if (!File.Exists(args[1])) return 8;
   for (int i=0; i<128; i++) { Console.WriteLine("O" + new string('x',4096)); Console.Error.WriteLine("E" + new string('y',4096)); }
   return 7;
  }
  if (args[0] == "raw") {
   byte[] data = new byte[262144];
   for (int i=0; i<data.Length; i+=4) { data[i]=0; data[i+1]=0xc3; data[i+2]=0xb8; data[i+3]=0x0a; }
   Console.OpenStandardOutput().Write(data,0,data.Length);
   Console.OpenStandardError().Write(data,0,data.Length);
   return 7;
  }
  return 10;
 }
}
''', encoding='utf-8')
    executable = folder / 'Child.exe'
    compiler = Path(os.environ['WINDIR']) / 'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
    result = subprocess.run([str(compiler), '/nologo', '/out:' + str(executable), str(source)],
                            capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    return executable


def test_native_argv_and_inherited_job_are_preserved(tmp_path, native_child):
    values = ['', 'two words', 'embedded"quote', 'tail\\', 'quote\\"tail\\',
              '\u00f8\u65e5\u672c', '&|<>^%$(literal)']
    encoded = [base64.b64encode(value.encode()).decode() for value in values]
    expected = '@(' + ','.join(_ps(value) for value in encoded) + ')'
    result = _run(tmp_path, f'''
$Expected={expected}
$Arguments=@('argv') + @($Expected | ForEach-Object {{ [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($_)) }})
$Actual=@(Invoke-SpacrNative -Command {_ps(native_child)} -Arguments $Arguments)
if ($LASTEXITCODE -ne 0 -or $Actual.Count -ne $Expected.Count) {{ throw 'Argument count or inherited Job failed' }}
for ($i=0; $i -lt $Expected.Count; $i++) {{ if ($Actual[$i] -cne $Expected[$i]) {{ throw ('Argument boundary failed: ' + $i) }} }}
''')
    assert result.returncode == 0, result.stdout + result.stderr


def test_both_streams_are_drained_live_and_nonzero_exit_is_retained(tmp_path, native_child):
    marker = tmp_path / 'first-line-seen'
    result = _run(tmp_path, f'''
$Lines=@(Invoke-SpacrNative -Command {_ps(native_child)} -Arguments @('stream', {_ps(marker)}) | ForEach-Object {{
    if ($_ -eq 'early') {{ [IO.File]::WriteAllText({_ps(marker)}, 'ready') }}
    $_
}})
if ($LASTEXITCODE -ne 7 -or $Lines.Count -ne 129) {{ throw 'Streaming output or exit was lost' }}
if (@($Lines | Where-Object {{ $_.StartsWith('O') -and $_.Length -eq 4097 }}).Count -ne 128) {{ throw 'Stdout corrupted' }}
function Get-SpacrInstallerMessage {{ param($Name, $Values) return ($Name + ':' + $Values[1]) }}
try {{ Invoke-Checked -Command {_ps(native_child)} -Arguments @('quiet'); throw 'Quiet failure admitted' }}
catch {{ if ($_.Exception.Message -ne 'command_failed:7' -or $LASTEXITCODE -ne 7) {{ throw }} }}
''')
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stderr.count(b'E' + b'y' * 4096) == 128


def test_raw_pip_capture_preserves_utf8_bytes_and_exit(tmp_path, native_child):
    output, error = tmp_path / 'stdout.bin', tmp_path / 'stderr.bin'
    result = _run(tmp_path, f'''
$Lines=@(Invoke-SpacrNative -Command {_ps(native_child)} -Arguments @('raw') -StandardOutputPath {_ps(output)} -StandardErrorPath {_ps(error)})
if ($LASTEXITCODE -ne 7 -or $Lines.Count -ne 0) {{ throw 'Raw capture polluted output or lost exit' }}
''')
    assert result.returncode == 0, result.stdout + result.stderr
    assert output.read_bytes() == error.read_bytes() == b'\x00\xc3\xb8\n' * 65536


def test_failed_native_stderr_tail_reaches_ps5_transcript_without_changing_streams(tmp_path, native_child):
    failed, successful = tmp_path / 'failed.log', tmp_path / 'successful.log'
    result = _run(tmp_path, f'''
if ($PSVersionTable.PSVersion.Major -ne 5) {{ throw 'This regression requires Windows PowerShell 5' }}
Start-Transcript -Path {_ps(failed)} | Out-Null
try {{
    $Lines=@(Invoke-SpacrNative -Command {_ps(native_child)} -Arguments @('transcript','failure'))
    if ($LASTEXITCODE -ne 7 -or $Lines.Count -ne 0) {{ throw 'Failed native exit or stdout stream changed' }}
    Write-Host 'continued after native stderr under Stop'
}} finally {{ Stop-Transcript | Out-Null }}
Start-Transcript -Path {_ps(successful)} | Out-Null
try {{
    $Lines=@(Invoke-SpacrNative -Command {_ps(native_child)} -Arguments @('transcript','success'))
    if ($LASTEXITCODE -ne 0 -or $Lines.Count -ne 0) {{ throw 'Successful native stderr became an error' }}
}} finally {{ Stop-Transcript | Out-Null }}
''')
    assert result.returncode == 0, result.stdout + result.stderr
    marker = 'final traceback: \u00f8\u65e5\u672c'
    transcript = failed.read_text(encoding='utf-8-sig')
    assert transcript.count(marker) == 1
    assert 'continued after native stderr under Stop' in transcript
    assert 'first failure line outside retained tail' not in transcript
    assert transcript.count('P' + 'z' * 4096) < 20
    assert result.stderr.count(marker.encode('utf-8')) == 1
    assert result.stderr.count(b'P' + b'z' * 4096) == 128
    assert result.stderr.count(b'successful native progress') == 1
    assert b'successful native progress' not in result.stdout


def test_isolated_python_utf8_output_and_errors_are_decoded(tmp_path):
    result = _run(tmp_path, f'''
$Code="import sys; print(chr(248)+chr(26085)); print(chr(248)+chr(26412), file=sys.stderr)"
$Lines=@(Invoke-SpacrNative -Command {_ps(sys.executable)} -Arguments @('-I','-X','utf8','-c',$Code))
if ($LASTEXITCODE -ne 0 -or $Lines.Count -ne 1 -or $Lines[0] -cne ([string][char]248 + [char]26085)) {{ throw 'Isolated Python stdout not UTF8' }}
''')
    assert result.returncode == 0, result.stdout + result.stderr
    assert '\u00f8\u672c'.encode('utf-8') in result.stderr


def test_work_cleanup_is_owned_and_does_not_replace_the_primary_error(tmp_path):
    source = INSTALLER.read_text(encoding='utf-8-sig')
    cleanup = 'function Remove-SpacrTree {' + source.split('function Remove-SpacrTree {', 1)[1].split(
        'function Remove-SpacrAbandonedPath {', 1)[0]
    finalizer = source.rsplit('} finally {', 1)[1].split('if ($TranscriptStarted)', 1)[0]
    result = _run(tmp_path, cleanup + f'''
$env:TEMP={_ps(tmp_path)}
$WorkDir=Join-Path $env:TEMP ('spacr-install-' + $PID)
[IO.Directory]::CreateDirectory((Join-Path $WorkDir 'nested')) | Out-Null
$Readonly=Join-Path $WorkDir 'nested\\owned.txt'
[IO.File]::WriteAllText($Readonly,'owned')
[IO.File]::SetAttributes($Readonly,[IO.FileAttributes]::ReadOnly)
$Foreign=Join-Path $env:TEMP 'foreign'
[IO.Directory]::CreateDirectory($Foreign) | Out-Null
$Sentinel=Join-Path $Foreign 'sentinel.txt'
[IO.File]::WriteAllText($Sentinel,'preserve')
Remove-SpacrInstallerWork
if ([IO.Directory]::Exists($WorkDir)) {{ throw 'Owned work remains' }}
$WorkDir=$Foreign
try {{ Remove-SpacrInstallerWork; throw 'Foreign root admitted' }}
catch {{ if ($_.Exception.NativeErrorCode -ne 87) {{ throw }} }}
if ([IO.File]::ReadAllText($Sentinel) -ne 'preserve') {{ throw 'Foreign sentinel changed' }}
$InstallerStarted=$true
$RetainWorkDir=$false
try {{
    try {{ throw 'primary install failure' }} catch {{ $OriginalError=$_; throw }} finally {{
{finalizer}
    }}
}} catch {{ if ($_.Exception.Message -ne 'primary install failure') {{ throw }} }}
if ($null -eq $WorkCleanupError) {{ throw 'Cleanup failure was not retained' }}
''')
    assert result.returncode == 0, result.stdout + result.stderr


def test_current_lease_release_preserves_sharing_locks_and_the_primary_error(tmp_path):
    source = INSTALLER.read_text(encoding='utf-8-sig')
    finalizer = source.rsplit('} finally {', 1)[1].rsplit('}', 1)[0]
    result = _run(tmp_path, f'''
$InstallerStarted=$false
$RetainWorkDir=$false
$TranscriptStarted=$false
$OriginalError=$null
$LeasePath=Join-Path {_ps(tmp_path)} 'owned-installer.lock'
$OwnedHandle=[IO.File]::Open($LeasePath,[IO.FileMode]::CreateNew,[IO.FileAccess]::ReadWrite,[IO.FileShare]::ReadWrite)
$InstallerLeaseHandles=@(@{{Path=$LeasePath;Handle=$OwnedHandle}})
try {{ }} finally {{
{finalizer}
}}
if ([IO.File]::Exists($LeasePath) -or $OwnedHandle.CanRead) {{ throw 'Current lease was not disposed and deleted' }}
$OwnedHandle=[IO.File]::Open($LeasePath,[IO.FileMode]::CreateNew,[IO.FileAccess]::ReadWrite,[IO.FileShare]::ReadWrite)
$OtherHandle=[IO.File]::Open($LeasePath,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::ReadWrite)
$InstallerLeaseHandles=@(@{{Path=$LeasePath;Handle=$OwnedHandle}})
try {{
    try {{
        try {{ throw 'primary install failure' }} catch {{ $OriginalError=$_; throw }} finally {{
{finalizer}
        }}
    }} catch {{ if ($_.Exception.Message -ne 'primary install failure') {{ throw }} }}
    if ($OwnedHandle.CanRead -or -not [IO.File]::Exists($LeasePath)) {{ throw 'Sharing lock or current handle ownership lost' }}
    if ($null -eq $WorkCleanupError -or ($WorkCleanupError.Exception.InnerException.HResult -band 65535) -ne 32) {{ throw 'Real sharing violation not retained' }}
}} finally {{ $OtherHandle.Dispose() }}
[IO.File]::Delete($LeasePath)
''')
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('rollback_fails', [False, True])
def test_outer_catch_retains_primary_when_locked_stage_removal_fails(tmp_path, rollback_fails):
    result = _run(tmp_path, f'''
$Tokens=$null
$ParseErrors=$null
$Ast=[Management.Automation.Language.Parser]::ParseFile({_ps(INSTALLER)},[ref]$Tokens,[ref]$ParseErrors)
if ($ParseErrors.Count) {{ throw 'Installer parse failed' }}
$Catches=@($Ast.FindAll({{ param($Node)
    $Node -is [Management.Automation.Language.CatchClauseAst] -and
    $Node.Body.Extent.Text -match '^\{{\s*\$OriginalError\s*=\s*\$_'
}},$true))
if ($Catches.Count -ne 1) {{ throw 'Expected unique actual outer catch' }}
$CatchText=$Catches[0].Body.Extent.Text
$CatchBody=[scriptblock]::Create($CatchText.Substring(1,$CatchText.Length-2))
foreach ($Helper in 'Remove-SpacrTree','ConvertTo-SpacrExtendedPath','Remove-SpacrEnvironment') {{
    $Definition=@($Ast.FindAll({{ param($Node)
        $Node -is [Management.Automation.Language.FunctionDefinitionAst] -and $Node.Name -eq $Helper
    }},$true))
    if ($Definition.Count -ne 1) {{ throw ('Expected one installer helper ' + $Helper) }}
    Invoke-Expression $Definition[0].Extent.Text
}}
function Get-SpacrInstallerMessage {{ param($Name) return $Name }}
$InstallerStarted=$true
$Committed=$false
$MetadataTouched=$false
$RetainWorkDir=$false
$StageVenv=Join-Path {_ps(tmp_path)} 'owned-stage'
$StageProfile=Join-Path {_ps(tmp_path)} 'owned-stage-profile.json'
[IO.Directory]::CreateDirectory($StageVenv) | Out-Null
[IO.File]::WriteAllText($StageProfile,'owned')
$Locked=Join-Path $StageVenv 'locked.dll'
$StageHandle=[IO.File]::Open($Locked,[IO.FileMode]::CreateNew,[IO.FileAccess]::ReadWrite,[IO.FileShare]::Read)
$OldVenv=$null
$RollbackHandle=$null
try {{
    if ({'$true' if rollback_fails else '$false'}) {{
        $OldVenv=Join-Path {_ps(tmp_path)} 'owned-previous'
        $VenvDir=Join-Path {_ps(tmp_path)} 'owned-runtime'
        [IO.Directory]::CreateDirectory($OldVenv) | Out-Null
        [IO.Directory]::CreateDirectory($VenvDir) | Out-Null
        [IO.File]::WriteAllText((Join-Path $OldVenv 'keep.txt'),'previous runtime')
        $RollbackHandle=[IO.File]::Open((Join-Path $VenvDir 'busy.dll'),[IO.FileMode]::CreateNew,[IO.FileAccess]::ReadWrite,[IO.FileShare]::Read)
    }}
    try {{
        try {{ throw 'primary install failure' }} catch {{ . $CatchBody }}
        throw 'Primary exception disappeared'
    }} catch {{
        if ($_.Exception.Message -ne 'primary install failure') {{ throw }}
    }}
    if (-not $RetainWorkDir) {{ throw 'Failed stage cleanup did not preserve diagnostics' }}
    if (-not [IO.File]::Exists($Locked)) {{ throw 'Locked stage evidence disappeared' }}
    if ($OldVenv -and [IO.File]::ReadAllText((Join-Path $OldVenv 'keep.txt')) -ne 'previous runtime') {{ throw 'Rollback evidence changed' }}
}} finally {{
    $StageHandle.Dispose()
    if ($RollbackHandle) {{ $RollbackHandle.Dispose() }}
}}
''')
    assert result.returncode == 0, result.stdout + result.stderr
    assert str(tmp_path / 'owned-stage').encode() in result.stdout
    # The console wraps a long warning at its width, wherever that falls.
    assert b'locked.dll' in result.stdout.replace(b'\r\n', b'')
    if rollback_fails:
        assert b'gui_failed' in result.stdout
        assert str(tmp_path / 'owned-previous').encode() in result.stdout
