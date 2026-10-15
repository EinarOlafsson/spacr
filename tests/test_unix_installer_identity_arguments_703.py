"""Execute the installer's real verification calls with a recording command."""
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get('SPACR_TEST_UNIX_SOURCE', ROOT / 'packaging/online/install_spacr_unix.sh'))
SHELLS = ['/bin/bash']
if os.environ.get('SPACR_TEST_BASH32'):
    SHELLS.append(os.environ['SPACR_TEST_BASH32'])


@pytest.mark.parametrize('shell', SHELLS)
@pytest.mark.parametrize('version', ['@SPACR_VERSION@', '1.5.1.5', '1.5.1.5.dev1'])
def test_both_runtime_checks_receive_exact_arguments_under_nounset(tmp_path, shell, version):
    source = SOURCE.read_text()
    setup = source[source.index('identity_args='):source.index('if ! "$cleanup_python" -I "$cleanup_module" "${cleanup_args[@]}"')]
    activated = source.rsplit('if ! "$cleanup_python" -I "$cleanup_module"', 1)[-1]
    activated = '"$cleanup_python" -I "$cleanup_module"' + activated.split('; then', 1)[0]
    record = tmp_path / 'calls.txt'
    env = dict(os.environ, DEFAULT_SPACR_VERSION=version,
               stage_python=str(tmp_path / 'staged runtime' / 'bin/python'),
               VENV_DIR=str(tmp_path / 'active runtime'),
               work_dir=str(tmp_path), record=str(record))
    script = ('set -Eeuo pipefail\n'
              'capture() { printf "%s\\n" CALL "$@" >> "$record"; }\n'
              'cleanup_python=capture\ncleanup_module="cleanup module.py"\n'
              + setup + '\n' + activated + '\n')
    result = subprocess.run([shell, '-c', script], env=env, text=True,
                            capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    calls = record.read_text().split('CALL\n')[1:]
    assert len(calls) == 2
    optional = {} if version.startswith('@') else {'--version': version}
    for call, python, extra in zip(calls,
            [env['stage_python'], env['VENV_DIR'] + '/bin/python'],
            [{'--retiring-prefix': env['VENV_DIR']}, {}]):
        args = call.splitlines()
        assert args[:3] == ['-I', 'cleanup module.py', 'verify-runtime']
        assert len(args[3:]) % 2 == 0
        assert dict(zip(args[3::2], args[4::2])) == {'--python': python, **optional, **extra}


def test_installer_remains_valid_bash_syntax():
    result = subprocess.run(['/bin/bash', '-n', str(SOURCE)], capture_output=True,
                            text=True, timeout=10)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('shell', SHELLS)
def test_owned_temporary_folder_is_physical_before_metadata_snapshot(tmp_path, shell):
    real = tmp_path / 'physical temp'
    real.mkdir()
    alias = tmp_path / 'temp alias'
    alias.symlink_to(real, target_is_directory=True)
    source = SOURCE.read_text()
    setup = source[source.index('work_dir="$(mktemp'):source.index('installer_tmp="$work_dir/')]
    result = subprocess.run([shell, '-c', 'set -Eeuo pipefail\n' + setup +
                             '\nprintf "%s" "$work_dir"'],
                            env=dict(os.environ, TMPDIR=str(alias)), text=True,
                            capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    folder = Path(result.stdout)
    assert folder.is_dir()
    assert folder.parent == real
    assert folder == folder.resolve()
