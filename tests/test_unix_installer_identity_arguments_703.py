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

# Every test executes the Unix installer under /bin/bash.
pytestmark = pytest.mark.posix_only


@pytest.mark.parametrize('shell', SHELLS)
@pytest.mark.parametrize('version', ['@SPACR_VERSION@', '1.5.1.5', '1.5.1.5.dev1'])
def test_both_runtime_checks_receive_exact_arguments_under_nounset(tmp_path, shell, version):
    source = SOURCE.read_text(encoding="utf-8")
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
    calls = record.read_text(encoding="utf-8").split('CALL\n')[1:]
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
    source = SOURCE.read_text(encoding="utf-8")
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


@pytest.mark.parametrize('platform,registration_code', [('linux', 0), ('linux', 1), ('macos', 0)])
def test_linux_cache_registration_follows_commit_and_cannot_undo_working_install(tmp_path, platform, registration_code):
    source = SOURCE.read_text(encoding="utf-8")
    tail = 'swap_started=0\n' + source.rsplit('swap_started=0\n', 1)[1]
    root = tmp_path / 'installed application'
    root.mkdir()
    working = root / 'working-runtime'
    working.write_text('verified', encoding='utf-8')
    old = root / 'previous-runtime'
    old.mkdir()
    record = tmp_path / 'registration.txt'
    script = '''set -Eeuo pipefail
swap_started=1
metadata_touched=1
capture() {
    old_state=retained
    if [[ ! -d "$old_venv" ]]; then old_state=removed; fi
    printf '%s\\n' "$swap_started" "$metadata_touched" "$old_state" "$@" > "$record"
    if [[ "$registration_code" != 0 ]]; then printf 'registration refused\\n' >&2; fi
    return "$registration_code"
}
cleanup_python=capture
spacr_say() { printf '%s\\n' "$*"; }
''' + tail
    result = subprocess.run(['/bin/bash', '-c', script], env=dict(os.environ,
        PLATFORM=platform, registration_code=str(registration_code),
        INSTALL_ROOT=str(root), old_venv=str(old), record=str(record),
        cleanup_module='verified cleanup module.py', NO_COMMAND_LAUNCHER='1', NO_LAUNCH='1'),
        text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert working.read_text(encoding='utf-8') == 'verified'
    assert 'installed' in result.stdout
    if platform == 'linux':
        assert record.read_text(encoding='utf-8').splitlines() == [
            '0', '0', 'removed', '-I', 'verified cleanup module.py',
            'register-installer-cache', '--install-root', str(root)]
        assert ('registration refused' in result.stderr) == bool(registration_code)
    else:
        assert not record.exists()


def test_generated_uninstaller_isolates_both_installed_helper_invocations(tmp_path):
    source = SOURCE.read_text(encoding="utf-8")
    start = source.index('    uninstall_path="$INSTALL_ROOT/uninstall-spacr.sh"')
    generation = source[start:source.index('\nfi\n\nswap_started=0', start)]
    root = tmp_path / 'installed application'
    python = root / 'venv/bin/python'
    python.parent.mkdir(parents=True)
    python.write_text('#!/bin/sh\nprintf "%s\\n" CALL "$@" >> "$record"\n', encoding='utf-8')
    python.chmod(0o755)
    record = tmp_path / 'helper-calls.txt'
    env = dict(os.environ, INSTALL_ROOT=str(root), VENV_DIR=str(root / 'venv'),
               LAUNCHER=str(tmp_path / 'bin/spacr'), DESKTOP_DIR=str(tmp_path / 'applications'),
               record=str(record))
    generated = subprocess.run(['/bin/bash', '-c', 'set -Eeuo pipefail\n' + generation],
                               env=env, capture_output=True, text=True, timeout=10)
    assert generated.returncode == 0, generated.stderr
    invoked = subprocess.run([str(root / 'uninstall-spacr.sh'), '--purge', '--yes'],
                             env=env, capture_output=True, text=True, timeout=10)
    assert invoked.returncode == 0, invoked.stderr
    calls = [call.splitlines() for call in record.read_text(encoding='utf-8').split('CALL\n')[1:]]
    assert calls == [['-I', '-m', 'spacr.install_cleanup', 'purge', '--yes'],
                     ['-I', '-m', 'spacr.install_cleanup', 'remove', '--only-root', str(root)]]
