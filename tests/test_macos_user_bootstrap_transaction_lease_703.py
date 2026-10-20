"""The user GUI wrapper leaves mutations and legacy exclusion to the leased backend."""
import os
from pathlib import Path
import select
import shlex
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
ONLINE = ROOT / 'packaging/online'
pytestmark = pytest.mark.skipif(os.name == 'nt', reason='POSIX bootstrap process contract')


def _support(tmp_path):
    support = tmp_path / 'support'
    support.mkdir()
    wrapper = support / 'install-for-user.sh'
    shutil.copyfile(ONLINE / 'install_for_macos_user.sh', wrapper)
    (support / 'installer_messages.sh').write_text('spacr_say() { printf "%s\\n" "$1"; }\n')
    source = (ONLINE / 'install_spacr_unix.sh').read_text(encoding="utf-8").split(
        '# @SPACR_INSTALLER_LEASE_BEGIN@\n', 1)[1].split('# @SPACR_INSTALLER_LEASE_END@', 1)[0]
    journal = tmp_path / 'backend-entry'
    backend = support / 'install-online.sh'
    stat_adapter = ('stat() { if [[ \"$1\" == \"-f\" ]]; then command stat -c \"$2\" \"$3\"; else command stat \"$@\"; fi; }\n'
                    if sys.platform.startswith('linux') else '')
    backend.write_text('#!/usr/bin/env bash\nset -Eeuo pipefail\n'
        'PLATFORM=macos\nINSTALL_ROOT="$4"\n'
        'spacr_say() { printf "%s\\n" "$1"; }\n'
        + stat_adapter +
        'if [[ -e "$INSTALL_ROOT" ]]; then state=exists; else state=absent; fi\n'
        'printf "%s\\n" "$state" > ' + shlex.quote(str(journal)) + '\n' + source +
        '\nprintf "acquired\\n"\n'
        'if [[ "${SPACR_TEST_WAIT:-0}" == "1" ]]; then read -r finish; fi\n'
        'if [[ "${SPACR_TEST_RETAIN:-0}" == "1" ]]; then SPACR_INSTALLER_RETAIN_LEASE=1; fi\n'
        'exit "${SPACR_TEST_EXIT:-0}"\n')
    backend.chmod(0o755)
    runtime = tmp_path / 'user runtime'
    env = dict(os.environ, SPACR_USER_INSTALL_ROOT=str(runtime), SPACR_NO_RELAUNCH='1')
    return wrapper, runtime, env, journal


def _run(wrapper, env):
    return subprocess.run(['/bin/sh', str(wrapper)], env=env, input='',
                          capture_output=True, text=True, timeout=10)


def _start(wrapper, env):
    process = subprocess.Popen(['/bin/sh', str(wrapper)], env={**env, 'SPACR_TEST_WAIT': '1'},
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    assert select.select([process.stdout], [], [], 10)[0]
    assert process.stdout.readline().strip() == 'acquired'
    return process


def test_gui_wrapper_does_not_create_application_or_marker_before_backend_lease(tmp_path):
    wrapper, runtime, env, journal = _support(tmp_path)
    result = _run(wrapper, env)
    assert result.returncode == 0, result.stderr
    assert journal.read_text(encoding="utf-8").strip() == 'absent'
    assert runtime.is_dir()
    assert not (runtime / '.installing').exists()
    assert not (runtime / '.venv.spacr-update-lock').exists()
    assert not (runtime.parent / f'.{runtime.name}.spacr-installer-lock').exists()


@pytest.mark.parametrize('kind', ['directory', 'file', 'dangling-link'])
def test_unknown_legacy_marker_is_preserved_without_entering_backend(tmp_path, kind):
    wrapper, runtime, env, journal = _support(tmp_path)
    runtime.mkdir()
    marker = runtime / '.installing'
    if kind == 'directory':
        marker.mkdir()
    elif kind == 'file':
        marker.write_bytes(b'unknown installation evidence')
    else:
        marker.symlink_to(tmp_path / 'unavailable original marker')
    before = marker.lstat()
    result = _run(wrapper, env)
    assert result.returncode == 75
    assert marker.lstat().st_ino == before.st_ino
    assert not journal.exists()
    if kind == 'file':
        assert marker.read_bytes() == b'unknown installation evidence'


def test_owned_backend_marker_excludes_both_modern_and_legacy_wrappers(tmp_path):
    wrapper, runtime, env, _ = _support(tmp_path)
    first = _start(wrapper, env)
    try:
        marker = runtime / '.installing'
        assert marker.is_dir()
        refused = _run(wrapper, env)
        assert refused.returncode == 75
        legacy = subprocess.run(['mkdir', str(marker)], capture_output=True, timeout=10)
        assert legacy.returncode != 0
    finally:
        first.communicate('release\n', timeout=10)
    assert first.returncode == 0
    assert not marker.exists()
    assert _run(wrapper, env).returncode == 0


def test_replaced_marker_is_never_removed_by_backend_or_gui_wrapper(tmp_path):
    wrapper, runtime, env, _ = _support(tmp_path)
    first = _start(wrapper, env)
    marker = runtime / '.installing'
    try:
        moved = tmp_path / 'original owned marker'
        marker.rename(moved)
        marker.mkdir()
        identity = marker.stat().st_ino
    finally:
        first.communicate('release\n', timeout=10)
    assert first.returncode == 0
    assert marker.is_dir() and marker.stat().st_ino == identity
    assert moved.is_dir()
    assert _run(wrapper, env).returncode == 75


def test_direct_backend_refuses_legacy_marker_and_releases_only_new_modern_leases(tmp_path):
    wrapper, runtime, env, _ = _support(tmp_path)
    marker = runtime / '.installing'
    marker.mkdir(parents=True)
    (marker / 'recovery-evidence').write_bytes(b'preserve exactly')
    identity = marker.stat().st_ino
    backend = wrapper.parent / 'install-online.sh'
    result = subprocess.run([str(backend), '--platform', 'macos', '--install-root', str(runtime)],
                            env=env, text=True, capture_output=True, timeout=10)
    assert result.returncode == 73, result.stdout + result.stderr
    assert marker.stat().st_ino == identity
    assert (marker / 'recovery-evidence').read_bytes() == b'preserve exactly'
    assert not (runtime / '.venv.spacr-update-lock').exists()
    assert not (runtime.parent / f'.{runtime.name}.spacr-installer-lock').exists()


def test_retained_rollback_keeps_all_three_leases_and_blocks_later_gui_start(tmp_path):
    wrapper, runtime, env, _ = _support(tmp_path)
    result = _run(wrapper, {**env, 'SPACR_TEST_RETAIN': '1', 'SPACR_TEST_EXIT': '7'})
    assert result.returncode == 7
    assert 'ready_opening' not in result.stdout
    leases = [runtime / '.installing', runtime / '.venv.spacr-update-lock',
              runtime.parent / f'.{runtime.name}.spacr-installer-lock']
    before = {path: path.stat().st_ino for path in leases}
    assert _run(wrapper, env).returncode == 75
    assert {path: path.stat().st_ino for path in leases} == before


def test_backend_failure_releases_owned_markers_without_false_ready_message(tmp_path):
    wrapper, runtime, env, _ = _support(tmp_path)
    result = _run(wrapper, {**env, 'SPACR_TEST_EXIT': '7'})
    assert result.returncode == 7
    assert 'ready_opening' not in result.stdout
    assert not (runtime / '.installing').exists()
    assert not (runtime / '.venv.spacr-update-lock').exists()
    assert not (runtime.parent / f'.{runtime.name}.spacr-installer-lock').exists()
