"""Actual shell transaction leases protect disposable runtimes and rollback."""
import json
import os
from pathlib import Path
import select
import stat
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
UNIX = ROOT / 'packaging/online/install_spacr_unix.sh'
WINDOWS = ROOT / 'packaging/online/install_spacr_windows.ps1'


def _lease_source():
    return UNIX.read_text().split('# @SPACR_INSTALLER_LEASE_BEGIN@\n', 1)[1].split(
        '# @SPACR_INSTALLER_LEASE_END@', 1)[0]


def _script(tmp_path, body):
    path = tmp_path / ('driver-' + str(len(list(tmp_path.glob('driver-*')))) + '.sh')
    path.write_text('set -Eeuo pipefail\nPLATFORM=linux\n'
                    'spacr_say() { printf "%s\\n" "$*" >&2; }\n' + _lease_source() + body)
    return path


def _environment(tmp_path, root):
    return dict(os.environ, INSTALL_ROOT=str(root), HOME=str(tmp_path / 'home'))


def _run(script, tmp_path, root):
    return subprocess.run(['bash', str(script)], env=_environment(tmp_path, root),
                          text=True, capture_output=True, timeout=10)


def _ready(process, expected='acquired'):
    assert select.select([process.stdout], [], [], 10)[0], 'lease owner did not reach its checkpoint'
    assert process.stdout.readline().strip() == expected


@pytest.fixture
def owner(tmp_path):
    processes = []

    def start(root, body='printf "acquired\\n"\nread -r answer\n'):
        process = subprocess.Popen(['bash', str(_script(tmp_path, body))],
                                   env=_environment(tmp_path, root), stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        processes.append(process)
        _ready(process)
        return process

    yield start
    for process in processes:
        if process.poll() is None:
            process.communicate('release\n', timeout=10)


def test_second_real_installer_cannot_mutate_an_owned_root(tmp_path, owner):
    root = tmp_path / 'private application'
    first = owner(root)
    (root / 'user-output').write_bytes(b'untouched')
    refused = _run(_script(tmp_path, 'printf broken > "$INSTALL_ROOT/user-output"\n'), tmp_path, root)
    assert refused.returncode == 73
    assert 'already_running' in refused.stderr
    assert (root / 'user-output').read_bytes() == b'untouched'
    assert stat.S_IMODE((root / '.venv.spacr-update-lock').stat().st_mode) == 0o700
    first.communicate('release\n', timeout=10)
    assert first.returncode == 0
    assert not (root / '.venv.spacr-update-lock').exists()
    assert not (root.parent / ('.' + root.name + '.spacr-installer-lock')).exists()
    subsequent = _run(_script(tmp_path, 'printf installed > "$INSTALL_ROOT/verified"\n'), tmp_path, root)
    assert subsequent.returncode == 0
    assert (root / 'verified').read_text() == 'installed'


def test_real_exit_rollback_remains_exclusive_until_original_runtime_is_restored(tmp_path, owner):
    root = tmp_path / 'application'
    (root / 'venv').mkdir(parents=True)
    (root / 'venv/version').write_text('previous')
    (root / 'data').mkdir()
    (root / 'data/output').write_text('science')
    cleanup = UNIX.read_text().split('cleanup() {\n', 1)[1].split('\ntrap ', 1)[0]
    body = '''
VENV_DIR="$INSTALL_ROOT/venv"
old_venv="$INSTALL_ROOT/.venv-previous-fixture"
stage_venv="$INSTALL_ROOT/.venv-staging-fixture"
stage_profile="$INSTALL_ROOT/.install-profile-staging-fixture"
work_dir="$INSTALL_ROOT/fixture-work"
mkdir "$work_dir"
command mv "$VENV_DIR" "$old_venv"
mkdir "$VENV_DIR"
printf candidate > "$VENV_DIR/version"
swap_started=1
metadata_touched=0
cleanup() {
''' + cleanup + '''
mv() {
    if [[ "$1" == "$old_venv" ]]; then printf 'rollback\\n'; read -r answer; fi
    command mv "$@"
}
trap 'cleanup; spacr_release_installer_lease' EXIT
printf 'acquired\\n'
read -r answer
exit 7
'''
    first = owner(root, body)
    first.stdin.write('fail\n')
    first.stdin.flush()
    _ready(first, 'rollback')
    blocked = _run(_script(tmp_path, 'printf wrong > "$INSTALL_ROOT/venv/version"\n'), tmp_path, root)
    assert blocked.returncode == 73
    assert not (root / 'venv').exists()
    assert (root / '.venv-previous-fixture/version').read_text() == 'previous'
    first.communicate('restore\n', timeout=10)
    assert first.returncode == 7
    assert (root / 'venv/version').read_text() == 'previous'
    assert (root / 'data/output').read_text() == 'science'
    assert not (root / '.venv.spacr-update-lock').exists()


def test_different_roots_can_install_concurrently(tmp_path, owner):
    first = owner(tmp_path / 'first')
    root = tmp_path / 'second'
    second = _run(_script(tmp_path, 'printf success > "$INSTALL_ROOT/verified"\n'), tmp_path, root)
    assert second.returncode == 0
    assert first.poll() is None
    assert (root / 'verified').read_text() == 'success'


@pytest.mark.parametrize('existing', ['outside', 'runtime'])
def test_existing_or_interrupted_lease_is_not_stolen_or_changed(tmp_path, existing):
    root = tmp_path / 'application'
    root.mkdir()
    lease = (root.parent / ('.' + root.name + '.spacr-installer-lock')
             if existing == 'outside' else root / '.venv.spacr-update-lock')
    lease.mkdir(mode=0o700)
    (lease / 'unknown-evidence').write_text('preserve')
    before = lease.stat()
    result = _run(_script(tmp_path, 'printf wrong > "$INSTALL_ROOT/mutation"\n'), tmp_path, root)
    assert result.returncode == 73
    assert (lease / 'unknown-evidence').read_text() == 'preserve'
    assert (lease.stat().st_dev, lease.stat().st_ino) == (before.st_dev, before.st_ino)
    assert not (root / 'mutation').exists()
    if existing == 'runtime':
        assert not (root.parent / ('.' + root.name + '.spacr-installer-lock')).exists()


def test_path_alias_cannot_bypass_current_root_lease(tmp_path, owner):
    root = tmp_path / 'actual application'
    first = owner(root)
    alias = tmp_path / 'different name'
    alias.symlink_to(root, target_is_directory=True)
    result = _run(_script(tmp_path, 'touch "$INSTALL_ROOT/mutation"\n'), tmp_path, alias)
    assert result.returncode == 73
    assert first.poll() is None
    assert not (root / 'mutation').exists()


def test_cleanup_does_not_remove_a_replaced_lease(tmp_path, owner):
    root = tmp_path / 'application'
    first = owner(root)
    lease = root.parent / ('.' + root.name + '.spacr-installer-lock')
    retained = tmp_path / 'original-lease'
    lease.rename(retained)
    lease.mkdir()
    first.communicate('release\n', timeout=10)
    assert first.returncode == 0
    assert lease.is_dir()
    assert retained.is_dir()


def test_failed_recovery_retains_both_leases_for_explicit_resolution(tmp_path):
    root = tmp_path / 'application'
    result = _run(_script(tmp_path, 'SPACR_INSTALLER_RETAIN_LEASE=1\nexit 9\n'), tmp_path, root)
    assert result.returncode == 9
    assert (root / '.venv.spacr-update-lock').is_dir()
    assert (root.parent / ('.' + root.name + '.spacr-installer-lock')).is_dir()


def test_gui_parent_returns_before_lease_and_only_the_backend_owns_it():
    source = UNIX.read_text()
    gui_route = source.index('if [[ "$PLATFORM" == "linux" && "${SPACR_INSTALL_GUI_CHILD:-0}" != "1" ]]')
    lease = source.index('# @SPACR_INSTALLER_LEASE_BEGIN@')
    assert gui_route < lease < source.index('touch "$INSTALL_LOG"')
    assert 'exit $?' in source[gui_route:lease]
    assert 'SPACR_INSTALL_GUI_CHILD' not in _lease_source()
    assert "trap 'cleanup; spacr_release_installer_lease' EXIT" in source
    assert 'else SPACR_INSTALLER_RETAIN_LEASE=1; fi' in source


def test_windows_wrapper_acquires_exclusive_handles_before_logging_and_cleans_owned_handles_last():
    source = WINDOWS.read_text()
    assert '[System.IO.FileMode]::CreateNew' in source
    assert '[System.IO.FileShare]::None' in source
    assert 'Get-Process' not in source
    outer = source.index('Handle = (Open-SpacrInstallerLease $OuterLease)')
    root = source.index('New-Item -ItemType Directory -Force -Path $InstallRoot | Out-Null')
    inner = source.index('Handle = (Open-SpacrInstallerLease $RuntimeLease)')
    assert outer < root < inner < source.index('Start-Transcript')
    assert source.index('if ($DryRun') < outer
    assert source.index('installer-metadata", "restore"') < source.index('$Lease.Handle.Dispose()')
    assert 'if (-not $InstallerStarted) { throw $OriginalError }' in source
    assert 'if ($InstallerStarted -and -not $RetainWorkDir)' in source
    assert 'if (-not $RetainWorkDir) {\n            Remove-Item -LiteralPath $Lease.Path' in source


def test_busy_refusal_reuses_actual_existing_installer_locales():
    for language in ['en', 'sv', 'de', 'es', 'zh_CN', 'pt', 'hi', 'ko', 'is', 'fr']:
        values = json.loads((ROOT / 'packaging/i18n' / (language + '.json')).read_text())
        assert values['already_running']
    assert 'spacr_say already_running' in _lease_source()
    assert 'Get-SpacrInstallerMessage "already_running"' in WINDOWS.read_text()


def test_nsis_bootstrap_files_are_private_to_this_installer_process():
    source = (ROOT / 'packaging/online/spacr_online_installer.nsi').read_text()
    section = source.split('SecSpaCR\n', 1)[1].split('SectionEnd', 1)[0]
    assert section.index('InitPluginsDir') < section.index('SetOutPath')
    assert '$TEMP\\spaCR-online-installer' not in section
    assert 'SetOutPath "$PLUGINSDIR\\spaCR-online-installer"' in section
    assert '-File "$PLUGINSDIR\\spaCR-online-installer\\install_spacr_windows.ps1"' in section
    assert 'Delete "$PLUGINSDIR\\spaCR-online-installer\\install_spacr_windows.ps1"' in section
