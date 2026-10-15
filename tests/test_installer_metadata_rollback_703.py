"""Failed installers restore the sidecars that describe the restored runtime."""
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys

import pytest

from spacr import install_cleanup as ic

ROOT = Path(__file__).resolve().parents[1]
UNIX = ROOT / 'packaging/online/install_spacr_unix.sh'


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _cli(*args):
    return subprocess.run([sys.executable, '-I', ic.__file__, 'installer-metadata', *map(str,args)],
                           capture_output=True, text=True, timeout=15)


def test_real_subprocess_restores_original_bytes_modes_links_and_absent_receipt(tmp_path):
    source = _write(tmp_path / 'installed/profile.json', 'old profile')
    source.chmod(0o640)
    os.utime(source, ns=(1234567890000000000,1234567890000000000))
    original = source.stat()
    receipt = tmp_path / 'installed/runtime-identity.json'
    user = _write(tmp_path / 'user/not-installer-data', 'valuable user data')
    launcher = tmp_path / 'installed/launcher'
    launcher.symlink_to(user)
    folder = tmp_path / 'snapshot'
    snapshot = _cli('snapshot', '--folder', folder, '--paths', source, receipt, launcher)
    assert snapshot.returncode == 0, snapshot.stdout + snapshot.stderr
    source.write_text('new profile')
    receipt.write_text('new identity')
    launcher.unlink()
    launcher.write_text('new launcher')
    for _ in range(2):
        restored = _cli('restore', '--folder', folder)
        assert restored.returncode == 0, restored.stdout + restored.stderr
        assert json.loads(restored.stdout)['state'] == 'restored'
        assert source.read_text() == 'old profile' and not receipt.exists()
        assert stat.S_IMODE(source.stat().st_mode) == 0o640
        assert source.stat().st_mtime_ns == original.st_mtime_ns
        assert launcher.is_symlink() and launcher.resolve() == user
        assert user.read_text() == 'valuable user data'
    assert (folder / 'metadata.json').exists()


@pytest.mark.parametrize('damage', ['backup-content', 'backup-link', 'journal-link', 'parent-link', 'traversal', 'duplicate'])
def test_restore_validates_all_backup_bindings_before_writing_any_target(tmp_path, damage):
    first = _write(tmp_path / 'live/first', 'original first')
    second = _write(tmp_path / 'live/second', 'original second')
    folder = tmp_path / 'snapshot'
    state = ic._installer_metadata_snapshot(str(folder), [str(first), str(second)])
    first.write_text('new first');second.write_text('new second')
    journal = folder / 'metadata.json'
    backup = folder / state['records'][1]['backup']
    if damage == 'backup-content': backup.write_text('corrupt backup')
    elif damage == 'backup-link':
        backup.unlink();backup.symlink_to(first)
    elif damage == 'journal-link':
        journal.rename(folder / 'original.json');journal.symlink_to(folder / 'original.json')
    elif damage == 'parent-link':
        first.parent.rename(tmp_path / 'original-live')
        (tmp_path / 'live').symlink_to(tmp_path / 'original-live', target_is_directory=True)
    elif damage == 'traversal':
        state['records'][1]['backup'] = '../live/first';journal.write_text(json.dumps(state))
    elif damage == 'duplicate':
        state['records'][1]['path'] = state['records'][0]['path'];journal.write_text(json.dumps(state))
    with pytest.raises((ValueError, OSError)):
        ic._installer_metadata_restore(str(folder))
    assert first.read_text() == 'new first' and second.read_text() == 'new second'
    assert journal.exists()


def test_directory_target_is_not_recursively_removed_and_backup_remains(tmp_path):
    source = _write(tmp_path / 'live/profile', 'original')
    folder = tmp_path / 'snapshot'
    ic._installer_metadata_snapshot(str(folder), [str(source)])
    source.unlink();source.mkdir();_write(source / 'new-user-data', 'kept')
    with pytest.raises(OSError): ic._installer_metadata_restore(str(folder))
    assert (source / 'new-user-data').read_text() == 'kept'
    assert any(path.is_file() and path.read_text() == 'original' for path in folder.iterdir())


def test_empty_snapshot_request_fails_without_creating_folder(tmp_path):
    folder = tmp_path / 'snapshot'
    result = _cli('snapshot', '--folder', folder)
    assert result.returncode == 1 and json.loads(result.stdout)['ok'] is False
    assert not folder.exists()


@pytest.mark.parametrize('corrupt_backup', [False, True])
def test_actual_linux_exit_trap_restores_runtime_and_matching_sidecars_or_retains_proof(tmp_path, corrupt_backup):
    root = tmp_path / 'installation'
    live = root / 'venv'
    old = root / 'previous-venv'
    _write(live / 'code', 'old runtime')
    profile = _write(root / 'install-profile.json', 'old profile')
    receipt = _write(root / 'runtime-identity.json', 'old receipt')
    launcher = _write(tmp_path / 'bin/spacr', 'old launcher')
    desktop = _write(tmp_path / 'applications/io.github.olafssonlab.spacr.desktop', 'old desktop')
    uninstaller = _write(root / 'uninstall-spacr.sh', 'old uninstaller')
    work = tmp_path / 'work';work.mkdir()
    folder = work / 'previous-metadata'
    cleanup = 'cleanup() {' + UNIX.read_text().split('cleanup() {',1)[1].split('\n}\ntrap cleanup EXIT',1)[0] + '\n}\n'
    bindings = {'work_dir':work,'VENV_DIR':live,'old_venv':old,'stage_profile':root/'stage-profile',
                'stage_venv':root/'stage-venv','metadata_backup':folder,
                'cleanup_python':sys.executable,'cleanup_module':ic.__file__}
    script = tmp_path / 'rollback.sh'
    commands = ['set -Eeuo pipefail', 'spacr_say() { printf "%s" "$1"; }']
    commands += [key+'='+shlex.quote(str(value)) for key,value in bindings.items()]
    commands += ['swap_started=0','metadata_touched=0',cleanup,'trap cleanup EXIT',
                 shlex.quote(sys.executable)+' -I '+shlex.quote(ic.__file__)+' installer-metadata snapshot --folder '+shlex.quote(str(folder))+' --paths '+
                 ' '.join(shlex.quote(str(path)) for path in [profile,receipt,launcher,desktop,uninstaller]),
                 'mv "$VENV_DIR" "$old_venv"','mkdir "$VENV_DIR"','printf new > "$VENV_DIR/code"',
                 'swap_started=1','metadata_touched=1']
    commands += ['printf new > '+shlex.quote(str(path)) for path in [profile,receipt,launcher,desktop,uninstaller]]
    if corrupt_backup:
        commands += ['printf corrupt > "$metadata_backup/0"']
    commands += ['exit 17']
    script.write_text('\n'.join(commands)+'\n')
    result = subprocess.run(['bash',str(script)],capture_output=True,text=True,timeout=20)
    assert result.returncode == 17, result.stderr
    assert (live/'code').read_text() == 'old runtime'
    if corrupt_backup:
        assert folder.exists() and (folder/'metadata.json').exists()
        assert profile.read_text() == 'new'
    else:
        assert profile.read_text() == 'old profile' and receipt.read_text() == 'old receipt'
        assert launcher.read_text() == 'old launcher' and desktop.read_text() == 'old desktop'
        assert uninstaller.read_text() == 'old uninstaller' and not work.exists()


def test_native_wiring_snapshots_before_exchange_and_never_streams_identity_into_existing_symlink():
    unix = UNIX.read_text()
    assert unix.index('installer-metadata snapshot') < unix.index('old_venv="$INSTALL_ROOT/.venv-previous-$$"')
    assert '> "$INSTALL_ROOT/runtime-identity.json"' not in unix
    assert '> "$work_dir/runtime-identity.new.json"' in unix
    assert unix.index('metadata_touched=1') < unix.index('mv "$work_dir/runtime-identity.new.json"')
    windows = (ROOT/'packaging/online/install_spacr_windows.ps1').read_text()
    assert windows.index('"installer-metadata", "snapshot"') < windows.index('$OldVenv = Join-Path')
    assert '$MetadataTouched -and $RuntimeRestored' in windows
    assert 'if (-not $RetainWorkDir)' in windows and 'throw $OriginalError' in windows
    assert '$Committed = $true' in windows and '-not $Committed -and $OldVenv' in windows
    assert '[System.Text.UTF8Encoding]::new($false)' in windows
