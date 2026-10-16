"""A failed frozen replacement keeps its prior runtime and unrelated data."""
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from spacr import install_cleanup as cleanup


def payload(root, version, *, manifest=True):
    root.mkdir(parents=True)
    (root / 'spacr.exe').write_text(version)
    internal = root / '_internal'
    metadata = internal / f'spacr-{version}.dist-info' / 'METADATA'
    metadata.parent.mkdir(parents=True)
    metadata.write_text(f'Name: spacr\nVersion: {version}\n')
    (internal / 'runtime.dll').write_bytes(version.encode())
    if manifest:
        files = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in root.rglob('*') if path.is_file()}
        (root / 'spacr-payload.json').write_text(json.dumps(
            {'schema': 1, 'version': version, 'files': files}))
    return root


def identity(argv):
    root = Path(argv[0]).parent
    Path(argv[-1]).write_text(json.dumps({
        'schema': 1, 'frozen': True, 'version': (root / 'spacr.exe').read_text(),
        'executable': argv[0], 'bundle_root': str(root),
        'import_root': str(root / '_internal'),
        'module': str(root / '_internal' / 'spacr' / '__init__.py')}))
    return 0, ''


@pytest.fixture
def installation(tmp_path):
    source = payload(tmp_path / 'download', '1.5.1.5')
    old = payload(tmp_path / 'installed', '1.5.1.4', manifest=False)
    (old / 'cache').mkdir()
    (old / 'cache' / 'annotations.csv').write_bytes(b'user data')
    (old / 'Uninstall.exe').write_bytes(b'existing uninstaller')
    return source, old


def test_legacy_replacement_keeps_unknown_data_and_recovery(installation):
    source, old = installation
    (old / '_internal' / 'user-model.bin').write_bytes(b'model')
    result = cleanup._install_frozen_windows(source, old, '1.5.1.5', run=identity)
    assert result['ok']
    assert (old / 'spacr.exe').read_text() == '1.5.1.5'
    assert (old / 'cache' / 'annotations.csv').read_bytes() == b'user data'
    assert (old / 'Uninstall.exe').read_bytes() == b'existing uninstaller'
    backup = Path(result['previous_runtime'])
    assert (backup / 'spacr.exe').read_text() == '1.5.1.4'
    assert (backup / '_internal' / 'user-model.bin').read_bytes() == b'model'
    assert not (old / '_internal' / 'user-model.bin').exists()


def test_known_payload_is_retired_and_same_version_reinstall_works(tmp_path):
    source = payload(tmp_path / 'download', '1.5.1.5')
    old = payload(tmp_path / 'installed', '1.5.1.5')
    (old / 'preferences.json').write_bytes(b'preferences')
    result = cleanup._install_frozen_windows(source, old, '1.5.1.5', run=identity)
    assert result['previous_runtime'] is None
    assert (old / 'preferences.json').read_bytes() == b'preferences'
    assert not list(tmp_path.glob('.spacr-frozen-install-*'))


@pytest.mark.parametrize('failure', ['copy', 'staged_identity', 'first_backup',
                                   'second_backup', 'first_activation',
                                   'second_activation', 'activated_identity'])
def test_failures_restore_the_working_copy(installation, monkeypatch, failure):
    source, old = installation
    replace = cleanup.os.replace
    copy = cleanup.shutil.copytree
    triggered = False

    def replace_once(src, dest):
        nonlocal triggered
        src, dest = Path(src), Path(dest)
        failures = {
            'first_backup': src == old / '_internal',
            'second_backup': src == old / 'spacr.exe',
            'first_activation': src.parent.name == 'new' and src.name == '_internal',
            'second_activation': src.parent.name == 'new' and src.name == 'spacr.exe',
        }
        if not triggered and failures.get(failure):
            triggered = True
            raise PermissionError('injected move failure')
        return replace(src, dest)

    def copy_once(src, dest, *args, **kwargs):
        if failure == 'copy':
            raise OSError('disk full')
        return copy(src, dest, *args, **kwargs)

    def verify(argv):
        live = Path(argv[0]).parent == old
        if (failure == 'activated_identity' and live
                or failure == 'staged_identity' and not live):
            return 2, 'import failed'
        return identity(argv)

    monkeypatch.setattr(cleanup.os, 'replace', replace_once)
    monkeypatch.setattr(cleanup.shutil, 'copytree', copy_once)
    with pytest.raises((OSError, ValueError)):
        cleanup._install_frozen_windows(source, old, '1.5.1.5', run=verify)
    assert (old / 'spacr.exe').read_text() == '1.5.1.4'
    assert (old / '_internal' / 'runtime.dll').read_bytes() == b'1.5.1.4'
    assert (old / 'cache' / 'annotations.csv').read_bytes() == b'user data'
    assert not list(old.parent.glob('.spacr-frozen-install-*'))
    assert not list(old.parent.glob('*.spacr-install-lock'))


def test_modified_activated_file_is_not_deleted_during_rollback(installation):
    source, old = installation

    def change(argv):
        if Path(argv[0]).parent == old:
            (old / 'spacr.exe').write_bytes(b'foreign replacement')
            return 2, 'failure'
        return identity(argv)

    with pytest.raises(ValueError):
        cleanup._install_frozen_windows(source, old, '1.5.1.5', run=change)
    assert (old / 'spacr.exe').read_bytes() == b'foreign replacement'
    backups = list(old.parent.glob('.spacr-frozen-install-*/previous/spacr.exe'))
    assert len(backups) == 1 and backups[0].read_text() == '1.5.1.4'
    assert (old / 'cache' / 'annotations.csv').read_bytes() == b'user data'


def test_newer_legacy_version_is_refused_without_running_or_mutating(tmp_path):
    source = payload(tmp_path / 'download', '1.5.1.5')
    old = payload(tmp_path / 'installed', '1.5.1.6', manifest=False)
    with pytest.raises(ValueError, match='downgrade'):
        cleanup._install_frozen_windows(source, old, '1.5.1.5',
                                        run=lambda argv: pytest.fail('must not run'))
    assert (old / 'spacr.exe').read_text() == '1.5.1.6'


@pytest.mark.parametrize('change', ['hash', 'extra', 'linked', 'unreadable'])
def test_untrusted_payload_refused_before_live_mutation(installation, monkeypatch, change):
    source, old = installation
    if change == 'hash':
        (source / 'spacr.exe').write_bytes(b'tampered')
    elif change == 'extra':
        (source / '_internal' / 'unlisted.dll').write_bytes(b'extra')
    elif change == 'linked':
        (source / '_internal' / 'runtime.dll').unlink()
        (source / '_internal' / 'runtime.dll').symlink_to(old / '_internal' / 'runtime.dll')
    else:
        walk = cleanup.os.walk

        def unreadable(root, *args, **kwargs):
            kwargs['onerror'](PermissionError('unreadable subtree'))
            yield from walk(root, *args, **kwargs)

        monkeypatch.setattr(cleanup.os, 'walk', unreadable)
    with pytest.raises((OSError, ValueError)):
        cleanup._install_frozen_windows(source, old, '1.5.1.5', run=identity)
    assert (old / 'spacr.exe').read_text() == '1.5.1.4'


def test_linked_legacy_metadata_is_not_used_for_version(installation, tmp_path):
    source, old = installation
    metadata = next((old / '_internal').glob('*.dist-info'))
    external = tmp_path / 'foreign.dist-info'
    shutil.move(metadata, external)
    metadata.symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match='identity_unverified'):
        cleanup._install_frozen_windows(source, old, '1.5.1.5', run=identity)
    assert (old / 'spacr.exe').read_text() == '1.5.1.4'


def test_identity_output_cannot_be_a_link(installation, tmp_path):
    source, _old = installation
    external = tmp_path / 'foreign.json'
    external.write_text('{}')

    def linked(argv):
        Path(argv[-1]).symlink_to(external)
        return 0, ''

    with pytest.raises((OSError, ValueError)):
        cleanup._frozen_windows_identity(source, '1.5.1.5', run=linked)
    assert external.read_text() == '{}'
