"""Pipeline JSON snapshot failures preserve the previous native file."""
import builtins
import json
import os
from pathlib import Path
import stat

import numpy as np
import pytest


@pytest.mark.parametrize('existing', [False, True])
def test_recursive_payload_never_creates_or_truncates_json(tmp_path, capsys, existing):
    from spacr.utils import _save_settings_json

    path = tmp_path / 'settings.json'
    if existing:
        path.write_bytes(b'previous valid snapshot')
    settings = {}
    settings['cycle'] = settings
    _save_settings_json(settings, str(path))
    assert 'Warning:' in capsys.readouterr().out
    assert path.read_bytes() == b'previous valid snapshot' if existing else not path.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == (['settings.json'] if existing else [])


class InterruptedWrite:
    def __init__(self, handle):
        self.handle = handle

    def __enter__(self):
        self.handle.__enter__()
        return self

    def __exit__(self, *args):
        return self.handle.__exit__(*args)

    def __getattr__(self, name):
        return getattr(self.handle, name)

    def write(self, text):
        self.handle.write(text[:4])
        self.handle.flush()
        raise OSError('interrupted native snapshot stream')


@pytest.mark.parametrize('existing', [False, True])
def test_interrupted_native_stream_preserves_old_file_and_cleans_stage(tmp_path, monkeypatch, capsys, existing):
    from spacr.utils import _save_settings_json

    path = tmp_path / 'settings.json'
    if existing:
        path.write_bytes(b'previous valid snapshot')
    original_open, original_fdopen = builtins.open, os.fdopen

    def opened(file, mode='r', *args, **kwargs):
        handle = original_open(file, mode, *args, **kwargs)
        return InterruptedWrite(handle) if 'w' in mode and Path(file).parent == tmp_path else handle

    def fdopened(fd, mode='r', *args, **kwargs):
        handle = original_fdopen(fd, mode, *args, **kwargs)
        return InterruptedWrite(handle) if 'w' in mode else handle

    with monkeypatch.context() as patch:
        patch.setattr(builtins, 'open', opened)
        patch.setattr(os, 'fdopen', fdopened)
        _save_settings_json({'value': 'new complete snapshot'}, str(path))
    assert 'Warning:' in capsys.readouterr().out
    assert path.read_bytes() == b'previous valid snapshot' if existing else not path.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == (['settings.json'] if existing else [])


@pytest.mark.parametrize('existing', [False, True])
def test_refused_final_publication_preserves_old_file(tmp_path, monkeypatch, capsys, existing):
    from spacr.utils import _save_settings_json

    path = tmp_path / 'settings.json'
    if existing:
        path.write_bytes(b'previous valid snapshot')
    original_replace = os.replace

    def refused(source, destination, *args, **kwargs):
        if Path(destination) == path:
            raise PermissionError('snapshot publication refused')
        return original_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, 'replace', refused)
    _save_settings_json({'value': 'new'}, str(path))
    assert 'Warning:' in capsys.readouterr().out
    assert path.read_bytes() == b'previous valid snapshot' if existing else not path.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == (['settings.json'] if existing else [])


def test_success_preserves_original_json_format_and_existing_mode(tmp_path):
    from spacr.utils import _save_settings_json

    path = tmp_path / 'settings.json'
    path.write_text('old', encoding="utf-8")
    path.chmod(0o640)
    settings = {'z': np.int64(3), 'a': (1, False, None), 'unicode': 'μm'}
    _save_settings_json(settings, str(path))
    assert path.read_text(encoding="utf-8") == json.dumps({'z': 3, 'a': [1, False, None], 'unicode': 'μm'}, indent=2, sort_keys=True)
    assert stat.S_IMODE(path.stat().st_mode) == 0o640
    assert sorted(p.name for p in tmp_path.iterdir()) == ['settings.json']


def test_existing_symlink_remains_a_link_to_the_updated_snapshot(tmp_path):
    from spacr.utils import _save_settings_json

    target = tmp_path / 'actual.json'
    target.write_text('old', encoding="utf-8")
    target.chmod(0o640)
    link = tmp_path / 'settings.json'
    try:
        link.symlink_to(target.name)
    except OSError as error:
        pytest.skip(str(error))
    _save_settings_json({'value': 3}, str(link))
    assert link.is_symlink()
    assert json.loads(target.read_text(encoding="utf-8")) == {'value': 3}
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert sorted(p.name for p in tmp_path.iterdir()) == ['actual.json', 'settings.json']


def test_missing_parent_still_warns_without_creating_folders(tmp_path, capsys):
    from spacr.utils import _save_settings_json

    path = tmp_path / 'missing' / 'settings.json'
    _save_settings_json({'value': 3}, str(path))
    assert 'Warning:' in capsys.readouterr().out
    assert not path.parent.exists()


def test_new_json_uses_normal_file_creation_permissions(tmp_path):
    from spacr.utils import _save_settings_json

    peer = tmp_path / 'normal.txt'
    with peer.open('w') as stream:
        stream.write('peer')
    path = tmp_path / 'settings.json'
    _save_settings_json({'value': 3}, str(path))
    assert json.loads(path.read_text(encoding="utf-8")) == {'value': 3}
    assert stat.S_IMODE(path.stat().st_mode) == stat.S_IMODE(peer.stat().st_mode)


def test_read_only_snapshot_still_refuses_overwrite(tmp_path, capsys):
    from spacr.utils import _save_settings_json

    path = tmp_path / 'settings.json'
    path.write_bytes(b'previous valid snapshot')
    path.chmod(0o440)
    if os.access(path, os.W_OK):
        pytest.skip('Current account can write read-only files')
    try:
        _save_settings_json({'value': 3}, str(path))
        assert 'Warning:' in capsys.readouterr().out
        assert path.read_bytes() == b'previous valid snapshot'
        assert stat.S_IMODE(path.stat().st_mode) == 0o440
        assert sorted(p.name for p in tmp_path.iterdir()) == ['settings.json']
    finally:
        path.chmod(0o640)
