"""Concurrent node jobs share one checksum-verified checkpoint download."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time

import pytest


CHILD = textwrap.dedent('''
    import hashlib, os, sys, time
    from pathlib import Path
    from spacr import model_zoo as zoo
    root, source, identity = map(Path, sys.argv[1:])
    payload = source.read_bytes()
    entry = zoo.ModelEntry(key='cells_r7', name=source.name, kind='cellpose',
        uri=source.as_uri(), version='7', sha256=hashlib.sha256(payload).hexdigest())
    zoo.catalogue = lambda **kwargs: [entry]
    zoo._spacr_home = lambda: root / 'node'
    def before_fetch():
        (root / (str(identity) + '.attempt')).touch()
        return False
    zoo._on_the_qt_gui_thread = before_fetch
    opener = zoo.open_uri
    def counted_open(uri, **kwargs):
        with (root / 'fetches.txt').open('a') as handle:
            handle.write(str(identity) + '\\n')
        deadline = time.monotonic() + 10
        while not (root / 'release').exists():
            if time.monotonic() > deadline:
                raise RuntimeError('test did not release temporary download')
            time.sleep(0.01)
        return opener(uri, **kwargs)
    zoo.open_uri = counted_open
    result = zoo._ensure_model_file('/laptop/' + entry.name)
    assert result.read_bytes() == payload
    print(str(result), flush=True)
''')


@pytest.mark.parametrize('previous_bytes', [False, True])
def test_concurrent_node_jobs_download_once_and_reuse_the_same_file(tmp_path, previous_bytes):
    source = tmp_path / 'cells_r7.CP_model'
    source.write_bytes(b'checked version seven checkpoint')
    old = tmp_path / 'node' / 'models' / source.name
    if previous_bytes:
        old.parent.mkdir(parents=True)
        old.write_bytes(b'preserved different-version weights')
    environment = dict(os.environ, SPACR_MODEL_DOWNLOAD='on')
    processes = [subprocess.Popen(
        [sys.executable, '-c', CHILD, str(tmp_path), str(source), str(index)],
        env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for index in range(2)]
    try:
        deadline = time.monotonic() + 10
        while not all((tmp_path / (str(index) + '.attempt')).exists() for index in range(2)):
            if time.monotonic() > deadline:
                raise AssertionError('temporary resolver children did not reach fetch policy')
            if any(process.poll() is not None for process in processes):
                raise AssertionError([process.communicate() for process in processes])
            time.sleep(0.01)
        (tmp_path / 'release').touch()
        results = [process.communicate(timeout=10) for process in processes]
        assert [process.returncode for process in processes] == [0, 0], results
        assert (tmp_path / 'fetches.txt').read_text().splitlines() in [['0'], ['1']]
        paths = [Path(stdout.strip()) for stdout, _stderr in results]
        assert paths[0] == paths[1]
        assert paths[0].read_bytes() == source.read_bytes()
        if previous_bytes:
            assert paths[0] != old
            assert old.read_bytes() == b'preserved different-version weights'
            assert set(paths[0].parent.iterdir()) == {old, paths[0]}
        else:
            assert list(paths[0].parent.iterdir()) == [paths[0]]
    finally:
        (tmp_path / 'release').touch()
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=10)


def _checkpoint(tmp_path, monkeypatch):
    import hashlib
    from spacr import model_zoo as zoo
    payload = b'checked version seven checkpoint'
    source = tmp_path / 'cells_r7.CP_model'
    source.write_bytes(payload)
    entry = zoo.ModelEntry(key='cells_r7', name=source.name, kind='cellpose',
                           uri=source.as_uri(), version='7',
                           sha256=hashlib.sha256(payload).hexdigest())
    monkeypatch.setenv('SPACR_MODEL_DOWNLOAD', 'on')
    monkeypatch.setattr(zoo, 'catalogue', lambda **kwargs: [entry])
    monkeypatch.setattr(zoo, '_spacr_home', lambda: tmp_path / 'node')
    monkeypatch.setattr(zoo, '_on_the_qt_gui_thread', lambda: False)
    return zoo, entry, source, payload


def test_failed_checksum_releases_lock_and_allows_a_corrected_retry(tmp_path, monkeypatch):
    import pytest
    zoo, entry, source, payload = _checkpoint(tmp_path, monkeypatch)
    source.write_bytes(b'corrupt mirror bytes')
    with pytest.raises(zoo.ChecksumMismatch):
        zoo._ensure_model_file(entry.key)
    assert list((tmp_path / 'node' / 'models').iterdir()) == []
    source.write_bytes(payload)
    result = zoo._ensure_model_file(entry.key)
    assert result.read_bytes() == payload
    assert list(result.parent.iterdir()) == [result]


def test_a_dead_downloader_lock_is_reclaimed_before_retry(tmp_path, monkeypatch):
    zoo, entry, source, payload = _checkpoint(tmp_path, monkeypatch)
    code = CHILD.replace(
        "result = zoo._ensure_model_file('/laptop/' + entry.name)",
        "zoo.fetch = lambda *args, **kwargs: os._exit(0)\n"
        "result = zoo._ensure_model_file('/laptop/' + entry.name)")
    result = subprocess.run(
        [sys.executable, '-c', code, str(tmp_path), str(source), 'dead'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert len(list((tmp_path / 'node' / 'models').glob('*.lock'))) == 1
    downloaded = zoo._ensure_model_file(entry.key)
    assert downloaded.read_bytes() == payload
    assert list(downloaded.parent.iterdir()) == [downloaded]


@pytest.mark.parametrize(('error_name', 'expected_errno'), [('PermissionError', 'EACCES'), ('UnknownError', 'EBUSY')])
def test_lock_permission_failure_preserves_existing_bytes_and_never_fetches(tmp_path, monkeypatch, error_name, expected_errno):
    import errno
    import pytest
    from PySide6 import QtCore
    zoo, entry, _source, _payload = _checkpoint(tmp_path, monkeypatch)
    cached = tmp_path / 'node' / 'models' / entry.name
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b'existing different checkpoint')
    original = QtCore.QLockFile

    class RefusedLock:
        PermissionError = original.PermissionError

        def __init__(self, path):
            self.path = path

        def setStaleLockTime(self, duration):
            pass

        def lock(self):
            return False

        def error(self):
            return getattr(original, error_name)

    monkeypatch.setattr(QtCore, 'QLockFile', RefusedLock)
    monkeypatch.setattr(zoo, 'fetch', lambda *args: pytest.fail('unlocked fetch'))
    with pytest.raises(OSError) as error:
        zoo._ensure_model_file(entry.key)
    assert error.value.errno == getattr(errno, expected_errno)
    assert cached.read_bytes() == b'existing different checkpoint'
    assert list(cached.parent.iterdir()) == [cached]
