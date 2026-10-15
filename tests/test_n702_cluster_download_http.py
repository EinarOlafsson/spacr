"""Cluster nodes fetch a missing zoo model over HTTP once, verified, into the shared cache.

Every download goes to a local fake HTTP server on 127.0.0.1; no test reaches
a real network. The cache folder stands in for a shared file system that
several jobs and nodes see at the same path.
"""
import hashlib
import http.server
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import threading
import time

import pytest

from spacr import model_zoo as zoo


class _Server:
    """A threaded HTTP server that serves fixed bytes and counts each GET."""

    def __init__(self, files, delay=0.0):
        self.files = dict(files)
        self.delay = delay
        self.gets = []
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                owner.gets.append(self.path)
                body = owner.files.get(self.path)
                if body is None:
                    self.send_error(404)
                    return
                time.sleep(owner.delay)
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.httpd = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def url(self, path):
        return f'http://127.0.0.1:{self.httpd.server_address[1]}{path}'

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


PAYLOAD = b'exact cells r7 checkpoint bytes'
NEWER = b'newer cells r8 checkpoint bytes'


def _entry(server, payload=PAYLOAD, name='cells_r7.CP_model', key='cells_r7', version='7'):
    return zoo.ModelEntry(key=key, name=name, kind='cellpose', uri=server.url('/' + name),
                          version=version, sha256=hashlib.sha256(payload).hexdigest())


@pytest.fixture
def node(tmp_path, monkeypatch):
    shared = tmp_path / 'shared' / 'models'
    monkeypatch.setenv('SPACR_MODEL_CACHE', str(shared))
    monkeypatch.setenv('SPACR_MODEL_DOWNLOAD', 'on')
    monkeypatch.setenv('NO_PROXY', '127.0.0.1')
    monkeypatch.setenv('no_proxy', '127.0.0.1')
    monkeypatch.setattr(zoo, '_on_the_qt_gui_thread', lambda: False)
    return shared


def _lock_path(shared, entry):
    identity = hashlib.sha256(f'{entry.name}\0{entry.sha256}'.encode()).hexdigest()
    return shared / f'.model-{identity}.lock'


def test_missing_laptop_path_downloads_verified_bytes_and_logs_one_line(node, monkeypatch, caplog):
    with _Server({'/cells_r7.CP_model': PAYLOAD}) as server:
        entry = _entry(server)
        monkeypatch.setattr(zoo, 'catalogue', lambda **kwargs: [entry])
        with caplog.at_level(logging.INFO, logger='spacr.model_zoo'):
            result = zoo._ensure_model_file('/Users/laptop/.spacr/models/cells_r7.CP_model',
                                            setting='cell_model_name')
    assert result == node / 'cells_r7.CP_model'
    assert result.read_bytes() == PAYLOAD
    assert server.gets == ['/cells_r7.CP_model']
    assert sorted(p.name for p in node.iterdir()) == ['cells_r7.CP_model']
    lines = [r.getMessage() for r in caplog.records if 'downloaded zoo model' in r.getMessage()]
    assert lines == [
        f"cell_model_name: /Users/laptop/.spacr/models/cells_r7.CP_model was not found here; "
        f"downloaded zoo model cells_r7 ModelEntry.version='7' (sha256 {entry.sha256}) to {result}."]


def test_checksum_mismatch_names_the_setting_and_leaves_no_file(node, monkeypatch):
    with _Server({'/cells_r7.CP_model': b'tampered mirror bytes'}) as server:
        entry = _entry(server)
        monkeypatch.setattr(zoo, 'catalogue', lambda **kwargs: [entry])
        with pytest.raises(zoo.ChecksumMismatch) as error:
            zoo._ensure_model_file('/laptop/cells_r7.CP_model', setting='cell_model_name')
    message = str(error.value)
    assert "cell_model_name='/laptop/cells_r7.CP_model'" in message
    assert entry.sha256 in message
    assert list(node.iterdir()) == []


def test_offline_node_refuses_with_an_exact_command_that_works(node, monkeypatch):
    with _Server({'/cells_r7.CP_model': PAYLOAD}) as server:
        entry = _entry(server)
        monkeypatch.setattr(zoo, 'catalogue', lambda **kwargs: [entry])
        monkeypatch.setattr(zoo, '_planning_model_entries', lambda: [entry])
        monkeypatch.setattr(zoo, 'fetch', lambda *a, **k: pytest.fail('offline node fetched'))
        monkeypatch.setenv('SPACR_MODEL_DOWNLOAD', 'off')
        with pytest.raises(zoo.ModelUnreadable) as error:
            zoo._ensure_model_file('/laptop/cells_r7.CP_model', setting='cell_model_name')
        assert server.gets == []
        message = str(error.value)
        cached = node / 'cells_r7.CP_model'
        command = message.splitlines()[-1]
        assert 'SPACR_MODEL_DOWNLOAD' in message and str(cached) in message
        assert command == zoo._manual_model_copy_command(entry, cached)
        assert server.url('/cells_r7.CP_model') in command and entry.sha256 in command
        if not (shutil.which('curl') and shutil.which('sha256sum')):
            pytest.skip('curl or sha256sum is not installed')
        done = subprocess.run(['sh', '-c', command], capture_output=True, text=True, timeout=30,
                              env=dict(os.environ, NO_PROXY='127.0.0.1'))
        assert done.returncode == 0, done.stderr
    assert zoo._ensure_model_file('/laptop/cells_r7.CP_model') == cached
    assert cached.read_bytes() == PAYLOAD


def test_offline_command_for_a_file_mirror_copies_and_checks():
    entry = zoo.ModelEntry(key='k', name='m.CP_model', kind='cellpose',
                           uri='file:///nas/models/m.CP_model', version='1', sha256='AB' * 32)
    command = zoo._manual_model_copy_command(entry, Path('/scratch/my models/m.CP_model'))
    assert command == ("mkdir -p '/scratch/my models' && cp /nas/models/m.CP_model "
                       "'/scratch/my models/m.CP_model' && echo '" + 'ab' * 32
                       + "  /scratch/my models/m.CP_model' | sha256sum -c -")


def test_pinned_version_downloads_exactly_the_recorded_identity(node, tmp_path, monkeypatch):
    with _Server({'/cells_r7.CP_model': PAYLOAD, '/cells_r8.CP_model': NEWER}) as server:
        rows = [{'key': 'cells_r7', 'name': 'cells_r7.CP_model', 'kind': 'cellpose',
                 'version': '7', 'uri': server.url('/cells_r7.CP_model'),
                 'sha256': hashlib.sha256(PAYLOAD).hexdigest()},
                {'key': 'cells_r8', 'name': 'cells_r8.CP_model', 'kind': 'cellpose',
                 'version': '8', 'uri': server.url('/cells_r8.CP_model'),
                 'sha256': hashlib.sha256(NEWER).hexdigest()}]
        monkeypatch.setattr(zoo, 'BUNDLED_REMOTE_MODELS', rows)
        monkeypatch.setattr(zoo, 'catalogue', lambda **k: pytest.fail('pin used provider discovery'))
        request = '/Users/laptop/models/cells_r8.CP_model'
        record = zoo._model_reference_from_record(request, rows[0])
        settings = {'cell_model_name': request, 'model_metadata': {
            'schema_version': 1, 'settings': {'cell_model_name': record}}}
        resolved = zoo._resolve_pinned_model_file(settings, 'cell_model_name')
    assert resolved.read_bytes() == PAYLOAD
    assert server.gets == ['/cells_r7.CP_model']
    assert settings['cell_model_name'] == request


def test_pinned_version_refuses_when_the_zoo_now_declares_other_bytes(node, monkeypatch):
    with _Server({'/cells_r7.CP_model': NEWER}) as server:
        row = {'key': 'cells_r7', 'name': 'cells_r7.CP_model', 'kind': 'cellpose',
               'version': '7', 'uri': server.url('/cells_r7.CP_model'),
               'sha256': hashlib.sha256(PAYLOAD).hexdigest()}
        record = zoo._model_reference_from_record('/laptop/cells_r7.CP_model', row)
        moved = dict(row, version='7.1', sha256=hashlib.sha256(NEWER).hexdigest())
        monkeypatch.setattr(zoo, 'BUNDLED_REMOTE_MODELS', [moved])
        settings = {'cell_model_name': '/laptop/cells_r7.CP_model', 'model_metadata': {
            'schema_version': 1, 'settings': {'cell_model_name': record}}}
        with pytest.raises(zoo.ModelZooError):
            zoo._resolve_pinned_model_file(settings, 'cell_model_name')
    assert server.gets == []
    assert not node.exists() or list(node.iterdir()) == []


CHILD = textwrap.dedent('''
    import hashlib, sys
    from spacr import model_zoo as zoo
    url, sha = sys.argv[1:]
    entry = zoo.ModelEntry(key='cells_r7', name='cells_r7.CP_model', kind='cellpose',
                           uri=url, version='7', sha256=sha)
    zoo.catalogue = lambda **kwargs: [entry]
    zoo._on_the_qt_gui_thread = lambda: False
    print(zoo._ensure_model_file('/laptop/cells_r7.CP_model', setting='cell_model_name'))
''')


def test_parallel_jobs_sharing_a_cache_download_once(node):
    with _Server({'/cells_r7.CP_model': PAYLOAD}, delay=1.5) as server:
        entry = _entry(server)
        env = dict(os.environ, NO_PROXY='127.0.0.1', no_proxy='127.0.0.1')
        jobs = [subprocess.Popen([sys.executable, '-c', CHILD, entry.uri, entry.sha256],
                                 env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True) for _ in range(4)]
        results = [job.communicate(timeout=60) for job in jobs]
    assert [job.returncode for job in jobs] == [0] * 4, results
    assert server.gets == ['/cells_r7.CP_model']
    assert {Path(out.strip()) for out, _err in results} == {node / 'cells_r7.CP_model'}
    assert sorted(p.name for p in node.iterdir()) == ['cells_r7.CP_model']


def _foreign_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('424242\nspacr\nspacr-other-node.invalid\n')


def test_live_lock_from_another_node_is_waited_for(node, monkeypatch, caplog):
    monkeypatch.setenv('SPACR_MODEL_LOCK_STALE_SECONDS', '1')
    with _Server({'/cells_r7.CP_model': PAYLOAD}) as server:
        entry = _entry(server)
        monkeypatch.setattr(zoo, 'catalogue', lambda **kwargs: [entry])
        lock = _lock_path(node, entry)
        _foreign_lock(lock)
        stop = threading.Event()

        def other_node():
            for _ in range(15):
                if stop.wait(0.2):
                    return
                os.utime(lock)
            staged = node / '.other-node-part'
            staged.write_bytes(PAYLOAD)
            os.replace(staged, node / entry.name)
            lock.unlink()

        holder = threading.Thread(target=other_node)
        holder.start()
        started = time.monotonic()
        try:
            with caplog.at_level(logging.INFO, logger='spacr.model_zoo'):
                result = zoo._ensure_model_file('/laptop/cells_r7.CP_model')
        finally:
            stop.set()
            holder.join()
    assert time.monotonic() - started >= 2.5
    assert result.read_bytes() == PAYLOAD
    assert server.gets == []
    messages = [r.getMessage() for r in caplog.records]
    assert any(m.startswith('Waiting for another job to finish downloading model cells_r7')
               for m in messages)
    assert not any('Removing download lock' in m for m in messages)


def test_abandoned_lock_from_a_crashed_node_is_reclaimed(node, monkeypatch, caplog):
    monkeypatch.setenv('SPACR_MODEL_LOCK_STALE_SECONDS', '1')
    with _Server({'/cells_r7.CP_model': PAYLOAD}) as server:
        entry = _entry(server)
        monkeypatch.setattr(zoo, 'catalogue', lambda **kwargs: [entry])
        _foreign_lock(_lock_path(node, entry))
        started = time.monotonic()
        with caplog.at_level(logging.INFO, logger='spacr.model_zoo'):
            result = zoo._ensure_model_file('/laptop/cells_r7.CP_model')
    assert time.monotonic() - started < 20
    assert result.read_bytes() == PAYLOAD
    assert server.gets == ['/cells_r7.CP_model']
    assert any('Removing download lock' in r.getMessage() for r in caplog.records)
    assert sorted(p.name for p in node.iterdir()) == ['cells_r7.CP_model']


def test_holder_refreshes_its_lock_while_downloading(node, monkeypatch):
    monkeypatch.setenv('SPACR_MODEL_LOCK_STALE_SECONDS', '1')
    with _Server({'/cells_r7.CP_model': PAYLOAD}, delay=1.5) as server:
        entry = _entry(server)
        monkeypatch.setattr(zoo, 'catalogue', lambda **kwargs: [entry])
        lock = _lock_path(node, entry)
        seen = set()
        stop = threading.Event()

        def watch():
            while not stop.wait(0.05):
                try:
                    seen.add(os.stat(lock).st_mtime_ns)
                except OSError:
                    pass

        watcher = threading.Thread(target=watch)
        watcher.start()
        try:
            result = zoo._ensure_model_file('/laptop/cells_r7.CP_model')
        finally:
            stop.set()
            watcher.join()
    assert result.read_bytes() == PAYLOAD
    assert len(seen) >= 3
    assert not lock.exists()


@pytest.mark.parametrize('value, expected', [('', 600.0), ('5', 5.0), ('0', 600.0), ('x', 600.0)])
def test_stale_lock_setting(monkeypatch, value, expected):
    monkeypatch.setenv('SPACR_MODEL_LOCK_STALE_SECONDS', value)
    assert zoo._model_lock_stale_seconds() == expected
