"""Actual Qt archive workers preserve unrelated files around each stream."""
import sys
import types

import pytest

from spacr.qt import hf_download as hf


def worker_case(kind, root):
    if kind == 'single':
        worker = hf._TarExampleWorker(root)
        worker.repo = 'owner/dataset'
        worker.archive = 'example.tar'
    else:
        worker = hf._ChosenArchivesWorker(root, archives=('example.tar',), repo='owner/dataset')
    seen = {'finished': [], 'progress': []}
    worker.finished.connect(lambda *values: seen['finished'].append(values))
    worker.progress.connect(lambda *values: seen['progress'].append(values))
    return worker, seen


class Response:
    headers = {'Content-Length': '3'}

    def __init__(self, chunks):
        self.chunks = chunks

    def raise_for_status(self):
        pass

    def iter_content(self, *, chunk_size):
        assert chunk_size == 1 << 20
        yield from self.chunks()


@pytest.mark.parametrize('kind', ['single', 'selected'])
@pytest.mark.parametrize('symbolic', [False, True])
def test_worker_download_does_not_own_preexisting_partial(kind, symbolic, tmp_path, monkeypatch):
    worker, seen = worker_case(kind, tmp_path)
    foreign = tmp_path / 'foreign.bin'
    foreign.write_bytes(b'foreign data')
    partial = tmp_path / 'example.tar.part'
    if symbolic:
        try:
            partial.symlink_to(foreign.name)
        except OSError as error:
            pytest.skip(f'this host cannot create symlinks: {error}')
    else:
        partial.write_bytes(b'other stream')
    before = partial.read_bytes()
    unpacked = []
    monkeypatch.setattr(hf, 'extract_example_archive', lambda archive, dest: unpacked.append(archive.read_bytes()))
    monkeypatch.setattr(hf, 'make_the_example_paths_absolute', lambda dest: None)
    monkeypatch.setitem(sys.modules, 'requests', types.SimpleNamespace(get=lambda *a, **k: Response(lambda: [b'new'])))
    worker.run()
    assert len(seen['finished']) == 1 and seen['finished'][0][0] is True
    assert unpacked == [b'new']
    assert seen['progress'] == [('example.tar', 0, 1), ('done', 1, 1)]
    assert foreign.read_bytes() == b'foreign data'
    assert partial.read_bytes() == before and partial.is_symlink() is symbolic
    assert sorted(p.name for p in tmp_path.iterdir()) == ['example.tar.part', 'foreign.bin']


@pytest.mark.parametrize('kind', ['single', 'selected'])
@pytest.mark.parametrize('failure', ['stream', 'short', 'cancel'])
def test_worker_failed_stream_removes_only_owned_stage(kind, failure, tmp_path, monkeypatch):
    worker, seen = worker_case(kind, tmp_path)
    final = tmp_path / 'example.tar'
    final.write_bytes(b'last good archive')
    partial = tmp_path / 'example.tar.part'
    partial.write_bytes(b'other stream')
    unpacked = []
    monkeypatch.setattr(hf, 'extract_example_archive', lambda archive, dest: unpacked.append(archive))
    monkeypatch.setattr(hf, 'make_the_example_paths_absolute', lambda dest: None)

    def chunks():
        if failure == 'cancel':
            worker.cancel()
        yield b'n'
        if failure == 'stream':
            raise OSError('interrupted stream')

    monkeypatch.setitem(sys.modules, 'requests', types.SimpleNamespace(get=lambda *a, **k: Response(chunks)))
    worker.run()
    assert len(seen['finished']) == 1 and seen['finished'][0][0] is False
    if failure == 'cancel':
        assert seen['finished'][0][3] == 'Cancelled by user.'
    assert not unpacked
    assert final.read_bytes() == b'last good archive'
    assert partial.read_bytes() == b'other stream'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['example.tar', 'example.tar.part']
