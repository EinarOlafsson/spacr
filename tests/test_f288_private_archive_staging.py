"""Real archive staging files belong to one interrupted or concurrent attempt."""
from concurrent.futures import ThreadPoolExecutor
import sys
import threading
import types

import pytest

from spacr import example_archives


class Response:
    def __init__(self, chunks, size=3):
        self.chunks = chunks
        self.headers = {'Content-Length': str(size)}

    def raise_for_status(self):
        pass

    def iter_content(self, *, chunk_size):
        yield from self.chunks()


def test_concurrent_same_archive_attempts_publish_only_their_own_stream(tmp_path, monkeypatch):
    first_written = threading.Event()
    second_published = threading.Event()

    def first_chunks():
        yield b'A'
        first_written.set()
        assert second_published.wait(5)
        yield b'AA'

    responses = iter([Response(first_chunks), Response(lambda: [b'BBB'])])
    monkeypatch.setitem(sys.modules, 'requests', types.SimpleNamespace(get=lambda *a, **k: next(responses)))
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(example_archives.download_archive, 'owner/dataset', 'example.tar', tmp_path)
        try:
            assert first_written.wait(5)
            second = example_archives.download_archive('owner/dataset', 'example.tar', tmp_path)
            assert second.read_bytes() == b'BBB'
        finally:
            second_published.set()
        assert first.result(timeout=5) == second
    assert second.read_bytes() == b'AAA'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['example.tar']


@pytest.mark.needs_symlink
@pytest.mark.parametrize('symbolic', [False, True])
def test_existing_partial_path_is_not_download_owned(tmp_path, monkeypatch, symbolic):
    foreign = tmp_path / 'foreign.bin'
    foreign.write_bytes(b'foreign data')
    partial = tmp_path / 'example.tar.part'
    if symbolic:
        partial.symlink_to(foreign.name)
    else:
        partial.write_bytes(b'other attempt')
    original = partial.read_bytes()
    monkeypatch.setitem(sys.modules, 'requests', types.SimpleNamespace(get=lambda *a, **k: Response(lambda: [b'new'])))
    result = example_archives.download_archive('owner/dataset', 'example.tar', tmp_path)
    assert result.read_bytes() == b'new'
    assert foreign.read_bytes() == b'foreign data'
    assert partial.read_bytes() == original
    assert partial.is_symlink() is symbolic


@pytest.mark.parametrize('failure', ['stream', 'short', 'progress'])
def test_failed_attempt_preserves_last_good_and_unrelated_partial(tmp_path, monkeypatch, failure):
    destination = tmp_path / 'example.tar'
    destination.write_bytes(b'last good archive')
    unrelated = tmp_path / 'example.tar.part'
    unrelated.write_bytes(b'other attempt')

    def chunks():
        yield b'n'
        if failure == 'stream':
            raise OSError('interrupted stream')

    def progress(written, expected):
        if failure == 'progress':
            raise RuntimeError('cancelled progress')

    monkeypatch.setitem(sys.modules, 'requests', types.SimpleNamespace(get=lambda *a, **k: Response(chunks)))
    with pytest.raises((OSError, RuntimeError)):
        example_archives.download_archive('owner/dataset', 'example.tar', tmp_path, progress=progress)
    assert destination.read_bytes() == b'last good archive'
    assert unrelated.read_bytes() == b'other attempt'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['example.tar', 'example.tar.part']
