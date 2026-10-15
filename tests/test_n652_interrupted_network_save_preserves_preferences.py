"""An interrupted preference write cannot replace the previous proxy settings."""
import errno
import io
import json

import pytest

from spacr import updater


def test_interrupted_network_write_preserves_settings_and_can_retry(tmp_path, monkeypatch):
    target = tmp_path / 'network.json'
    monkeypatch.setenv('SPACR_NETWORK_CONFIG', str(target))
    applied = []
    monkeypatch.setattr(updater, '_apply_network_settings', lambda: applied.append(True))
    updater._write_network_config('http://old.example.invalid:3128', '')
    original = target.read_bytes()
    applied.clear()
    opened = io.open
    partial_writes = []

    class InterruptedWriter:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def write(self, text):
            self.handle.write(text[:max(1, len(text) // 2)])
            self.handle.flush()
            partial_writes.append(True)
            raise OSError(errno.ENOSPC, 'controlled interrupted preference write')

        def __getattr__(self, name):
            return getattr(self.handle, name)

    def interrupt_open(file, mode='r', *args, **kwargs):
        handle = opened(file, mode, *args, **kwargs)
        return InterruptedWriter(handle) if 'w' in mode else handle

    with monkeypatch.context() as failure:
        failure.setattr(io, 'open', interrupt_open)
        for _ in range(2):
            with pytest.raises(OSError, match='interrupted preference write'):
                updater._write_network_config('http://new.example.invalid:8123', '')
            assert target.read_bytes() == original
            assert updater._read_network_config()['proxy'] == 'http://old.example.invalid:3128'
            assert list(tmp_path.iterdir()) == [target]
            assert applied == []
    assert len(partial_writes) == 2
    updater._write_network_config('http://new.example.invalid:8123', '')
    assert json.loads(target.read_text())['proxy'] == 'http://new.example.invalid:8123'
    assert applied == [True]
    assert list(tmp_path.iterdir()) == [target]
