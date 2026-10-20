"""Bounded local-file refusals; subprocess prevents a FIFO regression hanging tests."""
import os
import subprocess
import sys

import pytest

from spacr import attribution as cf


def test_fifo_refuses_without_waiting_for_a_writer(tmp_path):
    if not hasattr(os, 'mkfifo'):
        pytest.skip('POSIX FIFO primitive is unavailable')
    fifo = tmp_path / 'selected-checkpoint'
    os.mkfifo(fifo)
    script = '''
import sys
from spacr import attribution as cf
print('imports-ready', flush=True)
try:
    cf._cf_load_checkpoint(sys.argv[1], '0' * 64, {})
except ValueError as error:
    print(str(error), flush=True)
    raise SystemExit(0)
raise SystemExit(9)
'''
    child = subprocess.Popen([sys.executable, '-c', script, str(fifo)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = child.communicate(timeout=8)
    except subprocess.TimeoutExpired:
        child.kill()
        stdout, stderr = child.communicate()
        pytest.fail('selected FIFO did not refuse before blocking; ' + stdout + stderr)
    assert child.returncode == 0, stdout + stderr
    assert 'imports-ready' in stdout and 'counterfactual checkpoint' in stdout
    assert fifo.exists()


def test_growing_stream_cannot_read_past_the_explicit_checkpoint_budget(monkeypatch):
    monkeypatch.setattr(cf, '_CF_CHECKPOINT_MAX_BYTES', 8, raising=False)
    class Growing:
        reads = 0
        def seek(self, offset):
            assert offset == 0
        def read(self, size):
            self.reads += 1
            if self.reads > 5:
                pytest.fail('hash kept reading an appended stream beyond its budget')
            return b'ab'[:size]
    stream = Growing()
    with pytest.raises(ValueError, match='counterfactual checkpoint'):
        cf._cf_checkpoint_sha256(stream)
    assert stream.reads == 5


def test_missing_selected_checksum_refuses_before_accessing_any_path(monkeypatch):
    class UnopenedPath:
        def __fspath__(self):
            pytest.fail('missing checksum accessed selected path')

    monkeypatch.setattr(cf.torch, 'load', lambda *args, **kwargs: pytest.fail('missing checksum decoded state'))
    with pytest.raises(ValueError, match='counterfactual checkpoint'):
        cf._cf_load_checkpoint(UnopenedPath(), None, {})
