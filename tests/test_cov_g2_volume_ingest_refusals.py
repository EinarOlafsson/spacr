"""Raw volumetric TIFF ingest refuses every request it cannot honour."""
from __future__ import annotations


import pytest

from spacr import io
from tests.test_raw_volumetric_tiffs_f551 import _write


def test_a_volume_that_changes_while_hashed_is_refused(tmp_path, monkeypatch):
    path, _ = _write(tmp_path)
    stats = iter([1, 2])
    real_stat = io.os.stat

    class _Stat:
        def __init__(self, base, size):
            self.st_dev, self.st_ino = base.st_dev, base.st_ino
            self.st_size, self.st_mtime_ns = size, base.st_mtime_ns

    def changing(p, *a, **k):
        base = real_stat(p, *a, **k)
        if str(p) == str(path):
            return _Stat(base, next(stats))
        return base

    monkeypatch.setattr(io.os, "stat", changing)
    with pytest.raises(ValueError, match="changed while being read"):
        io._volume_file_hash(str(path))


