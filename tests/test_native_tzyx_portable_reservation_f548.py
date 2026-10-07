"""Native T/Z staging must reserve disk on systems without POSIX helpers."""

import errno
import os
from types import SimpleNamespace

import numpy as np
import pytest

from spacr import io


def test_native_workspace_checks_free_space_without_statvfs(tmp_path, monkeypatch):
    observed = []
    monkeypatch.delattr(io.os, "statvfs", raising=False)

    def disk_usage(path):
        observed.append(os.fspath(path))
        return SimpleNamespace(free=0)

    monkeypatch.setattr(io.shutil, "disk_usage", disk_usage)
    with pytest.raises(OSError) as error:
        io._native_workspace_preflight(
            tmp_path, (2, 2, 2, 1), 1, np.uint16, ("field.tif",))

    assert error.value.errno == errno.ENOSPC
    assert observed == [os.fspath(tmp_path)]
    monkeypatch.setattr(io.shutil, "disk_usage",
                        lambda path: SimpleNamespace(free=1024 * 1024))
    io._native_workspace_preflight(
        tmp_path, (2, 2, 2, 1), 1, np.uint16, ("field.tif",))


def test_native_map_reservation_writes_real_bytes_without_posix_helpers(
        tmp_path, monkeypatch):
    path = tmp_path / "selected.npy"
    mapped = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(2, 3, 4))
    expected = np.arange(mapped.size, dtype=np.float32).reshape(mapped.shape)
    monkeypatch.delattr(io.os, "posix_fallocate", raising=False)
    monkeypatch.delattr(io.os, "pwrite", raising=False)

    try:
        io._reserve_private_memmap(mapped)
        mapped[:] = expected
    finally:
        io._close_private_memmap(mapped)

    np.testing.assert_array_equal(np.load(path, allow_pickle=False), expected)


def test_native_map_reservation_refuses_zero_byte_fallback_write(
        tmp_path, monkeypatch):
    path = tmp_path / "selected.npy"
    mapped = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(2, 3, 4))
    writes = []
    monkeypatch.delattr(io.os, "posix_fallocate", raising=False)
    monkeypatch.delattr(io.os, "pwrite", raising=False)

    def no_write(descriptor, data):
        writes.append((descriptor, len(data)))
        return 0

    monkeypatch.setattr(io.os, "write", no_write)
    try:
        with pytest.raises(OSError) as error:
            io._reserve_private_memmap(mapped)
        assert error.value.errno == errno.ENOSPC
        assert writes and writes[0][1] > 0
    finally:
        io._close_private_memmap(mapped)
        path.unlink()

    assert not path.exists()
