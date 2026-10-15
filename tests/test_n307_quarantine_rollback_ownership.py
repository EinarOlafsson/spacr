"""Fallback quarantine failures never delete a different destination file."""
import errno
import os

import numpy as np
import pytest

from spacr import qc_quarantine as quarantine


@pytest.fixture
def transfer(tmp_path, monkeypatch):
    source = tmp_path / "source.npy"
    np.save(source, np.arange(12, dtype=np.uint16).reshape(3, 4))
    destination = tmp_path / "destination.npy"
    foreign = tmp_path / "foreign.npy"
    foreign.write_bytes(b"unrelated existing bytes")
    def unsupported_link(*args, **kwargs):
        raise OSError(errno.EXDEV, "simulated link across devices")
    monkeypatch.setattr(quarantine.os, "link", unsupported_link)
    return source, destination, foreign


def _replacement(destination, foreign, kind):
    if kind == "symlink":
        destination.symlink_to(foreign)
    else:
        destination.write_bytes(foreign.read_bytes())


@pytest.mark.parametrize("kind", ["regular", "symlink"])
def test_exclusive_creation_refusal_preserves_foreign_destination(transfer, monkeypatch, kind):
    source, destination, foreign = transfer
    original = source.read_bytes()
    actual_open = os.open
    def concurrent_creation(path, flags, *args, **kwargs):
        if os.fspath(path) == str(destination):
            _replacement(destination, foreign, kind)
        return actual_open(path, flags, *args, **kwargs)
    monkeypatch.setattr(quarantine.os, "open", concurrent_creation)
    with pytest.raises(FileExistsError):
        quarantine._move_without_overwrite(source, destination)
    assert source.read_bytes() == original
    assert destination.read_bytes() == b"unrelated existing bytes"
    assert destination.is_symlink() == (kind == "symlink")
    assert foreign.read_bytes() == b"unrelated existing bytes"


@pytest.mark.parametrize("kind", ["regular", "symlink"])
def test_failed_copy_preserves_replacement_destination(transfer, monkeypatch, kind):
    source, destination, foreign = transfer
    original = source.read_bytes()
    def interrupted(reader, writer, *args, **kwargs):
        writer.write(reader.read(7))
        destination.unlink()
        _replacement(destination, foreign, kind)
        raise OSError("interrupted after destination replacement")
    monkeypatch.setattr(quarantine.shutil, "copyfileobj", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        quarantine._move_without_overwrite(source, destination)
    assert source.read_bytes() == original
    assert destination.read_bytes() == b"unrelated existing bytes"
    assert destination.is_symlink() == (kind == "symlink")
    assert foreign.read_bytes() == b"unrelated existing bytes"


def test_failed_copy_removes_its_own_partial_destination(transfer, monkeypatch):
    source, destination, foreign = transfer
    original = source.read_bytes()
    def interrupted(reader, writer, *args, **kwargs):
        writer.write(reader.read(7))
        raise OSError("interrupted partial copy")
    monkeypatch.setattr(quarantine.shutil, "copyfileobj", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        quarantine._move_without_overwrite(source, destination)
    assert source.read_bytes() == original
    assert not destination.exists()
    assert foreign.read_bytes() == b"unrelated existing bytes"


def test_successful_fallback_keeps_bytes_mode_and_move_semantics(transfer):
    source, destination, foreign = transfer
    source.chmod(0o640)
    original = source.read_bytes()
    quarantine._move_without_overwrite(source, destination)
    assert not source.exists()
    assert destination.read_bytes() == original
    assert destination.stat().st_mode & 0o777 == 0o640
    np.testing.assert_array_equal(np.load(destination), np.arange(12, dtype=np.uint16).reshape(3, 4))
    assert foreign.read_bytes() == b"unrelated existing bytes"
