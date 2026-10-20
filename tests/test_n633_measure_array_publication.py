"""Measure transport arrays are retired only after complete publication."""
import errno
import os
from pathlib import Path

import numpy as np
import pytest

from spacr import example_archives as examples


def _transport(root, key="image"):
    values = np.arange(12, dtype=np.uint16).reshape(3, 4)
    archive = root / "field.npz"
    np.savez_compressed(archive, **{key: values})
    return archive, values


def test_interrupted_array_stream_retains_retryable_transport(tmp_path, monkeypatch):
    archive, values = _transport(tmp_path)
    original = archive.read_bytes()
    target = archive.with_suffix(".npy")

    def interrupted(output, array):
        if hasattr(output, "write"):
            output.write(b"incomplete numpy stream")
        else:
            Path(output).write_bytes(b"incomplete numpy stream")
        raise OSError("injected stream interruption")

    with monkeypatch.context() as scoped:
        scoped.setattr(np, "save", interrupted)
        examples.expand_measure_arrays(tmp_path)
    assert archive.read_bytes() == original
    assert not target.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["field.npz"]
    examples.expand_measure_arrays(tmp_path)
    assert np.array_equal(np.load(target), values)
    assert not archive.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["field.npy"]


@pytest.mark.parametrize("failure", ["fsync", "link"])
def test_failed_publication_keeps_transport_and_cleans_private_stage(
        tmp_path, monkeypatch, failure):
    archive, _ = _transport(tmp_path)
    original = archive.read_bytes()

    def interrupted(*args, **kwargs):
        raise OSError(errno.EOPNOTSUPP if failure == "link" else errno.EIO,
                      "injected publication failure")

    monkeypatch.setattr(os, failure, interrupted)
    examples.expand_measure_arrays(tmp_path)
    assert archive.read_bytes() == original
    assert sorted(p.name for p in tmp_path.iterdir()) == ["field.npz"]


@pytest.mark.parametrize("foreign", [
    "regular", pytest.param("dangling-link", marks=pytest.mark.needs_symlink)])
def test_concurrent_target_creation_never_overwrites_foreign_leaf(
        tmp_path, monkeypatch, foreign):
    archive, _ = _transport(tmp_path)
    original = archive.read_bytes()
    target = archive.with_suffix(".npy")
    native_link = os.link

    def publish(stage, destination):
        assert not target.exists()
        if foreign == "regular":
            target.write_bytes(b"concurrent user's bytes")
        else:
            target.symlink_to("missing-user-array")
        return native_link(stage, destination)

    monkeypatch.setattr(os, "link", publish)
    examples.expand_measure_arrays(tmp_path)
    assert archive.read_bytes() == original
    if foreign == "regular":
        assert target.read_bytes() == b"concurrent user's bytes"
    else:
        assert target.is_symlink()
        assert os.readlink(target) == "missing-user-array"
        assert not (tmp_path / "missing-user-array").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["field.npy", "field.npz"]


@pytest.mark.parametrize("key", ["image", "legacy-first-array"])
def test_success_preserves_array_dtype_shape_values_and_umask_mode(tmp_path, key):
    archive, values = _transport(tmp_path, key)
    prior_umask = os.umask(0o027)
    try:
        examples.expand_measure_arrays(tmp_path)
    finally:
        os.umask(prior_umask)
    target = archive.with_suffix(".npy")
    loaded = np.load(target)
    assert loaded.dtype == values.dtype
    assert loaded.shape == values.shape
    assert np.array_equal(loaded, values)
    if os.name == "posix":
        assert target.stat().st_mode & 0o777 == 0o640
    assert not archive.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["field.npy"]
