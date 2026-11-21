"""Local OME-Zarr handle reuse preserves pixels, invalidation and process isolation."""

from __future__ import annotations

import concurrent.futures
import json
import multiprocessing
import os
from pathlib import Path
import threading

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from spacr import ome_zarr as oz
from spacr.image_store import PlateWriter, read_image, read_labels


@pytest.fixture(autouse=True)
def fresh_opened_arrays():
    """Keep opened-handle evidence independent of other tests and release references."""
    oz._reset_open_array_cache()
    yield
    oz._reset_open_array_cache()


def _write(path, version="0.4", data=None):
    """Create a small image through the normal public OME-Zarr writer."""
    if data is None:
        data = np.arange(160, dtype=np.uint16).reshape(2, 8, 10)
    oz.write_ome_zarr(path, data, axes=("c", "y", "x"), ngff_version=version,
                     chunks=(1, 4, 5), compressor=None, write_empty_chunks=False)
    return data


def _observe_opens(monkeypatch):
    """Count actual public Zarr opens without changing returned array behavior."""
    original = zarr.open
    calls = []

    def opened(path, *args, **kwargs):
        """Retain the opened path and mode before delegating to Zarr."""
        calls.append((str(path), kwargs.get("mode")))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(zarr, "open", opened)
    return calls


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_repeated_full_and_region_reads_reuse_handle_without_sharing_pixels(tmp_path, monkeypatch, version):
    """One handle serves complete and sliced reads, but caller pixel edits never leak."""
    path = tmp_path / "image.zarr"
    expected = _write(path, version)
    calls = _observe_opens(monkeypatch)
    first = read_image(path)
    first[:] = 0
    second = read_image(path)
    region = oz.read_ome_zarr(path).read(region={"c": (0, 1), "y": (2, 6), "x": (3, 9)})
    np.testing.assert_array_equal(second, expected)
    np.testing.assert_array_equal(region, expected[:1, 2:6, 3:9])
    assert second.dtype == expected.dtype
    assert not np.shares_memory(first, second)
    assert calls == [(str((path / "0").resolve()), "r")]


def test_plate_image_and_labels_have_separate_reusable_handles(tmp_path, monkeypatch):
    """The normal image and label entry points share only their own array handles."""
    image = np.arange(160, dtype=np.uint16).reshape(2, 8, 10)
    labels = np.arange(80, dtype=np.uint32).reshape(8, 10)
    writer = PlateWriter(tmp_path, plate="P")
    writer.write_field("A01", 1, image, {"cell": labels})
    calls = _observe_opens(monkeypatch)
    for _ in range(2):
        np.testing.assert_array_equal(read_image(writer.path, key=("A01", 1)), image)
        np.testing.assert_array_equal(read_labels(writer.path, "cell", key=("A01", 1)), labels)
    assert len(calls) == 2
    assert {Path(path).relative_to(writer.path).as_posix() for path, _ in calls} == {
        "A/1/1/0", "A/1/1/labels/cell/0"}


def test_existing_chunk_pixel_updates_remain_visible_without_metadata_changes(tmp_path, monkeypatch):
    """Reusing a handle still reads newly written chunk bytes on each access."""
    path = tmp_path / "pixel-update.zarr"
    original = _write(path, "0.5", np.ones((2, 8, 10), dtype=np.uint16))
    calls = _observe_opens(monkeypatch)
    np.testing.assert_array_equal(read_image(path), original)
    before = oz._local_array_signature(path / "0")
    replacement = np.full_like(original, 9)
    writable = zarr.open(str(path / "0"), mode="a")
    writable[...] = replacement
    assert oz._local_array_signature(path / "0") == before
    np.testing.assert_array_equal(read_image(path), replacement)
    assert [mode for _, mode in calls] == ["r", "a"]


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_same_path_replacement_refreshes_shape_dtype_and_pixels(tmp_path, monkeypatch, version):
    """Overwriting a warmed group cannot retain old array metadata or pixel values."""
    path = tmp_path / "replaced.zarr"
    _write(path, version)
    calls = _observe_opens(monkeypatch)
    read_image(path)
    replacement = np.full((1, 3, 7), 70000, dtype=np.uint32)
    oz.write_ome_zarr(path, replacement, axes=("c", "y", "x"), ngff_version=version,
                     compressor=None, overwrite=True)
    actual = read_image(path)
    assert actual.shape == replacement.shape and actual.dtype == replacement.dtype
    np.testing.assert_array_equal(actual, replacement)
    assert len(calls) == 2


def test_in_place_array_resize_invalidates_the_opened_handle(tmp_path, monkeypatch):
    """A public Zarr metadata update changes subsequent complete-read geometry."""
    path = tmp_path / "resized.zarr"
    expected = _write(path)
    calls = _observe_opens(monkeypatch)
    read_image(path)
    zarr.open(str(path / "0"), mode="a").resize((2, 4, 10))
    actual = read_image(path)
    np.testing.assert_array_equal(actual, expected[:, :4, :])
    assert [mode for _, mode in calls] == ["r", "a", "r"]


def test_atomic_equal_length_metadata_replacement_refreshes_fill_value(tmp_path, monkeypatch):
    """Replacing metadata with its old mtime still invalidates the prior generation."""
    path = tmp_path / "fill.zarr"
    _write(path, data=np.zeros((2, 8, 10), dtype=np.uint16))
    assert {item.name for item in (path / "0").iterdir()} == {".zarray"}
    calls = _observe_opens(monkeypatch)
    assert not read_image(path).any()
    metadata = path / "0/.zarray"
    previous = metadata.stat()
    text = metadata.read_text(encoding="utf-8")
    assert '"fill_value": 0' in text
    replacement = text.replace('"fill_value": 0', '"fill_value": 7', 1)
    assert len(replacement.encode()) == previous.st_size
    temporary = metadata.with_name("replacement.json")
    temporary.write_text(replacement, encoding="utf-8")
    os.utime(temporary, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    os.replace(temporary, metadata)
    np.testing.assert_array_equal(read_image(path), np.full((2, 8, 10), 7, dtype=np.uint16))
    assert len(calls) == 2


def test_fresh_metadata_validation_and_deletion_are_not_hidden_by_reuse(tmp_path):
    """A warmed handle cannot conceal invalid group metadata or a removed array."""
    path = tmp_path / "invalidated.zarr"
    _write(path)
    read_image(path)
    attributes = path / ".zattrs"
    saved = attributes.read_bytes()
    payload = json.loads(saved)
    payload["multiscales"] = []
    attributes.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(oz.OmeZarrError, match="no `multiscales`"):
        read_image(path)
    attributes.write_bytes(saved)
    (path / "0/.zarray").unlink()
    with pytest.raises(oz.OmeZarrError, match="not a zarr array"):
        read_image(path)


def test_retargeting_a_store_symlink_reads_the_new_store(tmp_path, monkeypatch):
    """Path aliases follow their current target rather than the earlier opened array."""
    old = _write(tmp_path / "old.zarr")
    new = _write(tmp_path / "new.zarr", data=old + 1000)
    link = tmp_path / "alias.zarr"
    link.symlink_to(tmp_path / "old.zarr", target_is_directory=True)
    calls = _observe_opens(monkeypatch)
    np.testing.assert_array_equal(read_image(link), old)
    link.unlink()
    link.symlink_to(tmp_path / "new.zarr", target_is_directory=True)
    np.testing.assert_array_equal(read_image(link), new)
    assert len(calls) == 2


def test_public_runtime_configuration_changes_invalidate_the_handle(tmp_path, monkeypatch):
    """Zarr output memory order follows the active configuration on each call."""
    path = tmp_path / "order.zarr"
    expected = _write(path, "0.5")
    calls = _observe_opens(monkeypatch)
    with zarr.config.set({"array.order": "C"}):
        first = read_image(path)
        assert first.flags.c_contiguous
    with zarr.config.set({"array.order": "F"}):
        second = read_image(path)
        assert second.flags.f_contiguous and not second.flags.c_contiguous
    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(second, expected)
    assert len(calls) == 2


def test_nonserializable_public_config_uses_the_ordinary_open_path(tmp_path, monkeypatch):
    """An uncacheable config snapshot does not break an otherwise valid read."""
    path = tmp_path / "uncached.zarr"
    expected = _write(path)
    calls = _observe_opens(monkeypatch)

    def opaque_config():
        """Represent a valid extension object that JSON cannot fingerprint."""
        return {"extension": object()}

    monkeypatch.setattr(zarr.config, "to_dict", opaque_config)
    for _ in range(2):
        np.testing.assert_array_equal(read_image(path), expected)
    assert len(calls) == 2
    assert not oz._OPEN_ARRAY_CACHE


def test_missing_chunk_policy_changes_after_a_warmed_fill_read(tmp_path, monkeypatch):
    """A strict Zarr read must not inherit an earlier fill-missing-chunks policy."""
    from zarr.errors import ChunkNotFoundError

    path = tmp_path / "missing-chunks.zarr"
    expected = _write(path, data=np.zeros((2, 8, 10), dtype=np.uint16))
    assert {item.name for item in (path / "0").iterdir()} == {".zarray"}
    calls = _observe_opens(monkeypatch)
    with zarr.config.set({"array.read_missing_chunks": True}):
        np.testing.assert_array_equal(read_image(path), expected)
        with zarr.config.set({"array.read_missing_chunks": False}):
            with pytest.raises(ChunkNotFoundError):
                read_image(path)
        np.testing.assert_array_equal(read_image(path), expected)
    assert len(calls) == 2


def test_local_fallback_keeps_using_the_existing_reader(tmp_path, monkeypatch):
    """Explicit fallback never enters the opened-Zarr cache even when Zarr is installed."""
    path = tmp_path / "fallback.zarr"
    expected = _write(path)

    def unexpected_cache(store):
        """Fail if the explicit fallback starts using Zarr handles."""
        raise AssertionError("The fallback entered the opened-array cache")

    monkeypatch.setattr(oz, "_opened_local_array", unexpected_cache)
    np.testing.assert_array_equal(oz.read_ome_zarr(path).read(prefer_zarr=False), expected)


def test_concurrent_reads_reuse_a_warmed_handle_and_return_independent_arrays(tmp_path, monkeypatch):
    """Worker threads can read one array repeatedly without shared output buffers."""
    path = tmp_path / "threads.zarr"
    expected = _write(path, "0.5")
    calls = _observe_opens(monkeypatch)
    read_image(path)
    barrier = threading.Barrier(4)

    def read_together(index):
        """Start four reads concurrently and keep a different output mutation in each."""
        barrier.wait(timeout=15)
        data = read_image(path)
        np.testing.assert_array_equal(data, expected)
        data[0, 0, 0] = index + 1000
        return data

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(read_together, range(4)))
    assert [int(data[0, 0, 0]) for data in results] == [1000, 1001, 1002, 1003]
    np.testing.assert_array_equal(read_image(path), expected)
    assert len(calls) == 1


def _read_after_fork(path, connection):
    """Report inherited-cache state before exercising a normal read in the child."""
    try:
        connection.send((oz._OPEN_ARRAY_PID, len(oz._OPEN_ARRAY_CACHE),
                         int(read_image(path).sum())))
    finally:
        connection.close()


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="fork is unavailable")
def test_fork_resets_cached_handles_and_a_lock_held_by_another_thread(tmp_path):
    """A child must not inherit either opened handles or an unusably held cache lock."""
    path = tmp_path / "fork.zarr"
    expected = _write(path)
    read_image(path)
    assert oz._OPEN_ARRAY_CACHE
    held, release = threading.Event(), threading.Event()

    def hold_parent_lock():
        """Hold the cache lock across fork so retaining it would deadlock the child."""
        with oz._OPEN_ARRAY_LOCK:
            held.set()
            release.wait(timeout=30)

    holder = threading.Thread(target=hold_parent_lock)
    holder.start()
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_read_after_fork, args=(path, child))
    try:
        assert held.wait(timeout=15)
        process.start()
        child.close()
        release.set()
        assert parent.poll(30), "The child could not finish a read after fork"
        pid, inherited_count, total = parent.recv()
        process.join(timeout=15)
        assert process.exitcode == 0
        assert pid == process.pid and pid != os.getpid()
        assert inherited_count == 0 and total == int(expected.sum())
    finally:
        release.set()
        holder.join(timeout=15)
        if process.is_alive():
            process.terminate()
            process.join(timeout=15)
        parent.close()
        child.close()


def test_handle_cache_evicts_old_entries_without_retaining_pixels(tmp_path, monkeypatch):
    """A bounded cache reopens evicted handles and preserves all requested pixels."""
    paths = [tmp_path / f"image-{index}.zarr" for index in range(3)]
    expected = [_write(path, data=np.full((2, 8, 10), index, dtype=np.uint16))
                for index, path in enumerate(paths)]
    monkeypatch.setattr(oz, "_OPEN_ARRAY_LIMIT", 2)
    calls = _observe_opens(monkeypatch)
    for index in (0, 1, 0, 2, 1):
        np.testing.assert_array_equal(read_image(paths[index]), expected[index])
        assert len(oz._OPEN_ARRAY_CACHE) <= 2
    assert len(calls) == 4
