"""Raw volumetric TIFF ingest refuses every request it cannot honour."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
import tifffile

from spacr import io
from tests.test_raw_volumetric_tiffs_f551 import _settings, _write


@pytest.mark.parametrize("changes, message", [
    ({"timelapse": True}, "does not combine time and Z"),
    ({"z_axis": 2}, "z_axis must be 0"),
    ({"test_mode": True}, "turn test_mode off"),
    ({"illumination_correction": True}, "does not yet support"),
])
def test_unsupported_requests_are_refused(tmp_path, changes, message):
    _write(tmp_path)
    with pytest.raises(ValueError, match=message):
        io._preprocess_volume_tiffs(_settings(tmp_path, **changes))


def test_a_folder_without_tiff_volumes_is_refused(tmp_path):
    (tmp_path / "notes.png").write_bytes(b"x")
    with pytest.raises(ValueError, match="metadata-labelled ZYX TIFF"):
        io._preprocess_volume_tiffs(_settings(tmp_path))


def test_an_unmatched_filename_is_refused(tmp_path):
    _write(tmp_path)
    tifffile.imwrite(tmp_path / "random.tif", np.zeros((4, 8, 9), np.uint16),
                     metadata={"axes": "ZYX"})
    with pytest.raises(ValueError, match="filename metadata pattern"):
        io._preprocess_volume_tiffs(_settings(tmp_path))


def test_complex_or_multi_series_volumes_are_refused(tmp_path):
    path, _ = _write(tmp_path)
    tifffile.imwrite(path, np.zeros((4, 8, 9), np.complex64),
                     metadata={"axes": "ZYX"})
    with pytest.raises(ValueError, match="real numeric intensities"):
        io._preprocess_volume_tiffs(_settings(tmp_path))
    with tifffile.TiffWriter(path) as writer:
        writer.write(np.zeros((4, 8, 9), np.uint16), metadata={"axes": "ZYX"})
        writer.write(np.zeros((2, 3), np.uint8))
    with pytest.raises(ValueError, match="multiple TIFF series"):
        io._preprocess_volume_tiffs(_settings(tmp_path))


def test_an_object_channel_outside_the_source_is_refused(tmp_path):
    _write(tmp_path)
    with pytest.raises(ValueError, match="index the sorted source-channel"):
        io._preprocess_volume_tiffs(_settings(tmp_path, nucleus_channel=5))


def _ingested(tmp_path):
    _write(tmp_path)
    io._preprocess_volume_tiffs(_settings(tmp_path))
    return tmp_path / "stack" / ".spacr_volume_ingest.json"


def test_a_damaged_receipt_is_refused_on_reuse(tmp_path):
    receipt = _ingested(tmp_path)
    data = json.loads(receipt.read_text())
    receipt.write_text(json.dumps(dict(data, version=2)))
    with pytest.raises(ValueError, match="unsupported format"):
        io._preprocess_volume_tiffs(_settings(tmp_path))
    receipt.write_text(json.dumps(dict(data, stacks={})))
    with pytest.raises(ValueError, match="receipt is incomplete"):
        io._preprocess_volume_tiffs(_settings(tmp_path))


def test_an_extra_stack_in_the_output_is_refused(tmp_path):
    _ingested(tmp_path)
    np.save(tmp_path / "stack" / "plate1_A01_9_1.npy", np.zeros(1))
    with pytest.raises(ValueError, match="inventory differs"):
        io._preprocess_volume_tiffs(_settings(tmp_path))


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


def test_an_oversized_receipt_is_refused(tmp_path):
    receipt = _ingested(tmp_path)
    receipt.write_text(" " * (16 * 1024 * 1024 + 2))
    with pytest.raises(ValueError, match="exceeds 16 MiB"):
        io._preprocess_volume_tiffs(_settings(tmp_path))


def test_a_volume_with_nonfinite_intensities_is_refused(tmp_path):
    path, _ = _write(tmp_path)
    volume = np.ones((4, 8, 9), np.float32)
    volume[0, 0, 0] = np.nan
    tifffile.imwrite(path, volume, metadata={"axes": "ZYX"})
    with pytest.raises(ValueError, match="nonfinite intensities"):
        io._preprocess_volume_tiffs(_settings(tmp_path))


def test_an_input_changed_before_publication_is_refused(tmp_path, monkeypatch):
    path, _ = _write(tmp_path)
    real = io._volume_file_hash
    calls = []

    def drifting(target):
        digest = real(target)
        if str(target) == str(path):
            calls.append(1)
            if len(calls) == 3:
                return "changed"
        return digest

    monkeypatch.setattr(io, "_volume_file_hash", drifting)
    with pytest.raises(ValueError, match="changed before stack publication"):
        io._preprocess_volume_tiffs(_settings(tmp_path))
    assert not (tmp_path / "stack").exists()


def test_a_failed_publication_leaves_no_stack_even_if_cleanup_stumbles(
        tmp_path, monkeypatch):
    import shutil

    _write(tmp_path)

    def failing_link(source, destination):
        shutil.rmtree(os.path.dirname(source))
        raise OSError("link refused")

    real_rmdir = os.rmdir

    def stuck_rmdir(path, *a, **k):
        if os.path.basename(str(path)) == "stack":
            raise OSError("busy")
        return real_rmdir(path, *a, **k)

    monkeypatch.setattr(io.os, "link", failing_link)
    monkeypatch.setattr(io.os, "rmdir", stuck_rmdir)
    with pytest.raises(OSError, match="link refused"):
        io._preprocess_volume_tiffs(_settings(tmp_path))


def test_a_late_receipt_failure_unpublishes_the_stacks(tmp_path, monkeypatch):
    _write(tmp_path)
    real_link = os.link

    def receipt_refused(source, destination):
        if destination.endswith(".spacr_volume_ingest.json"):
            raise OSError("receipt refused")
        return real_link(source, destination)

    monkeypatch.setattr(io.os, "link", receipt_refused)
    with pytest.raises(OSError, match="receipt refused"):
        io._preprocess_volume_tiffs(_settings(tmp_path))
    assert not (tmp_path / "stack").exists()


def test_a_staging_folder_already_gone_after_publication_is_fine(tmp_path,
                                                                 monkeypatch):
    import shutil

    _write(tmp_path)
    real_link = os.link

    def link_then_tidy(source, destination):
        real_link(source, destination)
        if destination.endswith(".spacr_volume_ingest.json"):
            shutil.rmtree(os.path.dirname(source))

    monkeypatch.setattr(io.os, "link", link_then_tidy)
    io._preprocess_volume_tiffs(_settings(tmp_path))
    assert (tmp_path / "stack" / ".spacr_volume_ingest.json").exists()
