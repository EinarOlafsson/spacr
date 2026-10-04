"""Raw volumetric TIFF ingest refuses every request it cannot honour."""
from __future__ import annotations

import json

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
