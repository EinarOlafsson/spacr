"""Tests for spacr.pipeline_v2 — FilenameMapper + streaming stack builder.

The Cellpose streaming stage is exercised in
tests/qt/test_e2e_pipeline.py under the ``slow`` mark; here we cover
the file-mapping + stack-building code paths which don't need torch.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers — build a cellvoyager-shaped tiny plate on disk
# ---------------------------------------------------------------------------

def _write_tif(path: Path, shape=(32, 32), fill=None) -> None:
    import tifffile
    if fill is None:
        arr = (np.random.randint(0, 65535, size=shape)
                 .astype(np.uint16))
    else:
        arr = np.full(shape, fill, dtype=np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), arr)


def _make_plate(tmp: Path, wells=("A01", "A02"), fields=(1, 2),
                channels=(1, 2, 3, 4)) -> Path:
    """Build a spacr-cellvoyager-shaped plate under ``tmp``."""
    plate = tmp / "plate1"
    plate.mkdir(parents=True, exist_ok=True)
    for w in wells:
        for f in fields:
            for c in channels:
                name = (f"plate1_{w}_T01F{f:02d}L01A01Z01"
                        f"C{c:02d}.tif")
                _write_tif(plate / name)
    return plate


# ---------------------------------------------------------------------------
# FilenameMapper
# ---------------------------------------------------------------------------

def test_discover_parses_cellvoyager_layout(tmp_path):
    from spacr.pipeline_v2 import FilenameMapper
    plate = _make_plate(tmp_path, wells=("A01",), fields=(1,),
                          channels=(1, 2))
    mp = FilenameMapper.discover(plate)
    assert mp.metadata_type == "cellvoyager"
    assert len(mp.records) == 2
    r0 = mp.records[0]
    assert r0.plate == "plate1"
    assert r0.well == "A01"
    assert r0.field == 1
    assert r0.channel in (1, 2)


def test_discover_groups_channels_of_same_field_together(tmp_path):
    from spacr.pipeline_v2 import FilenameMapper
    plate = _make_plate(tmp_path, wells=("A01",), fields=(1, 2),
                          channels=(1, 2, 3, 4))
    mp = FilenameMapper.discover(plate)
    by_field = mp.by_field()
    # 2 fields × 1 well = 2 stack ids; 4 records each
    assert len(by_field) == 2
    for recs in by_field.values():
        assert len(recs) == 4


def test_discover_raises_on_empty_folder(tmp_path):
    from spacr.pipeline_v2 import FilenameMapper
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        FilenameMapper.discover(empty)


def test_save_and_load_csv_roundtrip(tmp_path):
    from spacr.pipeline_v2 import FilenameMapper
    plate = _make_plate(tmp_path, wells=("A01",), fields=(1,),
                          channels=(1, 2))
    mp = FilenameMapper.discover(plate)
    csv_path = mp.save_csv(plate / "filename_map.csv")
    assert csv_path.exists()
    assert csv_path.with_suffix(".json").exists()

    mp2 = FilenameMapper.load_csv(csv_path)
    assert mp2.metadata_type == mp.metadata_type
    assert len(mp2.records) == len(mp.records)
    for a, b in zip(mp.records, mp2.records):
        assert a.plate == b.plate
        assert a.well == b.well
        assert a.field == b.field
        assert a.channel == b.channel


def test_filename_map_csv_opens_in_excel_style(tmp_path):
    """First row is header; every subsequent row has 8 columns."""
    from spacr.pipeline_v2 import FilenameMapper
    plate = _make_plate(tmp_path, wells=("A01",), fields=(1,),
                          channels=(1, 2))
    mp = FilenameMapper.discover(plate)
    csv_path = mp.save_csv(plate / "filename_map.csv")
    with open(csv_path) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["original_path", "plate", "well", "field",
                        "channel", "time", "z", "stack_field_id"]
    for r in rows[1:]:
        assert len(r) == 8


# ---------------------------------------------------------------------------
# stream_originals_to_stack
# ---------------------------------------------------------------------------

def test_stream_originals_writes_one_npy_per_field(tmp_path):
    from spacr.pipeline_v2 import FilenameMapper, stream_originals_to_stack
    plate = _make_plate(tmp_path, wells=("A01", "A02"), fields=(1, 2),
                          channels=(1, 2, 3, 4))
    mp = FilenameMapper.discover(plate)
    stacks = stream_originals_to_stack(plate, mp,
                                         channels=(1, 2, 3, 4))
    assert len(stacks) == 4   # 2 wells × 2 fields
    for s in stacks:
        arr = np.load(s.path)
        assert arr.shape == (32, 32, 4)
        assert arr.dtype == np.uint16


def test_stream_originals_records_channel_order_sidecar(tmp_path):
    from spacr.pipeline_v2 import FilenameMapper, stream_originals_to_stack
    plate = _make_plate(tmp_path, wells=("A01",), fields=(1,),
                          channels=(1, 2, 3, 4))
    mp = FilenameMapper.discover(plate)
    stream_originals_to_stack(
        plate, mp, channels=(1, 2, 3, 4),
        channel_names=["nucleus", "cell", "pathogen", "organelle"],
    )
    meta = json.loads((plate / "merged" / "channel_order.json").read_text())
    assert meta["image_channels"] == ["nucleus", "cell", "pathogen",
                                         "organelle"]
    assert meta["mask_channels"] == []   # populated by streaming Cellpose


def test_stream_originals_writes_filename_map(tmp_path):
    from spacr.pipeline_v2 import FilenameMapper, stream_originals_to_stack
    plate = _make_plate(tmp_path, wells=("A01",), fields=(1,),
                          channels=(1, 2))
    mp = FilenameMapper.discover(plate)
    stream_originals_to_stack(plate, mp, channels=(1, 2))
    csv_path = plate / "filename_map.csv"
    assert csv_path.exists()
    with open(csv_path) as f:
        rows = list(csv.reader(f))
    assert rows[0][0] == "original_path"
    assert len(rows) >= 3   # header + at least 2 image rows


def test_stream_originals_fills_zeros_for_missing_channel(tmp_path):
    """If a field is missing one channel, the stack still has C=len(channels)
    with a zero plane at that index — downstream code shouldn't crash."""
    from spacr.pipeline_v2 import FilenameMapper, stream_originals_to_stack
    plate = _make_plate(tmp_path, wells=("A01",), fields=(1,),
                          channels=(1, 2))    # only 2 channels present
    mp = FilenameMapper.discover(plate)
    stacks = stream_originals_to_stack(plate, mp,
                                         channels=(1, 2, 3, 4))
    arr = np.load(stacks[0].path)
    assert arr.shape[-1] == 4
    # Channels 3, 4 (indices 2, 3) should be all-zeros
    assert arr[..., 2].sum() == 0
    assert arr[..., 3].sum() == 0


def test_stream_originals_is_deterministic_across_runs(tmp_path):
    """Same source → same output byte-for-byte. Regression test in case
    someone shuffles a dict somewhere."""
    from spacr.pipeline_v2 import FilenameMapper, stream_originals_to_stack
    plate = _make_plate(tmp_path, wells=("A01",), fields=(1,),
                          channels=(1, 2, 3, 4))
    mp = FilenameMapper.discover(plate)

    # First run
    s1 = stream_originals_to_stack(plate, mp, channels=(1, 2, 3, 4))
    hash1 = [_hash_npy(s.path) for s in s1]

    # Wipe merged/, run again
    import shutil
    shutil.rmtree(plate / "merged")
    s2 = stream_originals_to_stack(plate, mp, channels=(1, 2, 3, 4))
    hash2 = [_hash_npy(s.path) for s in s2]

    assert hash1 == hash2


def _hash_npy(path: Path) -> str:
    import hashlib
    return hashlib.sha256(np.load(path).tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# StackFile bookkeeping
# ---------------------------------------------------------------------------

def test_stack_file_records_channel_names(tmp_path):
    from spacr.pipeline_v2 import FilenameMapper, stream_originals_to_stack
    plate = _make_plate(tmp_path, wells=("A01",), fields=(1,),
                          channels=(1, 2))
    mp = FilenameMapper.discover(plate)
    stacks = stream_originals_to_stack(
        plate, mp, channels=(1, 2),
        channel_names=["nucleus", "cell"],
    )
    assert stacks[0].channels == ["nucleus", "cell"]
    assert stacks[0].shape == (32, 32, 2)


def test_pipeline_style_defaults_to_v2():
    """v2 (in-memory streaming, no .npz on disk) is the default pipeline."""
    from spacr.settings import set_default_settings_preprocess_generate_masks
    s = set_default_settings_preprocess_generate_masks(settings={})
    assert s["pipeline_style"] == "v2"
