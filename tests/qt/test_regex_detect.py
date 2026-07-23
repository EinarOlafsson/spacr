"""Tests for spacr.qt.regex_detect + folder_metadata + multi_format."""
from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# regex_detect
# ---------------------------------------------------------------------------

def test_apply_regex_returns_records_for_cellvoyager():
    from spacr.qt.regex_detect import apply_regex, CELLVOYAGER
    files = [
        "plate1_A01_T0001F001L01A01Z01C01.tif",
        "plate1_A01_T0001F001L01A01Z01C02.tif",
        "plate1_B03_T0001F002L01A01Z01C01.tif",
    ]
    records, missed = apply_regex(files, CELLVOYAGER)
    assert len(records) == 3
    assert missed == []
    assert records[0].get("plateID") == "plate1"
    assert records[0].get("chanID") == "01"


def test_apply_regex_isolates_non_matching_files():
    from spacr.qt.regex_detect import apply_regex, CELLVOYAGER
    files = ["good_A01_T0001F001L01A01Z01C01.tif", "nope.txt"]
    records, missed = apply_regex(files, CELLVOYAGER)
    assert len(records) == 1
    assert missed == ["nope.txt"]


def test_apply_regex_handles_invalid_regex():
    from spacr.qt.regex_detect import apply_regex
    # unbalanced group — should not raise
    records, missed = apply_regex(["foo.tif"], "(?P<x>")
    assert records == []
    assert missed == ["foo.tif"]


def test_validate_records_flags_missing_chanid():
    from spacr.qt.regex_detect import (
        MetadataRecord, validate_records,
    )
    records = [MetadataRecord("a.tif", {"plateID": "p1", "wellID": "A01",
                                            "fieldID": "1"})]
    warnings = validate_records(records, multichannel=True)
    assert any("chanID" in w for w in warnings)


def test_validate_records_flags_missing_location_field():
    from spacr.qt.regex_detect import (
        MetadataRecord, validate_records,
    )
    records = [MetadataRecord("a.tif", {"chanID": "01"})]
    warnings = validate_records(records, multichannel=True)
    assert any("location" in w.lower() or "wellID" in w for w in warnings)


def test_validate_records_ok_for_singlechannel_with_field_only():
    from spacr.qt.regex_detect import (
        MetadataRecord, validate_records,
    )
    records = [MetadataRecord("a.tif", {"fieldID": "1"})]
    # Only the "no plateID" soft warning should fire, no hard warnings
    warnings = validate_records(records, multichannel=False)
    hard = [w for w in warnings if "Missing required" in w
                                     or "Missing location" in w]
    assert hard == []


def test_auto_detect_regex_picks_cellvoyager_for_cellvoyager_files():
    from spacr.qt.regex_detect import auto_detect_regex
    files = [
        "plate1_A01_T0001F001L01A01Z01C01.tif",
        "plate1_A01_T0001F001L01A01Z01C02.tif",
        "plate1_B03_T0001F002L01A01Z01C01.tif",
    ]
    pattern, label, hits = auto_detect_regex(files)
    assert label == "cellvoyager"
    assert hits == 3


def test_auto_detect_regex_synthesises_when_nothing_fits():
    from spacr.qt.regex_detect import auto_detect_regex
    files = [
        "myrun_W1_F01_C01.tif",
        "myrun_W1_F02_C01.tif",
        "myrun_W2_F03_C02.tif",
    ]
    pattern, label, hits = auto_detect_regex(files)
    # It's ok if we synthesise or if canonical is chosen; the point
    # is a non-None result that matches all files.
    import re as _re
    rx = _re.compile(pattern)
    assert all(rx.match(f) for f in files), \
        f"auto-detected regex {pattern!r} did not match all files"


def test_tabulate_records_produces_aligned_table():
    from spacr.qt.regex_detect import (
        MetadataRecord, tabulate_records,
    )
    records = [
        MetadataRecord("a.tif", {"plateID": "p1", "wellID": "A01",
                                    "chanID": "01"}),
        MetadataRecord("b.tif", {"plateID": "p1", "wellID": "A02",
                                    "chanID": "02"}),
    ]
    out = tabulate_records(records, max_rows=10)
    assert "plateID" in out
    assert "wellID"  in out
    assert "A01"     in out
    assert "b.tif"   in out


def test_tabulate_records_samples_when_more_than_max_rows():
    from spacr.qt.regex_detect import (
        MetadataRecord, tabulate_records,
    )
    records = [
        MetadataRecord(f"{i:03d}.tif", {"fieldID": str(i)})
        for i in range(30)
    ]
    out = tabulate_records(records, max_rows=10)
    # count rows other than header + rule (both start with two spaces)
    data_rows = [line for line in out.splitlines()
                    if line.startswith("  ")
                    and not line.startswith("  -")
                    and "fieldID" not in line]
    assert len(data_rows) == 10


# ---------------------------------------------------------------------------
# folder_metadata
# ---------------------------------------------------------------------------

def test_detect_folder_metadata_recognises_well_field_layout(tmp_path):
    from spacr.qt.folder_metadata import detect_folder_metadata
    root = tmp_path / "dataset"
    for well in ("A01", "A02"):
        for field in ("F01", "F02"):
            d = root / well / field
            d.mkdir(parents=True)
            (d / "ch1.tif").write_bytes(b"II*\x00")
    template = detect_folder_metadata(root)
    assert template is not None
    assert "well" in template.depth_labels
    assert "field" in template.depth_labels


def test_detect_folder_metadata_returns_none_when_no_signal(tmp_path):
    from spacr.qt.folder_metadata import detect_folder_metadata
    root = tmp_path / "empty_layout"
    (root / "aa" / "bb").mkdir(parents=True)
    (root / "aa" / "bb" / "img.tif").write_bytes(b"II*\x00")
    assert detect_folder_metadata(root) is None


def test_well_from_index_is_384_shaped():
    from spacr.qt.folder_metadata import _well_from_index
    assert _well_from_index(0)  == "A01"
    assert _well_from_index(23) == "A24"
    assert _well_from_index(24) == "B01"
    assert _well_from_index(383) == "P24"


def test_assign_missing_fields_generates_sequential_names(tmp_path):
    from spacr.qt.folder_metadata import assign_missing_fields
    files = [tmp_path / f"raw_{i}.tif" for i in range(3)]
    for f in files: f.write_bytes(b"II*\x00")
    mappings = assign_missing_fields(
        files, plate="plate1",
        have_well=False, have_field=False, have_channel=False,
    )
    assert len(mappings) == 3
    names = [m.canonical for m in mappings]
    assert all(n.startswith("plate1_A01_F") for n in names)
    assert all(n.endswith("_C01.tif") for n in names)


def test_save_filename_map_writes_excel_readable_csv(tmp_path):
    import csv
    from spacr.qt.folder_metadata import (
        assign_missing_fields, save_filename_map,
    )
    src = tmp_path / "src.tif"; src.write_bytes(b"II*\x00")
    mappings = assign_missing_fields([src], plate="plate1")
    csv_path = save_filename_map(tmp_path / "map.csv", mappings)
    assert csv_path.exists()
    with open(csv_path) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["original_path", "canonical", "plate",
                        "well", "field", "channel", "time"]
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# multi_format
# ---------------------------------------------------------------------------

def test_describe_npy_reports_shape_and_dtype(tmp_path):
    import numpy as np
    from spacr.qt.multi_format import describe_file
    arr = np.zeros((4, 32, 32, 3), dtype=np.uint16)
    p = tmp_path / "cube.npy"
    np.save(p, arr)
    desc = describe_file(p)
    assert desc is not None
    assert desc.kind == "npy"
    assert desc.n_fields == 4
    assert desc.shape == (32, 32)
    assert "uint16" in desc.dtype


def test_describe_npz_reports_multiple_arrays(tmp_path):
    import numpy as np
    from spacr.qt.multi_format import describe_file
    p = tmp_path / "stack.npz"
    np.savez(p,
             a=np.zeros((32, 32), dtype=np.uint16),
             b=np.zeros((32, 32), dtype=np.uint16),
             c=np.zeros((32, 32), dtype=np.uint16))
    desc = describe_file(p)
    assert desc is not None
    assert desc.kind == "npz"
    assert desc.n_fields >= 3


def test_describe_file_returns_none_for_random_file(tmp_path):
    from spacr.qt.multi_format import describe_file
    p = tmp_path / "not_a_dataset.txt"
    p.write_text("just some notes")
    assert describe_file(p) is None
