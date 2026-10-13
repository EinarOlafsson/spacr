"""Tests for the extraction-preview bridge + editable metadata table."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pytest


@dataclass
class _FakeDesc:
    """Stand-in for multi_format.DatasetDescription."""
    path: Path
    n_fields: int = 1
    n_channels: int = 1
    n_timepoints: int = 1
    n_slices: int = 1


# ---------------------------------------------------------------------------
# plan_container_extraction
# ---------------------------------------------------------------------------

def test_plan_container_extraction_enumerates_fields_channels_times():
    from spacr.qt.ingest_preview import plan_container_extraction
    desc = _FakeDesc(path=Path("/data/movie.nd2"),
                     n_fields=3, n_channels=2, n_timepoints=2, n_slices=5)
    rows = plan_container_extraction(desc)
    # fields × channels × times (z is MIP-collapsed, not enumerated)
    assert len(rows) == 3 * 2 * 2
    r = rows[0]
    assert r["well"] == "plate1_A01"
    assert r["original"] == "/data/movie.nd2"
    assert r["canonical"] == "plate1_A01_T0001F001L01C01.tif"
    # last row reflects the max indices
    assert rows[-1]["canonical"] == "plate1_A01_T0002F003L01C02.tif"


def test_plan_container_defaults_to_single_plane():
    from spacr.qt.ingest_preview import plan_container_extraction
    rows = plan_container_extraction(_FakeDesc(path=Path("x.tif")))
    assert len(rows) == 1
    assert rows[0]["field"] == 1 and rows[0]["channel"] == 1


# ---------------------------------------------------------------------------
# plan_folder_extraction
# ---------------------------------------------------------------------------

def test_plan_folder_extraction_maps_each_image(tmp_path):
    from spacr.qt.ingest_preview import plan_folder_extraction
    # A flat folder with no folder metadata → auto-assigned wells/fields.
    for i in range(4):
        (tmp_path / f"img_{i}.tif").write_bytes(b"II*\x00")
    rows = plan_folder_extraction(tmp_path)
    assert len(rows) == 4
    assert all(r["canonical"].endswith(".tif") for r in rows)
    # canonical names are unique
    assert len({r["canonical"] for r in rows}) == 4


def test_plan_folder_extraction_empty(tmp_path):
    from spacr.qt.ingest_preview import plan_folder_extraction
    assert plan_folder_extraction(tmp_path) == []
    assert plan_folder_extraction(tmp_path / "missing") == []


def test_summarize_rows():
    from spacr.qt.ingest_preview import summarize_rows
    assert "no images" in summarize_rows([])
    rows = [{"well": "plate1_A01", "field": 1, "channel": 1, "time": 1},
            {"well": "plate1_A01", "field": 1, "channel": 2, "time": 1}]
    s = summarize_rows(rows)
    assert "2 images" in s and "2 channel(s)" in s


# ---------------------------------------------------------------------------
# rows_to_mappings / filename_map round-trip
# ---------------------------------------------------------------------------

def test_rows_to_mappings_and_write_csv(tmp_path):
    from spacr.qt.ingest_preview import plan_container_extraction, rows_to_mappings
    from spacr.qt.folder_metadata import save_filename_map
    rows = plan_container_extraction(
        _FakeDesc(path=Path("/d/a.lif"), n_fields=2, n_channels=2))
    mappings = rows_to_mappings(rows)
    assert len(mappings) == 4
    csv_path = save_filename_map(tmp_path / "filename_map.csv", mappings)
    text = Path(csv_path).read_text()
    assert "original_path" in text and "canonical" in text
    assert text.count("\n") == 5  # header + 4 rows


# ---------------------------------------------------------------------------
# MetadataTablePanel (needs Qt)
# ---------------------------------------------------------------------------

def test_metadata_panel_edit_recomputes_filename(qtbot):
    from spacr.qt.widgets.metadata_table import MetadataTablePanel
    from spacr.qt.ingest_preview import ROW_COLUMNS
    rows = [{"original": "/d/a.nd2", "plate": "plate1", "well": "plate1_A01",
             "field": 1, "channel": 1, "time": 1,
             "canonical": "plate1_A01_T0001F001L01C01.tif"}]
    panel = MetadataTablePanel(rows)
    qtbot.addWidget(panel)
    # Edit the channel cell → filename recomputes live.
    chan_col = ROW_COLUMNS.index("channel")
    panel._table.item(0, chan_col).setText("3")
    out = panel.rows()
    assert out[0]["channel"] == 3
    assert out[0]["canonical"] == "plate1_A01_T0001F001L01C03.tif"


def test_metadata_panel_bad_int_reverts_to_one(qtbot):
    from spacr.qt.widgets.metadata_table import MetadataTablePanel
    from spacr.qt.ingest_preview import ROW_COLUMNS
    rows = [{"original": "x", "plate": "plate1", "well": "plate1_A01",
             "field": 5, "channel": 1, "time": 1, "canonical": ""}]
    panel = MetadataTablePanel(rows)
    qtbot.addWidget(panel)
    fld_col = ROW_COLUMNS.index("field")
    panel._table.item(0, fld_col).setText("not-a-number")
    assert panel.rows()[0]["field"] == 1


def test_metadata_panel_write_filename_map(qtbot, tmp_path):
    from spacr.qt.widgets.metadata_table import MetadataTablePanel
    rows = [{"original": "/d/a.nd2", "plate": "plate1", "well": "plate1_A01",
             "field": 1, "channel": 1, "time": 1,
             "canonical": "plate1_A01_T0001F001L01C01.tif"}]
    panel = MetadataTablePanel(rows)
    qtbot.addWidget(panel)
    out = panel.write_filename_map(tmp_path / "filename_map.csv")
    assert Path(out).is_file()
    assert "canonical" in Path(out).read_text()
