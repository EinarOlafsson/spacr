"""The live watch plate paints only fields whose output is committed."""
from __future__ import annotations

import json
import os
import sqlite3

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _ledger(source, fields):
    work = source / "spacr_watch"
    work.mkdir(exist_ok=True)
    (work / "watch_ledger.json").write_text(
        json.dumps({"src": str(source), "fields": fields}), encoding="utf-8")


def _committed(source, *keys):
    database = source / "spacr_watch" / "measurements" / "measurements.db"
    database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS spacr_watch_fields "
                           "(field TEXT PRIMARY KEY, merged_at REAL)")
        connection.executemany("INSERT INTO spacr_watch_fields VALUES (?, 1)",
                               [(key,) for key in keys])


@pytest.mark.parametrize("key,expected", [
    ("experiment_plate_A01_0001_001", ("experiment_plate", 1, 1)),
    ("plate_A01_A0001_001", ("plate", 1, 1)),
    ("plate_A00_0001_001", None),
])
def test_watch_field_well_parsing_preserves_custom_plate_ids(key, expected):
    from spacr.qt.screens.plate_view import _watch_plate_well

    assert _watch_plate_well(key) == expected


def test_mask_watch_fills_wells_as_fields_finish(qtbot, tmp_path, monkeypatch):
    from spacr.qt import preferences
    from spacr.qt.screens.plate_view import _WatchLivePlate

    monkeypatch.setattr(preferences, "_is_alpha_visible", lambda *_: True)
    plate = _WatchLivePlate()
    qtbot.addWidget(plate)
    try:
        plate.begin(str(tmp_path), "mask")
        assert plate._grid.well_count(1, 1) == 0
        assert not plate.isHidden()

        _ledger(tmp_path, {
            "experiment_plate_A01_0001_001": {"status": "done"},
            "experiment_plate_A01_0001_002": {"status": "waiting"},
            "experiment_plate_A02_0001_001": {"status": "failed"},
            "other_B03_0001_001": {"status": "done"},
            "unmapped_field": {"status": "done"},
        })
        plate.refresh()
        assert plate._plate.currentText() == "experiment_plate"
        assert plate._grid.well_count(1, 1) == 1
        assert plate._grid.well_value(1, 2) is None
        assert "1 fields have no plate well" in plate._summary.text()
        plate._plate.setCurrentText("other")
        assert plate._grid.well_count(2, 3) == 1

        _ledger(tmp_path, {
            "experiment_plate_A01_0001_001": {"status": "done"},
            "experiment_plate_A01_0001_002": {"status": "done"},
        })
        plate.refresh()
        assert plate._plate.currentText() == "experiment_plate"
        assert plate._grid.well_count(1, 1) == 2
        plate.finish()
        assert not plate._timer.isActive()
    finally:
        plate.close()


def test_live_plate_keeps_last_commit_when_snapshots_are_unreadable(
        qtbot, tmp_path, monkeypatch):
    from spacr.qt import preferences
    from spacr.qt.screens.plate_view import _WatchLivePlate

    monkeypatch.setattr(preferences, "_is_alpha_visible", lambda *_: True)
    key = "plate1_A01_0001_001"
    _ledger(tmp_path, {key: {"status": "done"}})
    _committed(tmp_path, key)
    plate = _WatchLivePlate()
    qtbot.addWidget(plate)
    plate.begin(str(tmp_path), "mask_measure")
    plate._timer.stop()
    assert plate._grid.well_count(1, 1) == 1

    work = tmp_path / "spacr_watch"
    for invalid_fields in ([], None, "partial"):
        _ledger(tmp_path, invalid_fields)
        plate.refresh()
        assert plate._grid.well_count(1, 1) == 1

    _ledger(tmp_path, {key: {"status": "done"}})
    database = work / "measurements" / "measurements.db"
    with sqlite3.connect(database) as connection:
        connection.execute("DROP TABLE spacr_watch_fields")
    plate.refresh()
    assert plate._grid.well_count(1, 1) == 1
    assert plate._plate.currentText() == "plate1"


@pytest.mark.parametrize("pipeline", ["mask_measure", "mask_measure_classify"])
def test_measured_watch_waits_for_combined_database_commit(
        qtbot, tmp_path, pipeline, monkeypatch):
    from spacr.qt import preferences
    from spacr.qt.screens.plate_view import _WatchLivePlate

    monkeypatch.setattr(preferences, "_is_alpha_visible", lambda *_: True)
    keys = ("plate1_A01_0001_001", "plate1_A01_0001_002")
    _ledger(tmp_path, {key: {"status": "done"} for key in keys})
    plate = _WatchLivePlate()
    qtbot.addWidget(plate)
    try:
        plate.begin(str(tmp_path), pipeline)
        assert plate._grid.well_count(1, 1) == 0
        _committed(tmp_path, keys[0])
        plate.refresh()
        assert plate._grid.well_count(1, 1) == 1
        _committed(tmp_path, keys[1])
        plate.refresh()
        assert plate._grid.well_count(1, 1) == 2
        (tmp_path / "spacr_watch" / "watch_ledger.json").write_text(
            "{partial", encoding="utf-8")
        plate.refresh()
        assert plate._grid.well_count(1, 1) == 2
    finally:
        plate.close()
