"""Control Charts: the file dialogs, the form guards and the table picker.

A dialog the user dismisses writes and loads nothing; an export asked for
before anything was scored says so; a widget signal that arrives before a
table is loaded, or while one is loading, charts nothing; a form the engine
refuses becomes a sentence; and the table picker re-reads the open database.
"""
from __future__ import annotations

import os
import sqlite3

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

pytestmark = pytest.mark.qt

from spacr.qt.screens import control_chart as cc  # noqa: E402


def _campaign(plates: int = 30) -> pd.DataFrame:
    """Two controls, two wells each; the negative steps up after plate 20."""
    rows = []
    for index in range(plates):
        drift = 0.0 if index < 20 else 2.5 * (index - 19)
        for level, base in (("neg", 100.0), ("pos", 20.0)):
            for offset in (-1.0, 1.0):
                rows.append({"plateID": f"P{index + 1:02d}",
                             "run_order": index,
                             "well_type": level,
                             "batch": "A" if index % 2 else "B",
                             "signal": base + offset
                                       + (drift if level == "neg" else 0.0)})
    return pd.DataFrame(rows)


def _database(tmp_path) -> str:
    """A two-table measurement database; ``nucleus`` holds 13 plates."""
    path = str(tmp_path / "measurements.db")
    with sqlite3.connect(path) as db:
        _campaign(12).to_sql("cell", db, index=False)
        _campaign(13).to_sql("nucleus", db, index=False)
    return path


@pytest.fixture
def screen(qtbot):
    made = cc.ControlChartScreen(threaded=False)
    qtbot.addWidget(made)
    return made


def _levels(made):
    return [made._levels.item(i).text() for i in range(made._levels.count())]


def _items(box):
    return [box.itemText(i) for i in range(box.count())]


def _dialogs(monkeypatch, *, folder, table, save):
    """Answer the three QFileDialog statics; return the captions asked."""
    asked = []

    def _folder(*args, **kwargs):
        asked.append(args[1])
        return folder

    def _open(*args, **kwargs):
        asked.append(args[1])
        return table, ""

    def _save(*args, **kwargs):
        asked.append(args[1:3])
        return save, ""

    monkeypatch.setattr(cc.QFileDialog, "getExistingDirectory",
                        staticmethod(_folder))
    monkeypatch.setattr(cc.QFileDialog, "getOpenFileName", staticmethod(_open))
    monkeypatch.setattr(cc.QFileDialog, "getSaveFileName", staticmethod(_save))
    return asked


def test_the_file_dialogs_pass_on_a_chosen_path_and_drop_a_dismissed_one(
        screen, monkeypatch, tmp_path, subtests):
    """Every chooser hands its answer on unchanged, or does nothing on cancel."""
    with subtests.test(msg="exports before scoring ask nothing"):
        asked = _dialogs(monkeypatch, folder=str(tmp_path), table="",
                         save="")
        screen.choose_hit_export()
        assert screen._source.text() == "Nothing scored yet."
        screen._source.setText("")
        screen._choose_anomaly_export()
        assert screen._source.text() == "Nothing scored yet."
        screen._source.setText("")
        assert screen._export_anomalies("/tmp/never") is None
        assert screen._source.text() == "Nothing scored yet."
        screen._source.setText("")
        assert screen.export_hits("/tmp/never") is None
        assert screen._source.text() == "Nothing scored yet."
        assert screen.export_points(str(tmp_path / "x.csv")) is None
        assert screen._source.text() == "Nothing charted yet."
        assert asked == []
        assert not os.path.exists("/tmp/never")

    written, loaded = [], []
    monkeypatch.setattr(screen, "export_hits", written.append)
    monkeypatch.setattr(screen, "_export_anomalies", written.append)
    monkeypatch.setattr(screen, "_load_compounds", loaded.append)
    screen._hit_result = object()
    screen._anomaly = object()
    for case_id, folder, table, want_written, want_loaded in [
            ("dismissed", "", "", [], []),
            ("chosen", str(tmp_path), "compounds.csv",
             [str(tmp_path), str(tmp_path)], ["compounds.csv"])]:
        with subtests.test(msg=f"hit, anomaly and compound choosers {case_id}"):
            written.clear()
            loaded.clear()
            asked = _dialogs(monkeypatch, folder=folder, table=table, save="")
            screen.choose_hit_export()
            screen._choose_anomaly_export()
            screen._choose_compounds()
            assert asked == ["Write the hit report into",
                             "Write the anomaly report into",
                             "Compound table with SMILES"]
            assert written == want_written
            assert loaded == want_loaded
    screen._hit_result = None
    screen._anomaly = None

    screen.set_frame(_campaign(12))
    drawn = screen.result
    with subtests.test(msg="a cancelled table open keeps the chart"):
        asked = _dialogs(monkeypatch, folder="", table="", save="")
        screen.choose_table()
        assert asked == ["Open a measurement table"]
        assert screen.result is drawn
        assert screen._path is None
        assert screen._source.text() == "48 rows × 5 columns"

    with subtests.test(msg="a cancelled export writes nothing"):
        asked = _dialogs(monkeypatch, folder="", table="", save="")
        screen.choose_export()
        assert asked == [("Export the chart's points", "control_chart.csv")]
        assert list(tmp_path.glob("*.csv")) == []
        assert screen._source.text() == "48 rows × 5 columns"

    with subtests.test(msg="a chosen export writes one row per plate"):
        target = tmp_path / "points.csv"
        _dialogs(monkeypatch, folder="", table="", save=str(target))
        screen.choose_export()
        written_points = pd.read_csv(target)
        assert written_points["plate"].tolist() == [
            f"P{i:02d}" for i in range(1, 13)]
        assert screen._source.text() == "points written to points.csv"

    with subtests.test(msg="a chosen table is read"):
        path = _database(tmp_path)
        _dialogs(monkeypatch, folder="", table=path, save="")
        screen.choose_table()
        assert screen._path == path
        assert screen._source.text() == (
            "measurements.db · cell · 48 rows × 5 columns")


def test_form_signals_chart_only_once_a_table_has_settled(screen, subtests):
    """No frame, or a frame still loading, means no recompute."""
    with subtests.test(msg="a control column before any table draws nothing"):
        opening = screen.report.toPlainText()
        screen._control_column.addItem("well_type")
        assert screen._control_column.currentText() == "well_type"
        assert _levels(screen) == []
        assert screen.result is None
        assert screen.report.toPlainText() == opening

    with subtests.test(msg="the same signal with a frame refills the levels"):
        screen.set_frame(_campaign(12))
        screen._control_column.setCurrentText("batch")
        assert _levels(screen) == ["A", "B"]
        assert _items(screen._positive) == ["", "A", "B"]

    with subtests.test(msg="an edit while loading does not rechart"):
        screen.set_frame(_campaign(30))
        first = screen.result
        assert len(first.baseline_plates) == 20
        screen._loading = True
        try:
            screen._baseline.setValue(10)
            assert screen.result is first
        finally:
            screen._loading = False
        screen._baseline.setValue(12)
        assert screen.result is not first
        assert len(screen.result.baseline_plates) == 12

    with subtests.test(msg="reloading the same table keeps the user's picks"):
        frame = _campaign(12)
        screen.set_frame(frame)
        screen._order.setCurrentText("plateID")
        screen.set_frame(frame)
        assert screen._order.currentText() == "plateID"
        assert screen._plate.currentText() == "plateID"

    with subtests.test(msg="a form the engine refuses becomes a sentence"):
        screen.set_frame(_campaign(12))
        assert screen.result is not None
        failed = []
        screen.failed.connect(failed.append)
        screen._baseline.setRange(1, 500)
        screen._baseline.setValue(1)
        screen.recompute()
        assert screen.result is None
        assert screen.report.toPlainText() == failed[-1]
        assert "is not a baseline" in failed[-1]
        assert screen.violations.rowCount() == 0


def test_the_table_picker_rereads_the_open_database(screen, tmp_path,
                                                    subtests):
    """The picker is wired to the loader, and reads nothing with no file."""
    with subtests.test(msg="no file open: the picker's signal reads nothing"):
        screen._table_picker.addItems(["cell", "nucleus"])
        assert screen._table_picker.currentText() == "cell"
        assert screen._path is None
        assert screen.result is None
        assert screen._source.text() == "no table loaded"
        screen._on_table_picked("nucleus")
        assert screen._path is None
        assert screen.result is None

    path = _database(tmp_path)
    with subtests.test(msg="a named table is read and selected"):
        screen.load_path(path, table="nucleus")
        assert _items(screen._table_picker) == ["cell", "nucleus"]
        assert screen._table_picker.currentText() == "nucleus"
        assert screen._table_picker.isHidden() is False
        assert len(screen.result) == 13
        assert screen._source.text() == (
            "measurements.db · nucleus · 52 rows × 5 columns")

    with subtests.test(msg="picking another table re-reads the same file"):
        charted = screen.result
        screen._table_picker.setCurrentText("cell")
        assert screen._path == path
        assert screen.result is not charted
        assert len(screen.result) == 12
        assert screen._source.text() == (
            "measurements.db · cell · 48 rows × 5 columns")

    with subtests.test(msg="a blank pick reads nothing"):
        charted = screen.result
        screen._on_table_picked("")
        assert screen.result is charted
