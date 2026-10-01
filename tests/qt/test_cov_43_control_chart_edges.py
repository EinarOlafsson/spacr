"""Control Charts' hit, anomaly and compound actions at their edges.

A refused worker job is shown, not raised; an export asked for before
anything was scored says so; a folder or table dialog that is dismissed
writes nothing; a change made while a table is loading does not rescore;
and a CSV source is never read as a measurement database for host toxicity.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

pytestmark = pytest.mark.qt

from spacr.qt.screens import control_chart as cc  # noqa: E402


@pytest.fixture
def screen(qtbot):
    made = cc.ControlChartScreen(threaded=False)
    qtbot.addWidget(made)
    return made


def test_refused_jobs_are_shown_on_their_panels(screen):
    screen._on_hit_failed("no controls in the table")
    assert "no controls in the table" in screen.hit_summary.text()
    screen._on_anomaly_failed("too few control wells")
    assert "too few control wells" in screen.anomaly_summary.text()
    screen._on_chemistry_failed("rdkit missing")
    assert "rdkit missing" in screen._compound_label.text() or \
        "rdkit missing" in screen.chem_summary.text()


def test_exports_before_scoring_say_nothing_was_scored(screen, monkeypatch):
    asked = []
    monkeypatch.setattr(cc.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: asked.append(a) or ""))
    screen.choose_hit_export()
    assert screen._source.text() == "Nothing scored yet."
    screen._source.setText("")
    screen._choose_anomaly_export()
    assert screen._source.text() == "Nothing scored yet."
    screen._source.setText("")
    assert screen._export_anomalies("/tmp/never") is None
    assert screen._source.text() == "Nothing scored yet."
    assert asked == []


def test_a_dismissed_folder_or_table_dialog_writes_and_loads_nothing(
        screen, monkeypatch):
    monkeypatch.setattr(cc.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: ""))
    monkeypatch.setattr(cc.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    written = []
    monkeypatch.setattr(screen, "export_hits", written.append)
    monkeypatch.setattr(screen, "_export_anomalies", written.append)
    loaded = []
    monkeypatch.setattr(screen, "_load_compounds", loaded.append)
    screen._hit_result = object()
    screen._anomaly = object()
    screen.choose_hit_export()
    screen._choose_anomaly_export()
    screen._choose_compounds()
    assert written == [] and loaded == []


def test_chosen_folders_and_tables_are_used(screen, monkeypatch, tmp_path):
    monkeypatch.setattr(cc.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(tmp_path)))
    monkeypatch.setattr(cc.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: ("compounds.csv", "")))
    written, loaded = [], []
    monkeypatch.setattr(screen, "export_hits", written.append)
    monkeypatch.setattr(screen, "_export_anomalies", written.append)
    monkeypatch.setattr(screen, "_load_compounds", loaded.append)
    screen._hit_result = object()
    screen._anomaly = object()
    screen.choose_hit_export()
    screen._choose_anomaly_export()
    screen._choose_compounds()
    assert written == [str(tmp_path), str(tmp_path)]
    assert loaded == ["compounds.csv"]


def test_changes_while_loading_do_not_rescore(screen, monkeypatch):
    rescored = []
    monkeypatch.setattr(screen, "rescore_hits", lambda: rescored.append("h"))
    monkeypatch.setattr(screen, "_rescore_anomalies",
                        lambda: rescored.append("a"))
    screen._loading = True
    screen._on_hit_changed()
    screen._on_anomaly_changed()
    assert rescored == []
    screen._loading = False
    screen._on_hit_changed()
    screen._on_anomaly_changed()
    assert rescored == ["h", "a"]


def test_a_csv_source_is_not_read_for_host_toxicity(screen, monkeypatch):
    from spacr import sp_stats

    asked = []
    monkeypatch.setattr(sp_stats, "_host_toxicity",
                        lambda path: asked.append(path) or (None, None))
    monkeypatch.setattr(sp_stats, "_structure_activity",
                        lambda *a, **k: type("Chem", (), {
                            "report": lambda self: "linked"})())
    monkeypatch.setattr(screen, "_show_chemistry", lambda chem, msg: None)
    screen._hit_result = object()
    screen._compounds = object()
    screen._path = "/data/plate1.csv"
    screen._recompute_chemistry()
    screen._path = "/data/measurements.db"
    screen._recompute_chemistry()
    assert asked == [None, "/data/measurements.db"]
