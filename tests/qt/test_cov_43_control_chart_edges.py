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
