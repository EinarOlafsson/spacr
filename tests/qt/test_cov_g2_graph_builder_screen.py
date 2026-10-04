"""Graph Builder's file actions and its refusals before a table is loaded."""
from __future__ import annotations

import json

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QFileDialog, QInputDialog  # noqa: E402

from spacr.qt.screens import graph_builder as gb  # noqa: E402


@pytest.fixture
def screen(qtbot):
    widget = gb.GraphBuilderScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


def test_one_line_names_an_exception_with_no_message():
    class Quiet(Exception):
        def __str__(self):
            return ""

    assert gb._one_line(ValueError("bad\nworse")) == "worse"
    assert gb._one_line(Quiet()).endswith("Quiet")


def test_without_a_table_conditions_and_exports_are_refused(screen):
    screen.open_condition_dialog()
    with pytest.raises(ValueError, match="Load a source table"):
        screen.apply_condition_definition({})
    with pytest.raises(ValueError, match="Load a table before exporting"):
        screen.export_table("/tmp/never.csv")
    with pytest.raises(ValueError, match="Load a source table before saving"):
        screen.save_chart("/tmp/never.json")
    with pytest.raises(ValueError, match="Apply conditions"):
        screen.save_annotated_table("new")


def test_a_merge_without_provenance_cannot_open_crops(screen):
    screen._merge_definition = {"name": "merged"}
    screen._frame = pd.DataFrame({"a": [1]})
    screen._open_selection()
    assert "no verified image/object provenance" in screen._source.text()


def test_the_annotated_table_name_dialog(screen, monkeypatch):
    saved = []
    monkeypatch.setattr(screen, "save_annotated_table", saved.append)
    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("  ", True)))
    screen.choose_save_annotated_table()
    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("cells_ok", True)))
    screen.choose_save_annotated_table()
    assert saved == ["cells_ok"]


def test_export_reports_success_and_failure(screen, monkeypatch, tmp_path):
    target = tmp_path / "out.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(target), "")))
    screen.choose_export_table()
    assert "Could not export table" in screen._source.text()
    screen._frame = pd.DataFrame({"a": [1, 2]})
    screen.choose_export_table()
    assert "exported" in screen._source.text()
    assert pd.read_csv(target)["a"].tolist() == [1, 2]
    receipt = json.loads((tmp_path / "out.csv.conditions.json").read_text())
    assert set(receipt) == {"source", "merge_definition", "condition_annotation"}
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    screen.choose_export_table()


def test_exporting_over_the_source_is_refused(screen, tmp_path):
    source = tmp_path / "table.csv"
    source.write_text("a\n1\n")
    screen._frame = pd.DataFrame({"a": [1]})
    screen._condition_source = {"path": str(source), "table": None}
    with pytest.raises(ValueError, match="new export path"):
        screen.export_table(source)


def test_a_failed_export_leaves_no_partial_files(screen, monkeypatch, tmp_path):
    screen._frame = pd.DataFrame({"a": [1]})

    def refuse(*_args):
        raise OSError("read-only")

    monkeypatch.setattr(gb.os, "replace", refuse)
    with pytest.raises(OSError):
        screen.export_table(tmp_path / "out.csv")
    assert list(tmp_path.iterdir()) == []


def test_chart_file_dialogs_report_their_failures(screen, monkeypatch, tmp_path):
    chosen = str(tmp_path / "chart.json")
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (chosen, "")))
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (chosen, "")))
    screen.choose_save_chart()
    assert "Could not save chart" in screen._source.text()
    (tmp_path / "chart.json").write_text("{not json")
    screen.choose_load_chart()
    assert "Could not load chart" in screen._source.text()


def test_a_chart_is_not_saved_over_its_source(screen, tmp_path):
    source = tmp_path / "table.csv"
    source.write_text("a\n1\n")
    screen._condition_source = {"path": str(source), "table": None}
    with pytest.raises(ValueError, match="new chart path"):
        screen.save_chart(source)


def test_one_line_falls_back_when_the_traceback_is_blank(monkeypatch):
    import traceback

    monkeypatch.setattr(traceback, "format_exception_only",
                        lambda kind, exc: ["\n", "  \n"])
    assert gb._one_line(ValueError("")) == "ValueError"
    assert gb._one_line(ValueError("why")) == "why"


def test_cancelled_save_and_load_dialogs_change_nothing(screen, monkeypatch):
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    before = screen._source.text()
    assert screen.choose_save_chart() is None
    assert screen.choose_load_chart() is None
    assert screen._source.text() == before
