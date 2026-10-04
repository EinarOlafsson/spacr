"""The merge action of a table screen, driven without a real dialog."""
from __future__ import annotations

import types

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QWidget  # noqa: E402

from spacr.qt.widgets import derived_table_source as dts  # noqa: E402
from spacr.qt.widgets import merge_tables_dialog as mtd  # noqa: E402


class _Screen(QWidget, dts.DerivedTableSource):
    """The smallest host the mixin needs: a path, a picker and a status."""

    def __init__(self, path="/data/measurements.db"):
        super().__init__()
        self._path = path
        self._table_picker = types.SimpleNamespace(currentText=lambda: "cell")
        self._source = QLabel(self)
        self._frame = None
        self.used = []
        self._install_merge_button(QHBoxLayout(self))

    def use_derived_table(self, definition):
        self.used.append(definition)


def _dialog(answer, seen):
    class _Dialog:
        def __init__(self, path, parent, **kwargs):
            seen.append((path, kwargs))
            self.definition = {"name": "merged"}

        def exec(self):
            return answer

    return _Dialog


@pytest.fixture
def screen(qtbot, monkeypatch):
    monkeypatch.setattr(dts, "schemas", lambda path: {"cell": [], "nucleus": []})
    widget = _Screen()
    qtbot.addWidget(widget)
    return widget


def test_no_database_means_no_dialog(screen, monkeypatch):
    seen = []
    monkeypatch.setattr(mtd, "MergeTablesDialog", _dialog(QDialog.Accepted, seen))
    screen._path = ""
    screen.open_merge_dialog()
    assert seen == [] and screen.used == []


def test_an_accepted_dialog_uses_its_definition(screen, monkeypatch):
    seen = []
    monkeypatch.setattr(mtd, "MergeTablesDialog", _dialog(QDialog.Accepted, seen))
    screen._tables = ["cell", "gone"]
    screen.open_merge_dialog()
    assert seen == [("/data/measurements.db", {"selected": ["cell"]})]
    assert screen.used == [{"name": "merged"}]


def test_a_cancelled_dialog_keeps_the_table_and_reopens_the_last_merge(
        screen, monkeypatch):
    seen = []
    monkeypatch.setattr(mtd, "MergeTablesDialog", _dialog(QDialog.Rejected, seen))
    screen._merge_definition = {"name": "earlier"}
    screen.open_merge_dialog()
    assert seen[0][1] == {"selected": ["cell"],
                          "initial_definition": {"name": "earlier"}}
    assert screen.used == []


def test_a_failing_merge_is_reported_on_the_screen(screen, monkeypatch):
    def broken(path, parent, **kwargs):
        raise ValueError("no such table")

    monkeypatch.setattr(mtd, "MergeTablesDialog", broken)
    screen.open_merge_dialog()
    assert screen._source.text() == "Could not merge tables: no such table"


def test_a_derived_frame_without_provenance_disables_image_links(screen):
    annotate = QLabel(screen)
    to_annotate = QLabel(screen)
    screen._annotate, screen._to_annotate = annotate, to_annotate
    screen.builder = types.SimpleNamespace(
        canvas=types.SimpleNamespace(selected_count=lambda: 3))
    frame = pd.DataFrame({"a": [1]})
    frame.attrs["merge_definition"] = {"name": "merged"}
    screen._derived_frame_loaded(frame)
    assert screen._merge_button.isEnabled()
    assert not annotate.isEnabled() and not to_annotate.isEnabled()
    assert "provenance" in annotate.toolTip()
    screen._frame = frame
    assert screen._has_merge_image_provenance() is False

    frame.attrs["image_provenance"] = True
    screen._derived_frame_loaded(frame)
    assert annotate.isEnabled() and to_annotate.isEnabled()
    assert screen._has_merge_image_provenance() is True


def test_a_csv_source_cannot_be_merged(screen):
    screen._path = "/data/table.csv"
    screen._derived_frame_loaded(pd.DataFrame({"a": [1]}))
    assert not screen._merge_button.isEnabled()
    assert screen._has_merge_image_provenance() is True
