"""The Database Browser merges tables, also across plates and projects (item 1017).

The Merge tables window lists the open project first, takes more projects
(older measurements.db files and store projects alike), stacks one table or
joins object tables, previews, and saves a new table into a store project,
which the browser then opens. Two projects with a plate of the same name
stay apart through their labels, and the window says so.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

from spacr import store
from spacr.multi_database import SCREEN_COLUMN, SOURCE_COLUMN
from spacr.plate_measurements import merge_projects
from spacr.qt.screens.db_browser import DbBrowserScreen
from tests.test_merge_projects_1017 import _legacy, _store_project


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, qt_theme_applied, tmp_path):
    """Preferences in a temp folder; no run journal; a fresh shared store."""
    from PySide6.QtCore import QSettings

    from spacr import run_journal

    for fmt in (QSettings.NativeFormat, QSettings.IniFormat):
        QSettings.setPath(fmt, QSettings.UserScope, str(tmp_path / "qs"))
    QSettings("spacr", "qt").clear()
    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(run_journal, "runs_root", lambda: root)
    store.release_shared()
    yield
    store.release_shared()


@pytest.fixture
def screen(qtbot):
    """A synchronous Browser."""
    widget = DbBrowserScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


def test_the_merge_window_starts_with_the_open_project(screen, tmp_path):
    path = _legacy(tmp_path, "runA")
    assert screen.set_database(path)
    dialog = screen.open_merge()
    assert dialog.sources() == [path]
    assert dialog.labels() == ["measurements"]
    second = _legacy(tmp_path, "runB")
    assert dialog.add_source(second)
    assert dialog.labels() == ["runA", "runB"]
    assert not dialog._btn_save.isEnabled()
    assert screen.open_merge() is dialog
    assert dialog.sources() == [path, second]


def test_two_plates_of_the_same_name_merge_apart(screen, tmp_path):
    first = _legacy(tmp_path, "runA", plate="plate1", count=6)
    second = _legacy(tmp_path, "runB", plate="plate1", count=4, area=5.0)
    target = _store_project(tmp_path, "pool", plate="plate7")
    assert screen.set_database(first)
    dialog = screen.open_merge()
    assert dialog.add_source(second)
    assert not dialog.add_source(second)
    dialog.set_label(0, "monday")
    dialog.set_label(1, "tuesday")
    dialog.set_tables(["cell"])
    assert dialog.preview()
    assert "Plate plate1 is in monday and tuesday" in dialog.notes()
    assert "will hold 10 rows" in dialog.notes()
    frame = dialog.preview_frame()
    assert set(frame[SCREEN_COLUMN]) == {"monday", "tuesday"}
    dialog.set_name("both_runs")
    dialog.set_target(target)
    assert dialog._btn_save.isEnabled()
    assert dialog.save(), dialog.status_text()
    assert dialog.last_result["rows"] == 10
    assert screen.database_path() == target
    assert screen.current_table() == "both_runs"
    saved = store.read_frame(target, table="both_runs")
    in_code = merge_projects([first, second], "cell",
                             labels=["monday", "tuesday"])
    assert len(saved) == len(in_code) == 10
    keys = list(in_code.attrs["key_columns"])
    assert not saved.duplicated(keys).any()
    assert (saved.groupby(SOURCE_COLUMN).size().to_dict()
            == {"monday": 6, "tuesday": 4})


def test_a_join_of_an_older_file_and_a_store_project(screen, tmp_path):
    legacy = _legacy(tmp_path, "runA", plate="plate1", count=5)
    project = _store_project(tmp_path, "runB", plate="plate2", count=3)
    assert screen.set_database(project)
    dialog = screen.open_merge()
    dialog.add_source(legacy)
    dialog.set_way("join")
    dialog.set_tables(["cell", "nucleus"], anchor="cell")
    assert dialog.preview()
    frame = dialog.preview_frame()
    assert len(frame) == 8
    assert np.allclose(frame["nucleus_area"], frame["cell_area"])
    dialog.set_column_filter("nucleus")
    assert "cell_area" not in dialog.preview_columns()
    dialog.set_name("joined")
    dialog.set_target(project)
    assert dialog.save(), dialog.status_text()
    columns = set(store.read_frame(project, table="joined").columns)
    assert "nucleus_area" in columns and "cell_area" not in columns
    assert {"plateID", "object_label", SOURCE_COLUMN} <= columns


def test_a_changed_setting_needs_a_new_preview(screen, tmp_path):
    first = _legacy(tmp_path, "runA", plate="plate1")
    second = _legacy(tmp_path, "runB", plate="plate2")
    target = str(tmp_path / "new" / "measurements")
    screen.set_database(first)
    dialog = screen.open_merge()
    dialog.add_source(second)
    dialog.set_name("pooled")
    dialog.set_target(target)
    assert not dialog._btn_save.isEnabled()
    assert dialog.preview()
    assert dialog._btn_save.isEnabled()
    dialog.set_label(1, "other")
    assert not dialog._btn_save.isEnabled()
    assert "preview again" in dialog.status_text()
    assert not dialog.save()


def test_a_refused_merge_is_said_in_the_window(screen, tmp_path):
    first = _legacy(tmp_path, "runA", plate="plate1")
    second = _legacy(tmp_path, "runB", plate="plate2")
    screen.set_database(first)
    dialog = screen.open_merge()
    dialog.add_source(second)
    dialog.set_label(1, dialog.labels()[0])
    assert dialog.preview()
    assert "its own label" in dialog.status_text()
    assert dialog.preview_frame() is None
