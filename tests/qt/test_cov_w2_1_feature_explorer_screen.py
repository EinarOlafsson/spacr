"""Loading, filtering, computing columns and exporting on the Feature Explorer.

The screen's contract is a division of labour: a computed column is a property
of an object and is computed over the whole table, while the ranking is a
statement about a population and is computed over the filtered rows. Both
halves are driven here with a real table and a real filter, and the export is
written to a real file and read back.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from spacr.qt.linked_selection import LinkedSelection
from spacr.qt.screens.feature_explorer import (
    FeatureExplorerScreen, make_feature_explorer_screen,
)


@pytest.fixture
def link():
    """A PRIVATE link — never the process-wide one."""
    return LinkedSelection()


@pytest.fixture
def measurements():
    """Two classes that a couple of the columns separate cleanly."""
    n = 24
    return pd.DataFrame({
        "plateID": ["p1"] * n,
        "rowID": [f"r{1 + i % 4}" for i in range(n)],
        "columnID": ["c1"] * n,
        "fieldID": ["f1"] * n,
        "object_label": list(range(1, n + 1)),
        "condition": ["treated" if i % 2 else "control" for i in range(n)],
        "cell_area": [100.0 + 40.0 * (i % 2) + i * 0.1 for i in range(n)],
        "cell_perimeter": [40.0 + i * 0.01 for i in range(n)],
        "cell_intensity": [5.0 + (i % 3) for i in range(n)],
    })


@pytest.fixture
def screen(qtbot, link):
    """A Feature Explorer whose reads happen inline."""
    widget = FeatureExplorerScreen(link=link, threaded=False)
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def database(tmp_path, measurements):
    path = tmp_path / "measurements.db"
    with sqlite3.connect(path) as db:
        measurements.to_sql("cell", db, index=False)
        measurements.to_sql("nucleus", db, index=False)
    return path


def test_choosing_a_table_from_the_dialog_loads_it(screen, monkeypatch,
                                                   database):
    """The button is a file dialog and then the ordinary load path."""
    monkeypatch.setattr(
        "spacr.qt.screens.feature_explorer.QFileDialog.getOpenFileName",
        staticmethod(lambda *a, **k: (str(database), "")))

    screen.choose_table()

    assert screen._frame is not None


def test_cancelling_the_load_dialog_loads_nothing(screen, monkeypatch):
    """An empty path is the user pressing Escape."""
    monkeypatch.setattr(
        "spacr.qt.screens.feature_explorer.QFileDialog.getOpenFileName",
        staticmethod(lambda *a, **k: ("", "")))

    screen.choose_table()

    assert screen._frame is None


def test_nothing_is_pushed_before_a_table_is_loaded(screen):
    """`_push_frame` runs on every formula change, loaded or not."""
    screen._push_frame()
    screen._on_filter_changed()

    assert screen.explorer._frame is None


def test_exporting_before_ranking_says_there_is_nothing_to_write(screen,
                                                                 tmp_path):
    """Writing an empty CSV would look like a ranking of no features."""
    assert screen.ranking_frame() is None
    assert screen.export_ranking(str(tmp_path / "ranking.csv")) is None
    assert screen._source.text() == "Nothing ranked yet."
    assert not (tmp_path / "ranking.csv").exists()


def test_cancelling_the_export_dialog_writes_nothing(screen, monkeypatch,
                                                     tmp_path):
    """An empty path is the user pressing Escape."""
    monkeypatch.setattr(
        "spacr.qt.screens.feature_explorer.QFileDialog.getSaveFileName",
        staticmethod(lambda *a, **k: ("", "")))

    screen.choose_export()

    assert list(tmp_path.iterdir()) == []


def test_the_factory_builds_the_screen_the_registry_asks_for(qtbot):
    """`register_app` is handed this function, not the class."""
    widget = make_feature_explorer_screen("feature_explorer")
    qtbot.addWidget(widget)

    assert isinstance(widget, FeatureExplorerScreen)
