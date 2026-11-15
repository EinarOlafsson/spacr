"""The Volcano Explorer screen: loading a folder, and what a drag or drop does.

The screen built by :func:`spacr.qt.screens.volcano._make_screen` opens a
results folder (or refuses one with no table, naming it), accepts a drag only
when it carries URLs, and on a drop loads the first local path that holds a
table, skipping empty folders and remote URLs.  Real folders on disk; the
QMimeData is kept alive for the life of each event, because the event does
not own it and releasing it early crashes inside ``mimeData()``.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from PySide6.QtCore import QMimeData, QPoint, QPointF, QUrl, Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent

from spacr.qt.screens import volcano

pytestmark = pytest.mark.qt

NOTHING = "No results loaded"


def _results_frame(n=24, seed=5):
    """A regression results table with ``n`` guides."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "guide": [f"g{i}" for i in range(n)],
        "gene": [f"TGGT1_{200000 + i // 3}" for i in range(n)],
        "standardized_marginal_effect": rng.normal(0, 0.08, n),
        "adjusted_p_value": np.clip(rng.beta(0.6, 6, n), 1e-8, 1),
        "wells_with_guide": rng.integers(1, 12, n),
        "significant": False,
        "alpha": 0.05,
    })


@pytest.fixture
def make_screen(qtbot):
    """Build a fresh volcano screen."""
    def _build():
        widget = volcano._make_screen(app_key=volcano.APP_KEY, host=None)
        qtbot.addWidget(widget)
        return widget
    return _build


@pytest.fixture
def folders(tmp_path):
    """A results folder, an empty folder, and the table's absolute path."""
    good = tmp_path / "run_2024"
    good.mkdir()
    _results_frame().to_csv(good / "results.csv", index=False)
    empty = tmp_path / "nothing_here"
    empty.mkdir()
    return good, empty, os.path.abspath(good / "results.csv")


def test_load_names_the_table_or_the_folder_it_could_not_use(make_screen,
                                                             folders, subtests):
    """A new screen says nothing is loaded; load() answers True/False with its label."""
    good, empty, table = folders
    expected_rows = volcano.load_results(table)

    with subtests.test(msg="a new screen"):
        screen = make_screen()
        assert screen._path_label.text() == NOTHING
        assert screen.acceptDrops() is True
        assert screen.explorer.results().empty

    with subtests.test(msg="a folder with a table"):
        screen = make_screen()
        assert screen.load(good) is True
        assert screen._path_label.text() == table
        pd.testing.assert_frame_equal(screen.explorer.results(), expected_rows)

    with subtests.test(msg="a folder with no table"):
        screen = make_screen()
        assert screen.load(empty) is False
        assert screen._path_label.text() == f"No results CSV found under {empty}"
        assert screen.explorer.results().empty

    with subtests.test(msg="a failed load keeps the table already plotted"):
        screen = make_screen()
        screen.load(good)
        assert screen.load(empty) is False
        assert screen._path_label.text() == f"No results CSV found under {empty}"
        assert len(screen.explorer.results()) == 24


def _urls(*items):
    """QMimeData carrying local paths (str/Path) or literal URLs (QUrl)."""
    data = QMimeData()
    data.setUrls([item if isinstance(item, QUrl) else
                  QUrl.fromLocalFile(str(item)) for item in items])
    return data


def test_a_drag_is_accepted_only_when_it_carries_urls(make_screen, folders,
                                                      subtests):
    """Dragging a folder is accepted; dragging plain text is not."""
    good, _empty, _table = folders
    text_only = QMimeData()
    text_only.setText("results.csv")
    for case_id, data, accepted in [
            ("a dragged folder", _urls(good), True),
            ("a dragged remote url", _urls(QUrl("https://example.org/r.csv")),
             True),
            ("dragged text", text_only, False)]:
        with subtests.test(msg=case_id):
            screen = make_screen()
            event = QDragEnterEvent(QPoint(4, 4), Qt.CopyAction, data,
                                    Qt.LeftButton, Qt.NoModifier)
            event.setAccepted(False)
            screen.dragEnterEvent(event)
            assert event.isAccepted() is accepted
            assert screen._path_label.text() == NOTHING
            del event


def test_a_drop_loads_the_first_local_path_that_holds_a_table(make_screen,
                                                              folders, subtests):
    """Every drop is accepted; only a local path with a table changes the label."""
    good, empty, table = folders
    cases = [
        ("a dropped folder", (good,), table),
        ("an empty folder first does not stop the good one", (empty, good),
         table),
        ("the first folder that loads wins", (good, empty), table),
        ("a remote url loads nothing",
         (QUrl("https://example.org/results.csv"),), NOTHING),
        ("only an empty folder names it",
         (empty,), f"No results CSV found under {os.path.normpath(empty)}"),
    ]
    for case_id, items, label in cases:
        with subtests.test(msg=case_id):
            screen = make_screen()
            data = _urls(*items)
            event = QDropEvent(QPointF(4, 4), Qt.CopyAction, data,
                               Qt.LeftButton, Qt.NoModifier)
            screen.dropEvent(event)
            assert screen._path_label.text() == label
            assert event.isAccepted() is True
            assert len(screen.explorer.results()) == (
                24 if label == table else 0)
            del event, data
