"""A dropped database reaches the Measurements tab, and survives a save.

The three slices of instruction 130 built the input column, the tab and the
merge. These are the two seams BETWEEN them, which each slice could see was
missing but none could fix -- both live in files they were told not to touch.

1. `ml.normalize_regression_input_pairs` rebuilt every row as exactly
   {'score', 'count', 'plate'} and wrote it back over
   `settings['paired_data']`, so the database attachment was ERASED from the
   settings CSV a run saves. Nothing in the fit reads it -- the regression
   runs on scores and counts -- which is exactly why it would have gone
   unnoticed until someone reloaded a run and found the column empty.

2. `app_screen` built the panel with no `database_provider`, so the tab
   showed nothing; and `_on_results_tab_changed` returned early unless the
   widget was the Runs tab, so it never refreshed. Databases are dropped
   while the Measurements tab is NOT the visible one, which is the whole
   reason it needs an on-open refresh.
"""
from __future__ import annotations

import os
import pathlib
import sqlite3
import tempfile

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# --------------------------------------------------------------------------- #
#  The settings round trip
# --------------------------------------------------------------------------- #

def _normalise(rows):
    from spacr.ml import normalize_regression_input_pairs

    out = normalize_regression_input_pairs({"paired_data": list(rows)})
    return out[0] if isinstance(out, tuple) else out["paired_data"]


def test_the_database_survives_the_round_trip():
    rows = _normalise([{"score": "/s1.csv", "count": "/c1.csv",
                        "plate": "plate1", "database": "/p1/measurements.db"}])

    assert rows[0]["database"] == "/p1/measurements.db"


def test_a_row_without_one_is_still_legal():
    """A plate with no database must not fail: the regression runs on scores
    and counts, and the database only enables that plate in the tab."""
    rows = _normalise([{"score": "/s.csv", "count": "/c.csv",
                        "plate": "plate2"}])

    assert rows[0]["database"] is None
    assert rows[0]["score"] == "/s.csv"


def test_the_other_keys_are_not_disturbed():
    rows = _normalise([{"score": "/s.csv", "count": "/c.csv",
                        "plate": "plate1", "database": "/d.db"}])

    assert rows[0]["plate"] == "plate1"
    assert rows[0]["count"] == "/c.csv"


# --------------------------------------------------------------------------- #
#  The tab actually sees them
# --------------------------------------------------------------------------- #

@pytest.fixture
def plate(tmp_path):
    """A plate folder with a real measurements database beside its CSVs."""
    folder = tmp_path / "plate1"
    folder.mkdir()
    database = folder / "measurements.db"
    connection = sqlite3.connect(database)
    pd.DataFrame({"prcfo": ["plate1_r1_c1_f1_o1"],
                  "cell_area": [10.0],
                  "object_label_cell": [1]}).to_sql("cell", connection,
                                                    index=False)
    connection.close()
    (folder / "s.csv").write_text("prc,pred\nplate1_r1_c1,0.1\n")
    (folder / "c.csv").write_text(
        "rowID,columnID,grna_name,count,plateID\nr1,c1,TGGT1_1_1,5,plate1\n")
    return folder


@pytest.mark.qt
def test_the_provider_reads_the_input_table(qtbot, plate):
    pytest.importorskip("PySide6")
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    table = screen._settings_model._widgets["paired_data"]
    table.add_paths_for_side([str(plate / "s.csv")], "score")
    table.add_paths_for_side([str(plate / "c.csv")], "count")
    table.attach_database(str(plate / "measurements.db"))

    rows = screen._attached_database_rows()

    assert any(row.get("database") for row in rows), rows


@pytest.mark.qt
def test_the_provider_is_a_callable_not_a_snapshot(qtbot):
    """Databases are dropped AFTER the panel is built. A list captured at
    construction would never grow."""
    pytest.importorskip("PySide6")
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)

    assert callable(screen._attached_database_rows)
    assert screen._attached_database_rows() == []


@pytest.mark.qt
def test_opening_the_tab_refreshes_it(qtbot, plate):
    """It used to return early unless the widget was the Runs tab, so the
    Measurements tab never refreshed at all."""
    pytest.importorskip("PySide6")
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    table = screen._settings_model._widgets["paired_data"]
    table.add_paths_for_side([str(plate / "s.csv")], "score")
    table.add_paths_for_side([str(plate / "c.csv")], "count")
    table.attach_database(str(plate / "measurements.db"))

    tabs = screen._results_tabs
    index = next(i for i in range(tabs.count())
                 if tabs.widget(i) is screen._scan_panel)
    screen._on_results_tab_changed(index)

    assert screen._scan_panel.refresh_databases() == 1


@pytest.mark.qt
def test_the_runs_tab_still_refreshes(qtbot):
    """The early return was doing a job. Replacing it must not stop the Runs
    tab re-reading the sweep's table."""
    pytest.importorskip("PySide6")
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    runs = getattr(screen, "_sweep_runs", None)
    if runs is None:
        pytest.skip("this build has no Runs tab")

    called = []
    runs.load = lambda folder: called.append(folder) or True
    screen._sweep_destination = lambda: "/tmp/does-not-matter"
    tabs = screen._results_tabs
    index = next(i for i in range(tabs.count()) if tabs.widget(i) is runs)
    screen._on_results_tab_changed(index)

    assert called == ["/tmp/does-not-matter"]


@pytest.mark.qt
def test_a_screen_with_no_input_table_does_not_raise(qtbot):
    """The provider must not assume the paired-data widget exists -- the
    Measurements tab is built on the regression screen and other screens
    reach this code too."""
    pytest.importorskip("PySide6")
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    screen._settings_model._widgets.pop("paired_data", None)

    assert screen._attached_database_rows() == []
