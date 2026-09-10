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


def test_a_loaded_table_reaches_the_panel_the_filter_and_the_formulas(
        screen, measurements):
    """One `set_frame` has to arrive in three places or the screen disagrees."""
    screen.set_frame(measurements)

    assert screen._source.text() == "24 rows × 9 columns"
    assert screen.explorer._frame is not None
    assert len(screen.explorer._frame) == 24
    assert "condition" in [screen.explorer._label.itemText(i)
                           for i in range(screen.explorer._label.count())]


def test_a_database_offers_its_tables_and_loads_the_first(screen, database):
    """The picker is filled inline; only the read is off the GUI thread."""
    screen.load_path(str(database))

    assert [screen._table_picker.itemText(i)
            for i in range(screen._table_picker.count())] == ["cell", "nucleus"]
    assert "measurements.db · cell · 24 rows" in screen._source.text()
    assert not screen.is_busy()
    assert screen.active_jobs() == 0


def test_picking_another_table_reloads_the_same_file(screen, database):
    """The picker is wired to the loader, not to a cached frame."""
    screen.load_path(str(database))

    screen._on_table_picked("nucleus")

    assert "· nucleus" in screen._source.text()


def test_picking_a_table_with_no_file_loaded_does_nothing(screen):
    """The combo is populated programmatically; its signal must be harmless."""
    screen._on_table_picked("cell")

    assert screen._frame is None


def test_a_file_whose_tables_cannot_be_listed_says_so_inline(screen,
                                                             tmp_path):
    """No modal, and the screen keeps whatever it had."""
    broken = tmp_path / "not-a-database.db"
    broken.write_bytes(b"this is not sqlite")

    screen.load_path(str(broken))

    assert "could not read not-a-database.db" in screen._source.text()
    assert screen._frame is None


def test_a_failed_read_is_reported_on_the_source_line(screen, tmp_path):
    """The failure arrives from the worker and lands on the label."""
    screen._path = str(tmp_path / "cells.csv")

    screen._on_load_failed("No columns to parse from file")

    assert screen._source.text() == (
        "could not read cells.csv: No columns to parse from file")


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


def test_the_ranking_is_over_the_filtered_rows(screen, measurements):
    """A separation is a statement about a population; narrowing it matters."""
    from spacr.selection import CategoryFilter, DataFilter

    screen.set_frame(measurements)
    assert len(screen.explorer._frame) == 24

    screen._link.set_filter(DataFilter([CategoryFilter("rowID", ("r1",))]))
    screen._on_filter_changed()

    assert len(screen.explorer._frame) == 6


def test_a_computed_column_is_over_the_whole_table(screen, measurements):
    """A formula defines a property of an object; a slider must not move it."""
    screen.set_frame(measurements)
    from spacr.qt.widgets.formula import ColumnFormula

    assert screen.formulas.add_formula(
        ColumnFormula("roundness", "cell_area / cell_perimeter"))
    screen._on_formulas_changed()

    assert "roundness" in screen.explorer._frame.columns
    assert len(screen.explorer._frame) == 24


def test_a_filter_that_does_not_apply_leaves_the_rows_alone(screen,
                                                            measurements):
    """A filter drawn on another population must not empty this one."""
    from spacr.selection import CategoryFilter, DataFilter

    screen.set_frame(measurements)
    screen._link.set_filter(
        DataFilter([CategoryFilter("nothing_here", ("x",))]))

    assert len(screen._visible(measurements)) == 24


def test_nothing_is_pushed_before_a_table_is_loaded(screen):
    """`_push_frame` runs on every formula change, loaded or not."""
    screen._push_frame()
    screen._on_filter_changed()

    assert screen.explorer._frame is None


def test_the_export_carries_every_statistic_not_only_the_ranked_one(
        screen, measurements, tmp_path):
    """A reader asking whether the winner is a shift or a spread needs both."""
    screen.set_frame(measurements)
    screen.explorer._label.setCurrentText("condition")
    screen.explorer.rank_now()

    out = screen.export_ranking(str(tmp_path / "ranking.csv"))

    written = pd.read_csv(out)
    assert set(written.columns) >= {"feature", "rank", "separation",
                                    "statistic", "auc", "cohen_d", "ks",
                                    "mutual_info", "higher_in", "against",
                                    "min_n", "shape_not_shift"}
    assert written["rank"].tolist() == list(range(1, len(written) + 1))
    assert "Ranking written to ranking.csv" in screen._source.text()


def test_exporting_before_ranking_says_there_is_nothing_to_write(screen,
                                                                 tmp_path):
    """Writing an empty CSV would look like a ranking of no features."""
    assert screen.ranking_frame() is None
    assert screen.export_ranking(str(tmp_path / "ranking.csv")) is None
    assert screen._source.text() == "Nothing ranked yet."
    assert not (tmp_path / "ranking.csv").exists()


def test_choosing_an_export_path_writes_there(screen, measurements,
                                              monkeypatch, tmp_path):
    """The button is a save dialog and then the ordinary export."""
    screen.set_frame(measurements)
    screen.explorer._label.setCurrentText("condition")
    screen.explorer.rank_now()
    target = tmp_path / "chosen.csv"
    monkeypatch.setattr(
        "spacr.qt.screens.feature_explorer.QFileDialog.getSaveFileName",
        staticmethod(lambda *a, **k: (str(target), "")))

    screen.choose_export()

    assert target.exists()


def test_cancelling_the_export_dialog_writes_nothing(screen, monkeypatch,
                                                     tmp_path):
    """An empty path is the user pressing Escape."""
    monkeypatch.setattr(
        "spacr.qt.screens.feature_explorer.QFileDialog.getSaveFileName",
        staticmethod(lambda *a, **k: ("", "")))

    screen.choose_export()

    assert list(tmp_path.iterdir()) == []


def test_the_screen_exposes_the_spec_it_would_be_saved_from(screen,
                                                            measurements):
    """A saved analysis is restored from this."""
    screen.set_frame(measurements)
    screen.explorer._label.setCurrentText("condition")

    assert screen.spec.label == "condition"


def test_the_factory_builds_the_screen_the_registry_asks_for(qtbot):
    """`register_app` is handed this function, not the class."""
    widget = make_feature_explorer_screen("feature_explorer")
    qtbot.addWidget(widget)

    assert isinstance(widget, FeatureExplorerScreen)


def test_a_csv_is_read_without_asking_it_for_table_names(screen, tmp_path):
    """A CSV has no tables, and asking a database library for them is wrong.

    ``table_names`` is skipped by suffix, so the picker stays empty and hidden
    -- a CSV shown with an empty table dropdown above it reads as a file whose
    tables failed to load. This branch had never been taken: every existing
    test hands in a database.
    """
    import pandas as pd

    csv = tmp_path / "features.csv"
    pd.DataFrame({"condition": ["a", "b"] * 3,
                  "cell_area": [10.0, 12.0, 11.0, 13.0, 10.5, 12.5]}
                 ).to_csv(csv, index=False)

    screen.load_path(str(csv))

    assert screen._table_picker.count() == 0
    assert not screen._table_picker.isVisible()
    assert "features.csv" in screen._source.text()
    assert "could not read" not in screen._source.text()


@pytest.mark.parametrize("suffix", [".csv", ".tsv", ".txt", ".CSV"])
def test_every_text_suffix_skips_the_table_listing(screen, tmp_path,
                                                    monkeypatch, suffix):
    """The suffix test is case-insensitive and covers all three spellings.

    A user's export can be any of them, and a ``.CSV`` from Windows going down
    the database path would be reported as an unreadable database rather than
    read as the table it is.
    """
    from spacr.qt.screens import feature_explorer as fe

    asked = []

    def watched(path):
        asked.append(path)
        raise AssertionError("a text file was asked for its tables")

    monkeypatch.setattr(fe, "table_names", watched)

    target = tmp_path / f"features{suffix}"
    target.write_text("condition,cell_area\na,10\nb,12\n")

    screen.load_path(str(target))

    assert asked == [], f"{suffix} was routed through table_names"
