"""The views stop being islands: a lasso in one is a highlight in the rest.

:mod:`spacr.qt.linked_selection` is tested on its own in
``test_linked_selection.py``. This file tests the thing that actually matters
to a user, with the *real* screens over a *real* sqlite file: lasso a cluster
in the UMAP and the same objects light up in the database browser; narrow the
population and the table loses rows while the embedding keeps every point.

The distinction the whole design rests on is asserted directly here, because
it is the one that will get eroded first:

* a **filter** hides. Rows leave the table.
* a **selection** highlights. Nothing ever leaves anything.

Both screens run offscreen with ``threaded=False``, so every assertion is
about state that exists by the time the call returns.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

import pandas as pd

from spacr.qt.linked_selection import linked_selection
from spacr.qt.screens.db_browser import DbBrowserScreen
from spacr.qt.screens.plate_view import PlateViewScreen
from spacr.qt.widgets.umap_explorer import ImageUmapExplorer
from spacr.selection import (
    CategoryFilter,
    DataFilter,
    RangeFilter,
    Selection,
)

#: Six objects: two plate rows of three columns, one field each.
N_POINTS = 6


def _identity(i: int) -> dict:
    return {
        "plateID": "plate1",
        "rowID": f"r{i // 3 + 1}",
        "columnID": f"c{i % 3 + 1}",
        "fieldID": "f1",
        "object_label": i + 1,
        "cell_area": 100.0 * (i + 1),
    }


@pytest.fixture(autouse=True)
def _clean_link():
    """The link is process-wide; leaving state on it poisons other tests."""
    link = linked_selection()
    link.clear_filter()
    link.clear_selection()
    yield
    link.clear_filter()
    link.clear_selection()


@pytest.fixture
def measurements_db(tmp_path):
    """A real measurements.db whose ``cell`` table carries object keys."""
    folder = tmp_path / "plate1" / "measurements"
    folder.mkdir(parents=True)
    path = folder / "measurements.db"
    con = sqlite3.connect(path)
    try:
        con.execute(
            "CREATE TABLE cell (plateID TEXT, rowID TEXT, columnID TEXT, "
            "fieldID TEXT, object_label INTEGER, cell_area REAL)")
        con.executemany(
            "INSERT INTO cell VALUES (?, ?, ?, ?, ?, ?)",
            [tuple(_identity(i).values()) for i in range(N_POINTS)])
        con.commit()
    finally:
        con.close()
    return str(path)


@pytest.fixture
def browser(qtbot, measurements_db):
    screen = DbBrowserScreen(threaded=False)
    qtbot.addWidget(screen)
    assert screen.set_database(measurements_db)
    assert screen.select_table("cell")
    assert screen.loaded_rows() == N_POINTS
    return screen


@pytest.fixture
def umap(qtbot):
    """An embedding of the same six objects, in the same order.

    Points 0-2 sit near the origin and 3-5 far from it, so a lasso can catch
    a known half of them.
    """
    explorer = ImageUmapExplorer()
    qtbot.addWidget(explorer)
    explorer.set_payload({
        "embedding": np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.0],
                               [5.0, 5.0], [5.1, 5.1], [5.2, 5.0]]),
        "labels": np.array([0, 0, 0, 1, 1, 1]),
        # Only the `prcfo` — the identity `generate_image_umap` really
        # attaches. The key columns are rebuilt from it.
        "records": [
            {"prcfo": f"plate1_r{i // 3 + 1}_c{i % 3 + 1}_f1_o{i + 1}"}
            for i in range(N_POINTS)
        ],
        # ... plus the measured frame, so a filter naming a feature column
        # has something here to test.
        "frame": pd.DataFrame(
            {"cell_area": [100.0 * (i + 1) for i in range(N_POINTS)]}),
    })
    return explorer


def _lasso(explorer, x0, y0, x1, y1) -> None:
    """Drag a rectangular lasso, the way the LassoSelector calls back."""
    explorer._on_lasso([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------

def test_the_two_views_agree_on_what_names_an_object(umap, browser):
    """Both sides must land on the same key or nothing below works.

    The UMAP gets there through ``prcfo`` (which spells the object ``'o3'``)
    and the table through ``object_label`` (which spells it ``3``). If that
    conversion is ever dropped the linking silently stops matching anything,
    which looks exactly like "the user lassoed empty space".
    """
    from spacr.selection import OBJECT_KEY_COLUMNS, object_keys

    assert umap.point_keys() is not None
    table = browser._linked_frame(OBJECT_KEY_COLUMNS)
    assert list(umap.point_keys()) == list(object_keys(table))


# ---------------------------------------------------------------------------
# selection: it highlights, and it never hides
# ---------------------------------------------------------------------------

def test_a_lasso_in_the_umap_highlights_the_same_rows_in_the_table(
        umap, browser):
    _lasso(umap, -1.0, -1.0, 1.0, 1.0)

    assert umap._selected.tolist() == [0, 1, 2]
    assert linked_selection().selection.source == "umap"
    assert browser.selected_rows() == [0, 1, 2]


def test_selecting_rows_in_the_table_highlights_the_same_umap_points(
        umap, browser):
    browser.select_rows([3, 5])

    assert linked_selection().selection.source == "db_browser"
    assert umap.linked_points().tolist() == [3, 5]


def test_a_selection_hides_nothing_anywhere(umap, browser):
    """A lasso that highlighted by hiding would be destructive: the points it
    missed are exactly the ones you want to compare it against."""
    _lasso(umap, -1.0, -1.0, 1.0, 1.0)

    assert browser.hidden_rows() == []
    assert browser.loaded_rows() == N_POINTS
    assert len(browser.preview_rows()) == N_POINTS
    assert umap.visible_points().all()
    assert len(umap._scatter.get_offsets()) == N_POINTS


def test_a_selection_scrolls_the_table_to_what_was_selected(umap, browser):
    """Highlighting a row 40 000 rows off screen is not highlighting it."""
    calls = []
    browser._view.scrollTo = lambda index, *a: calls.append(index.row())

    _lasso(umap, 4.0, 4.0, 6.0, 6.0)

    assert calls == [3]


def test_a_lasso_that_caught_nothing_is_published_as_an_empty_selection(
        umap, browser):
    """Different from the resting state: "I looked and there was nothing
    there" is a result, and the table should show no rows selected rather
    than keep the previous highlight."""
    browser.select_rows([0, 1])
    _lasso(umap, 90.0, 90.0, 91.0, 91.0)

    selection = linked_selection().selection
    assert selection.is_active and len(selection) == 0
    assert browser.selected_rows() == []


def test_clearing_returns_everyone_to_rest(umap, browser):
    _lasso(umap, -1.0, -1.0, 1.0, 1.0)
    assert browser.selected_rows()

    umap.clear_linked_selection()

    assert not linked_selection().selection.is_active
    assert browser.selected_rows() == []
    assert umap.linked_points().tolist() == []


# ---------------------------------------------------------------------------
# echo suppression
# ---------------------------------------------------------------------------

def test_a_view_never_hears_the_selection_it_just_published(umap, browser):
    """Without this every lasso costs the drawing view a repaint of what it
    already drew — and a view that normalises what it publishes oscillates."""
    umap_heard, browser_heard = [], []
    umap.on_linked_selection_changed = umap_heard.append
    browser.on_linked_selection_changed = browser_heard.append

    _lasso(umap, -1.0, -1.0, 1.0, 1.0)

    assert umap_heard == [], "the UMAP heard the echo of its own lasso"
    assert len(browser_heard) == 1

    umap_heard.clear()
    browser_heard.clear()
    browser.select_rows([4])

    assert browser_heard == [], "the table heard the echo of its own selection"
    assert len(umap_heard) == 1


def test_the_umap_does_not_ring_its_own_lasso_as_a_foreign_selection(umap):
    """The observable half of echo suppression: the accent ring means
    "somebody else is looking at these", so the view that drew the lasso must
    not draw one around itself."""
    _lasso(umap, -1.0, -1.0, 1.0, 1.0)

    assert umap._selected.tolist() == [0, 1, 2]
    assert umap.linked_points().tolist() == []


def test_the_table_does_not_narrow_a_selection_it_was_handed(umap, browser):
    """The table holds one page; the selection may name the whole plate.

    Re-publishing what it was just told would replace a selection of every
    object with the handful this page happens to have loaded.
    """
    absent = ["plate9_r1_c1_f1_1", "plate9_r1_c1_f1_2"]
    linked_selection().set_selection(
        Selection.from_keys(list(umap.point_keys()[:2]) + absent,
                            source="umap"))

    assert browser.selected_rows() == [0, 1]
    assert len(linked_selection().selection) == 4
    assert linked_selection().selection.source == "umap"


# ---------------------------------------------------------------------------
# filter: it hides in the table, and only dims in the embedding
# ---------------------------------------------------------------------------

def test_a_filter_hides_table_rows_but_only_dims_umap_points(umap, browser):
    """The asymmetry is deliberate. Dropping points from a UMAP re-frames the
    axes around the survivors, and an embedding that moves when you tick a
    checkbox cannot be read; a table that shows rows outside the population
    it says it is showing cannot be trusted."""
    base = float(umap._display["point_alpha"])
    linked_selection().set_filter(
        DataFilter().add(CategoryFilter("columnID", ("c1",))))

    # The table: rows 0 and 3 are the c1 objects, the other four are gone.
    assert browser.visible_rows() == [0, 3]
    assert browser.hidden_rows() == [1, 2, 4, 5]
    assert all(browser._view.isRowHidden(r) for r in (1, 2, 4, 5))
    assert not any(browser._view.isRowHidden(r) for r in (0, 3))

    # The embedding: every point still drawn, four of them faded.
    assert len(umap._scatter.get_offsets()) == N_POINTS
    assert umap.visible_points().tolist() == [True, False, False,
                                              True, False, False]
    alphas = np.asarray(umap._scatter.get_alpha(), dtype=float)
    assert alphas[[0, 3]].tolist() == [base, base]
    assert (alphas[[1, 2, 4, 5]] < base).all()


def test_a_filter_on_a_measured_column_reaches_both_views(umap, browser):
    """cell_area lives in the table's own columns and in the UMAP's attached
    frame; the same declarative filter has to mean the same thing in both."""
    linked_selection().set_filter(
        DataFilter().add(RangeFilter("cell_area", low=350.0)))

    assert browser.visible_rows() == [3, 4, 5]
    assert umap.visible_points().tolist() == [False, False, False,
                                              True, True, True]


def test_clearing_the_filter_brings_every_row_and_every_point_back(
        umap, browser):
    linked_selection().set_filter(
        DataFilter().add(CategoryFilter("columnID", ("c1",))))
    linked_selection().clear_filter()

    assert browser.hidden_rows() == []
    assert not any(browser._view.isRowHidden(r) for r in range(N_POINTS))
    assert umap.visible_points().all()
    # A scalar again, not an array of identical values: the display settings
    # are read back off the artist elsewhere.
    assert umap._scatter.get_alpha() == pytest.approx(
        float(umap._display["point_alpha"]))


def test_both_views_say_they_are_filtered(umap, browser):
    """A view showing a third of its data without saying so is how a result
    computed on a third of a plate gets reported as the whole plate."""
    linked_selection().set_filter(
        DataFilter().add(CategoryFilter("columnID", ("c1",))))

    assert "filtered" in browser.status_text()
    assert "columnID" in browser.status_text()
    assert "filtered" in umap._status.text()
    assert "columnID" in umap._status.text()


def test_a_filter_naming_an_absent_column_hides_nothing_and_says_so(
        umap, browser):
    """Carried over from another table. An empty view is a worse answer than
    a complete one — but it must not pass itself off as a filtered one."""
    linked_selection().set_filter(
        DataFilter().add(RangeFilter("no_such_column", low=1.0)))

    assert browser.hidden_rows() == []
    assert "ignored" in browser.status_text()
    assert umap.visible_points().all()
    assert "ignored" in umap._status.text()


def test_an_embedding_with_no_identities_is_never_dimmed_away(qtbot):
    """The payload from an older run carries no ``prcfo``. Testing a category
    filter against a column of blanks would match nothing and fade the entire
    embedding out — which looks like a bug, not like a filter."""
    explorer = ImageUmapExplorer()
    qtbot.addWidget(explorer)
    explorer.set_payload({
        "embedding": np.array([[0.0, 0.0], [1.0, 1.0]]),
        "labels": np.array([0, 1]),
        "records": [{"display_name": "a.png"}, {"display_name": "b.png"}],
    })
    assert explorer.point_keys() is None

    linked_selection().set_filter(
        DataFilter().add(CategoryFilter("columnID", ("c1",))))

    assert explorer.visible_points().all()
    assert "ignored" in explorer._status.text()

    # And a lasso in it stays local rather than publishing keys it invented.
    _lasso(explorer, -1.0, -1.0, 0.5, 0.5)
    assert not linked_selection().selection.is_active


def test_a_filter_survives_the_next_chunk_of_rows(measurements_db, qtbot):
    """Row hiding is positional and Qt drops it on a model reset, so a page
    that arrives after the filter must be hidden too — otherwise scrolling
    quietly re-admits the rows the user filtered out."""
    screen = DbBrowserScreen(threaded=False)
    qtbot.addWidget(screen)
    screen.auto_count = False
    screen._page_size_box.setValue(25)
    assert screen.set_database(measurements_db)
    linked_selection().set_filter(
        DataFilter().add(CategoryFilter("columnID", ("c1",))))

    assert screen.select_table("cell")

    assert screen.hidden_rows() == [1, 2, 4, 5]


def test_a_filter_and_a_selection_do_not_interfere(umap, browser):
    """`visible()` applies the filter and never the selection: a selected row
    outside the population still hides, and a filtered-in row that nobody
    selected still shows."""
    linked_selection().set_selection(
        Selection.from_keys(list(umap.point_keys()), source="umap"))
    linked_selection().set_filter(
        DataFilter().add(CategoryFilter("columnID", ("c1",))))

    assert browser.visible_rows() == [0, 3]
    assert browser.selected_rows() == list(range(N_POINTS))


# ---------------------------------------------------------------------------
# the plate view, after the migration off its hand-rolled wiring
# ---------------------------------------------------------------------------

def _plate_frame() -> pd.DataFrame:
    rows = []
    for row in ("A", "B"):
        for column in (1, 2, 3):
            rows.append({"plateID": "p1", "rowID": row, "columnID": str(column),
                         "object_count": 50, "value": 1.0})
    return pd.DataFrame(rows)


@pytest.fixture
def plate(qtbot):
    screen = PlateViewScreen(threaded=False)
    qtbot.addWidget(screen)
    screen._frame = _plate_frame()
    screen._frame_key = ("", "", "value")
    screen._refresh_plate_combo(screen._frame)
    screen.recompute()
    return screen


def test_the_plate_view_is_linked_under_its_own_name(plate):
    assert plate.is_linked
    assert plate.link_source == "plate_view"


def test_the_plate_view_still_redraws_for_a_filter(plate):
    before = len(plate._layout_df)
    linked_selection().set_filter(
        DataFilter().add(CategoryFilter("columnID", ("1", "2"))))
    assert len(plate._layout_df) < before
    assert "filtered" in plate._filter_note


def test_a_selection_does_not_redraw_the_plate_view(plate):
    """A well is an aggregate of many objects, so there is nothing here for a
    selection to light up — and re-running the edge-effect statistics on
    every lasso would make the heatmap the slowest thing on screen."""
    before = plate._layout_df
    linked_selection().set_selection(
        Selection.from_keys(["p1_A_1_f1_1"], source="umap"))
    assert plate._layout_df is before


def test_the_plate_view_publishes_nothing(plate):
    """It reads the shared population; it does not drive it. A view that
    published on every redraw would fight the view the user is working in."""
    plate.recompute()
    plate.select_well(1, 1)
    assert not linked_selection().selection.is_active


def test_closing_the_plate_view_twice_is_silent(plate):
    """Qt closes a widget again on teardown; the mixin's disconnect is
    flag-guarded so the second one is a no-op rather than a
    `libpyside: Failed to disconnect` nobody can catch."""
    plate.close()
    assert not plate.is_linked
    plate.close()

    # Nothing is routed at the closed screen any more.
    linked_selection().set_filter(
        DataFilter().add(CategoryFilter("columnID", ("1",))))


def test_closing_a_linked_view_stops_it_listening(umap, browser):
    browser.close()
    umap.close()

    assert not browser.is_linked
    assert not umap.is_linked

    # Must not raise, and must not paint into either closed screen.
    linked_selection().set_filter(
        DataFilter().add(CategoryFilter("columnID", ("c1",))))
    linked_selection().set_selection(
        Selection.from_keys(["plate1_r1_c1_f1_1"], source="somewhere"))

    assert browser.hidden_rows() == []
    assert umap.linked_points().tolist() == []
