"""Tabulate: every aggregate and every n, worked out by hand first.

The table below is seven rows and is small enough that the whole answer can be
written down before the code runs, which is the only way to test an aggregator
that has an opinion about n=1, about NaN, and about the difference between "no
objects here" and "zero".

The frame::

    plateID rowID gene   area
    p1      r1    a      10
    p1      r1    b      20
    p1      r2    a      30
    p1      r2    a      34
    p2      r1    b      NaN
    p2      r1    b      NaN
    p2      r1    b       5

Pivoted with rows = (plateID, rowID) and columns = (gene,), the full cartesian
product is four rows by two columns, and every cell of it is known:

    (p1, r1) × a  ->  n 1, mean 10, sd blank (one object has no spread)
    (p1, r1) × b  ->  n 1, mean 20, sd blank
    (p1, r2) × a  ->  n 2, mean 32, sd 2*sqrt(2), sem 2, q75 33
    (p1, r2) × b  ->  EMPTY — nothing was measured there
    (p2, r1) × a  ->  EMPTY
    (p2, r1) × b  ->  3 source rows, n 1, mean 5 — measured, mostly unmeasurable
    (p2, r2) × a  ->  EMPTY   (p2 has no r2; the grid says so rather than
    (p2, r2) × b  ->  EMPTY    closing the gap)

The two cases that are easy to get wrong and are asserted hardest: an empty
cell must not read as ``0``, and ``(p2, r1) × b`` must read ``n = 1`` out of
three source rows rather than ``n = 3``.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from PySide6.QtCore import QMimeData, QPointF, Qt
from PySide6.QtGui import QDropEvent

from spacr.qt.linked_selection import LinkedSelection
from spacr.qt.widgets.graph_builder import COLUMN_MIME
from spacr.qt.widgets.pivot_builder import (
    AXIS_COLS, AXIS_ROWS, AXIS_VALUES, PivotPanel,
)
from spacr.qt.widgets.pivot_spec import (
    COUNT_ONLY, LOW_N, MAX, MEAN, MEDIAN, MIN, N, QUANTILE, SD, SEM,
    PivotError, PivotSpec, format_value, pivot,
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def hand_frame() -> pd.DataFrame:
    """The seven rows of the module docstring."""
    return pd.DataFrame({
        "plateID": ["p1", "p1", "p1", "p1", "p2", "p2", "p2"],
        "rowID": ["r1", "r1", "r2", "r2", "r1", "r1", "r1"],
        "gene": ["a", "b", "a", "a", "b", "b", "b"],
        "area": [10.0, 20.0, 30.0, 34.0, np.nan, np.nan, 5.0],
        "object_label": [1, 2, 3, 4, 5, 6, 7],
        "columnID": ["c1"] * 7,
        "fieldID": ["f1"] * 7,
    })


ALL_AGGS = (N, MEAN, MEDIAN, SD, SEM, MIN, MAX, QUANTILE)


@pytest.fixture
def hand() -> pd.DataFrame:
    return hand_frame()


@pytest.fixture
def table(hand) -> "object":
    return pivot(hand, PivotSpec(rows=("plateID", "rowID"), cols=("gene",),
                                 values=("area",), aggs=ALL_AGGS,
                                 quantile=0.75))


def at(result, row_levels, col_levels):
    """``(row, col)`` of a named cell, so the assertions read like the table."""
    return (result.row_levels.index(row_levels),
            result.col_levels.index(col_levels))


# ---------------------------------------------------------------------------
# The hand-computed table
# ---------------------------------------------------------------------------

def test_the_grid_is_the_full_product_including_the_combination_with_no_rows(
        table):
    """p2 has no r2, and the table says so rather than closing the gap."""
    assert table.shape == (4, 2)
    assert table.row_levels == (("p1", "r1"), ("p1", "r2"),
                                ("p2", "r1"), ("p2", "r2"))
    assert table.col_levels == (("a",), ("b",))
    assert table.n_cells == 8


def test_every_aggregate_of_every_cell(table):
    r, c = at(table, ("p1", "r2"), ("a",))
    assert table.n_at("area", r, c) == 2
    assert table.value_at("area", MEAN, r, c) == pytest.approx(32.0)
    assert table.value_at("area", MEDIAN, r, c) == pytest.approx(32.0)
    assert table.value_at("area", SD, r, c) == pytest.approx(2 * math.sqrt(2))
    assert table.value_at("area", SEM, r, c) == pytest.approx(2.0)
    assert table.value_at("area", MIN, r, c) == pytest.approx(30.0)
    assert table.value_at("area", MAX, r, c) == pytest.approx(34.0)
    assert table.value_at("area", QUANTILE, r, c) == pytest.approx(33.0)

    r, c = at(table, ("p1", "r1"), ("a",))
    assert table.n_at("area", r, c) == 1
    assert table.value_at("area", MEAN, r, c) == pytest.approx(10.0)
    assert table.value_at("area", MIN, r, c) == pytest.approx(10.0)
    assert table.value_at("area", MAX, r, c) == pytest.approx(10.0)

    r, c = at(table, ("p1", "r1"), ("b",))
    assert table.n_at("area", r, c) == 1
    assert table.value_at("area", MEAN, r, c) == pytest.approx(20.0)


def test_the_spread_of_one_object_is_blank_and_not_zero(table):
    """n=1 has no spread. A 0 there reads as 'perfectly reproducible'."""
    r, c = at(table, ("p1", "r1"), ("a",))
    assert math.isnan(table.value_at("area", SD, r, c))
    assert math.isnan(table.value_at("area", SEM, r, c))
    assert format_value(table.value_at("area", SD, r, c)) == ""


def test_an_empty_cell_reads_as_empty_and_never_as_zero(table):
    """The assertion this whole module exists for."""
    for row, col in ((("p1", "r2"), ("b",)),
                     (("p2", "r1"), ("a",)),
                     (("p2", "r2"), ("a",)),
                     (("p2", "r2"), ("b",))):
        r, c = at(table, row, col)
        assert table.is_empty(r, c), f"{row} x {col} should be empty"
        assert table.n_at("area", r, c) is None      # not 0
        assert table.sizes[r, c] == 0
        for agg in ALL_AGGS:
            value = table.value_at("area", agg, r, c)
            assert math.isnan(value), f"{agg} of an empty cell is {value}"
            assert format_value(value) == ""


def test_objects_measured_with_no_value_are_n_zero_which_is_not_empty(table):
    """(p2, r1) x b: three cells' worth of objects, one usable area.

    This is the distinction the empty rule must not swallow. Three objects
    were measured there; two produced no area. ``n`` is the number a mean was
    taken over, so it is 1 — not 3, and not blank.
    """
    r, c = at(table, ("p2", "r1"), ("b",))
    assert not table.is_empty(r, c)
    assert table.sizes[r, c] == 3
    assert table.n_at("area", r, c) == 1
    assert table.value_at("area", MEAN, r, c) == pytest.approx(5.0)


def test_a_value_column_that_is_entirely_nan_in_a_cell_reads_n_zero(hand):
    frame = hand.copy()
    frame.loc[frame.index[6], "area"] = np.nan     # now all three are NaN
    result = pivot(frame, PivotSpec(rows=("plateID", "rowID"), cols=("gene",),
                                    values=("area",), aggs=(N, MEAN)))
    r, c = at(result, ("p2", "r1"), ("b",))
    assert not result.is_empty(r, c)               # the objects are there
    assert result.n_at("area", r, c) == 0          # and none has a value
    assert math.isnan(result.value_at("area", MEAN, r, c))


def test_n_is_computed_whether_or_not_it_was_asked_for(hand):
    result = pivot(hand, PivotSpec(rows=("plateID",), values=("area",),
                                   aggs=(MEAN,)))
    assert N in result.spec.aggs
    assert result.n_at("area", 0, 0) == 4
    assert result.n_at("area", 1, 0) == 1


def test_the_n_range_is_over_values_not_rows(table):
    """Where they differ — a NaN-heavy column — the smaller one is the truth."""
    assert table.n_range() == (1, 2)
    assert int(table.sizes[table.present].max()) == 3
    assert "n per cell 1–2" in table.summary()


def test_low_n_cells_are_counted_for_the_reader(table):
    assert table.low_n_cells(threshold=LOW_N) == 4
    assert table.low_n_cells(threshold=1) == 3
    assert f"n ≤ {LOW_N}" in table.summary()


# ---------------------------------------------------------------------------
# Degenerate and hierarchical shapes
# ---------------------------------------------------------------------------

def test_no_keys_at_all_is_one_cell_over_the_whole_frame(hand):
    result = pivot(hand, PivotSpec(values=("area",), aggs=(N, MEAN)))
    assert result.shape == (1, 1)
    assert result.n_at("area", 0, 0) == 5
    assert result.value_at("area", MEAN, 0, 0) == pytest.approx(99 / 5)


def test_rows_only_is_a_group_summary(hand):
    result = pivot(hand, PivotSpec(rows=("plateID",), values=("area",),
                                   aggs=(N, MEAN)))
    assert result.shape == (2, 1)
    assert result.value_at("area", MEAN, 0, 0) == pytest.approx(23.5)
    assert result.value_at("area", MEAN, 1, 0) == pytest.approx(5.0)


def test_no_values_is_a_contingency_table_of_counts(hand):
    result = pivot(hand, PivotSpec(rows=("plateID",), cols=("gene",)))
    assert result.layer_keys == ((COUNT_ONLY, N),)
    assert result.value_at(COUNT_ONLY, N, 0, 0) == 3      # p1 x a
    assert result.value_at(COUNT_ONLY, N, 0, 1) == 1      # p1 x b
    assert result.is_empty(1, 0)                          # p2 x a: none
    assert math.isnan(result.value_at(COUNT_ONLY, N, 1, 0))


def test_the_plate_hierarchy_nests_in_drop_order(hand):
    plate_first = pivot(hand, PivotSpec(rows=("plateID", "rowID"),
                                        values=("area",)))
    row_first = pivot(hand, PivotSpec(rows=("rowID", "plateID"),
                                      values=("area",)))
    assert plate_first.row_levels[0] == ("p1", "r1")
    assert row_first.row_levels[0] == ("r1", "p1")
    assert plate_first.row_keys == ("plateID", "rowID")


def test_levels_sort_numerically_so_plate_2_precedes_plate_10():
    frame = pd.DataFrame({"plateID": ["10", "2", "2"], "area": [1.0, 2.0, 3.0]})
    result = pivot(frame, PivotSpec(rows=("plateID",), values=("area",)))
    assert result.row_levels == (("2",), ("10",))


def test_a_missing_key_becomes_its_own_level_rather_than_vanishing(hand):
    frame = hand.copy()
    frame.loc[frame.index[0], "gene"] = None
    result = pivot(frame, PivotSpec(rows=("gene",), values=("area",)))
    assert result.row_levels[0] == ("(missing)",)
    assert result.n_at("area", 0, 0) == 1
    # Nothing was lost on the way in.
    total = sum(int(result.layers[("area", N)][r, 0])
                for r in range(result.shape[0])
                if not result.is_empty(r, 0))
    assert total == int(frame["area"].notna().sum())


def test_a_non_numeric_value_column_says_so_rather_than_looking_empty(hand):
    result = pivot(hand, PivotSpec(rows=("plateID",), values=("gene",)))
    assert "not numeric" in result.notice
    assert "rows or columns instead" in result.notice


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------

def test_the_spec_round_trips_through_json_exactly():
    spec = PivotSpec(rows=("plateID", "rowID"), cols=("gene",),
                     values=("area", "intensity"), aggs=(N, MEAN, QUANTILE),
                     quantile=0.9)
    assert PivotSpec.from_json(spec.to_json()) == spec
    assert json.loads(spec.to_json())["rows"] == ["plateID", "rowID"]


def test_a_spec_from_another_build_still_opens():
    spec = PivotSpec.from_dict({"rows": ["plateID"], "future_option": 3})
    assert spec.rows == ("plateID",)
    assert N in spec.aggs


def test_a_column_cannot_be_on_both_axes():
    with pytest.raises(PivotError, match="both the row and the column axis"):
        PivotSpec(rows=("gene",), cols=("gene",))


def test_an_unknown_aggregation_is_refused_where_the_spec_is_built():
    with pytest.raises(PivotError, match="unknown aggregation"):
        PivotSpec(aggs=("geometric_mean",))


def test_a_quantile_outside_zero_to_one_is_refused():
    with pytest.raises(PivotError, match=r"\[0, 1\]"):
        PivotSpec(quantile=1.5)


def test_a_value_column_the_table_does_not_have_is_refused_with_a_way_out(hand):
    with pytest.raises(PivotError, match="not in this table"):
        pivot(hand, PivotSpec(rows=("plateID",), values=("nope",)))


def test_asking_for_a_layer_that_was_not_computed_says_what_is_there():
    result = pivot(hand_frame(), PivotSpec(rows=("plateID",),
                                           values=("area",), aggs=(N, MEAN)))
    with pytest.raises(PivotError, match="no median"):
        result.value_at("area", MEDIAN, 0, 0)


def test_duplicate_keys_collapse_rather_than_nesting_twice():
    spec = PivotSpec(rows=("plateID", "plateID"))
    assert spec.rows == ("plateID",)


# ---------------------------------------------------------------------------
# Export and the chart
# ---------------------------------------------------------------------------

def test_the_wide_frame_is_the_table_as_shown(table):
    wide = table.to_frame()
    assert list(wide.columns[:2]) == ["plateID", "rowID"]
    assert "a · mean(area)" in wide.columns
    assert "b · n(area)" in wide.columns
    assert len(wide) == 4
    row = wide[(wide["plateID"] == "p1") & (wide["rowID"] == "r2")].iloc[0]
    assert row["a · mean(area)"] == pytest.approx(32.0)
    assert math.isnan(row["b · mean(area)"])      # empty stays empty in the CSV


def test_the_csv_round_trips_with_the_empties_still_empty(table, tmp_path):
    path = table.to_csv(str(tmp_path / "t.csv"))
    back = pd.read_csv(path)
    assert len(back) == 4
    assert back["a · n(area)"].tolist()[:2] == [1.0, 2.0]
    assert math.isnan(back["b · mean(area)"].iloc[1])


def test_the_long_frame_is_one_row_per_non_empty_cell(table):
    long = table.to_long()
    assert len(long) == 4                       # eight cells, four with data
    assert set(long.columns) >= {"plateID", "rowID", "gene", "n", "mean",
                                 "sd", "cell_rows", "value_column"}
    picked = long[(long["plateID"] == "p2") & (long["gene"] == "b")].iloc[0]
    assert picked["n"] == 1
    assert picked["cell_rows"] == 3             # and the difference survives


def test_the_long_frame_carries_one_row_per_value_column(hand):
    frame = hand.assign(intensity=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    result = pivot(frame, PivotSpec(rows=("plateID",),
                                    values=("area", "intensity"),
                                    aggs=(N, MEAN)))
    long = result.to_long()
    assert set(long["value_column"]) == {"area", "intensity"}
    assert len(long) == 4                       # two plates x two measurements


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------

def drop(well, column: str) -> None:
    """Drop ``column`` onto a well the way the drag-and-drop actually does."""
    payload = QMimeData()
    payload.setData(COLUMN_MIME, column.encode("utf-8"))
    well._list.dropEvent(QDropEvent(QPointF(4, 4), Qt.CopyAction, payload,
                                    Qt.LeftButton, Qt.NoModifier))


@pytest.fixture
def panel(qtbot, hand) -> PivotPanel:
    widget = PivotPanel()
    qtbot.addWidget(widget)
    widget.set_frame(hand)
    return widget


def test_dropping_columns_builds_the_table(panel):
    drop(panel.wells[AXIS_ROWS], "plateID")
    drop(panel.wells[AXIS_ROWS], "rowID")
    drop(panel.wells[AXIS_COLS], "gene")
    drop(panel.wells[AXIS_VALUES], "area")
    result = panel.recompute()

    assert panel.spec().rows == ("plateID", "rowID")
    assert result.shape == (4, 2)
    assert result.value_at("area", MEAN, 1, 0) == pytest.approx(32.0)


def test_a_column_dropped_twice_lands_once(panel):
    drop(panel.wells[AXIS_ROWS], "plateID")
    drop(panel.wells[AXIS_ROWS], "plateID")
    assert panel.wells[AXIS_ROWS].columns() == ("plateID",)


def test_the_well_only_takes_a_column_payload(panel):
    text_only = QMimeData()
    # See the same payload in test_graph_builder.py: /tmp, not a home
    # directory, because nothing here opens it and the hygiene rule cannot
    # tell a decorative path from a one-machine precondition.
    text_only.setText("/tmp/a-file.tif")
    panel.wells[AXIS_ROWS]._list.dropEvent(QDropEvent(
        QPointF(1, 1), Qt.CopyAction, text_only, Qt.LeftButton, Qt.NoModifier))
    assert panel.wells[AXIS_ROWS].columns() == ()


def test_every_populated_cell_prints_its_n(panel):
    drop(panel.wells[AXIS_ROWS], "plateID")
    drop(panel.wells[AXIS_COLS], "gene")
    drop(panel.wells[AXIS_VALUES], "area")
    panel.recompute()

    # p1 x a: three objects, mean 74/3.
    text = panel.table.cell_text(0, 0)
    assert "n 3" in text
    assert "mean" in text
    # And the empty cell is blank, not "0" and not "nan".
    assert panel.table.cell_text(1, 0) == ""


def test_a_low_n_cell_is_marked_for_the_reader(panel):
    drop(panel.wells[AXIS_ROWS], "plateID")
    drop(panel.wells[AXIS_ROWS], "rowID")
    drop(panel.wells[AXIS_COLS], "gene")
    drop(panel.wells[AXIS_VALUES], "area")
    panel.recompute()
    keys = len(panel.result.row_keys)
    item = panel.table.item(0, keys)            # (p1, r1) x a: n = 1
    assert item.font().italic()
    assert f"n ≤ {LOW_N}" in item.toolTip()


def test_an_empty_cell_says_which_kind_of_nothing_it_is(panel):
    drop(panel.wells[AXIS_ROWS], "plateID")
    drop(panel.wells[AXIS_ROWS], "rowID")
    drop(panel.wells[AXIS_COLS], "gene")
    drop(panel.wells[AXIS_VALUES], "area")
    panel.recompute()
    keys = len(panel.result.row_keys)
    tooltip = panel.table.item(1, keys + 1).toolTip()   # (p1, r2) x b
    assert "No objects here at all" in tooltip
    assert "blank rather than zero" in tooltip


def test_the_preset_puts_the_well_hierarchy_on_the_rows(panel):
    panel.use_well_hierarchy()
    assert panel.wells[AXIS_ROWS].columns() == ("plateID", "rowID", "columnID")


def test_clearing_a_well_rebuilds_the_table(panel):
    drop(panel.wells[AXIS_ROWS], "plateID")
    drop(panel.wells[AXIS_VALUES], "area")
    panel.recompute()
    assert panel.result.shape == (2, 1)
    panel.wells[AXIS_ROWS].clear()
    panel.recompute()
    assert panel.result.shape == (1, 1)


def test_an_axis_column_the_new_table_lacks_is_dropped_not_carried(panel):
    drop(panel.wells[AXIS_ROWS], "plateID")
    drop(panel.wells[AXIS_VALUES], "area")
    panel.set_frame(pd.DataFrame({"gene": ["a", "b"], "area": [1.0, 2.0]}))
    assert panel.wells[AXIS_ROWS].columns() == ()
    assert panel.wells[AXIS_VALUES].columns() == ("area",)


def test_the_panel_exports_the_csv_it_is_showing(panel, tmp_path):
    drop(panel.wells[AXIS_ROWS], "plateID")
    drop(panel.wells[AXIS_COLS], "gene")
    drop(panel.wells[AXIS_VALUES], "area")
    panel.recompute()
    path = panel.export_csv(str(tmp_path / "out.csv"))
    back = pd.read_csv(path)
    assert list(back["plateID"]) == ["p1", "p2"]
    assert math.isnan(back["a · mean(area)"].iloc[1])


def test_asking_to_plot_emits_the_long_frame(panel):
    drop(panel.wells[AXIS_ROWS], "plateID")
    drop(panel.wells[AXIS_VALUES], "area")
    panel.recompute()
    emitted = []
    panel.plot_requested.connect(emitted.append)
    panel._on_plot()
    assert len(emitted) == 1
    assert list(emitted[0]["plateID"]) == ["p1", "p2"]
    assert list(emitted[0]["n"]) == [4.0, 1.0]


def test_an_empty_spec_asks_for_a_column_rather_than_drawing_nothing(panel):
    assert panel.recompute() is None
    assert "Drop a column onto Rows" in panel.notice.text()


# ---------------------------------------------------------------------------
# The screen
# ---------------------------------------------------------------------------

@pytest.fixture
def link() -> LinkedSelection:
    """A PRIVATE link — never the process-wide one."""
    return LinkedSelection()


def test_the_summary_goes_to_the_graph_builder_rather_than_a_second_plotter(
        qtbot, link, hand):
    from spacr.qt.screens.tabulate import TabulateScreen
    from spacr.qt.widgets.graph_spec import GraphSpec

    screen = TabulateScreen(link=link)
    qtbot.addWidget(screen)
    screen.set_frame(hand)
    drop(screen.pivot.wells[AXIS_ROWS], "plateID")
    drop(screen.pivot.wells[AXIS_VALUES], "area")
    screen.pivot.recompute()
    screen.pivot._on_plot()

    canvas = screen.graph.canvas
    canvas.set_spec(GraphSpec(x="plateID", y="mean"))
    assert canvas.render_data is not None
    assert canvas.render_data.n_total == 2
    # A summary row is a group, not an object — and the chart says so rather
    # than publishing an empty selection when someone brushes it.
    assert "no object keys" in canvas.notice()


def test_the_shared_filter_re_aggregates_rather_than_restyling(
        qtbot, link, hand):
    from spacr.qt.screens.tabulate import TabulateScreen
    from spacr.selection import CategoryFilter, DataFilter

    screen = TabulateScreen(link=link)
    qtbot.addWidget(screen)
    screen.set_frame(hand)
    drop(screen.pivot.wells[AXIS_ROWS], "plateID")
    drop(screen.pivot.wells[AXIS_VALUES], "area")
    screen.pivot.recompute()
    assert screen.pivot.result.n_at("area", 0, 0) == 4

    link.set_filter(DataFilter([CategoryFilter("gene", ("a",))]))
    screen._recompute_filtered()
    screen.pivot.recompute()
    assert screen.pivot.result.n_at("area", 0, 0) == 3       # b is gone
    assert screen.pivot.result.shape == (1, 1)               # and so is p2



@pytest.fixture
def registry_sandbox():
    """Restore the whole app registry after the test.

    A leaked row is a leaked tile, a leaked sidebar button and a leaked
    keyboard binding for every test that runs afterwards, so the list object is
    restored in place rather than trusting an unregister call.
    """
    from spacr.qt import app as app_mod
    apps = list(app_mod.APPS)
    factories = dict(app_mod.APP_FACTORIES)
    stages = dict(app_mod.APP_STAGE)
    meta = dict(app_mod.APP_META)
    yield app_mod
    app_mod.APPS[:] = apps
    app_mod.APP_FACTORIES.clear()
    app_mod.APP_FACTORIES.update(factories)
    app_mod.APP_STAGE.clear()
    app_mod.APP_STAGE.update(stages)
    app_mod.APP_META.clear()
    app_mod.APP_META.update(meta)
    app_mod._refresh_sections()


def test_the_screen_is_not_registered_until_app_py_says_so(qtbot):
    """One row in app.py's `_SELF_REGISTERING_APPS` turns it on; not this file."""
    from spacr.qt.app import APPS
    from spacr.qt.screens import tabulate as screen

    assert not any(row[0] == screen.APP_KEY for row in APPS)
    qtbot.addWidget(screen.make_tabulate_screen())


def test_registering_the_screen_reaches_every_reader_of_the_registry(
        registry_sandbox):
    """Driving `register()` is the same thing the one line will do."""
    from spacr.qt.screens import tabulate as screen
    app_mod = registry_sandbox

    assert screen.register() is True
    assert screen.register() is False           # idempotent, not a raise

    row = next(r for r in app_mod.APPS if r[0] == screen.APP_KEY)
    assert row[1] == screen.APP_NAME
    assert row[3] == app_mod.SECTION_EXPLORE
    assert app_mod.APP_FACTORIES[screen.APP_KEY] is screen.make_tabulate_screen

    from spacr import cli
    from spacr.qt.screens.app_screen import APP_INTROS, APP_TITLES
    assert APP_TITLES[screen.APP_KEY] == screen.APP_NAME
    assert APP_INTROS[screen.APP_KEY] == screen.APP_INTRO
    assert cli.INTERACTIVE_ONLY[screen.APP_KEY] == screen.APP_CLI_NOTE
