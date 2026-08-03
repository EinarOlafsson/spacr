"""The Graph Builder: its spec, its facet/scale engine, and its linking.

Assertions are on what the engine computes and on what the canvas actually
put on the axes — not on widget existence. A graph builder that lays out
beautifully and shares the wrong axis limits between panels is the failure
worth catching, because it is the one nobody notices.

The frame every test uses has a deliberately RAGGED facet structure: plate
``p1`` was measured in rows r1/r2 and plate ``p2`` in rows r2/r3. A 2x3 grid
over it therefore has two combinations with no rows at all, which is the case
the "draw empty panels" rule exists for.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from PySide6.QtCore import QMimeData, QPointF, Qt
from PySide6.QtGui import QDropEvent

from spacr.qt.linked_selection import LinkedSelection
from spacr.qt.widgets.graph_builder import (
    COLUMN_MIME, DropZone, GraphBuilderPanel, GraphCanvas,
)
from spacr.qt.widgets.graph_spec import (
    BAR, BINNED, BOX, CATEGORICAL, CONTINUOUS, EMPTY, FULL, HEATMAP,
    HISTOGRAM, SAMPLED, SCATTER, VIOLIN, GraphSpec, SpecError, column_kinds,
    facet_grid, infer_kind, plottable_columns, prepare_data, scales_for,
)
from spacr.selection import CategoryFilter, DataFilter, Selection, object_keys


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def ragged_frame() -> pd.DataFrame:
    """80 objects over two plates whose row sets only partly overlap.

    ``area`` is deliberately an order of magnitude bigger on p2 than on p1, so
    a per-panel autoscale and a shared scale cannot possibly agree by accident.
    """
    rng = np.random.default_rng(7)
    blocks = []
    for plate, rows, scale in (("p1", ("r1", "r2"), 1.0),
                               ("p2", ("r2", "r3"), 20.0)):
        for row in rows:
            n = 20
            blocks.append(pd.DataFrame({
                "plateID": [plate] * n,
                "rowID": [row] * n,
                "columnID": [f"c{i % 4 + 1}" for i in range(n)],
                "fieldID": ["f1"] * n,
                "object_label": range(1, n + 1),
                "area": np.linspace(1.0, 50.0, n) * scale,
                "intensity": rng.normal(size=n),
                "gene": [f"g{i % 3}" for i in range(n)],
                "cell_count": [i % 4 for i in range(n)],
                "note": [f"free text {plate}{row}{i}" for i in range(n)],
            }))
    return pd.concat(blocks, ignore_index=True)


@pytest.fixture
def frame() -> pd.DataFrame:
    return ragged_frame()


@pytest.fixture
def link() -> LinkedSelection:
    """A PRIVATE link, never the process-wide one — every other open view
    listens to that."""
    return LinkedSelection()


@pytest.fixture
def canvas(qtbot, link, frame) -> GraphCanvas:
    view = GraphCanvas(link=link, source="test_graph")
    qtbot.addWidget(view)
    view.set_frame(frame)
    return view


# ---------------------------------------------------------------------------
# The spec is a plain, serialisable object
# ---------------------------------------------------------------------------

def test_a_spec_round_trips_through_a_dict_and_through_json():
    """A chart has to survive being written to a settings file and read back.

    Every later item on the list codes against this object, so its
    serialisation is the contract, not an implementation detail.
    """
    spec = GraphSpec(x="area", y="intensity", colour="gene", size="cell_count",
                     facet_row="plateID", facet_col="rowID", kind=SCATTER,
                     roles={"cell_count": CONTINUOUS}, bins=17,
                     shared_x=True, shared_y=False, point_budget=1234, seed=9)

    assert GraphSpec.from_dict(spec.to_dict()) == spec
    assert GraphSpec.from_json(spec.to_json()) == spec

    payload = spec.to_dict()
    # The schema is fixed: every field present, every value plain JSON.
    assert set(payload) == {
        "x", "y", "colour", "size", "facet_row", "facet_col", "kind", "roles",
        "bins", "shared_x", "shared_y", "point_budget", "seed"}
    assert all(isinstance(v, (str, int, bool, dict, type(None)))
               for v in payload.values())


def test_a_spec_written_by_another_build_still_loads():
    """Unknown keys ignored, missing keys defaulted. A spec that will not load
    is a chart the user cannot get back."""
    spec = GraphSpec.from_dict({"x": "area", "hovercraft": "eels"})
    assert spec.x == "area"
    assert spec.y is None
    assert spec.bins == GraphSpec().bins


def test_a_spec_is_immutable_and_edits_return_a_new_one():
    spec = GraphSpec(x="area")
    moved = spec.with_channel("y", "intensity")
    assert spec.y is None and moved.y == "intensity"
    assert moved.with_channel("y", None).y is None
    # "" and None are the same empty zone, so `if spec.x:` is the whole test.
    assert GraphSpec(x="").x is None


def test_a_spec_that_cannot_mean_anything_raises_where_it_was_built():
    with pytest.raises(SpecError):
        GraphSpec(kind="pie")
    with pytest.raises(SpecError):
        GraphSpec(bins=0)
    with pytest.raises(SpecError):
        GraphSpec().with_channel("z", "area")
    with pytest.raises(SpecError):
        GraphSpec(roles={"area": "ordinal"})


# ---------------------------------------------------------------------------
# The column classifier is the filter panel's, not a second one
# ---------------------------------------------------------------------------

def test_column_kinds_are_the_local_data_filters_rule_reread(frame):
    """One classifier for the app: the filter offering ticks and the plot
    treating the same column as continuous would be two mental models of one
    table."""
    from spacr.qt.widgets.data_filter_panel import classify_columns
    kinds = column_kinds(frame)
    base = classify_columns(frame)
    translation = {"range": CONTINUOUS, "category": CATEGORICAL,
                   "skip": "skip"}
    assert kinds == {name: translation[kind] for name, kind in base.items()}
    assert kinds["area"] == CONTINUOUS
    assert kinds["plateID"] == CATEGORICAL
    assert kinds["note"] == "skip"          # free text: not an axis
    assert "note" not in plottable_columns(frame)


def test_a_role_override_beats_the_table_wide_rule(frame):
    """`cell_count` has four levels, so the rule calls it categorical. A user
    who wants it as a number says so rather than editing the table."""
    assert column_kinds(frame)["cell_count"] == CATEGORICAL
    spec = GraphSpec(x="cell_count", y="area")
    assert spec.resolved_kind(spec.kinds_for(frame)) == BOX
    numeric = spec.with_role("cell_count", CONTINUOUS)
    assert numeric.resolved_kind(numeric.kinds_for(frame)) == SCATTER


# ---------------------------------------------------------------------------
# What gets dropped decides what gets drawn
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("channels, expected", [
    ({}, EMPTY),
    ({"x": "area"}, HISTOGRAM),                       # one continuous
    ({"y": "intensity"}, HISTOGRAM),                  # ...on either axis
    ({"x": "gene"}, BAR),                             # one categorical
    ({"x": "area", "y": "intensity"}, SCATTER),       # two continuous
    ({"x": "gene", "y": "area"}, BOX),                # one of each
    ({"x": "area", "y": "gene"}, BOX),                # ...either way round
    ({"x": "gene", "y": "plateID"}, HEATMAP),         # two categorical
])
def test_the_default_plot_type_follows_the_columns_dropped(
        frame, channels, expected):
    spec = GraphSpec(**channels)
    assert infer_kind(spec, column_kinds(frame)) == expected
    assert spec.resolved_kind(column_kinds(frame)) == expected


def test_colour_and_facets_never_change_what_kind_of_chart_this_is(frame):
    """Dragging `gene` onto colour must not turn a scatter into something
    else, or the chart the user built stops being the one they are reading."""
    kinds = column_kinds(frame)
    base = GraphSpec(x="area", y="intensity")
    for channel, column in (("colour", "gene"), ("size", "cell_count"),
                            ("facet_row", "plateID"), ("facet_col", "rowID")):
        assert base.with_channel(channel, column).resolved_kind(kinds) == SCATTER


def test_a_pinned_kind_beats_the_inference_and_survives_a_new_drop(frame):
    kinds = column_kinds(frame)
    pinned = GraphSpec(x="gene", y="area", kind=VIOLIN)
    assert pinned.resolved_kind(kinds) == VIOLIN
    # Still pinned after another drag — that is the difference between a pin
    # and an inference that happened to agree.
    assert pinned.with_channel("colour", "plateID").resolved_kind(kinds) == VIOLIN
    assert pinned.with_kind(None).resolved_kind(kinds) == BOX


def test_the_canvas_infers_from_the_column_the_user_actually_dropped(
        qtbot, link, frame):
    """The end-to-end version of the table above, through a real drop event."""
    panel = GraphBuilderPanel(link=link)
    qtbot.addWidget(panel)
    panel.set_frame(frame)

    def drop(channel: str, column: str) -> None:
        payload = QMimeData()
        payload.setData(COLUMN_MIME, column.encode("utf-8"))
        panel.zone(channel).dropEvent(QDropEvent(
            QPointF(4, 4), Qt.CopyAction, payload, Qt.LeftButton,
            Qt.NoModifier))

    drop("x", "area")
    assert panel.spec.x == "area"
    assert panel.spec.resolved_kind(panel.canvas.kinds) == HISTOGRAM

    drop("y", "intensity")
    assert panel.spec.resolved_kind(panel.canvas.kinds) == SCATTER

    # Replace x with a categorical column: box plot, no code path in between.
    drop("x", "gene")
    assert panel.spec.resolved_kind(panel.canvas.kinds) == BOX


def test_a_drop_zone_only_takes_a_column_payload(qtbot):
    zone = DropZone("x")
    qtbot.addWidget(zone)
    text_only = QMimeData()
    text_only.setText("/home/someone/a-file.tif")
    zone.dropEvent(QDropEvent(QPointF(1, 1), Qt.CopyAction, text_only,
                              Qt.LeftButton, Qt.NoModifier))
    assert zone.column is None


def test_clearing_a_zone_re_renders_without_that_channel(canvas):
    canvas.set_spec(GraphSpec(x="area", y="intensity"))
    assert canvas.spec.resolved_kind(canvas.kinds) == SCATTER
    canvas.set_channel("y", None)
    assert canvas.spec.resolved_kind(canvas.kinds) == HISTOGRAM
    assert canvas.render_data is not None


# ---------------------------------------------------------------------------
# Faceting — the empty panels are the point
# ---------------------------------------------------------------------------

def test_two_way_faceting_draws_every_combination_including_the_empty_ones(
        frame):
    """p1 has no r3 and p2 has no r1. Both panels are still in the grid.

    Closing the gaps would tell the reader "there is no p1/r3" when the truth
    is "p1/r3 was never measured" — a different fact, and the one the grid is
    there to show.
    """
    spec = GraphSpec(x="area", y="intensity",
                     facet_row="plateID", facet_col="rowID")
    grid = facet_grid(frame, spec)

    assert grid.shape == (2, 3)
    assert grid.n_panels == 6                      # 2 x 3, not 4 non-empty
    empty = {(p.row_level, p.col_level) for p in grid.panels if p.is_empty}
    assert empty == {("p1", "r3"), ("p2", "r1")}
    assert sum(p.n for p in grid.panels) == len(frame)   # nothing lost
    # Levels are numeric-aware sorted and identical along every row.
    assert grid.row_levels == ("p1", "p2")
    assert grid.col_levels == ("r1", "r2", "r3")


def test_one_way_faceting_is_a_single_row_or_column(frame):
    spec = GraphSpec(x="area", facet_col="rowID")
    grid = facet_grid(frame, spec)
    assert grid.shape == (1, 3)
    assert grid.is_faceted
    unfaceted = facet_grid(frame, GraphSpec(x="area"))
    assert unfaceted.shape == (1, 1)
    assert not unfaceted.is_faceted
    assert unfaceted.panels[0].n == len(frame)


def test_a_missing_facet_value_gets_its_own_panel_rather_than_vanishing(frame):
    """A facet that silently drops a tenth of the table is the failure the
    empty-panel rule exists to prevent, and NaN is how it usually happens."""
    holed = frame.copy()
    holed.loc[holed.index[:10], "rowID"] = np.nan
    grid = facet_grid(holed, GraphSpec(x="area", facet_col="rowID"))
    assert "(missing)" in grid.col_levels
    assert sum(p.n for p in grid.panels) == len(holed)


def test_the_canvas_draws_an_axes_for_every_panel_empty_ones_included(canvas):
    canvas.set_spec(GraphSpec(x="area", y="intensity",
                              facet_row="plateID", facet_col="rowID"))
    axes = canvas.panel_axes()
    assert len(axes) == 6
    assert set(axes) == {(r, c) for r in range(2) for c in range(3)}
    assert canvas.axes_at(0, 2) is not None            # p1 / r3 — empty
    assert canvas.grid.panel(0, 2).is_empty


def test_a_facet_column_with_too_many_levels_is_capped_and_says_so():
    """The grid stops growing, and the notice says how much is not drawn.
    Quietly drawing twelve of forty-seven panels is the same lie as sampling
    without saying so."""
    n = 470
    wide = pd.DataFrame({
        "plateID": ["p"] * n, "rowID": ["r"] * n, "columnID": ["c"] * n,
        "fieldID": ["f"] * n, "object_label": range(n),
        "area": np.linspace(1, 100, n),
        "gene": [f"g{i:03d}" for i in range(n)],
    })
    grid = facet_grid(wide, GraphSpec(x="area", facet_col="gene"))
    assert grid.shape == (1, 12)
    assert grid.hidden_rows == n - 12
    assert "levels" in grid.notice and "row(s) outside" in grid.notice


# ---------------------------------------------------------------------------
# Shared axes
# ---------------------------------------------------------------------------

def test_shared_axes_really_are_shared_and_wide_enough_for_every_panel(canvas,
                                                                      frame):
    """Identical limits across panels, and limits that bound ALL the data.

    Matplotlib's ``sharex`` makes panels agree with each other — on whatever
    the first panel happened to autoscale to. Computing the limits over the
    whole frame is what makes them right, and p2's `area` is twenty times p1's
    precisely so that the difference between the two is not subtle.
    """
    canvas.set_spec(GraphSpec(x="area", y="intensity",
                              facet_row="plateID", facet_col="rowID"))
    axes = canvas.panel_axes()
    x_limits = {ax.get_xlim() for ax in axes.values()}
    y_limits = {ax.get_ylim() for ax in axes.values()}
    assert len(x_limits) == 1, "panels disagree about x"
    assert len(y_limits) == 1, "panels disagree about y"

    (low, high), = x_limits
    assert low <= frame["area"].min() and high >= frame["area"].max()
    (ylow, yhigh), = y_limits
    assert ylow <= frame["intensity"].min() and yhigh >= frame["intensity"].max()

    # ...and the empty panel carries the same limits, so it is readable as
    # "nothing here" rather than as a different chart.
    assert canvas.axes_at(0, 2).get_xlim() == (low, high)


def test_turning_sharing_off_lets_the_panels_disagree(canvas):
    """The negative control. Without it, a bug that pins every panel to the
    same limits by accident would pass the test above."""
    canvas.set_spec(GraphSpec(x="area", y="intensity", facet_row="plateID",
                              shared_x=False, shared_y=False))
    limits = {ax.get_xlim() for ax in canvas.panel_axes().values()}
    assert len(limits) == 2


def test_a_faceted_histogram_shares_its_count_axis_too(canvas):
    """Sharing the value axis of an aggregate is the same rule as sharing a
    data axis; forgetting it is the usual way a faceted histogram lies."""
    canvas.set_spec(GraphSpec(x="area", facet_col="rowID"))
    assert canvas.scales.count_limit is not None
    limits = {ax.get_ylim() for ax in canvas.panel_axes().values()}
    assert len(limits) == 1


def test_a_categorical_axis_keeps_the_same_level_order_in_every_panel(canvas):
    """Tick *positions* everywhere, tick *labels* on the bottom row.

    Shared axes draw the level names once, under the grid, which is what
    makes a column of panels read as one axis — but every panel must put
    ``g1`` at the same place or the columns do not line up.
    """
    canvas.set_spec(GraphSpec(x="gene", y="area", facet_row="plateID"))
    scales = canvas.scales
    assert scales.x_levels == ("g0", "g1", "g2")
    axes = canvas.panel_axes()
    positions = {tuple(ax.get_xticks()) for ax in axes.values()}
    assert positions == {(0, 1, 2)}
    bottom = axes[(max(r for r, _c in axes), 0)]
    assert [t.get_text() for t in bottom.get_xticklabels()] == \
        list(scales.x_levels)


def test_a_lone_y_column_is_still_scaled_on_the_axis_it_is_drawn_on(canvas,
                                                                    frame):
    """A histogram draws its column left-to-right whichever zone it came from.

    Scaling ``spec.y`` as if it were the vertical axis would leave the
    horizontal one unscaled, and a faceted "histogram of Y" would quietly
    stop sharing its bins — panels that look comparable and are not.
    """
    canvas.set_spec(GraphSpec(y="area", facet_col="rowID"))
    scales = canvas.scales
    assert canvas.spec.resolved_kind(canvas.kinds) == HISTOGRAM
    assert scales.x_limits is not None and scales.y_limits is None
    assert scales.x_edges is not None
    low, high = scales.x_limits
    assert low <= frame["area"].min() and high >= frame["area"].max()
    assert len({ax.get_xlim() for ax in canvas.panel_axes().values()}) == 1
    bottom = canvas.axes_at(0, 0)
    assert bottom.get_xlabel() == "area"
    assert bottom.get_ylabel() == "count"


def test_brushing_a_histogram_sweeps_bins_not_counts(canvas, frame):
    """The vertical axis of a histogram is a count, not a variable, so a drag
    that starts high and ends low must not exclude anything."""
    canvas.set_spec(GraphSpec(x="area"))
    published = canvas.brush(-5.0, 3.0, 200.0, 0.0)
    expected = int((frame["area"] <= 200.0).sum())
    assert len(published) == expected > 0
    assert expected < len(frame)          # p2's area runs past 200


def test_a_constant_column_still_gets_a_usable_axis():
    """One distinct value would be a zero-width axis, which matplotlib widens
    differently per panel — breaking sharing in the hardest case to notice."""
    n = 20
    flat = pd.DataFrame({
        "plateID": ["p1"] * 10 + ["p2"] * 10, "rowID": ["r"] * n,
        "columnID": ["c"] * n, "fieldID": ["f"] * n,
        "object_label": range(n),
        "area": [5.0] * n, "intensity": np.linspace(0, 1, n)})
    # A one-value numeric column is categorical by the shared rule, so the
    # continuous path is reached the way a user would reach it: by saying so.
    spec = GraphSpec(x="area", y="intensity", facet_row="plateID",
                     roles={"area": CONTINUOUS})
    kinds = spec.kinds_for(flat)
    scales = scales_for(flat, spec, kinds, facet_grid(flat, spec))
    low, high = scales.x_limits
    assert low < 5.0 < high


# ---------------------------------------------------------------------------
# Large data — stated, never silent
# ---------------------------------------------------------------------------

def big_frame(n: int = 120_000) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        "plateID": ["p1"] * n, "rowID": ["r1"] * n, "columnID": ["c1"] * n,
        "fieldID": ["f1"] * n, "object_label": range(n),
        "area": rng.normal(size=n), "intensity": rng.normal(size=n),
        "gene": [f"g{i % 3}" for i in range(n)],
    })


def test_an_aggregate_uses_every_row_however_big_the_table_is():
    """A histogram is already a reduction; sampling first would move the
    answer for no gain."""
    big = big_frame()
    spec = GraphSpec(x="area")
    data = prepare_data(big, spec, column_kinds(big))
    assert data.strategy == FULL
    assert data.n_shown == len(big) == data.n_total
    assert data.is_complete


def test_a_scatter_past_the_budget_bins_rather_than_dropping_rows():
    big = big_frame()
    spec = GraphSpec(x="area", y="intensity")
    data = prepare_data(big, spec, column_kinds(big))
    assert data.strategy == BINNED
    assert data.n_shown == data.n_total
    assert data.is_complete
    assert "every row is counted" in data.notice


def test_a_scatter_that_needs_per_point_marks_samples_and_says_so():
    """A categorical colour cannot be carried by a density raster, so this one
    really is a subset — and the notice carries the count, the fraction and
    the word."""
    big = big_frame()
    spec = GraphSpec(x="area", y="intensity", colour="gene")
    data = prepare_data(big, spec, column_kinds(big))
    assert data.strategy == SAMPLED
    assert data.n_shown == spec.point_budget < data.n_total
    assert not data.is_complete
    assert "random" in data.notice and "50,000 of 120,000" in data.notice


def test_the_sample_is_the_same_sample_every_time():
    """A screenshot in a report and the screen it came from must not differ by
    a random draw."""
    big = big_frame()
    spec = GraphSpec(x="area", y="intensity", size="area", seed=11)
    kinds = column_kinds(big)
    from dataclasses import replace
    first = prepare_data(big, spec, kinds).frame
    second = prepare_data(big, spec, kinds).frame
    assert first.index.equals(second.index)
    # ...and the seed is the only thing that moves it, so a different draw is
    # something the user asked for rather than something that happened.
    other = prepare_data(big, replace(spec, seed=12), kinds).frame
    assert not first.index.equals(other.index)


def test_the_canvas_puts_the_large_data_notice_on_screen(qtbot, link):
    """A subset that does not announce itself is a result nobody can check."""
    view = GraphCanvas(link=link, source="big")
    qtbot.addWidget(view)
    view.set_frame(big_frame())
    view.set_spec(GraphSpec(x="area", y="intensity", colour="gene"))
    assert "random 50,000 of 120,000" in view.notice()
    view.set_spec(GraphSpec(x="area", y="intensity"))
    assert "density" in view.notice()


# ---------------------------------------------------------------------------
# Linking: a selection highlights, a filter hides
# ---------------------------------------------------------------------------

def test_a_brush_here_reaches_a_second_linked_view(qtbot, link, frame):
    """Two canvases, one link: brushing in the first highlights in the second."""
    first = GraphCanvas(link=link, source="graph_a")
    second = GraphCanvas(link=link, source="graph_b")
    for view in (first, second):
        qtbot.addWidget(view)
        view.set_frame(frame)
        view.set_spec(GraphSpec(x="area", y="intensity"))

    assert second.selected_count() == 0
    published = first.brush(-1.0, -10.0, 30.0, 10.0)

    assert published is not None
    expected = int(((frame["area"] >= -1.0) & (frame["area"] <= 30.0)).sum())
    assert len(published) == expected > 0
    assert published.source == "graph_a"
    assert link.selection.keys.equals(published.keys)
    assert second.selected_count() == expected


def test_a_brush_names_rows_that_were_never_drawn(qtbot, link):
    """The brush is a predicate over the frame, not a hit test on the marks,
    so a sampled panel still selects everything inside the rectangle."""
    big = big_frame(60_000)
    view = GraphCanvas(link=link, source="big")
    qtbot.addWidget(view)
    view.set_frame(big)
    view.set_spec(GraphSpec(x="area", y="intensity", colour="gene"))
    assert view.render_data.strategy == SAMPLED

    published = view.brush(-10.0, -10.0, 10.0, 10.0)
    assert len(published) == len(big) > view.render_data.n_shown


def test_a_selection_highlights_and_a_filter_hides(qtbot, link, frame):
    """The whole semantic difference, in one test.

    A selection is the subset the user pointed at *inside* the population; a
    filter is the population. Conflating them is what makes a lasso
    destructive.
    """
    view = GraphCanvas(link=link, source="graph")
    qtbot.addWidget(view)
    view.set_frame(frame)
    view.set_spec(GraphSpec(x="area", y="intensity"))
    assert view.render_data.n_shown == len(frame)

    # A selection: same number of rows on screen, some of them ringed.
    p1_keys = object_keys(frame[frame["plateID"] == "p1"])
    link.set_selection(Selection(keys=p1_keys, source="somewhere_else"))
    assert view.render_data.n_shown == len(frame), "a selection hid rows"
    assert view.selected_count() == 40
    assert "40 highlighted" in view.notice()

    # A filter: rows genuinely leave, and the axes re-scale to what is left.
    link.set_filter(DataFilter().add(CategoryFilter("plateID", ("p1",))))
    qtbot.waitUntil(lambda: view.render_data.n_shown == 40, timeout=2000)
    assert view.render_data.n_shown == 40
    (low, high) = view.axes_at(0, 0).get_xlim()
    assert high < frame["area"].max(), "the axes did not follow the filter"


def test_a_filter_that_keeps_nothing_draws_an_empty_grid_rather_than_crashing(
        qtbot, link, frame):
    """"Your filter matches no cells" is an answer. A traceback is not."""
    view = GraphCanvas(link=link, source="graph")
    qtbot.addWidget(view)
    view.set_frame(frame)
    view.set_spec(GraphSpec(x="area", y="intensity", facet_col="rowID"))
    link.set_filter(DataFilter().add(CategoryFilter("plateID", ())))
    view.render_now()
    assert view.render_data.n_shown == 0
    assert len(view.panel_axes()) == 1
    assert all(p.is_empty for p in view.grid.panels)
    assert "no rows" in view.notice()


def test_linked_visible_applies_the_filter_and_not_the_selection(link, frame):
    """The mixin's own contract, asserted where this screen depends on it."""
    view = GraphCanvas(link=link, source="graph")
    try:
        link.set_selection(Selection(
            keys=object_keys(frame.iloc[:5]), source="elsewhere"))
        assert len(view.linked_visible(frame)) == len(frame)
        link.set_filter(DataFilter().add(CategoryFilter("plateID", ("p2",))))
        assert len(view.linked_visible(frame)) == 40
    finally:
        view.unlink_selection()
        view.deleteLater()


def test_a_view_does_not_repaint_for_the_echo_of_its_own_brush(qtbot, link,
                                                              frame):
    view = GraphCanvas(link=link, source="graph")
    qtbot.addWidget(view)
    view.set_frame(frame)
    view.set_spec(GraphSpec(x="area", y="intensity"))
    seen = []
    view.on_linked_selection_changed = lambda sel: seen.append(sel)
    view.brush(-1.0, -10.0, 30.0, 10.0)
    assert seen == []


def test_a_table_without_object_keys_still_plots_and_says_it_cannot_link(
        qtbot, link):
    """Degrading to "no brushing" beats refusing to draw a CSV someone
    exported without the key columns."""
    plain = pd.DataFrame({"area": np.linspace(0, 1, 30),
                          "intensity": np.linspace(1, 0, 30)})
    view = GraphCanvas(link=link, source="plain")
    qtbot.addWidget(view)
    view.set_frame(plain)
    view.set_spec(GraphSpec(x="area", y="intensity"))
    assert view.render_data.n_shown == 30
    assert "no object keys" in view.notice()
    assert view.brush(0.0, 0.0, 1.0, 1.0) is None


def test_a_filter_naming_a_column_this_table_lacks_is_reported_not_swallowed(
        qtbot, link):
    plain = pd.DataFrame({"area": np.linspace(0, 1, 30),
                          "intensity": np.linspace(1, 0, 30)})
    view = GraphCanvas(link=link, source="plain")
    qtbot.addWidget(view)
    view.set_frame(plain)
    view.set_spec(GraphSpec(x="area", y="intensity"))
    link.set_filter(DataFilter().add(CategoryFilter("plateID", ("p1",))))
    view.render_now()
    assert view.render_data.n_shown == 30
    assert "does not apply here" in view.notice()


# ---------------------------------------------------------------------------
# The canvas draws what the spec says
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec", [
    GraphSpec(x="area", y="intensity"),
    GraphSpec(x="area"),
    GraphSpec(x="gene"),
    GraphSpec(x="gene", y="area"),
    GraphSpec(x="gene", y="area", kind=VIOLIN),
    GraphSpec(x="gene", y="plateID"),
    GraphSpec(x="area", y="intensity", kind="line"),
    GraphSpec(x="area", y="intensity", colour="gene", size="cell_count"),
    GraphSpec(x="area", y="intensity", colour="intensity"),
    GraphSpec(x="area", y="intensity", facet_row="plateID", facet_col="rowID"),
])
def test_every_plot_type_renders_without_raising(canvas, spec):
    canvas.set_spec(spec)
    assert canvas.render_data is not None
    assert canvas.panel_axes()


def test_an_empty_spec_says_what_to_do_instead_of_drawing_nothing(canvas):
    canvas.set_spec(GraphSpec())
    assert canvas.render_data is None
    assert canvas.spec.is_empty


def test_a_column_the_new_table_lacks_is_dropped_from_the_spec(canvas):
    """Half-resolving would draw a chart of fewer variables than the zones
    claim."""
    canvas.set_spec(GraphSpec(x="area", y="intensity", colour="gene"))
    canvas.set_frame(pd.DataFrame({"area": [1.0, 2.0], "intensity": [1.0, 2.0]}))
    assert canvas.spec.colour is None
    assert canvas.spec.x == "area"


def test_the_series_order_is_fixed_so_a_filter_does_not_repaint_the_survivors(
        canvas, frame):
    from spacr.qt.widgets.graph_builder import categorical_colours
    order = categorical_colours()
    canvas.set_spec(GraphSpec(x="area", y="intensity", colour="gene"))
    assert canvas.scales.colour_levels == ("g0", "g1", "g2")
    assert canvas._series_colour(0) == order[0]
    # The ninth level folds into one grey rather than inventing a hue.
    from spacr.qt.widgets.graph_builder import OTHER_COLOUR
    assert canvas._series_colour(len(order)) == OTHER_COLOUR


# ---------------------------------------------------------------------------
# The registration seam
# ---------------------------------------------------------------------------

@pytest.fixture
def registry_sandbox():
    """Restore the whole app registry after the test.

    A leaked row is a leaked tile, a leaked sidebar button and a leaked
    Ctrl+N binding for every test that runs afterwards, so this restores
    the list object in place rather than trusting `unregister_app`.
    """
    from spacr.qt import app as app_mod
    apps = list(app_mod.APPS)
    factories = dict(app_mod.APP_FACTORIES)
    stages = dict(app_mod.APP_STAGE)
    yield app_mod
    app_mod.APPS[:] = apps
    app_mod.APP_FACTORIES.clear()
    app_mod.APP_FACTORIES.update(factories)
    app_mod.APP_STAGE.clear()
    app_mod.APP_STAGE.update(stages)
    app_mod._refresh_sections()


def test_registering_the_screen_reaches_every_reader_of_the_registry(
        registry_sandbox):
    """No row in `app.py`'s table, no branch in `_build_screen`.

    `register()` is deliberately not called at import — see its docstring —
    so this drives it and asserts the result, which is the same thing the
    one line that wires it in will do.
    """
    from spacr.qt.screens import graph_builder as screen_mod
    app_mod = registry_sandbox

    assert screen_mod.register() is True
    row = next((r for r in app_mod.APPS if r[0] == screen_mod.APP_KEY), None)
    assert row is not None
    assert row[1] == screen_mod.APP_NAME
    assert row[3] == app_mod.SECTION_EXPLORE
    assert app_mod.SECTION_EXPLORE in app_mod.SECTIONS
    assert row in app_mod.section_members(app_mod.SECTION_EXPLORE)
    assert app_mod.registered_factory(screen_mod.APP_KEY) is not None
    assert app_mod.app_stage(screen_mod.APP_KEY) == app_mod.STAGE_ALPHA
    # Registering twice must not raise — the module may be imported twice.
    assert screen_mod.register() is False


def test_the_registry_row_carries_everything_its_side_tables_need(
        registry_sandbox):
    """The four strings the shipped suite requires of every registered app.

    They live beside the screen rather than being invented in four other
    files, so wiring the app in is copying, not writing.
    """
    from spacr.qt.screens import graph_builder as screen_mod
    for text in (screen_mod.APP_NAME, screen_mod.APP_DESCRIPTION,
                 screen_mod.APP_INTRO, screen_mod.APP_CLI_NOTE):
        assert text.strip()
    assert screen_mod.APP_KEY == "graph_builder"


def test_the_registered_factory_is_what_the_window_would_build(qtbot,
                                                               registry_sandbox):
    """`MainWindow._build_screen` calls whatever `registered_factory` returns."""
    from spacr.qt.screens import graph_builder as screen_mod
    app_mod = registry_sandbox
    screen_mod.register()
    factory = app_mod.registered_factory(screen_mod.APP_KEY)
    screen = factory(app_key=screen_mod.APP_KEY)
    qtbot.addWidget(screen)
    assert isinstance(screen, screen_mod.GraphBuilderScreen)


def test_the_registered_factory_builds_the_screen(qtbot):
    from spacr.qt.screens.graph_builder import make_graph_builder_screen
    screen = make_graph_builder_screen()
    qtbot.addWidget(screen)
    assert screen.builder is not None
    assert screen.filters is not None


def test_the_widget_registers_its_own_qss_block():
    from spacr.qt.theme import stylesheet, widget_qss_names
    assert "GraphBuilder" in widget_qss_names()
    assert "GraphDropZone" in stylesheet()


def test_the_screen_plots_a_frame_and_shares_the_filter_panels_link(qtbot,
                                                                    frame):
    from spacr.qt.screens.graph_builder import GraphBuilderScreen
    private = LinkedSelection()
    screen = GraphBuilderScreen(link=private)
    qtbot.addWidget(screen)
    screen.set_frame(frame)
    screen.builder.set_spec(GraphSpec(x="area", y="intensity"))
    assert screen.builder.canvas.render_data.n_shown == len(frame)

    screen.filters.add_column("plateID")
    screen.filters.flush()
    assert not private.filter.is_empty
    qtbot.waitUntil(
        lambda: screen.builder.canvas.render_data.n_shown == len(frame),
        timeout=2000)


def test_the_screen_reads_a_csv_and_a_sqlite_table(qtbot, tmp_path, frame):
    from spacr.qt.screens.graph_builder import (
        GraphBuilderScreen, read_table, table_names)
    import sqlite3

    csv = tmp_path / "measurements.csv"
    frame.to_csv(csv, index=False)
    assert len(read_table(str(csv))) == len(frame)

    db = tmp_path / "measurements.db"
    with sqlite3.connect(db) as conn:
        frame.to_sql("cell", conn, index=False)
        frame.head(5).to_sql("nucleus", conn, index=False)
    assert table_names(str(db))[0] == "cell"      # preferred order
    assert len(read_table(str(db), "nucleus")) == 5

    screen = GraphBuilderScreen(link=LinkedSelection())
    qtbot.addWidget(screen)
    screen.load_path(str(db))
    assert screen.builder.well.columns()
    assert "cell" in screen._source.text()
