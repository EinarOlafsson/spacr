"""Trellis: what the grid admits it is not showing, and what the canvas does with it.

Panels side by side are read as comparable, so every place the grid drops
something -- levels beyond the cap, rows outside them, a sample instead of the
whole table, a panel with no usable values -- has to reach the notice or the
count axis.  The canvas then draws the grid (or a message instead of one),
writes per-panel limits, brushes one panel into the returned keys, and follows
another view's selection by re-styling marks or redrawing bars.

One general test per behaviour family, each walking a case table with pytest's
``subtests`` fixture; every frame is small enough to count by hand.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from spacr.qt.linked_selection import LinkedSelection  # noqa: E402
from spacr.qt.widgets.graph_spec import (  # noqa: E402
    BAR, CATEGORICAL, CONTINUOUS, HISTOGRAM, SCATTER, GraphSpec,
)
from spacr.qt.widgets.trellis_spec import (  # noqa: E402
    SCALE_FREE, TrellisSpec, trellis,
)
from spacr.qt.widgets.trellis_view import TrellisCanvas  # noqa: E402
from spacr.selection import Selection  # noqa: E402

pytestmark = pytest.mark.qt

NUMERIC = {"area": CONTINUOUS, "intensity": CONTINUOUS}

#: Six keyed objects; the (p2, b) panel of the 2 x 2 grid is empty.
_LAYOUT = (("p1", "a"), ("p1", "a"), ("p1", "b"),
           ("p2", "a"), ("p2", "a"), ("p2", "a"))

_LOAD = ("Load a table, then drag a column onto X or Y and a grouping column "
         "onto Facet ↓ or Facet →.")


def _keyed():
    return pd.DataFrame([
        {"plateID": plate, "rowID": "r1", "columnID": "c1", "fieldID": "f1",
         "object_label": index + 1, "gene": gene,
         "area": 10.0 * index + 10.0, "intensity": 100.0 - 5.0 * index}
        for index, (plate, gene) in enumerate(_LAYOUT)])


def _scatter_spec(**kwargs):
    return TrellisSpec(graph=GraphSpec(x="area", y="intensity",
                                       facet_row="plateID", facet_col="gene",
                                       roles=NUMERIC), **kwargs)


def _histogram_spec(**kwargs):
    return TrellisSpec(graph=GraphSpec(x="area", facet_row="plateID",
                                       facet_col="gene", bins=2,
                                       roles=NUMERIC), **kwargs)


def _gene_histogram(**kwargs):
    return TrellisSpec(graph=GraphSpec(x="area", facet_col="gene",
                                       kind=HISTOGRAM, bins=2,
                                       roles={"area": CONTINUOUS}), **kwargs)


def _hand():
    return pd.DataFrame({"plateID": ["p1", "p1", "p1", "p2", "p2", "p2"],
                         "gene": ["a", "a", "b", "a", "a", "a"],
                         "area": [10.0, 20.0, 100.0, 30.0, 40.0, 50.0]})


def _rows_only():
    return TrellisSpec(graph=GraphSpec(x="area", facet_row="plateID", bins=2,
                                       roles={"area": CONTINUOUS}))


def _row_histogram():
    return TrellisSpec(graph=GraphSpec(x="area", facet_row="plateID", bins=2,
                                       kind=HISTOGRAM,
                                       roles={"area": CONTINUOUS}))


def _make(qtbot, frame, link=None):
    canvas = TrellisCanvas(link=link or LinkedSelection())
    qtbot.addWidget(canvas)
    if frame is not None:
        canvas.set_frame(frame)
    return canvas


def test_trellis_reports_dropped_levels_subsets_and_count_ceilings(subtests):
    """Capped levels, a sampled frame, empty panels and missing values each reach the notice or the count axis exactly."""
    with subtests.test(msg="levels_beyond_the_cap"):
        frame = pd.DataFrame({"plateID": [f"p{i % 20:02d}" for i in range(60)],
                              "area": np.arange(60, dtype=float)})
        result = trellis(frame, TrellisSpec(graph=GraphSpec(
            x="area", facet_col="plateID", bins=2, roles={"area": CONTINUOUS})))
        assert result.grid.notice == ("plateID has 20 levels; the first 12 are "
                                      "drawn; 24 row(s) outside the drawn levels")
        assert result.notice == result.grid.notice
    with subtests.test(msg="subset_notice_carried_onto_the_grid"):
        frame = pd.DataFrame({"x": np.arange(20.0), "y": np.arange(20.0),
                              "g": ["a", "b"] * 10})
        result = trellis(frame, TrellisSpec(graph=GraphSpec(
            x="x", y="y", facet_col="g", kind=SCATTER, point_budget=4,
            roles={"x": CONTINUOUS, "y": CONTINUOUS})))
        assert result.data.strategy == "binned"
        assert result.data.notice == ("20 rows drawn as a 30×30 density — "
                                      "every row is counted")
        assert result.notice == result.data.notice
    with subtests.test(msg="all_missing_panel_adds_no_height"):
        frame = pd.DataFrame({"plateID": ["p1", "p1", "p1", "p2", "p2"],
                              "area": [1.0, 2.0, 3.0, np.nan, np.nan]})
        result = trellis(frame, _row_histogram())
        assert result.n_at(1, 0) == 2
        assert result.scales_at(0, 0).count_limit == pytest.approx(2 * 1.08)
        assert result.scales_at(1, 0).count_limit == pytest.approx(2 * 1.08)
    with subtests.test(msg="nothing_to_bin_sets_no_ceiling"):
        frame = pd.DataFrame({"plateID": ["p1", "p1", "p2", "p2"],
                              "area": [np.nan] * 4})
        result = trellis(frame, _row_histogram())
        assert result.scales_at(0, 0).x_edges is None
        assert result.scales_at(0, 0).count_limit is None
        assert result.scales_at(1, 0).count_limit is None
    with subtests.test(msg="every_panel_empty_says_nothing_about_n"):
        hand = _hand()
        empty = trellis(hand.iloc[0:0], _rows_only(), levels_source=hand)
        assert empty.n_range() is None
        assert empty.summary() == "2 × 1 = 2 panels · 2 with no rows"
        assert "n = 3 each" in trellis(hand, _rows_only()).summary()
    with subtests.test(msg="row_only_brush_keeps_the_other_row_out"):
        hand = _hand()
        result = trellis(hand, _rows_only())
        assert result.shape == (2, 1)
        mask = result.brush(5.0, -1e9, 45.0, 1e9, row=1, col=0)
        assert mask.tolist() == [False, False, False, True, True, False]
    with subtests.test(msg="missing_category_is_a_bar"):
        frame = pd.DataFrame({"plateID": ["p1", "p1", "p2", "p2"],
                              "gene": ["a", "b", None, None]})
        result = trellis(frame, TrellisSpec(graph=GraphSpec(
            x="gene", facet_row="plateID", kind=BAR,
            roles={"gene": CATEGORICAL})))
        assert [panel.n for panel in result.panels] == [2, 2]
        assert result.panels[1].scales.count_limit == pytest.approx(2 * 1.08)


def test_trellis_canvas_draws_messages_limits_and_count_axes(qtbot, subtests):
    """Empty tables show the load message; y and count limits are exact per scale mode; edits are debounced."""
    for case_id, frame in (("no_table", None), ("empty_table", _keyed().iloc[0:0])):
        with subtests.test(msg=case_id):
            canvas = _make(qtbot, frame)
            canvas.set_trellis_spec(_histogram_spec())
            assert canvas.trellis is None
            assert canvas.panel_axes() == {}
            assert [t.get_text() for t in canvas.figure().axes[0].texts] == [_LOAD]
    with subtests.test(msg="deferred_change_waits_for_debounce"):
        canvas = _make(qtbot, _keyed())
        canvas.set_trellis_spec(_histogram_spec())
        drawn = canvas.trellis
        canvas.set_trellis_spec(_histogram_spec(wrap=1), immediate=False)
        assert canvas._debounce.isActive() is True
        assert canvas.trellis is drawn
        canvas.render_now()
        assert canvas._debounce.isActive() is False
        assert canvas.trellis is not drawn
    for case_id, spec, distinct in (("free_scatter_y", _scatter_spec(scale_y=SCALE_FREE), 4),
                                    ("shared_scatter_y", _scatter_spec(), 1)):
        with subtests.test(msg=case_id):
            canvas = _make(qtbot, _keyed())
            canvas.set_trellis_spec(spec)
            limits = {ax.get_ylim() for ax in canvas.panel_axes().values()}
            assert len(limits) == distinct
    with subtests.test(msg="categorical_y_ticks_at_levels"):
        canvas = _make(qtbot, _keyed())
        canvas.set_trellis_spec(TrellisSpec(graph=GraphSpec(
            x="area", y="gene", facet_row="plateID", roles=NUMERIC)))
        ax = canvas.panel_axes()[(0, 0)]
        assert [t.get_text() for t in ax.get_yticklabels()] == ["a", "b"]
        assert ax.get_ylim() == pytest.approx((-0.5, 1.5))
    seen = pd.DataFrame({"gene": ["a", "a", "a", "b"],
                         "area": [1.0, 2.0, 3.0, 10.0]})
    blind = pd.DataFrame({"gene": ["a", "a", "b", "b"], "area": [np.nan] * 4})
    for scale, real_limits in ((None, [(0.0, 3.24), (0.0, 3.24)]),
                               (SCALE_FREE, [(0.0, 3.24), (0.0, 1.08)])):
        mode = scale or "shared"
        with subtests.test(msg=f"{mode}_count_axis_of_nothing_left_alone"):
            canvas = _make(qtbot, blind)
            kwargs = {"scale_y": scale} if scale else {}
            canvas.set_trellis_spec(_gene_histogram(**kwargs))
            assert [p.scales.count_limit for p in canvas.trellis.panels] == [None, None]
            empty_axes = canvas.panel_axes()[(0, 0)]
            assert [bar.get_height() for bar in empty_axes.patches] == [0, 0]
            low, high = empty_axes.get_ylim()
            assert low < 0 < high
        with subtests.test(msg=f"{mode}_count_axis_of_real_bars"):
            canvas = _make(qtbot, seen)
            canvas.set_trellis_spec(_gene_histogram(**kwargs))
            drawn = canvas.panel_axes()
            assert [bar.get_height() for bar in drawn[(0, 0)].patches] == [3, 0]
            assert [bar.get_height() for bar in drawn[(0, 1)].patches] == [0, 1]
            assert drawn[(0, 0)].get_ylim() == pytest.approx(real_limits[0])
            assert drawn[(0, 1)].get_ylim() == pytest.approx(real_limits[1])


def test_trellis_canvas_brushes_one_panel_and_follows_other_selections(qtbot, subtests):
    """Brushes return exact keys or None; foreign selections restyle scatters, redraw histograms and are counted."""
    keyed = _keyed()
    with subtests.test(msg="brush_publishes_by_default"):
        link = LinkedSelection()
        canvas = _make(qtbot, keyed, link)
        canvas.set_trellis_spec(_scatter_spec())
        fired = []
        link.selection_changed.connect(lambda: fired.append(link.selection))
        published = canvas.brush(-1e9, -1e9, 1e9, 1e9, row=1, col=0)
        assert published is link.selection
        assert fired == [published]
        assert set(published.keys) == set(Selection.from_frame(
            keyed.iloc[3:6], source=canvas.link_source).keys)
    with subtests.test(msg="panel_not_in_grid"):
        canvas = _make(qtbot, keyed)
        canvas.set_trellis_spec(_scatter_spec())
        assert canvas.brush(0, 0, 1e9, 1e9, row=9, col=9, publish=False) is None
    with subtests.test(msg="nothing_drawn_yet"):
        canvas = _make(qtbot, keyed)
        canvas.set_trellis_spec(TrellisSpec())
        assert canvas.trellis is None
        assert canvas.brush(0, 0, 1, 1, publish=False) is None
    with subtests.test(msg="no_object_keys"):
        frame = pd.DataFrame({"gene": ["a", "a", "b", "b"],
                              "area": [1.0, 2.0, 3.0, 4.0],
                              "intensity": [4.0, 3.0, 2.0, 1.0]})
        canvas = _make(qtbot, frame)
        canvas.set_trellis_spec(TrellisSpec(graph=GraphSpec(
            x="area", y="intensity", facet_col="gene", roles=NUMERIC)))
        assert canvas.brush(-1e9, -1e9, 1e9, 1e9, publish=False) is None
        assert ("no object keys in this table — brushing cannot publish a "
                "selection") in canvas.notice().split(" · ")
    with subtests.test(msg="notice_counts_highlighted_rows"):
        link = LinkedSelection()
        canvas = _make(qtbot, keyed, link)
        canvas.set_trellis_spec(_scatter_spec())
        link.set_selection(Selection.from_frame(keyed.iloc[:3], source="test"))
        canvas.render_now()
        assert canvas.notice().split(" · ")[-1] == "3 highlighted"
    with subtests.test(msg="scatter_restyles_without_redraw"):
        link = LinkedSelection()
        canvas = _make(qtbot, keyed, link)
        canvas.set_trellis_spec(_scatter_spec())
        before = canvas.panel_axes()
        assert canvas.trellis.n_empty == 1
        link.set_selection(Selection.from_frame(keyed.iloc[2:4], source="other"))
        assert canvas.panel_axes() == before
        assert canvas.selected_count() == 2
        assert canvas.notice().split(" · ")[-1] == "2 highlighted"
    with subtests.test(msg="histogram_redraws"):
        link = LinkedSelection()
        canvas = _make(qtbot, keyed, link)
        canvas.set_trellis_spec(_histogram_spec())
        before = dict(canvas.panel_axes())
        link.set_selection(Selection.from_frame(keyed.iloc[:2], source="other"))
        after = canvas.panel_axes()
        assert set(after) == set(before)
        assert all(after[key] is not before[key] for key in after)
        assert canvas.selected_count() == 2
