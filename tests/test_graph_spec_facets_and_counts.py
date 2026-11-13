"""Table-driven tests of facet grids, scales and the shared count axis in
``spacr.qt.widgets.graph_spec``.

The spec turns dropped columns into a chart.  These are the places where there
is less to draw than the caller assumed -- a grid larger than the page, a
counted column that is not in the table, panels with no rows, a histogram with
no bins -- and each has to answer with an exact value a renderer can draw.
Pure Python: no Qt widget is built here.
"""
from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets.graph_spec import (BAR, HISTOGRAM, GraphSpec,
                                         _count_limit, column_kinds,
                                         facet_grid, scales_for)


def _frame(n=24):
    generator = np.random.default_rng(0)
    return pd.DataFrame({
        "area": generator.normal(100.0, 20.0, n),
        "intensity": generator.normal(500.0, 60.0, n),
        "plateID": [f"p{i % 4 + 1}" for i in range(n)],
        "wellID": [f"w{i % 6 + 1}" for i in range(n)],
    })


class _Panel:
    """A grid cell whose own filter selects ``rows``."""

    def __init__(self, rows, is_empty=False):
        self._rows = rows
        self.is_empty = is_empty

    def frame(self, _frame):
        return self._rows


def _stub_grid(*panels):
    return types.SimpleNamespace(panels=list(panels))


def test_a_grid_larger_than_the_page_loses_columns_before_rows():
    """A capped grid keeps more row levels than column levels and says so."""
    frame = _frame(n=48)
    grid = facet_grid(frame, GraphSpec(x="area", facet_row="plateID",
                                       facet_col="wellID"), max_panels=8)
    assert grid.row_levels == ("p1", "p2", "p3")
    assert grid.col_levels == ("w1", "w2")
    assert len(grid.panels) == 6
    assert grid.hidden_rows == 36
    assert grid.notice == ("grid capped at 8 panels; 36 row(s) outside the "
                           "drawn levels")


def test_scales_place_levels_once_for_every_panel(subtests):
    """Categorical axes share one level->tick map; continuous ones have none."""
    frame = _frame()
    cases = [
        ("categorical_x", GraphSpec(x="plateID"),
         {"p1": 0, "p2": 1, "p3": 2, "p4": 3}, None),
        ("continuous_xy", GraphSpec(x="area", y="intensity"), None, None),
    ]
    for case_id, spec, want_x, want_y in cases:
        with subtests.test(msg=case_id):
            scales = scales_for(frame, spec, column_kinds(frame))
            assert scales.x_positions() == want_x
            assert scales.y_positions() == want_y
            assert scales.count_limit is None
    with subtests.test(msg="shared_count_axis_is_the_tallest_bar_anywhere"):
        frame48 = _frame(n=48)
        spec = GraphSpec(x="plateID", facet_row="wellID")
        scales = scales_for(frame48, spec, column_kinds(frame48),
                            grid=facet_grid(frame48, spec))
        assert scales.count_limit == pytest.approx(4 * 1.08)


def test_the_shared_count_limit_is_the_tallest_bar_with_headroom(subtests):
    """_count_limit = 1.08 x the tallest bar over non-empty panels, else None."""
    frame = _frame()
    facet_cases = [
        ("counted_column_absent", frame, GraphSpec(x="not_a_column"), BAR, None),
        ("histogram_without_bins", frame, GraphSpec(x="area"), HISTOGRAM, None),
        ("empty_facet_panels_skipped", frame,
         GraphSpec(x="plateID", facet_row="wellID", facet_col="plateID"), BAR,
         2 * 1.08),
        ("all_missing_column_is_one_bar",
         pd.DataFrame({"plate": ["p1"] * 3 + ["p2"] * 5,
                       "condition": ["a", "a", "b"] + [None] * 5}),
         GraphSpec(kind=BAR, x="condition", facet_col="plate"), BAR, 5 * 1.08),
        ("half_the_grid_empty",
         pd.DataFrame({"plate": ["p1", "p1", "p2"], "row": ["A", "A", "B"],
                       "condition": ["a", "a", "b"]}),
         GraphSpec(kind=BAR, x="condition", facet_col="plate", facet_row="row"),
         BAR, 2 * 1.08),
    ]
    for case_id, data, spec, kind, want in facet_cases:
        with subtests.test(msg=case_id):
            grid = facet_grid(data, spec)
            limit = _count_limit(data, spec, grid, kind, None, None)
            if want is None:
                assert limit is None
            else:
                assert limit == pytest.approx(want)
    with subtests.test(msg="the_fixtures_have_empty_panels"):
        spec = GraphSpec(x="plateID", facet_row="wellID", facet_col="plateID")
        grid = facet_grid(frame, spec)
        assert (sum(p.is_empty for p in grid.panels), len(grid.panels)) == (12, 24)

    spec = types.SimpleNamespace(x="well", y=None)
    nothing = pd.DataFrame({"well": pd.Series([], dtype=object)})
    stub_cases = [
        ("tallest_across_panels",
         [_Panel(pd.DataFrame({"well": ["a", "b"]})),
          _Panel(pd.DataFrame({"well": ["a"] * 5 + ["b"]}))], 5 * 1.08),
        ("panel_filtered_to_nothing_contributes_nothing",
         [_Panel(nothing), _Panel(pd.DataFrame({"well": ["a", "a"]}))], 2 * 1.08),
        ("every_panel_empty_means_no_limit", [_Panel(nothing)], None),
        ("declared_empty_panel_is_skipped",
         [_Panel(pd.DataFrame({"well": ["a"] * 9}), is_empty=True),
          _Panel(pd.DataFrame({"well": ["a"]}))], 1 * 1.08),
    ]
    for case_id, panels, want in stub_cases:
        with subtests.test(msg=case_id):
            limit = _count_limit(pd.DataFrame({"well": ["a", "a", "b"]}), spec,
                                 _stub_grid(*panels), BAR, None, None)
            if want is None:
                assert limit is None
            else:
                assert limit == pytest.approx(want)
