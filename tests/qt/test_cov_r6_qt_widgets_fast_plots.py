"""The guarantees that make six of the fast plots' guards dead code.

Round 6 was pointed at 227 uncovered items in this module. All but nine of
them were already closed by the four fast-plot test files beside this one;
the nine that remained are six branches that NO caller can take, and the
lines behind them. They are not silenced here -- there is no pragma in this
repository -- because a branch that is unreachable today is unreachable only
for a reason, and the reason is always somebody else's invariant:

* ``_violin_profile``'s ``peak <= 0`` sits after a guard that has already
  established ``max > min`` over finite values, and the histogram is taken
  over exactly that range, so every value it was given falls in a bin.
* ``_rows_of``'s object-dtype test is a conversion, not a choice:
  pyqtgraph stores per-point ``data`` in an object column, always, and the
  two callers only ever hand it a live ``ScatterPlotItem``.
* ``_q_ramp``'s empty-``finite`` arm cannot run, because ``set_results``
  only reaches for the ramp once ``np.isfinite(q).any()`` is true.
* ``_offer_graph_kinds`` classifies the same frame twice -- once inside
  ``offer``, once through ``shape_of`` -- so the second call cannot raise
  or come back blank when the first one did not.
* ``VolcanoPlot._detail``'s local-FDR test is guarded by the q above it: a
  finite q means a finite raw p, and a finite raw p means a finite local FDR.
* ``EffectRankPlot.set_results`` returns on an empty table long before it
  writes the names onto the axis, so ``shown`` is never zero there.

So each test below pins the INVARIANT rather than the guard: if one of these
ever stops holding, the branch stops being dead and this file is where that
is noticed. Every one drives the case the invariant forbids AND the case it
allows, in the same test, because an assertion that something is absent is
worth nothing on a widget that draws nothing at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

import pyqtgraph as pg                                       # noqa: E402
from PySide6.QtGui import QColor                             # noqa: E402
from PySide6.QtWidgets import QCheckBox                      # noqa: E402

from spacr.qt.widgets import fast_plots as fp                # noqa: E402

pytestmark = pytest.mark.qt


# ------------------------------------------------------------------ fixtures

@pytest.fixture
def plot(qtbot):
    """A live FastPlot with both axes named, nothing drawn on it yet."""
    widget = fp.FastPlot(title="fast", x_label="ex", y_label="why")
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def volcano(qtbot):
    widget = fp.VolcanoPlot()
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def ranked(qtbot):
    widget = fp.EffectRankPlot()
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def spec_plot(qtbot):
    """A grouped plot that remembers the spec it was drawn from."""
    from spacr.qt.widgets.grouped_plot import GroupedPlot, PlotSpec

    frame = pd.DataFrame({"group": ["a", "a", "b", "b"],
                          "value": [1.0, 2.0, 3.0, 4.0]})
    widget = GroupedPlot(PlotSpec(frame=frame, value="value", group="group",
                                  kind="box"))
    qtbot.addWidget(widget)
    return widget


def _coefficients(n=12, *, p=None):
    """A coefficient table the shape ``perform_regression`` writes."""
    return pd.DataFrame({
        "feature": [f"fraction:grna[g{i}]" for i in range(n)],
        "grna": [f"g{i}" for i in range(n)],
        "coefficient": np.linspace(-2.0, 2.0, n),
        "p_value": (np.linspace(1e-6, 0.9, n) if p is None
                    else np.asarray(p, dtype=float)),
    })


def _scatters(widget) -> list:
    return [item for item in widget.plot.plotItem.items
            if isinstance(item, pg.ScatterPlotItem)]


def _brush_names(item) -> list:
    return [brush.color().name() for brush in item.data["brush"]]


def _legend_labels(widget) -> list:
    legend = getattr(widget.plot.plotItem, "legend", None)
    return [] if legend is None else [label.text for _, label in legend.items]


def _legend_box(widget):
    return [box for box in widget.findChildren(QCheckBox)
            if box.text().startswith("legend")][0]


def _show_as(menu):
    for action in menu.actions():
        if action.menu() is not None and action.text() == "Show as":
            return action.menu()
    return None


def _remember_entry(menu):
    show_as = _show_as(menu)
    if show_as is None:
        return None
    for action in show_as.actions():
        if action.text().startswith("Always start with"):
            return action
    return None


# ------------------------------------------------------- the violin's bins

def test_a_violin_bins_the_range_it_measured_so_a_bar_is_always_filled():
    """The outline's peak is the half-width it was asked for. Every time.

    The profile is taken over ``range=(min, max)`` of the SAME values it is
    counting, so every one of them lands in a bin and the tallest bin cannot
    be empty -- which is why ``peak <= 0`` below the histogram has no input
    that reaches it. The one case that does refuse is a group with no spread
    at all, and that is caught above the histogram, not below it.

    Driven on the helper rather than through ``add_group_mark`` because the
    caller filters the values down to the finite ones first, so a plot cannot
    hand this function anything the guard above has not already seen.
    """
    for values in ([0.0, 1.0],
                   list(np.linspace(-3.0, 3.0, 200)),
                   [1e-9, 2e-9],
                   [0.0, 0.0, 5.0]):
        centres, density = fp._violin_profile(np.asarray(values, float), 0.4)

        assert centres is not None, f"{values!r} was refused a profile"
        assert float(np.max(density)) == pytest.approx(0.4), (
            "the widest bin is not the half-width the caller asked for")
        assert density[0] == 0.0 and density[-1] == 0.0, (
            "the outline was left open at the ends")
        assert len(centres) == len(density)

    # The refusal, driven in the same test: no spread, no shape.
    assert fp._violin_profile(np.array([2.0, 2.0, 2.0]), 0.4) == (None, None)


# -------------------------------------------------- the rows under a scatter

def test_the_rows_under_a_scatter_come_back_as_positions_in_the_frame(ranked):
    """A ranked plot draws out of frame order, and the colouring follows it.

    ``add_scatter`` stores the frame row behind each point in pyqtgraph's own
    per-point ``data`` column, which is an OBJECT column whatever is put in
    it -- so ``_rows_of`` always has the conversion to make, and its
    "already integers" arm belongs to no caller. What the conversion buys is
    here: the second point drawn is the frame's LAST row, and colouring by a
    column paints it with that row's value rather than with the second one's.
    """
    frame = pd.DataFrame({"feature": ["a", "b", "c"],
                          "coefficient": [-2.0, 1.0, 2.0],
                          "p_value": [0.01, 0.5, 0.02]})

    assert ranked.set_results(frame) == 3

    item = _scatters(ranked)[0]
    rows = fp.FastPlot._rows_of(item)
    assert item.data["data"].dtype == object, (
        "pyqtgraph stopped storing per-point data in an object column")
    assert rows.dtype == np.dtype("int64")
    assert list(rows) == [0, 2, 1], "the ranking no longer reorders the rows"

    assert ranked.colour_by_column("coefficient") == 3
    painted = _brush_names(item)
    lookup = pg.colormap.get("viridis").getLookupTable(
        nPts=fp.COLORMAP_STEPS, alpha=True)
    coldest = QColor(*(int(c) for c in lookup[0])).name()
    hottest = QColor(*(int(c) for c in lookup[fp.COLORMAP_STEPS - 1])).name()

    assert painted[0] == coldest, "the -2 coefficient is not the ramp's floor"
    assert painted[1] == hottest, (
        "the second point took the second ROW's colour instead of its own")


# ------------------------------------------------------------- the q ramp

def test_a_table_with_no_q_at_all_never_reaches_for_the_ramp(volcano):
    """The ramp is asked for, and the plot still declines it: there is no q.

    ``set_results`` only builds the ramp once at least one q is finite, which
    is why the ramp's own "no finite q" arm is unreachable -- the ramp is
    never entered in that state. What the user sees instead is the
    called/not-called colouring, and a key with no q stop in it, because a
    scale over a quantity with no values is a key that names nothing.
    """
    volcano.set_q_colour("ramp")
    blank = _coefficients(8, p=[np.nan] * 8)

    volcano.set_results(blank)
    _legend_box(volcano).setChecked(True)
    without = _legend_labels(volcano)
    assert not [text for text in without if text.startswith("q ")], (
        f"a q ramp was drawn over a table with no q: {without}")

    # The same widget, the same request, with p values behind it.
    assert volcano.set_results(_coefficients(8)) == 8
    with_q = _legend_labels(volcano)
    assert [text for text in with_q if text.startswith("q ")], (
        f"the ramp key never appeared even with real q values: {with_q}")


# --------------------------------------------------- "Show as", and its shape

def test_every_shape_the_show_as_menu_is_built_for_can_be_remembered(
        spec_plot):
    """``offer`` and ``shape_of`` classify the same frame, so both answer.

    The submenu is built from ``offer``, which classifies the frame first;
    the "Always start with…" entry then classifies it AGAIN through
    ``shape_of``. That second call cannot fail where the first one did not,
    which is what makes the ``except`` and the empty-shape arm beneath it
    dead -- and the way to keep it that way is to check that every data shape
    the menu is built for still names itself.
    """
    from dataclasses import replace

    from spacr.graph_types import shape_of

    shapes = {
        "categorical_continuous": (pd.DataFrame({"g": ["a", "a", "b"],
                                                 "v": [1.0, 2.0, 3.0]}),
                                   "g", "v"),
        "continuous_continuous": (pd.DataFrame({"g": [2.0, 1.0, 2.0],
                                                "v": [1.0, 2.0, 3.0]}),
                                  "g", "v"),
        "ordered_continuous": (pd.DataFrame({"g": [1.0, 2.0, 3.0],
                                             "v": [1.0, 2.0, 3.0]}),
                               "g", "v"),
        "continuous_only": (pd.DataFrame({"v": [1.0, 2.0, 3.0]}), "", "v"),
    }
    for expected, (frame, group, value) in shapes.items():
        spec_plot.spec = replace(spec_plot.spec, frame=frame, group=group,
                                 value=value, kind="box")

        assert shape_of(frame, group, value) == expected
        entry = _remember_entry(spec_plot.build_style_menu())
        assert entry is not None, f"{expected} offered no way to pin a kind"
        assert entry.toolTip().startswith("Draw this kind first")

    # And the one thing that does withhold it: no kind to remember.
    spec_plot.spec = replace(spec_plot.spec, kind="")
    assert _remember_entry(spec_plot.build_style_menu()) is None


# ------------------------------------------------------ q and the local FDR

def test_a_row_that_has_a_q_always_has_a_local_fdr_to_go_with_it(volcano):
    """The two are computed over the same family, from the same raw p.

    ``adjust_p_values`` leaves a non-finite p non-finite, and ``local_fdr``
    does the same, so within one family the rows carrying a q are exactly the
    rows carrying a local FDR. That is why the click report's local-FDR test
    can never fail for a row whose q it has already printed -- and a row with
    no q never gets that far, which the nuisance term here shows.
    """
    frame = _coefficients(24)
    frame.loc[24] = ["Intercept", "", 0.4, np.nan]

    volcano.set_results(frame, drop_untested=False)

    q = volcano._q_values
    lfdr = volcano.local_fdr_values()
    finite = np.isfinite(q)
    assert finite.any() and not finite.all(), "the table was not mixed"
    assert np.isfinite(lfdr[finite]).all(), (
        "a row carrying a q was left without a local FDR")
    assert not np.isfinite(lfdr[~finite]).any(), (
        "a row with no q was given a local FDR anyway")

    assert "local FDR=" in volcano._detail(0)
    assert volcano._detail(24) == "not in the tested family, so no q"


# ------------------------------------------------- the names on the rank axis

def test_the_rank_axis_is_named_only_once_a_coefficient_survived(ranked):
    """An empty table returns before the axis is touched, so it stays bare.

    Which is what makes ``if shown:`` below the tick labels dead: by the time
    the names are written the frame is known to hold at least one row, and
    forty is the most that are ever labelled. The empty case is refused
    much earlier, with a sentence rather than an axis of nothing.
    """
    empty = pd.DataFrame({"feature": [], "coefficient": [], "p_value": []})

    assert ranked.set_results(empty) == 0
    assert "No coefficients to rank" in ranked._status.text()
    assert not ranked.plot.getAxis("left")._tickLevels, (
        "an empty table wrote tick labels onto the axis")

    assert ranked.set_results(_coefficients(1)) == 1
    levels = ranked.plot.getAxis("left")._tickLevels
    assert levels and len(levels[0]) == 1, (
        "the one surviving coefficient was not named on the axis")
    low, high = ranked.plot.getViewBox().viewRange()[1]
    assert low < -0.5 and high > 0.5, (
        f"the single row was not framed on the y axis: {(low, high)}")
