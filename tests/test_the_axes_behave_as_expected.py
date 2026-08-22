"""The log axes told the truth about the marks beside them -- and did not.

Instruction 148 A, and it is a CORRECTNESS bug rather than a nit.

    FastPlot._apply_log  ->  self.plot.setLogMode(x, y)

``PlotItem.setLogMode`` relabels the axes and then walks its items calling
``setLogMode`` on each one that HAS the method. ``PlotDataItem`` has it.
``ScatterPlotItem`` DOES NOT -- and every point on every plot in this module
is a ``ScatterPlotItem``: the volcano, the Q-Q, the control panel, the
guide-agreement plot, the effect ranking. So ticking "log y" relabelled the
axis to a log scale and left the dots exactly where they were, and a reader
took p-values off a ruler that did not describe the marks beside it.

Measured before the fix, on a volcano of thirty points: the axis reported
``logMode True`` while ``ScatterPlotItem.data['y']`` was byte-for-byte the
same array, and the only screen movement was the 0.7 px the left axis grew by
when its tick labels got shorter.

WHAT IS TESTED HERE is the fix's whole contract, because three separate things
read the coordinates and all three have to read the same ones: the dots move,
a hover still reports the REAL p-value rather than its logarithm, and a click
still lands on the row the user aimed at. Plus the refusal: a value at or
below zero has no logarithm, and instruction 148 decided the answer is to
REFUSE the axis with the count in the reason -- never to drop the points,
which would leave a volcano whose visible point count nobody can account for.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

pytestmark = pytest.mark.qt


#: p-values spread over eleven decades, because the whole difference between a
#: linear and a logarithmic axis is what happens to the middle of a range like
#: that. On a linear -log10(p) axis every point but the strongest is squashed
#: into the bottom tenth; a log scale spreads them, and a dot that does not
#: move between the two is the bug.
P_VALUES = np.geomspace(1e-11, 0.5, 12)


def _frame(coefficients=None) -> pd.DataFrame:
    n = len(P_VALUES)
    if coefficients is None:
        # STRICTLY POSITIVE, so log x is a question this table can answer.
        coefficients = np.linspace(0.1, 2.5, n)
    return pd.DataFrame({
        "feature": [f"fraction:grna[g{i}_1]" for i in range(n)],
        "coefficient": np.asarray(coefficients, dtype="float64"),
        "p_value": P_VALUES,
    })


@pytest.fixture()
def volcano(qtbot):
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    plot = VolcanoPlot()
    qtbot.addWidget(plot)
    plot.resize(700, 500)
    plot.show()
    qtbot.waitExposed(plot)
    plot.set_results(_frame())
    return plot


def _scatter(plot):
    from pyqtgraph import ScatterPlotItem

    items = [item for item in plot.plot.listDataItems()
             if isinstance(item, ScatterPlotItem)]
    assert items, "the plot drew no points at all"
    return items[0]


def _screen(plot, index: int):
    """Where dot ``index`` is PAINTED, in scene pixels.

    Read through the spot itself and mapped by Qt, so it is the position a
    user's eye and a user's mouse both find -- not a number this module
    computed and could be wrong about in the same direction twice.
    """
    scatter = _scatter(plot)
    point = scatter.points()[index]
    return scatter.mapToScene(point.pos())


def _log_entry(plot, axis: str):
    """The right-click entry that logs ``axis``.

    Instruction 148 C moved it off the checkbox strip under the plot, so this
    is now the control a user actually presses.
    """
    from spacr.qt.widgets.fast_plots import menu_entries

    for action in menu_entries(plot.build_style_menu()):
        if action.text().startswith(f"Log {axis} axis"):
            return action
    raise AssertionError(f"no log {axis} entry on the menu")


def _tick_log(plot, axis: str, on: bool = True) -> None:
    """Turn a log axis on the way a user does: off the right-click menu."""
    entry = _log_entry(plot, axis)
    if entry.isChecked() != on:
        entry.trigger()


# --------------------------------------------------------------------------- #
#  The dots move
# --------------------------------------------------------------------------- #

def test_ticking_log_y_moves_the_dots(volcano, qtbot):
    """The acceptance test instruction 148 names: two dots' SCREEN positions,
    read before and after, and they have to be different positions."""
    before = [_screen(volcano, 3), _screen(volcano, 7)]

    _tick_log(volcano, "y")
    qtbot.wait(1)

    after = [_screen(volcano, 3), _screen(volcano, 7)]
    moved = [abs(a.y() - b.y()) for a, b in zip(after, before)]
    assert min(moved) > 20, (
        f"the axis says log and the dots did not move: {moved} px "
        f"({before} -> {after})")


def test_the_drawn_position_is_the_logarithm_of_the_value(volcano):
    """Not merely 'different' -- the RIGHT different."""
    _tick_log(volcano, "y")

    drawn = np.asarray(_scatter(volcano).data["y"], dtype="float64")
    assert np.allclose(drawn, np.log10(-np.log10(P_VALUES)))


def test_unticking_puts_every_point_back_exactly_where_it_was(volcano):
    """The drawn coordinates are derived from the data every time, so this is
    the identity rather than a round trip through log10 and back."""
    original = np.array(_scatter(volcano).data["y"], dtype="float64")

    _tick_log(volcano, "y")
    _tick_log(volcano, "y", on=False)

    assert np.array_equal(np.asarray(_scatter(volcano).data["y"]), original)


def test_the_axis_label_says_the_axis_is_logged(volcano):
    """A tick in a menu nobody has open is not notice. The axis carries it."""
    assert volcano.plot.getAxis("left").labelText == "-log10(p)"

    _tick_log(volcano, "y")

    assert volcano.plot.getAxis("left").labelText == "-log10(p) (log scale)"
    assert volcano.plot.getAxis("bottom").labelText == "coefficient", (
        "logging y renamed the x axis")


def test_the_label_loses_the_note_again_when_the_scale_comes_off(volcano):
    _tick_log(volcano, "y")
    _tick_log(volcano, "y", on=False)

    assert volcano.plot.getAxis("left").labelText == "-log10(p)"


def test_a_threshold_line_is_logged_with_the_points_it_is_drawn_over(volcano):
    """A plot that logs its dots and not its p=0.05 line is a WORSE lie than
    the one this fixes: the line would then call the wrong points called."""
    from pyqtgraph import InfiniteLine

    volcano.set_results(_frame(), alpha=0.05)
    lines = [item for item in volcano.plot.plotItem.items
             if isinstance(item, InfiniteLine) and item.angle % 180 == 0]
    assert lines, "the volcano drew no significance line"
    before = float(lines[0].value())

    _tick_log(volcano, "y")

    assert float(lines[0].value()) == pytest.approx(np.log10(before))


# --------------------------------------------------------------------------- #
#  What must not break: the hover, the click, the ring
# --------------------------------------------------------------------------- #

def test_hovering_still_reports_the_real_value_not_its_logarithm(volcano):
    """pyqtgraph's own tip reads the DRAWN position, so on a logged axis it
    would say "y: 0.30" where -log10(p) is 2 -- a tooltip in units nobody
    asked for and no label explains."""
    _tick_log(volcano, "y")
    scatter = _scatter(volcano)
    index = 5
    drawn_y = float(scatter.data["y"][index])
    expected = -np.log10(P_VALUES[index])

    tip = volcano._point_tip(float(scatter.data["x"][index]), drawn_y,
                             int(scatter.data["data"][index]))

    assert f"y: {expected:.3g}" in tip, tip
    assert f"y: {drawn_y:.3g}" not in tip, tip


def test_a_real_click_on_a_logged_dot_still_returns_its_row(volcano, qtbot):
    """Driven through Qt at the pixel the dot is painted at. The hit test and
    the painter must read the same coordinates; calling the handler directly
    would test everything except the part that is easy to get wrong."""
    from PySide6.QtCore import QEvent, Qt as QtCore_Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication

    _tick_log(volcano, "y")
    qtbot.wait(1)
    got = []
    volcano.key_selected.connect(got.append)
    index = 6
    where = volcano.plot.mapFromScene(_screen(volcano, index))

    for kind in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease):
        QApplication.sendEvent(volcano.plot.viewport(), QMouseEvent(
            kind, where, QtCore_Qt.LeftButton, QtCore_Qt.LeftButton,
            QtCore_Qt.NoModifier))

    assert got == [_frame()["feature"][index]], (
        f"a click where the dot is drawn selected {got}")


def test_the_selection_ring_lands_on_the_dot_the_user_sees(volcano):
    """`_row_xy` holds DATA coordinates, so the ring has to be transformed
    like everything else -- otherwise it rings where the point used to be."""
    _tick_log(volcano, "y")
    index = 4

    assert volcano.highlight_key(_frame()["feature"][index])

    scatter = _scatter(volcano)
    ring = volcano._highlight
    assert float(ring.data["x"][0]) == pytest.approx(
        float(scatter.data["x"][index]))
    assert float(ring.data["y"][0]) == pytest.approx(
        float(scatter.data["y"][index]))


def test_a_ring_drawn_before_the_scale_changes_follows_the_dot(volcano):
    volcano.highlight_key(_frame()["feature"][4])

    _tick_log(volcano, "y")

    scatter = _scatter(volcano)
    assert float(volcano._highlight.data["y"][0]) == pytest.approx(
        float(scatter.data["y"][4]))


# --------------------------------------------------------------------------- #
#  Non-positive values are REFUSED, never dropped
# --------------------------------------------------------------------------- #

def test_log_x_is_refused_and_the_reason_carries_the_count(volcano):
    """Instruction 148's own sentence, to the comma."""
    coefficients = np.linspace(0.1, 2.5, len(P_VALUES))
    coefficients[:4] = [-1.0, -0.5, 0.0, -0.2]
    volcano.set_results(_frame(coefficients))

    assert volcano.log_reason("x") == (
        "log x: 4 of 12 points are at or below zero and have no logarithm")


def test_a_refused_axis_leaves_the_control_disabled_and_explaining_itself(
        volcano):
    """Instruction 106: never silently absent, never present-but-inert."""
    coefficients = np.linspace(0.1, 2.5, len(P_VALUES))
    coefficients[0] = 0.0
    volcano.set_results(_frame(coefficients))

    entry = _log_entry(volcano, "x")
    assert not entry.isEnabled()
    assert volcano.log_reason("x") in entry.text()
    assert entry.toolTip() == volcano.log_reason("x")
    assert _log_entry(volcano, "y").isEnabled()


def test_asking_for_a_refused_scale_changes_nothing_at_all(volcano):
    """Not "draws a subset": nothing. A plot drawn from rows the user did not
    choose is the failure this refusal exists to prevent."""
    coefficients = np.linspace(0.1, 2.5, len(P_VALUES))
    coefficients[0] = -0.3
    volcano.set_results(_frame(coefficients))
    before = np.array(_scatter(volcano).data["x"], dtype="float64")

    assert volcano.set_log_axes(x=True) == (False, False)

    assert np.array_equal(np.asarray(_scatter(volcano).data["x"]), before)
    assert len(_scatter(volcano).data) == len(P_VALUES), (
        "points went missing rather than the scale being refused")


def test_a_redraw_that_admits_a_zero_takes_the_scale_off_and_says_so(volcano):
    """The level filter admits a coefficient of zero AFTER log x is on. The
    points must not silently become NaN and leave the plot."""
    volcano.set_log_axes(x=True)
    assert volcano.log_axes() == (True, False)

    coefficients = np.linspace(0.1, 2.5, len(P_VALUES))
    coefficients[2] = 0.0
    volcano.set_results(_frame(coefficients))

    assert volcano.log_axes() == (False, False)
    assert np.array_equal(
        np.asarray(_scatter(volcano).data["x"], dtype="float64"),
        coefficients)
    assert "log x" in volcano._status.text()


def test_a_histogram_refuses_log_y_because_its_bars_start_at_zero(qtbot):
    """"50 of 100 values are at or below zero" is true and tells the reader
    nothing they can act on. The baseline is the answer."""
    from spacr.qt.widgets.fast_plots import PValueHistogram

    plot = PValueHistogram()
    qtbot.addWidget(plot)
    plot.set_p_values(np.random.default_rng(0).random(200))

    assert plot.log_reason("y") == (
        "log y: the bars are measured from zero, which has no logarithm")
    assert plot.set_log_axes(y=True) == (False, False)


def test_an_axis_that_names_its_groups_is_not_a_quantity(qtbot):
    """A control panel's x carries "nc" and "pc" at hand-placed positions.
    The logarithm of a position that stands for a name is not a number."""
    from spacr.qt.widgets.fast_plots import ControlSeparation

    plot = ControlSeparation()
    qtbot.addWidget(plot)
    rng = np.random.default_rng(2)
    plot.set_groups({"negative": rng.normal(-1, 0.3, 40),
                     "positive": rng.normal(1, 0.3, 35)})

    assert plot.log_reason("x") == (
        "log x: this axis names its groups rather than measuring a quantity")


# --------------------------------------------------------------------------- #
#  A histogram is bars, and a bar is found in data units
# --------------------------------------------------------------------------- #

def test_a_click_finds_the_right_bar_with_log_x_on(qtbot):
    """`mapSceneToView` answers in DRAWN units, so a bin looked up with those
    lands in the wrong bar the moment the axis is logged.

    Driven on a bare :class:`BinnedPlot` because both shipped histograms
    refuse log x for reasons of their own -- the p-value one bins over
    ``(0, 1)`` and the effect one draws a reference line at zero -- and the
    coordinate conversion under test belongs to the base class either way.
    """
    from PySide6.QtCore import QEvent, QPointF, Qt as QtCore_Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication
    from spacr.qt.widgets.fast_plots import BinnedPlot

    plot = BinnedPlot(title="Counts", x_label="value", y_label="rows")
    qtbot.addWidget(plot)
    plot.resize(700, 500)
    plot.show()
    qtbot.waitExposed(plot)
    values = np.geomspace(0.05, 40.0, 300)
    plot.set_keys([f"g{i}" for i in range(len(values))])
    plot._fill_bins(values, 10)
    plot.add_bars()
    assert plot.set_log_axes(x=True) == (True, False), plot.log_reason("x")
    got = []
    plot.keys_selected.connect(got.append)

    target = 3
    middle = float((plot._edges[target] + plot._edges[target + 1]) / 2)
    viewbox = plot.plot.plotItem.vb
    viewbox.updateAutoRange()
    (_x0, _x1), (y0, y1) = viewbox.viewRange()
    where = plot.plot.mapFromScene(viewbox.mapViewToScene(
        QPointF(np.log10(middle), (y0 + y1) / 2)))
    for kind in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease):
        QApplication.sendEvent(plot.plot.viewport(), QMouseEvent(
            kind, where, QtCore_Qt.LeftButton, QtCore_Qt.LeftButton,
            QtCore_Qt.NoModifier))
    QApplication.processEvents()

    assert got, "a real click on a logged histogram reached nothing"
    assert set(got[0]) == set(plot.keys_in_bin(target))


def test_the_bars_themselves_move_when_the_axis_is_logged(qtbot):
    """A bar is a rectangle in x, so logging the axis has to re-measure both
    of its edges -- an item pyqtgraph would not have moved either."""
    from spacr.qt.widgets.fast_plots import BinnedPlot

    plot = BinnedPlot(title="Counts", x_label="value", y_label="rows")
    qtbot.addWidget(plot)
    values = np.geomspace(0.05, 40.0, 300)
    plot._fill_bins(values, 10)
    bars = plot.add_bars()
    edges = np.asarray(plot._edges, dtype="float64")

    plot.set_log_axes(x=True)

    assert np.allclose(np.asarray(bars.opts["x0"], dtype="float64"),
                       np.log10(edges[:-1]))
    assert np.allclose(np.asarray(bars.opts["x1"], dtype="float64"),
                       np.log10(edges[1:]))


# --------------------------------------------------------------------------- #
#  Section B: a limit the user typed is the limit that is used
# --------------------------------------------------------------------------- #

def test_a_typed_limit_is_in_data_units_with_the_scale_on(volcano):
    """pyqtgraph's ranges are in DRAWN units, so before this a user who typed
    "1e-6 to 1" on a logged axis got a view from a millionth to one OF THE
    LOGARITHMS -- twenty decades away from what they asked for."""
    volcano.set_log_axes(y=True)

    volcano.set_axis_limits(y=(0.5, 8.0))

    (y_from, y_to) = volcano.axis_limits()[1]
    assert (round(y_from, 6), round(y_to, 6)) == (0.5, 8.0)
    drawn = volcano.plot.getViewBox().viewRange()[1]
    assert drawn[0] == pytest.approx(np.log10(0.5))
    assert drawn[1] == pytest.approx(np.log10(8.0))


def test_the_dialog_says_which_units_it_is_asking_for(volcano, monkeypatch):
    """The one axis where the two differ is the one nobody can guess."""
    from PySide6.QtWidgets import QInputDialog

    volcano.set_log_axes(y=True)
    prompts = []

    def _ask(parent, title, prompt, *args, **kwargs):
        prompts.append(prompt)
        return (1.0, True)

    monkeypatch.setattr(QInputDialog, "getDouble", staticmethod(_ask))

    volcano._ask_axis_limits()

    assert prompts == ["X from:", "X to:",
                       "Y from in data units, not log10:",
                       "Y to in data units, not log10:"]


def test_the_same_data_window_is_kept_when_the_scale_changes(volcano):
    """The user pinned a range of p-values, not a range of logarithms."""
    volcano.set_axis_limits(y=(0.5, 8.0))

    volcano.set_log_axes(y=True)

    (y_from, y_to) = volcano.axis_limits()[1]
    assert (round(y_from, 6), round(y_to, 6)) == (0.5, 8.0)


def test_a_typed_limit_survives_a_level_change(volcano):
    """A redraw with a different family of rows rebuilds every item."""
    volcano.set_axis_limits(x=(-0.25, 0.25))

    volcano.set_results(_frame(np.linspace(3.0, 9.0, len(P_VALUES))))

    (x_from, x_to) = volcano.axis_limits()[0]
    assert (round(x_from, 6), round(x_to, 6)) == (-0.25, 0.25)


def test_a_typed_limit_survives_a_recolour(volcano):
    volcano.set_axis_limits(y=(1.0, 4.0))

    volcano.colour_by_column("p_value")

    (y_from, y_to) = volcano.axis_limits()[1]
    assert (round(y_from, 6), round(y_to, 6)) == (1.0, 4.0)


def test_a_typed_limit_survives_a_redraw_with_the_scale_on(volcano):
    volcano.set_log_axes(y=True)
    volcano.set_axis_limits(y=(0.5, 8.0))

    volcano.set_results(_frame())

    (y_from, y_to) = volcano.axis_limits()[1]
    assert (round(y_from, 6), round(y_to, 6)) == (0.5, 8.0)


def test_a_non_positive_limit_on_a_logged_axis_is_refused_and_says_so(volcano):
    """Substituting a bound the user did not type is how a figure comes to
    show a range nobody chose."""
    volcano.set_log_axes(y=True)
    before = volcano.axis_limits()[1]

    volcano.set_axis_limits(y=(0.0, 8.0))

    assert volcano.axis_limits()[1] == before
    assert "no logarithm" in volcano._status.text()


def test_back_to_automatic_forgets_the_pin_for_good(volcano):
    """Not merely re-enables auto-range: a scale change afterwards must not
    put the old window back."""
    volcano.set_axis_limits(y=(1.0, 4.0))
    volcano.auto_range_axes()
    automatic = volcano.axis_limits()[1]

    volcano.set_log_axes(y=True)
    volcano.set_log_axes(y=False)

    assert volcano.axis_limits()[1] == pytest.approx(automatic)


def test_panning_by_hand_forgets_the_typed_limit(volcano):
    """A drag is a user saying "not that window any more"."""
    volcano.set_axis_limits(y=(1.0, 4.0))

    volcano.plot.getViewBox().sigRangeChangedManually.emit((False, True))

    assert volcano._pinned == {"x": None, "y": None}


# --------------------------------------------------------------------------- #
#  Section C: grid and log are right-click entries, not a strip of checkboxes
# --------------------------------------------------------------------------- #

def test_log_and_grid_are_entries_on_the_right_click_menu(volcano):
    from spacr.qt.widgets.fast_plots import menu_entries

    entries = {action.text(): action
               for action in menu_entries(volcano.build_style_menu())}

    for wanted in ("Log x axis", "Log y axis", "Grid"):
        matched = [text for text in entries if text.startswith(wanted)]
        assert matched, f"{wanted} is not on the menu: {sorted(entries)}"
        assert entries[matched[0]].isCheckable(), wanted


def test_log_is_under_axes_and_grid_is_under_appearance(volcano):
    """Named groups, and the obvious ones: a scale is a fact about the axis,
    a grid is a fact about how the plot looks."""
    menu = volcano.build_style_menu()
    where = {}
    for action in menu.actions():
        submenu = action.menu()
        if submenu is None:
            continue
        for entry in submenu.actions():
            where[entry.text()] = action.text()

    assert where.get("Log x axis") == "Axes", where
    assert where.get("Log y axis") == "Axes", where
    assert where.get("Grid") == "Appearance", where


def test_the_strip_under_the_plot_no_longer_carries_them(volcano):
    """"grid log x and y can be right click options not under the graph
    options"."""
    from PySide6.QtWidgets import QCheckBox

    labels = {box.text().lower() for box in volcano.findChildren(QCheckBox)}

    assert "log x" not in labels, labels
    assert "log y" not in labels, labels
    assert "grid" not in labels, labels
    # "legend (2)" rather than "legend": instruction 149 gave the volcano's
    # colour a job -- it carries the FDR call -- so even a plot with no
    # category column now has a two-entry legend, and the box says how many.
    assert any(label.startswith("legend") for label in labels), (
        "the legend was moved too")


def test_the_menu_entry_shows_the_scale_that_is_in_force(volcano):
    """A checkable entry that does not show its own state is one the user has
    to press to find out what it was."""
    assert not _log_entry(volcano, "y").isChecked()

    _tick_log(volcano, "y")

    assert _log_entry(volcano, "y").isChecked()
    assert not _log_entry(volcano, "x").isChecked()


def test_the_grid_entry_turns_the_grid_off_and_on(volcano):
    from spacr.qt.widgets.fast_plots import menu_entries

    assert volcano.grid_shown()
    grid = next(a for a in menu_entries(volcano.build_style_menu())
                if a.text() == "Grid")

    grid.trigger()

    assert not volcano.grid_shown()
    assert not next(a for a in menu_entries(volcano.build_style_menu())
                    if a.text() == "Grid").isChecked()
