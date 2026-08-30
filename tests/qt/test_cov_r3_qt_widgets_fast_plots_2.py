"""What ``fast_plots`` does when the data, the dialog or the axis is missing.

Every path here is one a user reaches by accident: a group whose column was
all blanks, a radar spoke with no number, a selection naming a gene whose
point was never plotted, a dialog they pressed Cancel on. The figure has to
stay a figure through all of them -- no whisker claiming a spread that
overflowed, no ring at a coordinate nobody computed, no half-applied restyle
from a cancelled prompt. Each test drives the degenerate input AND its
healthy twin, so "nothing happened" is measured against something happening.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

import pyqtgraph as pg                                      # noqa: E402
from PySide6.QtGui import QColor                            # noqa: E402
from PySide6.QtWidgets import QInputDialog                  # noqa: E402

from spacr.figure_style import (                            # noqa: E402
    saved_figure_appearance as _real_look)
from spacr.qt.widgets import fast_plots as fp               # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture
def plot(qtbot):
    """A bare FastPlot: both axes named, nothing drawn on it yet."""
    widget = fp.FastPlot(title="fast", x_label="ex", y_label="why")
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def keyed(qtbot):
    """Three named rows, only two of which were actually plotted.

    The shape a volcano has after an unusable p-value drops a gene: the key
    is still known, the coordinate is not.
    """
    widget = fp.FastPlot(title="fast", x_label="ex", y_label="why")
    qtbot.addWidget(widget)
    widget.set_keys(["a", "b", "c"])
    widget.add_scatter(np.array([0.0, 1.0]), np.array([0.0, 1.0]),
                       rows=[0, 1])
    return widget


def _answers(monkeypatch, **canned):
    """Make each QInputDialog getter answer without opening a window."""
    for name, value in canned.items():
        monkeypatch.setattr(QInputDialog, name,
                            staticmethod(lambda *a, _v=value, **k: _v))


# ------------------------------------------------------------- group marks

def test_a_violin_of_one_repeated_value_is_drawn_as_points_instead(plot):
    """A flat group must not be given a shape it does not have.

    Every observation identical means the density has no width, so the
    outline would be a vertical sliver that reads as a distribution. The
    mark falls back to points and still reports the same count, so a caller
    totalling observations across groups is not silently short by one.
    """
    flat = plot.add_group_mark(0.0, [2.5, 2.5, 2.5, 2.5], "violin")
    spread = plot.add_group_mark(1.0, [1.0, 2.0, 3.0, 9.0], "violin")

    assert flat == 4, "the fallback dropped the group's observations"
    assert spread == 4
    outlines = [item for item in plot.plot.plotItem.items
                if isinstance(item, pg.PlotCurveItem)]
    assert len(outlines) == 1, (
        f"{len(outlines)} violin outlines drawn for one shaped group")


def test_an_unknown_mark_names_the_marks_it_does_know(plot):
    """A typo in a mark name has to say what the legal names are.

    ``add_group_mark`` is reached from saved settings and from the graph-kind
    menu, so a stale stored value lands here. A silently empty plot leaves
    the user staring at a blank panel; the message lists every legal mark.
    """
    assert plot.add_group_mark(0.0, [1.0, 2.0], "box") == 2
    with pytest.raises(ValueError) as raised:
        plot.add_group_mark(0.0, [1.0, 2.0], "candlestick")
    assert "candlestick" in str(raised.value)
    assert "violin" in str(raised.value), (
        "the refusal did not list the marks that would have worked")


def test_a_group_of_one_gets_a_dot_and_no_interval(plot):
    """One observation has no error bar, and inventing one is a claim.

    The line mark draws a marker plus a spread whisker. With n=1 the
    standard error is undefined, so the whisker is omitted rather than drawn
    as zero -- zero length reads as "measured, and there was no spread".
    """
    before = len(plot.plot.plotItem.items)
    assert plot.add_group_mark(0.0, [4.0], "line") == 1
    alone = len(plot.plot.plotItem.items) - before

    before = len(plot.plot.plotItem.items)
    assert plot.add_group_mark(1.0, [4.0, 6.0, 5.0], "line") == 3
    together = len(plot.plot.plotItem.items) - before

    assert together == alone + 1, (
        "the single observation was given an interval it cannot have")


def test_a_group_that_is_all_blanks_draws_nothing_at_all(plot):
    """A column of blanks must not put an empty mark on the axis.

    A grouped plot hands each level a slice; a level whose values are all
    NaN would draw a bar of height nan, which pyqtgraph renders as an
    invisible artist that still widens the axis.
    """
    before = len(plot.plot.plotItem.items)
    assert plot.add_group_mark(0.0, [np.nan, np.nan], "bar") == 0
    assert len(plot.plot.plotItem.items) == before, (
        "an all-blank group still added artists to the plot")
    assert plot.add_group_mark(1.0, [1.0, 3.0], "bar") == 2
    assert len(plot.plot.plotItem.items) > before


def test_a_spread_that_overflows_leaves_the_bar_without_a_whisker(plot):
    """A whisker is drawn only when the spread is a number.

    ``spread_of`` answers a non-finite value once the squared deviations
    overflow. Drawn, that is a whisker running off the view, which reads as
    an enormous measured uncertainty rather than an arithmetic failure.
    """
    before = len(plot.plot.plotItem.items)
    with np.errstate(over="ignore", invalid="ignore"):
        assert plot.add_group_mark(0.0, [1e308, -1e308], "bar",
                                   spread="var") == 2
    overflowed = len(plot.plot.plotItem.items) - before

    before = len(plot.plot.plotItem.items)
    assert plot.add_group_mark(1.0, [1.0, 3.0], "bar", spread="var") == 2
    measurable = len(plot.plot.plotItem.items) - before

    assert measurable == overflowed + 1, (
        "an unrepresentable spread was still drawn as a whisker")


# ---------------------------------------------------------------- clicking

def test_a_click_that_hit_nothing_leaves_the_status_line_alone(plot):
    """Two clicks that resolve to no row must not disturb the plot.

    pyqtgraph delivers ``sigClicked`` with an empty list when the hit test
    finds nothing, and with a point carrying ``None`` for data when the item
    was drawn without row identifiers. Either one reaching the formatting
    code emits a bogus index to every linked table.
    """
    seen = []
    plot.point_clicked.connect(seen.append)

    plot._on_points_clicked(None, [])
    plot._on_points_clicked(None, [SimpleNamespace(data=lambda: None)])
    assert seen == [], f"a click on nothing emitted {seen}"

    plot._on_points_clicked(None, [SimpleNamespace(data=lambda: 0)])
    assert seen == [0], "a click on a real row emitted nothing"


def test_a_point_with_no_name_still_reports_its_row(plot):
    """A plot with no labels, no frame and no keys is still clickable.

    A diagnostic plot is handed bare arrays: nothing to write in the status
    line and no identifier to broadcast. The click must still emit the row
    index -- that is what a linked image view uses -- and must not blank the
    headline the panel put there.
    """
    seen = []
    plot.point_clicked.connect(seen.append)
    plot.set_status_note("kept")

    plot._on_points_clicked(None, [SimpleNamespace(data=lambda: 2)])

    assert seen == [2], "an unnamed point did not report its row"
    assert "kept" in plot._status.text(), (
        "an unnamed point overwrote the status line with an empty caption")
    assert plot.selected_keys() == [], (
        "a point with no key was announced as a selection")


# --------------------------------------------------------------- selection

def test_a_key_whose_point_was_never_plotted_reports_not_found(keyed):
    """Saying "not drawn" beats ringing something near it.

    A gene can be named by the results table and absent from the figure --
    an unusable p-value, a nuisance term the plot leaves off. The ring needs
    a remembered coordinate, and there is none, so the answer is False
    rather than a ring appearing at the origin.
    """
    assert keyed.highlight_key("a") is True
    assert keyed.highlight_key("c") is False, (
        "a key with no plotted coordinate claimed to have been ringed")
    assert keyed._highlight is None, "a ring was drawn for an absent point"


def test_a_selection_naming_the_same_gene_twice_rings_it_once(keyed):
    """A repeated key is one selection member, not two.

    The selection arrives from a table where a click can repeat a row, and
    from ``toggle_key``. Keeping the duplicate makes the count in the status
    line disagree with the rings on screen, which is what the count is for.
    """
    assert keyed.highlight_keys(["a", "a", "b"]) == 2
    assert keyed.selected_keys() == ["a", "b"], (
        "the duplicate survived into the selection")


def test_undrawn_members_of_a_selection_are_kept_but_not_ringed(keyed):
    """Membership is never conditional on the point having been drawn.

    A rectangle or a table can select a gene this figure does not plot. The
    key stays in the selection so every linked view receives it; only the
    ring is skipped, and the returned count says how many were drawn so a
    caller can tell the two numbers apart.
    """
    drawn = keyed.highlight_keys(["c", "zz", "b"])

    assert drawn == 1, f"expected one ring for three keys, got {drawn}"
    assert keyed.selected_keys() == ["c", "zz", "b"], (
        "keys with no plotted point were dropped from the selection")


def test_a_rectangle_skips_unnamed_rows_and_never_repeats_a_gene(qtbot):
    """Rubber-band selection returns identifiers, in pick order, once each.

    Two rows can carry the same gene and a row can carry none at all -- a
    fit with no gene-level term gives exactly that. The unnamed row selects
    nothing and the repeated one is a single member, or the count shown
    exceeds the genes actually selected.
    """
    widget = fp.FastPlot(title="fast", x_label="ex", y_label="why")
    qtbot.addWidget(widget)
    widget.set_keys(["a", None, "a"])
    widget.add_scatter(np.array([0.0, 1.0, 2.0]), np.array([0.0, 0.0, 0.0]),
                       rows=[0, 1, 2])

    assert widget.select_in_rect(-1.0, -1.0, 3.0, 1.0) == ["a"]
    assert widget.select_in_rect(-1.0, -1.0, 3.0, 1.0, add=True) == ["a"], (
        "extending the selection with what it already held duplicated it")


# ------------------------------------------------- curves, bars and radars

def test_charts_with_no_finite_numbers_offer_no_table_to_export(plot):
    """An empty ``data.csv`` beside a figure is worse than none at all.

    ``curve_frame`` and ``ranked_frame`` are what the export bundle writes
    beside the picture. A sweep with no finite point and a feature-importance
    table whose scores failed to parse both have to answer None: a zero-row
    csv reads as a curve that was measured and found empty.
    """
    assert plot.curve_frame() is None
    assert plot.add_curve([np.nan, 1.0], [np.nan, np.nan]) == 0
    assert plot.curve_frame() is None, (
        "a curve with no finite points still produced a table")
    assert plot.add_curve([1.0, 2.0], [3.0, 4.0]) == 2
    assert list(plot.curve_frame()["y"]) == [3.0, 4.0]

    assert plot.add_ranked_bars(["a", "b"], [np.nan, np.nan]) == 0
    assert plot.ranked_frame() is None, (
        "a chart with no bars still offered a table to export")
    assert plot.add_ranked_bars(["a", "b"], [1.0, 2.0]) == 2
    assert list(plot.ranked_frame()["name"]) == ["b", "a"]


def test_a_radar_drops_blank_spokes_and_refuses_a_two_sided_polygon(plot):
    """Below three spokes there is no polygon to draw.

    Blank spokes are dropped rather than plotted at zero, because a zero
    radius is a measurement. That can take a five-spoke radar under the three
    it needs, and the answer then is zero: a two-sided "polygon" is a line
    segment the reader would read as a shape.
    """
    assert plot.radar_frame() is None
    assert plot.add_radar(["a", "b", "c", "d", "e"],
                          [1.0, np.nan, 2.0, 3.0, np.nan]) == 3
    assert list(plot.radar_frame()["name"]) == ["a", "c", "d"], (
        "the blank spokes were kept in the exported table")
    assert plot.add_radar(["a", "b", "c", "d"],
                          [1.0, np.nan, np.nan, 3.0]) == 0


def test_a_beeswarm_skips_a_feature_whose_column_is_all_blank(plot):
    """One unusable feature must not cost the other rows their points.

    SHAP contributions arrive with a column of NaN whenever a feature was
    constant in the fold. Drawing it puts a row of points at NaN; skipping it
    keeps the other features on their rows and reports the honest total.
    """
    matrix = np.array([[1.0, np.nan], [2.0, np.nan], [-1.0, np.nan]])
    assert plot.add_beeswarm(["kept", "blank"], matrix) == 3, (
        "the blank feature was counted among the drawn points")
    assert plot.add_beeswarm(["kept", "also"],
                             np.array([[1.0, 2.0], [2.0, 1.0]])) == 4


def test_missing_feature_values_take_the_grey_and_not_a_colour(plot):
    """A blank feature value has no place on the colour scale.

    The beeswarm encodes the feature's own value as the point's colour.
    Mapping NaN through the lookup lands it at one end of the scale, reading
    as "lowest value" rather than "not measured", so it takes the muted
    colour instead.
    """
    lookup = np.array([[i, 255 - i, 0, 255] for i in range(fp.COLORMAP_STEPS)])
    grey = QColor(fp.MISSING_COLOUR)

    blank = fp.FastPlot._shade([np.nan, np.nan], lookup)
    assert [c.name() for c in blank] == [grey.name()] * 2

    mixed = fp.FastPlot._shade([0.0, np.nan, 1.0], lookup)
    assert mixed[1].name() == grey.name()
    assert mixed[0].name() != grey.name(), (
        "a measured value was greyed out with the missing ones")


# ------------------------------------------------------ shapes and fading

def test_no_levels_means_no_fading_and_no_alphas(plot):
    """The opacity channel must be a no-op when there is nothing to encode.

    ``_categorical_opacity`` is called before the column is known to hold
    anything. With no levels it hands the brushes straight back: a ramp over
    zero levels divides by ``count - 1`` and would fade every point alike,
    silently deleting the channel.
    """
    assert fp.FastPlot._opacity_alphas(0) == []
    assert fp.FastPlot._opacity_alphas(3) == [70, 162, 255]

    brushes = [pg.mkBrush(QColor("#112233"))]
    assert fp.FastPlot._categorical_opacity(brushes, [], 1) is brushes, (
        "an empty column still rebuilt the brush list")


def test_the_screen_size_remembers_only_the_first_floors(plot):
    """The original size floors are captured once, not on every resize.

    ``clear_screen_size`` puts back what the layout gave the widget.
    Re-reading the bounds on a second ``set_screen_size`` captures the FIXED
    size just imposed, so releasing the plot would pin it there forever and
    the splitter could never grow it again.
    """
    plot.setMinimumHeight(240)
    plot.set_screen_size(400, 300)
    remembered = plot._size_bounds
    plot.set_screen_size(500, 320)

    assert plot._size_bounds == remembered, (
        "the second call overwrote the floors with the fixed size")
    plot.clear_screen_size()
    assert plot.minimumHeight() == 240


def test_a_collapsed_plot_is_not_reshaped_for_the_page(plot):
    """A canvas with no width cannot be given a proportion.

    The export holds the plot item at the page's aspect for one render. A
    widget never shown can report a zero-width geometry, and width times
    ratio is then a zero-height scene -- an empty file. The context yields
    False instead and the exporter writes the plot as it is.
    """
    from PySide6.QtCore import QRectF

    plot.set_canvas_shape("square")
    item = plot.plot.plotItem
    item.setMinimumSize(0, 0)          # a widget that was never shown
    item.setGeometry(QRectF(0.0, 0.0, 0.0, 0.0))
    with plot._held_at_the_page_shape() as held:
        assert held is False, "a zero-width canvas was reshaped anyway"

    item.setGeometry(QRectF(0.0, 0.0, 400.0, 200.0))
    with plot._held_at_the_page_shape() as held:
        assert held is True


# ------------------------------------------------- axes, menu and dialogs

def test_a_viewbox_with_no_drain_hook_is_left_alone(plot):
    """The pending-range drain is optional, and calling None would crash.

    pyqtgraph 0.14 queues auto-range updates instead of applying them, so a
    read in the same event-loop turn sees the previous table's range unless
    the queue is drained. Older releases carry no such method, and spaCR
    supports both.
    """
    box = plot.plot.getViewBox()
    calls = []
    box.updateAutoRange = lambda: calls.append("drained")
    plot._sync_auto_range()
    assert calls == ["drained"], "the pending range update was never applied"

    box.updateAutoRange = None
    plot._sync_auto_range()
    assert calls == ["drained"], (
        "a viewbox without the hook was called through anyway")


def test_an_axis_the_widget_will_not_hand_over_is_left_out(plot):
    """The line control colours the axes it can actually reach.

    ``axis_items`` is what puts the spines and tick marks under "Line
    colour". A widget that answers None for an edge must shrink the list
    rather than put a None in it: the caller calls ``setPen`` on every entry.
    """
    real = plot.plot.getAxis
    assert len(plot.axis_items()) == 4

    plot.plot.getAxis = lambda edge: None if edge == "top" else real(edge)
    remaining = plot.axis_items()
    assert len(remaining) == 3, f"expected three axes, got {len(remaining)}"
    assert all(axis is not None for axis in remaining)


def _menu_labels(menu) -> list:
    """Every action label in a menu, submenus included."""
    out = []
    for action in menu.actions():
        out.append(action.text())
        if action.menu() is not None:
            out.extend(_menu_labels(action.menu()))
    return out


def test_a_plot_that_cannot_write_its_correction_does_not_offer_to(plot):
    """The "write this out" entry belongs only where there is a writer.

    A recorrected plot disagrees with the results.csv beside it, and the
    entry is how the two are made to agree. On a plot with no writer it is
    an entry that does nothing -- the inert control the design forbids -- so
    the correction submenu appears without it.
    """
    plot.offer_corrections([("BH", lambda: None, True)])
    plot.offer_smoothers(lambda name: None)

    labels = _menu_labels(plot.build_style_menu())
    assert "Diagnostic curve (decides no hit)" in labels, (
        "the offered smoothers never reached the menu")
    assert not any(text.startswith("Write this correction") for text in labels)

    plot._correction_writer = lambda: None
    assert any(text.startswith("Write this correction")
               for text in _menu_labels(plot.build_style_menu())), (
        "a plot that can write its correction was not offered the entry")


def _print_look():
    """The real print appearance, bound before any test patches the module."""
    return _real_look("print")


class _PlainAxis:
    """An axis with a pen and no ``textPen`` -- older pyqtgraph's shape."""

    def __init__(self, colour: str):
        self._pen = pg.mkPen(colour)

    def pen(self):
        return self._pen

    def setPen(self, pen):
        self._pen = pen


def test_a_screen_save_repaints_no_chrome_at_all(plot, monkeypatch):
    """"Save as it looks" has to mean exactly that.

    The print look flips the axes and their text so a dark figure is legible
    on paper. In screen mode nothing may be touched: a user who asked for the
    screen colours and got #222222 axes has lost the figure they set up.
    """
    import spacr.figure_style as style

    monkeypatch.setattr(style, "saved_figure_appearance",
                        lambda: SimpleNamespace(mode="screen"))
    plot.plot.getAxis("bottom").setPen(pg.mkPen("#ffffff"))
    assert fp.FastPlot._wear_the_print_look(plot.plot) == []

    monkeypatch.setattr(style, "saved_figure_appearance", _print_look)
    assert fp.FastPlot._wear_the_print_look(plot.plot), (
        "a print save left white axes invisible on a white page")


def test_the_print_look_survives_a_missing_axis_and_an_old_one(monkeypatch):
    """Repainting for the page must not depend on pyqtgraph's version.

    ``textPen`` arrived in a later release, and a plot can be built with
    fewer than four axes. Either reaching the repaint unguarded raises inside
    the exporter, losing the whole figure at save time.
    """
    import spacr.figure_style as style

    monkeypatch.setattr(style, "saved_figure_appearance", _print_look)
    axes = {"bottom": _PlainAxis("#ffffff"), "left": _PlainAxis("#ffffff")}
    item = SimpleNamespace(getAxis=lambda edge: axes.get(edge),
                           titleLabel=None)

    undo = fp.FastPlot._wear_the_print_look(item)

    assert len(undo) == 2, f"expected one repaint per present axis, got {undo}"
    assert axes["bottom"].pen().color().name() == "#222222"
    for put_back in undo:
        put_back()
    assert axes["bottom"].pen().color().name() == "#ffffff", (
        "the undo did not put the screen colour back")


def test_a_cancelled_prompt_leaves_the_plot_exactly_as_it_was(plot,
                                                              monkeypatch):
    """Cancel must be a true no-op across all three restyle prompts.

    A dialog that half-applies is worse than one that does nothing: the user
    has no way back to what they had. Each prompt is answered once and then
    cancelled, so the accepted value is what a cancel must fail to change.
    """
    _answers(monkeypatch, getDouble=(2.0, True))
    plot._ask_aspect_ratio()
    assert plot.aspect_ratio() == 2.0
    _answers(monkeypatch, getDouble=(7.0, False))
    plot._ask_aspect_ratio()
    assert plot.aspect_ratio() == 2.0, "a cancelled prompt locked the scales"

    monkeypatch.setattr(fp, "pick_colour", lambda *a, **k: QColor("#123456"))
    plot._ask_line_colour()
    assert plot.line_colour() == "#123456"
    monkeypatch.setattr(fp, "pick_colour", lambda *a, **k: QColor())
    plot._ask_line_colour()
    assert plot.line_colour() == "#123456", (
        "a cancelled colour dialog still recoloured the lines")

    asked = []
    monkeypatch.setattr(plot, "shape_by_column", asked.append)
    _answers(monkeypatch, getItem=("condition", True))
    plot._ask_shape_column()
    assert asked == ["condition"]
    _answers(monkeypatch, getItem=("condition", False))
    plot._ask_shape_column()
    assert asked == ["condition"], (
        "a cancelled shape prompt still remapped the point shapes")
