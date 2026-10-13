"""The translator's remaining refusals: the artist it cannot paint, place or read.

Every branch here is one the scene translation takes when an artist does not
give it what it asked for -- a colour that is not paint, a position that is
nowhere, an axis that raises when asked what its ticks say, a bar drawn in
axes fraction that converts to infinity. The rule the module lives by is that
NONE of those may be fatal and none may be invented around: the mark is left
off and the page is still written.

Each test drives the refusal and the ordinary case side by side, because "no
item was added" is only interesting next to the input that does add one.
"""

import os

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt          # noqa: E402

from spacr.figures import scene as sc    # noqa: E402
from spacr.figures.scene import (SceneReport, _plain_text, build_scene,
                                 pyqtgraph_ready)                # noqa: E402


@pytest.fixture
def closed_figures():
    """Close whatever a test drew, whichever way it ends."""
    yield
    plt.close("all")


@pytest.fixture
def qt_ready(qapp):
    ready, why = pyqtgraph_ready()
    if not ready:
        pytest.skip(why)
    return True


class _Collector:
    """The one method of a pyqtgraph ``PlotItem`` the translators call."""

    def __init__(self):
        self.items = []

    def addItem(self, item, **_ignored):
        self.items.append(item)


def _look():
    return sc._Look(None)


# ---------------------------------------------------------------------------
# starting a QApplication where there is no display
# ---------------------------------------------------------------------------

def test_a_machine_with_no_display_is_told_to_render_offscreen(qt_ready,
                                                              monkeypatch):
    """The offscreen platform is set only when there is nothing to draw on.

    Both halves matter: setting it on a machine that HAS a display would take
    a running GUI's own exports off the screen it is showing them on.
    """
    started = []

    class _StubApplication:
        _instance = None

        def __init__(self, argv):
            started.append(list(argv))

        @classmethod
        def instance(cls):
            return cls._instance

    monkeypatch.setattr("PySide6.QtWidgets.QApplication", _StubApplication)
    monkeypatch.setattr(sc, "_APPLICATION", None)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)

    ok, why = pyqtgraph_ready()
    assert (ok, why) == (True, "")
    assert os.environ["QT_QPA_PLATFORM"] == "offscreen"
    assert started == [[]], "no QApplication was started"
    assert sc._APPLICATION is not None

    # The same call on a machine with a display leaves the platform alone.
    monkeypatch.setattr(sc, "_APPLICATION", None)
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    assert pyqtgraph_ready()[0] is True
    assert "QT_QPA_PLATFORM" not in os.environ


# ---------------------------------------------------------------------------
# a Line2D
# ---------------------------------------------------------------------------

def test_a_one_point_line_is_not_a_reference_line(closed_figures):
    """A reference line is a direction, and one point has none.

    ``axhline`` puts two points down; a single-point marker drawn in the same
    blended transform is a mark, and treating it as an infinite line would
    draw a rule across the whole panel.
    """
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots()
    assert sc._reference_line(Line2D([0.5], [0.5]), axes) is None
    assert sc._reference_line(Line2D([0.0, 1.0], [0.5, 0.5]), axes) == "h"


def test_a_reference_line_painted_in_none_is_not_drawn(qt_ready,
                                                       closed_figures):
    """`color='none'` is a line the panel deliberately left off the page."""
    figure, axes = plt.subplots()
    invisible = axes.axhline(0.5, color="none")
    visible = axes.axhline(0.6, color="#888888")

    plot = _Collector()
    assert sc._add_line(plot, invisible, axes, _look()) == 0
    assert plot.items == []
    assert sc._add_line(plot, visible, axes, _look()) == 1
    assert len(plot.items) == 1


def test_a_series_with_no_line_colour_still_gets_its_markers(qt_ready,
                                                             closed_figures):
    """A marker-only style spelt as `color='none'` is points, not nothing."""
    import pyqtgraph as pg

    figure, axes = plt.subplots()
    unpainted, = axes.plot([0, 1, 2], [0, 1, 2], color="none", marker="o")
    painted, = axes.plot([0, 1, 2], [2, 1, 0], color="#123456", marker="o")

    plot = _Collector()
    assert sc._add_line(plot, unpainted, axes, _look()) == 1
    assert isinstance(plot.items[0], pg.ScatterPlotItem), (
        "the curve was drawn with a colour the panel did not give it")

    plot = _Collector()
    assert sc._add_line(plot, painted, axes, _look()) == 2
    assert isinstance(plot.items[0], pg.PlotDataItem)


# ---------------------------------------------------------------------------
# a scatter
# ---------------------------------------------------------------------------

def test_a_scatter_that_kept_no_sizes_gets_the_default_diameter(
        qt_ready, closed_figures):
    """A PathCollection with its sizes cleared is still a scatter.

    The default is matplotlib's own s=20, and it has to reach every point,
    not just the first.
    """
    figure, axes = plt.subplots()
    collection = axes.scatter([0, 1, 2], [0, 1, 2], c="#AA3333")
    collection.set_sizes([])
    assert collection.get_sizes().size == 0

    plot = _Collector()
    assert sc._add_path_collection(plot, collection, _look()) == 1
    sizes = list(plot.items[0].data["size"])
    assert len(sizes) == 3
    assert sizes[0] == sizes[1] == sizes[2]
    expected = float(np.sqrt(20.0) * _look().scale)
    assert sizes[0] == pytest.approx(expected)

    collection.set_sizes([80.0])
    plot = _Collector()
    assert sc._add_path_collection(plot, collection, _look()) == 1
    assert plot.items[0].data["size"][0] != pytest.approx(expected)


def test_a_scatter_with_fewer_sizes_than_points_wraps_them(qt_ready,
                                                           closed_figures):
    """Short of one size per point, the sizes repeat rather than run out.

    Truncating to the sizes it has would drop points off the panel; resizing
    is the only answer that draws every offset the panel plotted.
    """
    figure, axes = plt.subplots()
    collection = axes.scatter([0, 1, 2], [0, 1, 2], c="#AA3333")
    collection.set_sizes([16.0, 64.0])

    plot = _Collector()
    assert sc._add_path_collection(plot, collection, _look()) == 1
    sizes = list(plot.items[0].data["size"])
    assert len(sizes) == 3, "a point was left off for want of a size"
    assert sizes[2] == pytest.approx(sizes[0])
    assert sizes[1] != pytest.approx(sizes[0])


def test_a_transparent_point_gets_no_brush_and_its_neighbour_still_does(
        qt_ready, closed_figures):
    """One unfilled point in a scatter must not unfill the rest of it."""
    from PySide6.QtCore import Qt

    figure, axes = plt.subplots()
    collection = axes.scatter([0, 1], [0, 1],
                              facecolors=[(1.0, 0.0, 0.0, 0.0),
                                          (0.0, 0.0, 1.0, 1.0)])

    plot = _Collector()
    assert sc._add_path_collection(plot, collection, _look()) == 1
    brushes = list(plot.items[0].data["brush"])
    assert brushes[0].style() == Qt.BrushStyle.NoBrush, (
        "a fully transparent face was painted")
    assert brushes[1].style() == Qt.BrushStyle.SolidPattern
    assert brushes[1].color().alpha() == 255


# ---------------------------------------------------------------------------
# a LineCollection
# ---------------------------------------------------------------------------

def test_stems_painted_in_nothing_are_not_drawn(qt_ready, closed_figures):
    """`vlines(color='none')` is geometry with no ink, and adds no item."""
    from matplotlib.collections import LineCollection

    figure, axes = plt.subplots()
    invisible = LineCollection([[(0.0, 0.0), (0.0, 1.0)]], colors="none")
    visible = LineCollection([[(0.0, 0.0), (0.0, 1.0)]], colors="#444444")

    plot = _Collector()
    assert sc._add_line_collection(plot, invisible, _look()) == 0
    assert plot.items == []
    assert sc._add_line_collection(plot, visible, _look()) == 1
    assert len(plot.items) == 1


# ---------------------------------------------------------------------------
# a Rectangle drawn in axes fraction
# ---------------------------------------------------------------------------

def test_a_backdrop_that_converts_to_nowhere_is_dropped(qt_ready,
                                                        closed_figures):
    """A fraction that lands off the number line is not a place to draw.

    `_skip_box` draws its backdrop in axes fraction; a corner that converts
    to infinity would reach `BarGraphItem` as a NaN geometry and take the
    whole bar set with it.
    """
    from matplotlib.patches import Rectangle

    figure, axes = plt.subplots()
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    nowhere = Rectangle((float("inf"), 0.0), 1.0, 1.0,
                        transform=axes.transAxes, facecolor="#DDDDDD")
    axes.add_patch(nowhere)
    somewhere = Rectangle((0.1, 0.1), 0.5, 0.5,
                          transform=axes.transAxes, facecolor="#DDDDDD")
    axes.add_patch(somewhere)

    plot = _Collector()
    assert sc._add_rectangles(plot, [nowhere], axes, _look()) == 0
    assert plot.items == []
    assert sc._add_rectangles(plot, [somewhere], axes, _look()) == 1


# ---------------------------------------------------------------------------
# mathtext
# ---------------------------------------------------------------------------

def test_deeply_nested_groups_are_stripped_to_the_last_one():
    """The stripping pass runs until the braces stop changing, not once.

    Six passes is the budget, and a label that needs all six has to come out
    as clean as one that needs two -- an unstripped brace is what makes a
    label "not understood" and costs the whole figure its renderer.
    """
    assert _plain_text("$" + "{" * 6 + "a" + "}" * 6 + "$") == ("a", True)
    assert _plain_text("${{a}}$") == ("a", True)


# ---------------------------------------------------------------------------
# a Text
# ---------------------------------------------------------------------------

def test_a_caption_that_cannot_be_placed_is_left_off_the_scene(
        qt_ready, closed_figures):
    """Nowhere is not a place, and a TextItem at infinity poisons the view."""
    report = SceneReport()
    figure, axes = plt.subplots()
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    nowhere = axes.text(float("inf"), 0.5, "a caption",
                        transform=axes.transAxes)
    somewhere = axes.text(0.5, 0.5, "a caption", transform=axes.transAxes)

    plot = _Collector()
    assert sc._add_text(plot, nowhere, axes, _look(), report) == 0
    assert plot.items == []
    assert sc._add_text(plot, somewhere, axes, _look(), report) == 1


def test_a_caption_box_with_no_paint_gets_neither_fill_nor_border(
        qt_ready, closed_figures):
    """A bbox spelt in 'none' is a box the panel asked for and left empty."""
    from PySide6.QtCore import Qt

    report = SceneReport()
    figure, axes = plt.subplots()
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    empty = axes.text(0.2, 0.5, "bare", transform=axes.transAxes,
                      bbox=dict(facecolor="none", edgecolor="none"))
    filled = axes.text(0.6, 0.5, "boxed", transform=axes.transAxes,
                       bbox=dict(facecolor="#EEEEEE", edgecolor="#333333"))

    plot = _Collector()
    assert sc._add_text(plot, empty, axes, _look(), report) == 1
    assert plot.items[0].fill.style() == Qt.BrushStyle.NoBrush
    assert plot.items[0].border.style() == Qt.PenStyle.NoPen

    plot = _Collector()
    assert sc._add_text(plot, filled, axes, _look(), report) == 1
    assert plot.items[0].fill.style() != Qt.BrushStyle.NoBrush
    assert plot.items[0].border.style() != Qt.PenStyle.NoPen


def test_a_caption_whose_font_cannot_be_read_is_still_written(
        qt_ready, closed_figures):
    """The type size is a nicety; the words are not.

    An artist mid-teardown raises when asked its font size, and losing the
    caption over that would take a panel's whole explanation with it.
    """
    report = SceneReport()
    figure, axes = plt.subplots()
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    text = axes.text(0.5, 0.5, "a caption", transform=axes.transAxes)

    asked = []

    def _refuse():
        asked.append("fontsize")
        raise RuntimeError("this artist is being removed")

    text.get_fontsize = _refuse

    plot = _Collector()
    assert sc._add_text(plot, text, axes, _look(), report) == 1
    assert asked == ["fontsize"]
    assert plot.items[0].toPlainText() == "a caption"


# ---------------------------------------------------------------------------
# an axis that will not describe itself
# ---------------------------------------------------------------------------

def test_an_axis_that_refuses_its_tick_labels_costs_only_the_ticks(
        qt_ready, closed_figures):
    """Two reads of the same labels -- the type size and the categories.

    Both are wrapped, and both have to survive an axis that raises, because
    the alternative is losing a whole page of panels to one axis that was
    being torn down while the scene was built.
    """
    def _categorical():
        figure, axes = plt.subplots()
        axes.bar(["plate1", "plate2", "plate3"], [1, 2, 3])
        figure.canvas.draw()
        return figure, axes

    figure, axes = _categorical()
    widget, report = build_scene(figure)
    try:
        carried = widget.getItem(0, 0).getAxis("bottom")._tickLevels
        assert carried, "the categorical labels were not carried at all"
        assert {"plate1", "plate2", "plate3"} <= {label
                                                  for _, label in carried[0]}
    finally:
        widget.deleteLater()

    figure, axes = _categorical()
    asked = []

    def _refuse(*_args, **_kwargs):
        asked.append("ticklabels")
        raise RuntimeError("this axis is being removed")

    axes.xaxis.get_ticklabels = _refuse
    widget, report = build_scene(figure)
    try:
        assert asked, "the axis was never asked"
        assert report.complete, report.reason()
        assert report.items >= 1, "the bars went with the tick labels"
        assert not widget.getItem(0, 0).getAxis("bottom")._tickLevels
    finally:
        widget.deleteLater()


def test_a_tick_label_in_mathtext_is_left_to_pyqtgraph(qt_ready,
                                                       closed_figures):
    """A formula the translator will not guess at is not printed as source.

    The whole set goes, not the half it could read: a categorical axis
    showing one of its three names and a blank where the other two were is a
    more convincing lie than an axis pyqtgraph labelled itself.
    """
    figure, axes = plt.subplots()
    axes.plot([0, 1], [0, 1])
    axes.set_xticks([0.0, 1.0])
    assert _plain_text(r"$\frac{a}{b}$")[1] is False
    axes.set_xticklabels([r"$\frac{a}{b}$", "plate1"])
    axes.set_yticks([0.0, 1.0])
    axes.set_yticklabels(["row A", "row B"])
    figure.canvas.draw()

    widget, report = build_scene(figure)
    try:
        plot = widget.getItem(0, 0)
        assert not plot.getAxis("bottom")._tickLevels, (
            "a mathtext tick label was carried across raw")
        carried = plot.getAxis("left")._tickLevels
        assert {"row A", "row B"} <= {label for _, label in carried[0]}
    finally:
        widget.deleteLater()


# ---------------------------------------------------------------------------
# a colour bar
# ---------------------------------------------------------------------------

def test_a_colour_bar_whose_axis_refuses_is_still_a_ramp(qt_ready,
                                                         closed_figures):
    """The bar is the point; the size of its tick type is not."""
    figure, axes = plt.subplots()
    image = axes.imshow(np.arange(9.0).reshape(3, 3))
    bar = figure.colorbar(image, ax=axes)
    figure.canvas.draw()

    asked = []

    def _refuse(*_args, **_kwargs):
        asked.append("ticklabels")
        raise RuntimeError("this axis is being removed")

    bar.ax.yaxis.get_ticklabels = _refuse

    widget, report = build_scene(figure)
    try:
        assert asked, "the colour bar's axis was never asked"
        assert report.complete, report.reason()
        assert report.axes == 2 and report.items >= 2
    finally:
        widget.deleteLater()


# ---------------------------------------------------------------------------
# an artist the translator knows about but does not draw
# ---------------------------------------------------------------------------

def test_an_inset_axes_is_known_chrome_rather_than_a_missing_artist(
        qt_ready, closed_figures):
    """An Axes inside an Axes is on the carried list and is not an item.

    It must not reach `report.missing`: naming it would send every panel with
    an inset back to matplotlib for an artist that is drawn as its own plot.
    """
    figure, axes = plt.subplots()
    axes.plot([0, 1], [0, 1])
    inset = axes.inset_axes([0.6, 0.1, 0.3, 0.3])
    inset.plot([0, 1], [1, 0])
    figure.canvas.draw()
    assert type(inset).__name__ in sc.CARRIED

    widget, report = build_scene(figure)
    try:
        assert report.complete, report.reason()
        assert report.missing == []
    finally:
        widget.deleteLater()


# ---------------------------------------------------------------------------
# a legend
# ---------------------------------------------------------------------------

def test_a_legend_entry_with_no_text_is_skipped_not_blanked(qt_ready,
                                                            closed_figures):
    """An unlabelled handle costs a row of empty legend, so it is dropped."""
    figure, axes = plt.subplots()
    first, = axes.plot([0, 1], [0, 1], color="#AA3333")
    second, = axes.plot([0, 1], [1, 0], color="#3333AA")
    legend = axes.legend(handles=[first, second], labels=["", "named"])
    figure.canvas.draw()
    assert [text.get_text() for text in legend.get_texts()] == ["", "named"]

    widget, report = build_scene(figure)
    try:
        item = [child for child in widget.getItem(0, 0).getViewBox().childItems()
                if type(child).__name__ == "LegendItem"]
        assert len(item) == 1
        assert len(item[0].items) == 1, "the unlabelled handle got a row"
    finally:
        widget.deleteLater()


# ---------------------------------------------------------------------------
# laying the scene out
# ---------------------------------------------------------------------------

def test_a_layout_with_no_application_does_not_ask_one_to_run(qt_ready,
                                                              monkeypatch):
    """`_lay_out` runs under a headless export as well as under a GUI.

    With an application it spends the event loop to make the paint happen;
    with none there is nothing to spend, and calling `processEvents` on the
    class rather than an instance would raise inside the one function that
    must not.
    """
    spent = []

    class _StubApplication:
        _instance = None

        @classmethod
        def instance(cls):
            return cls._instance

        @staticmethod
        def processEvents():
            spent.append("processEvents")

    class _Widget:
        def __init__(self):
            self.activated = 0
            self.grabbed = 0
            self.ci = self

        @property
        def layout(self):
            return self

        def activate(self):
            self.activated += 1

        def grab(self):
            self.grabbed += 1

    monkeypatch.setattr("PySide6.QtWidgets.QApplication", _StubApplication)
    widget = _Widget()
    sc._lay_out(widget)
    assert (widget.activated, widget.grabbed) == (2, 1)
    assert spent == [], "an event loop was spent where there is no application"

    _StubApplication._instance = _StubApplication
    widget = _Widget()
    sc._lay_out(widget)
    assert (widget.activated, widget.grabbed) == (2, 1)
    assert spent == ["processEvents"]


# ---------------------------------------------------------------------------
# tidying up after a render
# ---------------------------------------------------------------------------

def test_a_widget_that_will_not_be_deleted_does_not_lose_the_file(
        qt_ready, tmp_path, monkeypatch, closed_figures):
    """The file is already on disk when the widget is dropped.

    Raising out of the `finally` would replace a written figure with a
    traceback -- the exact trade `render_figure` promises never to make.
    """
    written = tmp_path / "panel.png"
    written.write_bytes(b"not really a png")
    refused = []

    class _Widget:
        def deleteLater(self):
            refused.append("deleteLater")
            raise RuntimeError("this widget has gone")

    monkeypatch.setattr(sc, "build_scene",
                        lambda figure, **_kwargs: (_Widget(), SceneReport()))
    monkeypatch.setattr(sc, "export_scene",
                        lambda widget, path: str(written))

    path, report = sc.render_figure(None, str(written), announce=False)
    assert refused == ["deleteLater"]
    assert path == str(written)
    assert report.complete
