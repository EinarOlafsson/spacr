"""Table-driven tests of the matplotlib-to-pyqtgraph scene translation.

``spacr.figures.scene`` writes a generated figure as a pyqtgraph scene that
looks like the screen, and its contract is that the scene is FAITHFUL or
ABSENT: an artist with no paint, no data or no place is left off, an artist the
translator does not know makes the report incomplete, and nothing raises into a
run whose fit is already finished.  One general test per unit walks a case
table with pytest's ``subtests`` fixture: the whole-figure translation
(``build_scene``: item counts, completeness, carried ticks and legend rows),
each artist translator (lines, scatters, stems, texts, legends), the text and
coordinate helpers, the renderer choice, the Qt start-up, the layout pass and
the two writers (``render_figure`` / ``write_figure``).

Every figure is a real matplotlib figure drawn with real artists, because the
thing under test is a translation between two libraries.  The stand-ins are
limited to what a real object cannot do: a ``QApplication`` class this process
must not start a second of, and widgets or artists that raise mid-teardown.
"""
from __future__ import annotations

import os

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from spacr.figures import scene as sc  # noqa: E402
from spacr.figures.scene import SceneReport  # noqa: E402

MATHTEXT = "mathtext this translator does not know"


@pytest.fixture(autouse=True)
def _figures_closed_and_renderer_unpinned(monkeypatch):
    """Close every figure a case drew; keep ``SPACR_FIGURE_RENDERER`` out."""
    monkeypatch.delenv("SPACR_FIGURE_RENDERER", raising=False)
    yield
    plt.close("all")


@pytest.fixture
def qt_ready(qapp):
    """A live QApplication that pyqtgraph can paint with, or a skip."""
    ready, why = sc.pyqtgraph_ready()
    if not ready:
        pytest.skip(why)
    return qapp


@pytest.fixture
def plot(qt_ready):
    """One live pyqtgraph plot item to translate single artists into."""
    import pyqtgraph as pg

    widget = pg.GraphicsLayoutWidget()
    yield widget.addPlot()
    widget.deleteLater()


class _Collector:
    """The one ``PlotItem`` method the translators call, recording its items."""

    def __init__(self):
        self.items = []

    def addItem(self, item, **_ignored):  # noqa: N802 (Qt naming)
        self.items.append(item)


def _raise(message):
    """A callable that raises ``RuntimeError(message)`` whatever it is given."""
    def _refuse(*_args, **_kwargs):
        raise RuntimeError(message)
    return _refuse


def _tick_labels(widget, name, row=0, col=0):
    """The labels pyqtgraph was told to write on one axis, or None."""
    levels = widget.getItem(row, col).getAxis(name)._tickLevels
    return None if not levels else [label for _, label in levels[0]]


def _built(figure):
    """``build_scene`` on a drawn figure; the widget is deleted afterwards."""
    figure.canvas.draw()
    widget, report = sc.build_scene(figure)
    widget.deleteLater()
    return report


# ---------------------------------------------------------------------------
# build_scene: one figure per artist shape
# ---------------------------------------------------------------------------

def _reference_lines(ax):
    ax.plot([0, 1, 2], [1, 2, 3])
    ax.axhline(0.0, color="#888888", linestyle="--")
    ax.axvline(1.0, color="#888888", linestyle=":")


def _marker_only_line(ax):
    ax.plot([0, 1, 2], [1, 2, 3], linestyle="None", marker="o",
            markerfacecolor="#3355AA", markersize=6)


def _scatters(ax):
    ax.scatter([0, 1, 2], [1, 2, 3], s=[18, 36, 72], c="#AA3333")
    ax.scatter([], [])


def _polygons(ax):
    ax.add_collection(PolyCollection(
        [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)], [(0.0, 0.0), (1.0, 1.0)]],
        closed=False, facecolors=["#66AACC", "#66AACC"], alpha=0.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def _unfilled_polygon(ax):
    ax.add_collection(PolyCollection([[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]],
                                     facecolors="none"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def _chrome_and_annotations(ax):
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.text(0.05, 0.9, "n = 200", transform=ax.transAxes,
            bbox={"facecolor": "white", "edgecolor": "#333333"})
    ax.text(0.5, 0.5, "in data coordinates")
    ax.text(0.5, 0.4, "")
    ax.figure.suptitle("The whole page")


def _formula(ax):
    ax.plot([0, 1], [0, 1])
    ax.text(0.5, 0.5, r"$\frac{\partial f}{\partial x}$")


def _arrow(ax):
    ax.plot([0, 1], [0, 1])
    ax.annotate("here", xy=(0.5, 0.5), xytext=(0.7, 0.2),
                arrowprops={"arrowstyle": "->"})


def _legend(ax):
    ax.plot([0, 1], [0, 1], label="up", color="#3366AA")
    ax.plot([0, 1], [1, 0], label="down", color="#AA3366")
    ax.legend()


def _limited(draw):
    """Draw, then pin the view to the unit square."""
    def _draw(ax):
        draw(ax)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    return _draw


def _backdrop(ax):
    ax.plot([0, 1], [0, 1])
    ax.add_patch(Rectangle((0.1, 0.1), 0.8, 0.8, transform=ax.transAxes,
                           facecolor="#DDDDDD"))


def _inset(ax):
    ax.plot([0, 1], [0, 1])
    ax.inset_axes([0.6, 0.1, 0.3, 0.3]).plot([0, 1], [1, 0])


def _colour_bar_whose_axis_refuses(ax):
    image = ax.imshow(np.arange(9.0).reshape(3, 3))
    bar = ax.figure.colorbar(image, ax=ax)
    ax.figure.canvas.draw()
    bar.ax.yaxis.get_ticklabels = _raise("this axis is being removed")


# (case id, how the axes is drawn, axes, items, missing, notes)
SCENES = [
    ("a reference line is one infinite line each", _reference_lines,
     1, 3, [], []),
    ("a marker-only line becomes points", _marker_only_line, 1, 1, [], []),
    ("a line with no data adds nothing", lambda ax: ax.plot([], []),
     1, 0, [], []),
    ("a scatter is one item and an empty scatter none", _scatters,
     1, 1, [], []),
    ("forty stems are one item", lambda ax: ax.vlines(
        range(40), 0, np.linspace(0.1, 1.0, 40), color="#444444"),
     1, 1, [], []),
    ("a two-vertex path is not a polygon", _polygons, 1, 1, [], []),
    ("an unfilled polygon is not drawn", _unfilled_polygon, 1, 0, [], []),
    ("a confidence band is a polygon", lambda ax: ax.fill_between(
        np.linspace(0, 1, 20), np.linspace(0, 1, 20) - 0.1,
        np.linspace(0, 1, 20) + 0.1, color="#66AACC", alpha=0.4),
     1, 1, [], []),
    ("histogram bars are one item", lambda ax: ax.hist(
        np.random.default_rng(0).normal(size=200), bins=20,
        color="#8888CC", edgecolor="white"), 1, 1, [], []),
    ("axis labels are configuration and captions are items",
     _chrome_and_annotations, 1, 3, [], []),
    ("a formula it will not guess at stops the scene", _formula,
     1, 1, [MATHTEXT], []),
    ("an annotation arrow is named as dropped", _arrow, 1, 1,
     ["Annotation arrow"],
     ["The annotation arrow cannot yet be carried by pyqtgraph."]),
    ("an image keeps its colour map", lambda ax: ax.imshow(
        np.arange(64, dtype=float).reshape(8, 8), cmap="viridis"),
     1, 1, [], []),
    ("an image with no data is not drawn",
     lambda ax: ax.imshow(np.zeros((0, 0))), 1, 0, [], []),
    ("a legend is one item beside its two lines", _legend, 1, 3, [], []),
    ("an artist nobody taught the translator is named",
     lambda ax: ax.quiver([0, 1], [0, 1], [1, 1], [1, -1]),
     1, 0, ["Quiver"], []),
    ("a bar with no size is not drawn",
     _limited(lambda ax: ax.bar([0.0], [0.0], width=0.0)), 1, 0, [], []),
    ("a bar outside the view is clamped away",
     _limited(lambda ax: ax.bar([5.0], [1.0], width=0.4)), 1, 0, [], []),
    ("a backdrop in axes fraction is converted", _backdrop, 1, 2, [], []),
    ("a caption painted in none is not drawn", lambda ax: (
        ax.plot([0, 1], [0, 1]), ax.text(0.5, 0.5, "invisible",
                                         color="none")), 1, 1, [], []),
    ("an inset axes is carried chrome, not a missing artist", _inset,
     1, 1, [], []),
    ("a colour bar whose axis refuses is still a ramp",
     _colour_bar_whose_axis_refuses, 2, 2, [], []),
]


def test_build_scene_counts_what_each_artist_becomes(qt_ready, subtests):
    """Each artist shape becomes an exact number of items or a named refusal."""
    for case, draw, axes, items, missing, notes in SCENES:
        with subtests.test(msg=case):
            figure, ax = plt.subplots()
            draw(ax)
            report = _built(figure)
            assert (report.axes, report.items, report.missing) == (
                axes, items, missing)
            assert report.notes == notes
            assert report.complete is (not missing)
            plt.close(figure)


def _categorical(ax):
    ax.bar(["plate1", "plate2", "plate3"], [1, 2, 3])


def _categorical_refusing(ax):
    _categorical(ax)
    ax.xaxis.get_ticklabels = _raise("this axis is being removed")


def _log_axes(ax):
    ax.plot([1, 10, 100], [1e-3, 1e-2, 1e-1])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.text(0.5, 0.5, "corner", transform=ax.transAxes)


def _mathtext_ticks(ax):
    ax.plot([0, 1], [0, 1])
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels([r"$\frac{a}{b}$", "plate1"])
    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(["row A", "row B"])


def _unlabelled_legend_entry(ax):
    first, = ax.plot([0, 1], [0, 1], color="#AA3333")
    second, = ax.plot([0, 1], [1, 0], color="#3333AA")
    ax.legend(handles=[first, second], labels=["", "named"])


def _legend_rows(widget):
    return [[label.text for _, label in child.items]
            for child in widget.getItem(0, 0).getViewBox().childItems()
            if type(child).__name__ == "LegendItem"]


# (case id, how the axes is drawn, bottom tick labels, left tick labels,
#  legend rows, items)
CARRIED_CHROME = [
    ("a categorical axis keeps the names the panel wrote", _categorical,
     ["plate1", "plate2", "plate3"], None, [], 1),
    ("an axis that refuses its labels costs only the ticks",
     _categorical_refusing, None, None, [], 1),
    ("a log axis writes its powers as Unicode, not mathtext", _log_axes,
     ["10\u207b\u00b9", "10\u2070", "10\u00b9", "10\u00b2", "10\u00b3"],
     ["10\u207b\u2074", "10\u207b\u00b3", "10\u207b\u00b2",
      "10\u207b\u00b9", "10\u2070"], [], 2),
    ("one mathtext tick label sends the whole axis to pyqtgraph",
     _mathtext_ticks, None, ["row A", "row B"], [], 1),
    ("an unlabelled legend entry gets no row", _unlabelled_legend_entry,
     None, None, [["named"]], 3),
]


def test_build_scene_carries_tick_labels_and_legend_rows(qt_ready, subtests):
    """Explicit tick names and labelled legend rows are carried, nothing else."""
    for case, draw, bottom, left, legends, items in CARRIED_CHROME:
        with subtests.test(msg=case):
            figure, ax = plt.subplots()
            draw(ax)
            figure.canvas.draw()
            widget, report = sc.build_scene(figure)
            try:
                assert report.missing == []
                assert report.items == items
                assert _tick_labels(widget, "bottom") == bottom
                assert _tick_labels(widget, "left") == left
                assert _legend_rows(widget) == legends
            finally:
                widget.deleteLater()
                plt.close(figure)

    with subtests.test(msg="a page with no ground is painted transparent"):
        from PySide6.QtCore import Qt

        figure, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        figure.canvas.draw()
        for ground, style in ((None, Qt.BrushStyle.NoBrush),
                              ("#FFFFFF", Qt.BrushStyle.SolidPattern)):
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(sc._Look, "ground",
                              property(lambda self, g=ground: g))
                widget, report = sc.build_scene(figure)
            try:
                assert report.complete
                assert widget.backgroundBrush().style() == style
            finally:
                widget.deleteLater()


# ---------------------------------------------------------------------------
# the artist translators
# ---------------------------------------------------------------------------

def test_add_line_draws_reference_lines_curves_and_markers(qt_ready, subtests):
    """A Line2D becomes an infinite line, a curve, points, or nothing."""
    import pyqtgraph as pg

    figure, ax = plt.subplots()
    cases = [
        ("a reference line painted in none", ax.axhline(0.5, color="none"),
         []),
        ("a visible reference line", ax.axhline(0.6, color="#888888"),
         [pg.InfiniteLine]),
        ("a marker-only style spelt color='none'",
         ax.plot([0, 1, 2], [0, 1, 2], color="none", marker="o",
                 markerfacecolor="#AA3366")[0], [pg.ScatterPlotItem]),
        ("a painted line with markers",
         ax.plot([0, 1, 2], [2, 1, 0], color="#123456", marker="o")[0],
         [pg.PlotDataItem, pg.ScatterPlotItem]),
        ("a line with no stroke and no marker",
         ax.plot([0, 1], [0, 1], color="none")[0], []),
    ]
    for case, line, kinds in cases:
        with subtests.test(msg=case):
            collected = _Collector()
            look = sc._Look(None)
            assert sc._add_line(collected, line, ax, look) == len(kinds)
            assert [type(item) for item in collected.items] == kinds


def test_add_path_collection_sizes_brushes_and_records_colours(plot, subtests):
    """A scatter's area becomes a recycled diameter; paint is per point."""
    from PySide6.QtCore import Qt

    figure, ax = plt.subplots()
    look = sc._Look(dpi=100.0)
    root = np.sqrt

    def scatter(sizes=None, **style):
        collection = ax.scatter([0, 1, 2], [0, 1, 2], **style)
        if sizes is not None:
            collection.set_sizes(sizes)
        return collection

    size_cases = [
        ("sizes cleared take matplotlib's s=20", [], [root(20.0)] * 3),
        ("one size reaches every point", [80.0], [root(80.0)] * 3),
        ("a short size array is recycled", [16.0, 64.0], [4.0, 8.0, 4.0]),
        ("a full size array is used as given", [1.0, 4.0, 9.0],
         [1.0, 2.0, 3.0]),
    ]
    for case, sizes, diameters in size_cases:
        with subtests.test(msg=case):
            assert sc._add_path_collection(
                plot, scatter(sizes, c="#AA3333"), look) == 1
            drawn = [spot.size() for spot in plot.items[-1].points()]
            assert drawn == pytest.approx([d * look.scale for d in diameters])

    with subtests.test(msg="an empty scatter adds nothing"):
        count = len(plot.items)
        assert sc._add_path_collection(plot, ax.scatter([], []), look) == 0
        assert len(plot.items) == count

    with subtests.test(msg="a transparent point is unfilled, its neighbour not"):
        collected = _Collector()
        two = ax.scatter([0, 1], [0, 1], facecolors=[(1.0, 0.0, 0.0, 0.0),
                                                     (0.0, 0.0, 1.0, 1.0)])
        assert sc._add_path_collection(collected, two, sc._Look(None)) == 1
        brushes = list(collected.items[0].data["brush"])
        assert [b.style() for b in brushes] == [Qt.BrushStyle.NoBrush,
                                                Qt.BrushStyle.SolidPattern]
        assert brushes[1].color().alpha() == 255

    with subtests.test(msg="hollow points record no colour, filled ones do"):
        hollow_look, filled_look = sc._Look(dpi=100.0), sc._Look(dpi=100.0)
        hollow = ax.scatter([0, 1], [0, 1], facecolors="none",
                            edgecolors="blue")
        filled = ax.scatter([0, 1], [1, 0], color="#3366AA")
        assert sc._add_path_collection(plot, hollow, hollow_look) == 1
        assert sc._add_path_collection(plot, filled, filled_look) == 1
        assert hollow_look.data_colours == []
        assert filled_look.data_colours == ["#3366AA", "#3366AA"]


def test_add_line_collection_needs_ink_and_a_whole_segment(plot, subtests):
    """Stems become one batched item, or nothing when there is nothing to ink."""
    figure, ax = plt.subplots()
    cases = [
        ("stems painted in none", ax.vlines([0.1, 0.2], 0, 1, colors="none"),
         0, None),
        ("a collection painted none by hand",
         LineCollection([[(0.0, 0.0), (0.0, 1.0)]], colors="none"), 0, None),
        ("an empty collection", LineCollection([], colors="green"), 0, None),
        ("isolated one-point segments",
         LineCollection([[[0.0, 1.0]], [[2.0, 3.0]]], colors="green"), 0,
         None),
        ("two visible stems", ax.vlines([0.1, 0.2], 0, 1, colors="green"), 1,
         ([0.1, 0.1, 0.2, 0.2], [0.0, 1.0, 0.0, 1.0])),
        ("one whole segment",
         LineCollection([[[0.0, 1.0], [2.0, 3.0]]], colors="#444444"), 1,
         ([0.0, 2.0], [1.0, 3.0])),
    ]
    for case, collection, added, xy in cases:
        with subtests.test(msg=case):
            collected = _Collector()
            assert sc._add_line_collection(
                collected, collection, sc._Look(None)) == added
            assert len(collected.items) == added
            if xy is not None:
                curve = collected.items[0]
                assert list(curve.xData) == pytest.approx(xy[0])
                assert list(curve.yData) == pytest.approx(xy[1])


def test_add_text_places_boxes_and_survives_an_unreadable_font(
        plot, subtests, monkeypatch):
    """A caption is placed or left off; its box and font are niceties."""
    from PySide6.QtCore import Qt

    figure, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    in_axes = {"transform": ax.transAxes}
    cases = [
        ("an infinite position is nowhere",
         ax.text(float("inf"), 0.5, "lost", **in_axes), None),
        ("a NaN position is nowhere",
         ax.text(float("nan"), 0.5, "lost", **in_axes), None),
        ("a box spelt in none has neither fill nor border",
         ax.text(0.2, 0.5, "bare", bbox=dict(facecolor="none",
                                             edgecolor="none"), **in_axes),
         (Qt.BrushStyle.NoBrush, Qt.PenStyle.NoPen)),
        ("a painted box keeps its fill and border",
         ax.text(0.6, 0.5, "boxed", bbox=dict(facecolor="#EEEEEE",
                                              edgecolor="#333333"), **in_axes),
         (Qt.BrushStyle.SolidPattern, Qt.PenStyle.SolidLine)),
    ]
    for case, text, box in cases:
        with subtests.test(msg=case):
            report = SceneReport()
            collected = _Collector()
            added = sc._add_text(collected, text, ax, sc._Look(None), report)
            assert (added, report.missing) == (0 if box is None else 1, [])
            if box is not None:
                item = collected.items[0]
                assert (item.fill.style(), item.border.style()) == box
                assert item.toPlainText() == text.get_text()

    with subtests.test(msg="a font size that cannot be read keeps the words"):
        text = ax.text(0.5, 0.5, "a caption", **in_axes)
        asked = []

        def _refuse():
            asked.append("fontsize")
            raise RuntimeError("this artist is being removed")

        text.get_fontsize = _refuse
        collected = _Collector()
        assert sc._add_text(collected, text, ax, sc._Look(None),
                            SceneReport()) == 1
        assert asked == ["fontsize"]
        assert collected.items[0].toPlainText() == "a caption"


def test_a_font_qt_refuses_costs_no_label_range_or_colour_bar(
        plot, monkeypatch):
    """A QFont that cannot be built leaves text, title, range and ramp intact."""
    look = sc._Look(dpi=100.0)
    figure, ax = plt.subplots()
    ax.plot([0, 1, 2], [1, 4, 9])
    ax.set_title("Panel")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 10)
    annotation = ax.text(0.5, 0.5, "still here", transform=ax.transAxes)
    bar_figure, bar_ax = plt.subplots()
    bar = bar_figure.colorbar(
        bar_ax.imshow(np.arange(9).reshape(3, 3), cmap="viridis"), ax=bar_ax)
    bar.ax.set_ylabel("counts")
    figure.canvas.draw()
    bar_figure.canvas.draw()
    report = SceneReport()
    monkeypatch.setattr(sc, "_font", _raise("Qt refused this font"))

    assert sc._add_text(plot, annotation, ax, look, report) == 1
    assert plot.items[-1].textItem.toPlainText() == "still here"

    sc._configure_axes(plot, ax, look, report)
    assert plot.titleLabel.text == "Panel"
    assert plot.viewRange() == [pytest.approx([0.0, 2.0]),
                                pytest.approx([0.0, 10.0])]

    sc._translate_colorbar(plot, bar.ax, look, report)
    assert (report.items, report.missing) == (1, [])
    assert plot.getAxis("right").labelText == "counts"
    assert plot.viewRange()[1] == pytest.approx([0.0, 8.0])


class _Entry:
    """A legend text: its words and its ink."""

    def __init__(self, body):
        self.body = body

    def get_text(self):
        return self.body

    def get_color(self):
        return "#333333"


class _RefusingHandle:
    """A legend handle that raises for every colour it is asked for."""

    get_color = get_facecolor = get_edgecolor = staticmethod(
        _raise("this handle has been removed"))


class _Legend:
    """A finished legend: texts and the handles beside them."""

    def __init__(self, *labels, handle=None):
        self._texts = [_Entry(label) for label in labels]
        self.legend_handles = [handle or _RefusingHandle() for _ in labels]

    def get_texts(self):
        return self._texts


def test_add_legend_draws_one_key_of_labelled_entries(plot, subtests):
    """Only labelled entries get a row; a handle with no colour gets grey."""
    import pyqtgraph as pg

    figure, ax = plt.subplots()
    red, = ax.plot([0, 1], [0, 1], color="#FF0000")
    blue, = ax.plot([0, 1], [1, 0], color="#0000FF")
    real = ax.legend(handles=[red, blue], labels=["", "kept"])
    figure.canvas.draw()
    cases = [
        ("a legend with no entries", _Legend(), 0, None),
        ("a legend of blanks only", _Legend("", ""), 0, None),
        ("a real legend with one blank entry", real, 1, [("kept", "#0000ff")]),
        ("a handle that refuses every colour", _Legend("an entry"), 1,
         [("an entry", "#b4b4b4")]),
    ]
    for case, legend, added, rows in cases:
        with subtests.test(msg=case):
            before = [child for child in plot.getViewBox().childItems()
                      if isinstance(child, pg.LegendItem)]
            assert sc._add_legend(plot, legend, sc._Look(None)) == added
            made = [child for child in plot.getViewBox().childItems()
                    if isinstance(child, pg.LegendItem)
                    and child not in before]
            if rows is None:
                assert made == []
                continue
            assert len(made) == 1
            assert [(label.text,
                     sample.item.opts["pen"].color().name().lower())
                    for sample, label in made[0].items] == rows


# ---------------------------------------------------------------------------
# text and coordinates
# ---------------------------------------------------------------------------

PLAIN_TEXT = [
    ("a plain label", "plain label", ("plain label", True)),
    ("an empty label", "", ("", True)),
    ("two nested groups", "${{a}}$", ("a", True)),
    ("six nested groups", "$" + "{" * 6 + "a" + "}" * 6 + "$", ("a", True)),
    ("seven nested groups", "$" + "{" * 7 + "x" + "}" * 7 + "$", ("x", True)),
    ("a superscript", r"$x^2$", ("x²", True)),
    ("a known symbol", r"$\alpha \times 10$", ("α × 10", True)),
    ("a fraction is not guessed at", r"$\frac{\partial f}{\partial x}$",
     (r"\frac\partial f\partial x", False)),
]


def test_plain_text_translates_known_mathtext_and_refuses_the_rest(subtests):
    """Known mathtext becomes Unicode; anything else comes back not understood."""
    for case, raw, expected in PLAIN_TEXT:
        with subtests.test(msg=case):
            assert sc._plain_text(raw) == expected


def test_in_data_coordinates_places_or_refuses_a_point(subtests):
    """A position converts to data coordinates, keeps its own, or is nowhere."""
    figure, ax = plt.subplots()
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 10)
    unreadable = ax.text(0.5, 0.5, "a caption")
    unreadable.get_transform = _raise("this artist is being removed")
    log_figure, log_ax = plt.subplots()
    log_ax.set_xscale("log")
    log_ax.set_yscale("log")
    log_ax.set_xlim(1.0, 100.0)
    log_ax.set_ylim(0.1, 1000.0)
    cases = [
        ("data coordinates are kept", ax, ax.text(1.5, 7.0, "d"), (1.5, 7.0)),
        ("an unreadable transform keeps the text's own position", ax,
         unreadable, (0.5, 0.5)),
        ("axes fraction converts against the limits", ax,
         ax.text(0.5, 0.25, "f", transform=ax.transAxes), (1.0, 2.5)),
        ("an infinite fraction is nowhere", ax,
         ax.text(float("inf"), 0.5, "f", transform=ax.transAxes), None),
        ("a fraction point on linear axes", ax, sc._FractionPoint(0.5, 0.5),
         (1.0, 5.0)),
        ("a fraction point on log axes takes the geometric midpoint", log_ax,
         sc._FractionPoint(0.5, 0.5), (10.0, 10.0)),
        ("a NaN fraction point is nowhere", ax,
         sc._FractionPoint(float("nan"), 0.5), None),
    ]
    for case, axes, point, expected in cases:
        with subtests.test(msg=case):
            got = sc._in_data_coordinates(point, axes)
            if expected is None:
                assert got is None
            else:
                assert got == pytest.approx(expected)


# ---------------------------------------------------------------------------
# the report and the renderer choice
# ---------------------------------------------------------------------------

def test_scene_report_is_complete_only_without_missing_entries(subtests):
    """``reason`` counts each missing name once, sorted; empty when complete."""
    cases = [
        ("an empty report", {}, True, ""),
        ("notes and colours alone do not block",
         {"axes": 2, "items": 7, "data_colours": ["#3366AA"],
          "notes": ["fallback"]}, True, ""),
        ("one missing capability", {"missing": ["mathtext"]}, False,
         "pyqtgraph cannot yet carry mathtext x1"),
        ("repeats are counted, names sorted",
         {"missing": ["Quiver", "Quiver", "Arc"]}, False,
         "pyqtgraph cannot yet carry Arc x1, Quiver x2"),
    ]
    for case, values, complete, reason in cases:
        with subtests.test(msg=case):
            report = SceneReport(**values)
            assert (report.complete, report.reason()) == (complete, reason)
            for name, value in values.items():
                assert getattr(report, name) == value


GALLERY = "the gallery shows matplotlib figures here"
NO_QT = "pyqtgraph is unavailable here: no Qt"


def test_scene_renderer_answers_with_a_reason_for_every_fallback(
        monkeypatch, subtests):
    """The renderer comes with a reason that is empty only for pyqtgraph."""
    # (case id, force, environment, gallery block, pyqtgraph ready, answer)
    cases = [
        ("matplotlib asked for", "matplotlib", None, "", (True, ""),
         ("matplotlib", "matplotlib was asked for")),
        ("matplotlib from the environment", None, "matplotlib", "",
         (True, ""), ("matplotlib", "matplotlib was asked for")),
        ("a forced value that is no renderer means auto", "crayons", None,
         "", (True, ""), ("pyqtgraph", "")),
        ("auto with a gallery that cannot show it", "auto", None, GALLERY,
         (True, ""), ("matplotlib", GALLERY)),
        ("pyqtgraph asked for skips the gallery question", "pyqtgraph", None,
         GALLERY, (True, ""), ("pyqtgraph", "")),
        ("pyqtgraph asked for on a machine without it", "pyqtgraph", None,
         "", (False, NO_QT), ("matplotlib", NO_QT)),
        ("auto on a machine without it", None, None, "", (False, NO_QT),
         ("matplotlib", NO_QT)),
    ]
    for case, force, environment, blocked, ready, answer in cases:
        with subtests.test(msg=case), monkeypatch.context() as patch:
            if environment is None:
                patch.delenv("SPACR_FIGURE_RENDERER", raising=False)
            else:
                patch.setenv("SPACR_FIGURE_RENDERER", environment)
            patch.setattr(sc, "_the_gallery_could_not_show_it",
                          lambda blocked=blocked: blocked)
            patch.setattr(sc, "pyqtgraph_ready", lambda ready=ready: ready)
            assert sc.scene_renderer(force) == answer

    from spacr import figure_sink

    blocked = ("a figure sink is attached and no file sink is, so a rendered "
               "file would not reach the gallery (spacr/qt/bridge.py installs "
               "only set_sink)")
    for case, figures, files, answer in (
            ("no sink at all lets auto use pyqtgraph", None, None, ""),
            ("a figure sink without a file sink blocks auto", print, None,
             blocked),
            ("a figure sink and a file sink both reach the gallery", print,
             print, "")):
        with subtests.test(msg=case), monkeypatch.context() as patch:
            patch.setattr(figure_sink, "_sink", figures)
            patch.setattr(figure_sink, "_file_sink", files)
            assert sc._the_gallery_could_not_show_it() == answer


# ---------------------------------------------------------------------------
# starting Qt and laying the scene out
# ---------------------------------------------------------------------------

def test_pyqtgraph_ready_goes_offscreen_only_without_a_display(
        qt_ready, monkeypatch, subtests):
    """No display means the offscreen platform; a refused start is a reason."""
    import PySide6.QtWidgets

    started = []

    class _Starts:
        @staticmethod
        def instance():
            return None

        def __init__(self, argv):
            started.append(list(argv))

    class _Refuses:
        @staticmethod
        def instance():
            return None

        def __init__(self, argv):
            raise RuntimeError("a second application is not allowed")

    cases = [
        ("no display: offscreen, one application started", _Starts, None,
         (True, ""), "offscreen", [[]]),
        ("a display: the platform is left alone", _Starts, ":0",
         (True, ""), None, [[]]),
        ("an application that cannot start is a reason", _Refuses, None,
         (False, "no QApplication could be started: "
                 "a second application is not allowed"), "offscreen", []),
    ]
    for case, application, display, answer, platform, starts in cases:
        with subtests.test(msg=case):
            started.clear()
            monkeypatch.setattr(PySide6.QtWidgets, "QApplication",
                                application)
            monkeypatch.setattr(sc, "_APPLICATION", None)
            for name in ("DISPLAY", "WAYLAND_DISPLAY", "QT_QPA_PLATFORM"):
                monkeypatch.delenv(name, raising=False)
            if display:
                monkeypatch.setenv("DISPLAY", display)
            assert sc.pyqtgraph_ready() == answer
            assert os.environ.get("QT_QPA_PLATFORM") == platform
            assert started == starts
            assert isinstance(sc._APPLICATION, application) is bool(starts)


class _RecordingWidget:
    """A widget recording the layout pass ``_lay_out`` asks of it."""

    def __init__(self):
        self.calls = []
        layout = type("_Layout", (), {
            "activate": lambda _self: self.calls.append("activate")})()
        self.ci = type("_CI", (), {"layout": layout})()

    def grab(self):
        self.calls.append("grab")


class _RefusingWidget:
    """A widget gone mid-export: asking for its central item raises."""

    def __init__(self):
        self.calls = []

    @property
    def ci(self):
        self.calls.append("ci")
        raise RuntimeError("this widget has gone")


def test_lay_out_paints_once_and_never_raises(monkeypatch, subtests):
    """Activate, paint, spend events only if an application runs, activate."""
    import PySide6.QtWidgets

    def application(running, calls):
        class _Application:
            @staticmethod
            def instance():
                return _Application if running else None

            @staticmethod
            def processEvents():  # noqa: N802 (Qt naming)
                calls.append("processEvents")
        return _Application

    cases = [
        ("no application spends no events", False, _RecordingWidget,
         ["activate", "grab", "activate"]),
        ("a running application spends its events", True, _RecordingWidget,
         ["activate", "grab", "processEvents", "activate"]),
        ("a widget that refuses is asked once and is not fatal", True,
         _RefusingWidget, ["ci"]),
    ]
    for case, running, make, calls in cases:
        with subtests.test(msg=case):
            widget = make()
            monkeypatch.setattr(PySide6.QtWidgets, "QApplication",
                                application(running, widget.calls))
            sc._lay_out(widget)
            assert widget.calls == calls


# ---------------------------------------------------------------------------
# writing it out
# ---------------------------------------------------------------------------

class _UndeletableWidget:
    """A widget whose ``deleteLater`` raises, as one Qt already took does."""

    def __init__(self):
        self.asked = []

    def deleteLater(self):  # noqa: N802 (Qt naming)
        self.asked.append("deleteLater")
        raise RuntimeError("this widget has gone")


def _write_scene(_widget, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("scene")
    return str(path)


def test_render_figure_reports_every_failure_instead_of_raising(
        qt_ready, tmp_path, monkeypatch, subtests):
    """A failed render is ``(None, report)`` naming why; a written one a path."""
    undeletable = _UndeletableWidget()

    def explode(_figure, **_options):
        raise RuntimeError("the scene could not be laid out")

    # (case id, ready, build_scene, export_scene, written?, missing, last note)
    cases = [
        ("pyqtgraph unavailable", (False, "pyqtgraph is unavailable here"),
         None, None, False, ["pyqtgraph"], "pyqtgraph is unavailable here"),
        ("an exception inside the translation", (True, ""), explode, None,
         False, ["RuntimeError"],
         "RuntimeError: the scene could not be laid out"),
        ("an exporter that writes nothing", (True, ""),
         lambda figure, **_o: (None, SceneReport()),
         lambda widget, path: None, False,
         ["the exporter wrote nothing"], None),
        ("an incomplete translation is not exported", (True, ""),
         lambda figure, **_o: (None, SceneReport(missing=["Quiver"])),
         _raise("an incomplete scene was exported"), False, ["Quiver"], None),
        ("a widget that will not be deleted keeps the file", (True, ""),
         lambda figure, **_o: (undeletable, SceneReport()), _write_scene,
         True, [], None),
    ]
    figure, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    for case, ready, build, export, written, missing, note in cases:
        with subtests.test(msg=case), monkeypatch.context() as patch:
            patch.setattr(sc, "pyqtgraph_ready", lambda ready=ready: ready)
            if build is not None:
                patch.setattr(sc, "build_scene", build)
            if export is not None:
                patch.setattr(sc, "export_scene", export)
            path, report = sc.render_figure(
                figure, str(tmp_path / case.replace(" ", "_") / "panel.png"),
                fmt="png", announce=False)
            assert report.missing == missing
            assert (report.notes[-1] if report.notes else None) == note
            if written:
                assert os.path.basename(path) == "panel.png"
                assert open(path, encoding="utf-8").read() == "scene"
                assert undeletable.asked == ["deleteLater"]
            else:
                assert path is None


def test_write_figure_falls_back_to_matplotlib_and_says_why(
        qt_ready, tmp_path, monkeypatch, subtests):
    """The picture is always written; the reason is empty only for pyqtgraph."""
    def simple(ax):
        ax.plot([0, 1, 2], [1, 2, 3], color="#3366AA")
        ax.set_title("A panel")

    def formula(ax):
        ax.plot([0, 1], [0, 1])
        ax.text(0.5, 0.5, r"$\frac{a}{b}$")

    # (case id, draw, renderer, exporter writes nothing?, renderer, reason)
    cases = [
        ("a complete scene is written by pyqtgraph", simple, "pyqtgraph",
         False, "pyqtgraph", ""),
        ("an untranslatable formula writes the matplotlib page", formula,
         "pyqtgraph", False, "matplotlib",
         f"pyqtgraph cannot yet carry {MATHTEXT} x1"),
        ("an exporter that writes nothing is a fallback", simple, "pyqtgraph",
         True, "matplotlib",
         "pyqtgraph cannot yet carry the exporter wrote nothing x1"),
        ("matplotlib asked for, without a gallery", simple, "matplotlib",
         False, "matplotlib", "matplotlib was asked for"),
    ]
    for case, draw, asked, silent, renderer, why in cases:
        with subtests.test(msg=case), monkeypatch.context() as patch:
            if silent:
                patch.setattr(sc, "export_scene", lambda widget, path: None)
            figure, ax = plt.subplots()
            draw(ax)
            figure.canvas.draw()
            stem = tmp_path / case.replace(" ", "_") / "panel"
            stem.parent.mkdir()
            written, chosen, reason = sc.write_figure(
                figure, str(stem), fmt="png", renderer=asked, announce=False)
            assert (chosen, reason) == (renderer, why)
            assert os.path.basename(str(written)) == "panel.png"
            assert os.path.isfile(str(written))
