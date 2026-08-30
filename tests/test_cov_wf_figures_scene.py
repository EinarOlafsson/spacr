"""The refusals and fall-backs inside the matplotlib-to-pyqtgraph translation.

``SceneReport``'s contract is that a scene is FAITHFUL or ABSENT, so what is
worth pinning is the edges: the artist with no paint on it, the label that
cannot be placed, the formula nested deeper than the stripper's loop, the font
Qt will not build, the axis that will not say where its ticks are. Each must
leave the picture either correct or unwritten, and none may raise into a run
whose fit is already finished.

Every figure here is a real matplotlib figure with real artists on it. The
exceptions are the ``QApplication`` this process must not start a second of and
the widget that refuses deletion -- neither of which a real object can do.
"""
from __future__ import annotations

import os

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

pytest.importorskip("PySide6")

from spacr.figures import scene as sc  # noqa: E402
from spacr.figures.scene import (  # noqa: E402
    SceneReport,
    _add_legend,
    _add_line,
    _add_line_collection,
    _add_path_collection,
    _add_rectangles,
    _add_text,
    _carry_ticks,
    _configure_axes,
    _lay_out,
    _Look,
    _plain_text,
    _reference_line,
    _translate_axes,
    _translate_colorbar,
    pyqtgraph_ready,
    render_figure,
)


@pytest.fixture(autouse=True)
def closed_figures():
    """Close whatever a test drew, whichever way it ends."""
    yield
    plt.close("all")


@pytest.fixture
def plot():
    """One live pyqtgraph plot item to translate artists into."""
    ready, why = pyqtgraph_ready()
    if not ready:
        pytest.skip(why)
    import pyqtgraph as pg

    widget = pg.GraphicsLayoutWidget()
    yield widget.addPlot()
    widget.deleteLater()


@pytest.fixture
def look():
    """The appearance the translation paints with, at a round 100 dpi."""
    return _Look(dpi=100.0)


def _refuse_the_font(monkeypatch):
    """Make every QFont this module builds fail, as a broken font cache does."""
    def _boom(*_args, **_keywords):
        raise RuntimeError("Qt refused this font")

    monkeypatch.setattr(sc, "_font", _boom)


class _RefusesASecondApplication:
    """A ``QApplication`` class reporting none living and allowing no new one.

    Qt permits one application per process and this one already has it, so the
    real class cannot be asked what happens on the very first call.
    """

    @staticmethod
    def instance():
        return None

    def __init__(self, argv):
        raise RuntimeError("a second application is not allowed")


def test_a_machine_with_no_display_is_told_to_paint_offscreen(monkeypatch):
    """A headless run gets the offscreen platform BEFORE Qt is started: without
    it the first ``QApplication`` aborts the whole process ("could not connect
    to display"), taking the finished fit with it rather than merely losing the
    pyqtgraph version of one panel. What follows is a reason, not a raise.
    """
    import PySide6.QtWidgets

    # The real call first: pyqtgraph MIRRORS `QApplication` into its own Qt
    # module when it is imported, so it has to be imported before the stand-in
    # is installed or every later test inherits it.
    started, why_not = pyqtgraph_ready()
    if not started:
        pytest.skip(why_not)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    monkeypatch.setattr(PySide6.QtWidgets, "QApplication",
                        _RefusesASecondApplication)

    ready, why = pyqtgraph_ready()

    assert os.environ["QT_QPA_PLATFORM"] == "offscreen"
    assert ready is False
    assert why == ("no QApplication could be started: "
                   "a second application is not allowed")


def test_a_reference_line_is_read_from_its_geometry_and_from_its_paint(
        plot, look):
    """A single point has no direction to be a reference line in, and
    ``axhline(color='none')`` is one the panel left invisible on purpose.

    Either mistake draws an ``InfiniteLine`` clean across the panel at a height
    nobody asked for -- a rule the matplotlib page does not have, which is the
    one thing the two renderers must never disagree about.
    """
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots()
    single = Line2D([0.5], [0.5], transform=axes.transAxes)
    axes.add_line(single)
    assert _reference_line(single, axes) is None
    assert _reference_line(axes.axhline(1.0), axes) == "h"

    assert _add_line(plot, axes.axhline(0.4, color="none"), axes, look) == 0
    assert _add_line(plot, axes.axhline(0.6, color="red"), axes, look) == 1
    assert len(plot.items) == 1


def test_a_line_with_no_stroke_still_draws_its_markers(plot, look):
    """A marker-only series carries no line colour and its points ARE the data:
    dropping them because the stroke resolved to nothing reports a complete
    translation of a panel with an empty axes in it.
    """
    figure, axes = plt.subplots()
    line, = axes.plot([0, 1], [0, 1], color="none", marker="o",
                      markerfacecolor="#AA3366")

    assert _add_line(plot, line, axes, look) == 1
    assert type(plot.items[-1]).__name__ == "ScatterPlotItem"


def test_a_scatter_size_array_is_recycled_over_all_of_its_points(plot, look):
    """matplotlib defaults to s=20 and recycles a short size array over the
    offsets, and so must this: a scatter that came out at diameter zero is an
    empty panel the report calls complete, and one that came out truncated is a
    volcano plot silently missing its last third.
    """
    figure, axes = plt.subplots()
    sizeless = axes.scatter([0, 1, 2], [0, 1, 2])
    sizeless.set_sizes([])
    assert _add_path_collection(plot, sizeless, look) == 1
    assert [spot.size() for spot in plot.items[-1].points()] == pytest.approx(
        [np.sqrt(20.0) * look.scale] * 3)

    short = axes.scatter([0, 1, 2], [3, 4, 5])
    short.set_sizes([10.0, 20.0])
    assert _add_path_collection(plot, short, look) == 1
    assert [spot.size() for spot in plot.items[-1].points()] == pytest.approx(
        [np.sqrt(10.0) * look.scale, np.sqrt(20.0) * look.scale,
         np.sqrt(10.0) * look.scale])


def test_an_open_scatter_records_no_colour_for_the_legibility_check(plot):
    """``facecolors='none'`` is the hollow marker the QC panels use and it puts
    no ink on the page to be read. Recording its non-colour fires the "will not
    read on paper" warning on every panel that draws hollow points, and a
    warning that fires on everything is one nobody reads.
    """
    figure, axes = plt.subplots()
    hollow = axes.scatter([0, 1], [0, 1], facecolors="none", edgecolors="blue")
    filled = axes.scatter([0, 1], [1, 0], color="#3366AA")

    hollow_look, filled_look = _Look(dpi=100.0), _Look(dpi=100.0)
    assert _add_path_collection(plot, hollow, hollow_look) == 1
    assert _add_path_collection(plot, filled, filled_look) == 1

    assert hollow_look.data_colours == []
    assert set(filled_look.data_colours) == {"#3366AA"}


def test_stems_with_no_colour_add_nothing(plot, look):
    """A stem plot drawn in an invisible colour has nothing to carry, and the
    item it would otherwise add takes pyqtgraph's default pen -- so the panel
    comes back with a black comb across it that the panel never drew.
    """
    figure, axes = plt.subplots()

    assert _add_line_collection(plot, axes.vlines([0.1, 0.2], 0, 1,
                                                  colors="none"), look) == 0
    assert _add_line_collection(plot, axes.vlines([0.1, 0.2], 0, 1,
                                                  colors="green"), look) == 1
    assert len(plot.items) == 1


def test_what_cannot_be_placed_in_data_coordinates_is_left_off(plot, look):
    """A backdrop and an annotation are both positioned in axes fraction, and a
    fraction that is not a number cannot be converted.

    Placing the bar anyway gives the whole ``BarGraphItem`` a NaN range and it
    paints nothing at all; ``TextItem.setPos`` takes NaN silently and the label
    then travels with the view instead of sitting where it was written.
    """
    from matplotlib.patches import Rectangle

    figure, axes = plt.subplots()
    report = SceneReport()
    lost = Rectangle((float("nan"), 0.2), 0.3, 0.3, transform=axes.transAxes)
    placed = Rectangle((0.1, 0.2), 0.3, 0.3, transform=axes.transAxes)
    axes.add_patch(lost)
    axes.add_patch(placed)
    assert _add_rectangles(plot, [lost], axes, look) == 0
    assert _add_rectangles(plot, [placed], axes, look) == 1

    nowhere = axes.text(float("nan"), 0.5, "lost", transform=axes.transAxes)
    somewhere = axes.text(0.5, 0.5, "placed", transform=axes.transAxes)
    assert _add_text(plot, nowhere, axes, look, report) == 0
    assert _add_text(plot, somewhere, axes, look, report) == 1
    assert report.missing == []


def test_a_formula_nested_deeper_than_the_loop_is_still_stripped():
    """Six passes of brace-stripping, then a final sweep, and the label comes out
    as the reader's characters either way. A label that keeps a brace is
    reported as mathtext nobody taught the translator, which sends an otherwise
    perfect panel back to matplotlib.
    """
    assert _plain_text("$" + "{" * 7 + "x" + "}" * 7 + "$") == ("x", True)
    assert _plain_text("${{y}}$") == ("y", True)


def test_a_transparent_label_box_draws_neither_fill_nor_border(plot, look):
    """``bbox=dict(facecolor='none', edgecolor='none')`` is padding, not a box:
    drawing the default brush for it puts a solid panel behind every annotation
    that only wanted a little room around its text.
    """
    from PySide6.QtCore import Qt

    figure, axes = plt.subplots()
    report = SceneReport()
    invisible = axes.text(0.5, 0.5, "padded", transform=axes.transAxes,
                          bbox=dict(facecolor="none", edgecolor="none"))
    boxed = axes.text(0.2, 0.2, "boxed", transform=axes.transAxes,
                      bbox=dict(facecolor="yellow", edgecolor="black"))

    assert _add_text(plot, invisible, axes, look, report) == 1
    plain_item = plot.items[-1]
    assert _add_text(plot, boxed, axes, look, report) == 1
    boxed_item = plot.items[-1]

    assert plain_item.fill.style() == Qt.NoBrush
    assert plain_item.border.style() == Qt.NoPen
    assert boxed_item.fill.style() == Qt.SolidPattern
    assert boxed_item.border.style() == Qt.SolidLine


def test_a_font_qt_will_not_build_costs_no_label_axes_or_colour_bar(
        plot, look, monkeypatch):
    """Type size is a preference; the text, the range and the ramp are the
    figure. A font that cannot be built -- a broken fontconfig cache is the
    usual cause -- must leave the annotation on the panel, the title and range
    on the axes and the colour bar keyed, rather than take a panel that is
    already half drawn back to matplotlib over its typography.
    """
    figure, axes = plt.subplots()
    axes.plot([0, 1, 2], [1, 4, 9])
    axes.set_title("Panel")
    axes.set_xlim(0, 2)
    axes.set_ylim(0, 10)
    annotation = axes.text(0.5, 0.5, "still here", transform=axes.transAxes)
    bar_figure, bar_axes = plt.subplots()
    image = bar_axes.imshow(np.arange(9).reshape(3, 3), cmap="viridis")
    bar = bar_figure.colorbar(image, ax=bar_axes)
    bar.ax.set_ylabel("counts")
    figure.canvas.draw()
    bar_figure.canvas.draw()
    report = SceneReport()
    _refuse_the_font(monkeypatch)

    assert _add_text(plot, annotation, axes, look, report) == 1
    assert plot.items[-1].textItem.toPlainText() == "still here"

    _configure_axes(plot, axes, look, report)
    assert plot.titleLabel.text == "Panel"
    assert plot.viewRange() == [pytest.approx([0.0, 2.0]),
                                pytest.approx([0.0, 10.0])]

    _translate_colorbar(plot, bar.ax, look, report)
    assert report.items == 1
    assert plot.getAxis("right").labelText == "counts"
    assert plot.viewRange()[1] == pytest.approx([0.0, 8.0])


class _WillNotSayWhereItsTicksAre:
    """An axis whose tick locations cannot be read."""

    def get_ticklocs(self):
        raise RuntimeError("this axis will not say where its ticks are")


class _AxesWithUnreadableTicks:
    """The two attributes :func:`_carry_ticks` reads off an Axes."""

    def __init__(self):
        self.xaxis = _WillNotSayWhereItsTicksAre()
        self.yaxis = _WillNotSayWhereItsTicksAre()


def test_ticks_that_cannot_be_carried_are_left_to_pyqtgraph(plot):
    """Carrying explicit ticks is an improvement, not a requirement.

    An axis that raises when asked, and a set of labels holding mathtext nobody
    taught the translator, both leave pyqtgraph to write its own -- carrying the
    unreadable ones put `$\\mathdefault{10^{-1}}$` down the side of the
    design-spectrum panel. An axis that DOES answer keeps the panel's own names.
    """
    _carry_ticks(plot, _AxesWithUnreadableTicks())
    assert plot.getAxis("bottom")._tickLevels is None

    figure, axes = plt.subplots()
    axes.plot([0, 1, 2], [1, 2, 3])
    axes.set_xticks([0, 1, 2])
    axes.set_xticklabels([r"$\frac{a}{b}$", "beta", "gamma"])
    figure.canvas.draw()
    _carry_ticks(plot, axes)
    assert plot.getAxis("bottom")._tickLevels is None

    axes.set_xticklabels(["alpha", "beta", "gamma"])
    figure.canvas.draw()
    _carry_ticks(plot, axes)
    assert [label for _, label in plot.getAxis("bottom")._tickLevels[0]] == [
        "alpha", "beta", "gamma"]


def test_an_inset_axes_is_not_reported_as_an_artist_nobody_taught(plot, look):
    """An inset is a CHILD of its host axes as well as a panel of its own, and
    it is translated as its own plot -- so meeting it again among the host's
    children must not mark the translation incomplete, which would send every
    page with an inset on it back to matplotlib for an artist that WAS carried.
    An artist genuinely nobody has taught the translator is still named.
    """
    figure, axes = plt.subplots()
    axes.plot([0, 1], [0, 1])
    axes.inset_axes([0.6, 0.6, 0.3, 0.3])
    figure.canvas.draw()
    report = SceneReport()

    _translate_axes(plot, axes, look, report)
    assert report.missing == []

    quiver_figure, quiver_axes = plt.subplots()
    quiver_axes.quiver([0, 1], [0, 1], [1, 1], [1, -1])
    quiver_figure.canvas.draw()
    _translate_axes(plot, quiver_axes, look, report)
    assert report.missing == ["Quiver"]


def test_a_legend_entry_with_no_text_is_left_out(plot, look):
    """An empty label is a spacer the panel used, not an entry to draw: a blank
    row in the key is a swatch beside nothing, which reads as a series whose
    name went missing. The entries that DO have text still come through.
    """
    import pyqtgraph as pg

    figure, axes = plt.subplots()
    first, = axes.plot([0, 1], [0, 1], color="#FF0000")
    second, = axes.plot([0, 1], [1, 0], color="#0000FF")
    legend = axes.legend(handles=[first, second], labels=["", "kept"])
    figure.canvas.draw()

    assert _add_legend(plot, legend, look) == 1

    drawn = [child for child in plot.getViewBox().childItems()
             if isinstance(child, pg.LegendItem)]
    assert len(drawn) == 1
    assert [label.text for _, label in drawn[0].items] == ["kept"]


class _RecordingLayout:
    """The ``ci.layout`` :func:`_lay_out` activates, which records that it was."""

    def __init__(self, calls):
        self.calls = calls

    def activate(self):
        self.calls.append("activate")


class _RecordingWidget:
    """A widget that records the layout pass :func:`_lay_out` asks it for."""

    def __init__(self):
        self.calls = []
        self.ci = type("_CI", (), {"layout": _RecordingLayout(self.calls)})()

    def grab(self):
        self.calls.append("grab")


def test_a_scene_is_laid_out_whether_or_not_an_application_is_running(
        monkeypatch):
    """The paint is what teaches an ``AxisItem`` how wide it needs to be, and it
    has to happen before anything measures the scene. With an application there
    are events to spend between the two layout passes; on a render started
    before any application exists there are not, and the layout must still be
    activated twice -- or the x label sits on the tick labels in the file.
    """
    import PySide6.QtWidgets

    class _NoApplication:
        @staticmethod
        def instance():
            return None

    monkeypatch.setattr(PySide6.QtWidgets, "QApplication", _NoApplication)
    applicationless = _RecordingWidget()
    _lay_out(applicationless)
    assert applicationless.calls == ["activate", "grab", "activate"]

    running = _RecordingWidget()

    class _RunningApplication:
        @staticmethod
        def instance():
            return _RunningApplication

        @staticmethod
        def processEvents():
            running.calls.append("processEvents")

    monkeypatch.setattr(PySide6.QtWidgets, "QApplication", _RunningApplication)
    _lay_out(running)
    assert running.calls == ["activate", "grab", "processEvents", "activate"]


class _RefusesToBeDeleted:
    """A widget whose ``deleteLater`` raises, as one Qt already took does."""

    def deleteLater(self):
        raise RuntimeError("this widget is already gone")


def test_a_widget_that_refuses_to_be_deleted_does_not_lose_the_figure(
        monkeypatch, tmp_path):
    """The file is written before the widget is cleaned up, and the clean-up is
    the least important thing in the function: letting a failed ``deleteLater``
    out of the ``finally`` replaces a written figure with an exception report
    and sends the caller off to draw a page over the one already on disk.
    """
    report = SceneReport()

    def _write(_widget, path):
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("scene")
        return str(path)

    monkeypatch.setattr(sc, "build_scene",
                        lambda figure, **options: (_RefusesToBeDeleted(),
                                                   report))
    monkeypatch.setattr(sc, "export_scene", _write)
    figure, axes = plt.subplots()

    written, got = render_figure(figure, str(tmp_path / "panel.png"),
                                 announce=False)

    assert got is report
    assert got.complete is True
    assert os.path.basename(written).startswith("panel.")
    assert os.path.exists(written)
