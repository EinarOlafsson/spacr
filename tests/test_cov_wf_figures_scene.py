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


import matplotlib
import pytest

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

pytest.importorskip("PySide6")

from spacr.figures.scene import (  # noqa: E402
    SceneReport,
    _add_line,
    _carry_ticks,
    _Look,
    _reference_line,
    _translate_axes,
    pyqtgraph_ready,
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


def test_an_offset_annotation_is_anchored_and_keeps_its_label(plot, look):
    """Volcano labels use point offsets as well as a data-coordinate anchor."""
    figure, axes = plt.subplots()
    axes.scatter([0.2], [0.8])
    axes.annotate("EAF1 g2", (0.2, 0.8), xytext=(5, -12),
                  textcoords="offset points")
    report = SceneReport()

    _translate_axes(plot, axes, look, report)

    assert report.complete, report.reason()
    labels = [item.textItem.toPlainText() for item in plot.items
              if type(item).__name__ == "TextItem"]
    assert "EAF1 g2" in labels


class _WillNotSayWhereItsTicksAre:
    """An axis whose tick locations cannot be read."""

    def get_ticklocs(self):
        raise RuntimeError("this axis will not say where its ticks are")


class _AxesWithUnreadableTicks:
    """The two attributes :func:`_carry_ticks` reads off an Axes."""

    def __init__(self):
        self.xaxis = _WillNotSayWhereItsTicksAre()
        self.yaxis = _WillNotSayWhereItsTicksAre()


class _TickLabel:
    """The one ``Text`` operation :func:`_carry_ticks` needs."""

    def __init__(self, text):
        self._text = text

    def get_text(self):
        return self._text


class _TickAxis:
    """A deterministic categorical axis, including malformed pairs."""

    def __init__(self, locations, labels):
        self._locations = locations
        self._labels = labels

    def get_ticklocs(self):
        return self._locations

    def get_ticklabels(self):
        return [_TickLabel(text) for text in self._labels]


class _AxesWithTicks:
    """The x/y axis pair :func:`_carry_ticks` reads."""

    def __init__(self, xaxis):
        self.xaxis = xaxis
        self.yaxis = _TickAxis([], [])


def test_mismatched_tick_labels_are_not_carried(plot):
    """Locations and labels are an indivisible mapping.

    A formatter returning fewer labels than locations must leave the target
    axis untouched; zipping it would silently mislabel data.  A matched
    categorical pair remains the positive counterpart and is carried intact.
    """
    mismatch = _AxesWithTicks(_TickAxis([0.0, 1.0], ["plate 1"]))
    _carry_ticks(plot, mismatch)
    assert plot.getAxis("bottom")._tickLevels is None

    matched = _AxesWithTicks(_TickAxis([0.0, 1.0], ["plate 1", "plate 2"]))
    _carry_ticks(plot, matched)
    assert plot.getAxis("bottom")._tickLevels == [
        [(0.0, "plate 1"), (1.0, "plate 2")]
    ]


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


