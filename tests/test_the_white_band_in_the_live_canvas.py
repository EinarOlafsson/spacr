"""The white band inside the live canvas is the x-axis, and always was.

Follow-up to c645a753, which made the panel, the stack, the canvas host, the
thumbnail strip and all three surfaces of the QGraphicsView see-through, and
recorded one thing as NOT fixed:

    "a white band remains inside the live matplotlib canvas. It is not the
     figure: it survives setting the axes patch invisible and survives
     forcing it red."

MEASURED OVER THE WHOLE CANVAS RATHER THAN ONE SCANLINE, THERE IS NO BAND.
97.6% of the canvas rectangle shows the ground through, and what does not is
the figure's own ink. The reading that started this was taken at a single y,
and that y landed exactly on the bottom spine of the axes:

    x=200  ground      left of the axes
    x=290  white   |
    x=470  white   |--  the axes' bottom spine, ONE PIXEL TALL
    x=740  white   |
    x=830  ground       right of the axes

which is why x=200 and x=830 came back transparent: they are outside the
AXES' horizontal extent, not outside a band. Down the same column there are
exactly two white rows in 552, at canvas-local y=66 and y=491 -- the top spine
and the bottom one. Hide the spines and both go; a figure with no axes leaves
the canvas 100% see-through.

It is white because THIS PANEL MAKES IT WHITE. `_style_figure_colors` sets
every spine, tick and label to the theme's foreground, so on a dark theme the
axis line is white by design -- the same intention as the text colour control
added in the same commit. And the two experiments read as ruling out the
figure both leave the spine alone: `ax.patch.set_visible(False)` and
`ax.set_facecolor('#ff0000')` change what is UNDER the axis line, and the
sample was taken ON it.

THE GROUND IS PAINTED UNDER THE PANEL, NOT BY A STYLESHEET ON ITS HOST. A
`setStyleSheet("background: magenta")` on a host CASCADES to every descendant,
so a canvas that paints its host's magenta and a canvas that lets the magenta
through are the same pixels, and a probe built that way reports transparency
that is not there. Ablating all six of the things `show_live_canvas` does to
the canvas reads as "no effect at all" through a cascading host and as
"the canvas goes fully opaque" through this one. The rig below fills a QImage
and renders the children onto it.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

pytestmark = pytest.mark.qt

#: Nothing in a figure or a theme is this colour, so every pixel still holding
#: it is ground that nothing painted over.
GROUND = (255, 0, 255)


def _over_the_ground(host):
    """``(height, width, 3)`` of ``host``'s children drawn onto the ground."""
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QColor, QImage, QPainter, QRegion
    from PySide6.QtWidgets import QWidget

    image = QImage(host.size(), QImage.Format_RGB32)
    image.fill(QColor(*GROUND))
    painter = QPainter(image)
    try:
        host.render(painter, QPoint(0, 0), QRegion(),
                    QWidget.RenderFlag.DrawChildren)
    finally:
        painter.end()
    raw = np.frombuffer(memoryview(image.constBits()), dtype=np.uint8,
                        count=image.sizeInBytes())
    raw = raw.reshape(image.height(), image.bytesPerLine() // 4,
                      4)[:, :image.width()]
    return raw[..., [2, 1, 0]]          # Format_RGB32 is BGRA in memory


class _Rendered:
    """A FigureQueue drawn over the ground, and the pixels it produced."""

    def __init__(self, pixels, host, queue):
        from PySide6.QtCore import QPoint

        self.pixels = pixels
        self.host = host
        self.queue = queue
        canvas = queue._canvas
        origin = canvas.mapTo(host, QPoint(0, 0))
        self.x, self.y = origin.x(), origin.y()
        self.width, self.height = canvas.width(), canvas.height()

    @property
    def canvas_pixels(self):
        return self.pixels[self.y:self.y + self.height,
                           self.x:self.x + self.width]

    @property
    def ground_fraction(self) -> float:
        area = self.canvas_pixels
        if not area.size:
            return 0.0
        return float((area == np.array(GROUND)).all(axis=-1).mean())

    def white_rows_at(self, column: int):
        """Canvas-local rows that are pure white in the given host column."""
        strip = self.pixels[self.y:self.y + self.height, column]
        return np.flatnonzero((strip == 255).all(axis=-1)).tolist()


def _render(qtbot, tweak=None, host_size=(900, 620)):
    from spacr.qt.widgets.figure_queue import FigureQueue
    from PySide6.QtWidgets import QVBoxLayout, QWidget

    host = QWidget()
    qtbot.addWidget(host)
    host.resize(*host_size)
    layout = QVBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    queue = FigureQueue()
    layout.addWidget(queue)

    figure = plt.figure(figsize=(4, 3))
    axis = figure.add_subplot(111)
    axis.plot([0, 1], [0, 1])
    queue.add_figure(figure)
    try:
        if tweak is not None:
            tweak(queue, figure, axis)
        assert queue._canvas is not None, "the live canvas never came up"
        queue._canvas.draw()
        host.show()
        qtbot.waitExposed(host)
        for _ in range(8):
            qtbot.wait(1)
        return _Rendered(_over_the_ground(host), host, queue)
    finally:
        plt.close(figure)


# --------------------------------------------------------------------------- #
#  There is no band
# --------------------------------------------------------------------------- #

def test_the_ground_shows_through_nearly_the_whole_canvas(qtbot):
    """The measurement a single scanline could not make.

    A band would take tens of per cent of the rectangle. What is actually
    there is the figure's ink, which is a couple.
    """
    shown = _render(qtbot)
    assert shown.ground_fraction > 0.95, (
        f"only {shown.ground_fraction:.1%} of the canvas is see-through; "
        f"something inside it is painting a background again")


def test_the_white_at_y_530_is_the_axis_line_and_goes_with_it(qtbot):
    """Two white rows in 552, and both of them are spines.

    This is the whole of the reported "band": hide the spines and there is
    nothing white left in the column at all.
    """
    with_spines = _render(qtbot)
    rows = with_spines.white_rows_at(470)
    assert 0 < len(rows) <= 4, (
        f"{len(rows)} white rows at x=470 is a band; two is an axis frame: "
        f"{rows[:10]}")

    def hide_spines(queue, figure, axis):
        for spine in axis.spines.values():
            spine.set_visible(False)

    without = _render(qtbot, hide_spines)
    assert without.white_rows_at(470) == [], (
        "white survived the spines being hidden, so it is not the axis")
    assert without.ground_fraction >= with_spines.ground_fraction


def test_the_transparent_pixels_are_where_the_axes_are_not(qtbot):
    """Why x=200 and x=830 read see-through and x=290..740 did not.

    Along the spine's own row the run of white starts and ends with the axes;
    five rows above it, every one of those same columns is ground. A band
    would still be there.
    """
    shown = _render(qtbot)
    rows = shown.white_rows_at(470)
    assert rows, "no spine found; the figure styling changed"
    spine = rows[-1] + shown.y

    row = shown.pixels[spine, shown.x:shown.x + shown.width]
    white = np.flatnonzero((row == 255).all(axis=-1))
    assert len(white) > 200, "the spine should run the width of the axes"

    above = shown.pixels[spine - 5, shown.x:shown.x + shown.width]
    still_ground = (above[white] == np.array(GROUND)).all(axis=-1)
    assert still_ground.mean() > 0.95, (
        "five rows above the axis line the same columns are not ground, so "
        "there really is something filling the axes rectangle")


def test_a_figure_with_no_axes_leaves_the_canvas_completely_transparent(qtbot):
    """Nothing between the figure and the ground paints anything at all."""

    def drop_the_axes(queue, figure, axis):
        figure.delaxes(axis)

    shown = _render(qtbot, drop_the_axes)
    assert shown.ground_fraction == 1.0, (
        f"an empty figure still covered {1 - shown.ground_fraction:.1%} of "
        f"the canvas")


def test_the_agg_buffer_is_already_transparent_where_there_is_no_ink(qtbot):
    """Upstream of Qt entirely: matplotlib hands over an RGBA buffer that is
    transparent except for the ink, so nothing downstream has to undo a
    background -- it only has to avoid adding one."""
    from spacr.qt.widgets.figure_queue import FigureQueue
    from PySide6.QtWidgets import QVBoxLayout, QWidget

    host = QWidget()
    qtbot.addWidget(host)
    host.resize(900, 620)
    layout = QVBoxLayout(host)
    queue = FigureQueue()
    layout.addWidget(queue)
    figure = plt.figure(figsize=(4, 3))
    figure.add_subplot(111).plot([0, 1], [0, 1])
    queue.add_figure(figure)
    try:
        assert queue._canvas is not None
        queue._canvas.draw()
        buffer = np.asarray(queue._canvas.buffer_rgba())
        opaque = float((buffer[..., 3] == 255).mean())
        assert opaque < 0.05, (
            f"{opaque:.1%} of the Agg buffer is opaque; the figure itself is "
            f"painting a background")
    finally:
        plt.close(figure)


# --------------------------------------------------------------------------- #
#  What a buried canvas looks like, so the guard above means something
# --------------------------------------------------------------------------- #

def test_a_stock_matplotlib_canvas_is_the_opaque_one(qtbot):
    """The same figure on a canvas nobody dressed comes out solid.

    `FigureCanvasQT.__init__` ends with `setPalette(QPalette(QColor("white")))`
    and `paintEvent` opens with `painter.eraseRect(rect)`, which fills with
    that palette's background brush before the RGBA buffer is blitted over it.
    That is the white rectangle -- and it is what `show_live_canvas` clears by
    zeroing the palette's Window and Base and by putting a transparent
    background in the widget's own stylesheet.

    Ablated in turn against this rig: either one of those two alone holds the
    canvas at 97.6% ground, and dropping BOTH puts it at 0.0%. The attribute
    flags are belt and braces; the palette is the mechanism.
    """
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from PySide6.QtWidgets import QVBoxLayout, QWidget

    host = QWidget()
    qtbot.addWidget(host)
    host.resize(900, 620)
    layout = QVBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    figure = plt.figure(figsize=(4, 3))
    figure.patch.set_facecolor("none")
    axis = figure.add_subplot(111)
    axis.patch.set_facecolor("none")
    axis.plot([0, 1], [0, 1])
    canvas = FigureCanvasQTAgg(figure)
    layout.addWidget(canvas)
    try:
        canvas.draw()
        host.show()
        qtbot.waitExposed(host)
        for _ in range(8):
            qtbot.wait(1)
        pixels = _over_the_ground(host)
        ground = (pixels == np.array(GROUND)).all(axis=-1).mean()
        assert ground < 0.05, (
            f"a stock canvas let {ground:.1%} of the ground through, so it is "
            f"no longer what show_live_canvas is undoing -- re-measure the "
            f"guard above before trusting it")
    finally:
        plt.close(figure)


def test_the_canvas_palette_is_the_thing_that_has_to_stay_cleared(qtbot):
    """Named so a change to it fails here rather than in a screenshot."""
    from spacr.qt.widgets.figure_queue import FigureQueue
    from PySide6.QtGui import QPalette
    from PySide6.QtWidgets import QVBoxLayout, QWidget

    host = QWidget()
    qtbot.addWidget(host)
    layout = QVBoxLayout(host)
    queue = FigureQueue()
    layout.addWidget(queue)
    figure = plt.figure(figsize=(4, 3))
    figure.add_subplot(111).plot([0, 1], [0, 1])
    queue.add_figure(figure)
    try:
        canvas = queue._canvas
        assert canvas is not None
        for role in (QPalette.Window, QPalette.Base):
            assert canvas.palette().color(role).alpha() == 0, (
                f"palette {role} is opaque again; eraseRect fills with it "
                f"every paint, before the figure is blitted on top")
    finally:
        plt.close(figure)
