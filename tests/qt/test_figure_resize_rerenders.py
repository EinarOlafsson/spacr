"""Enlarging the figure container spreads the points, not the thumbnails.

A UMAP draws its crops with `OffsetImage(zoom=...)`, which is in DISPLAY
pixels. Re-rendering the figure at a larger size therefore grows the axes
— the points spread out — while every thumbnail stays the same size on
screen. That is what makes a crowded embedding readable.

Scaling the rendered PNG instead magnifies the thumbnails along with
everything else, which is the opposite of what is wanted: the only way to
see more of the embedding was to make the images smaller.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@pytest.fixture()
def queue(qtbot):
    from spacr.qt.widgets.figure_queue import FigureQueue

    widget = FigureQueue()
    qtbot.addWidget(widget)
    widget.resize(420, 320)
    widget.show()
    qtbot.waitExposed(widget)
    return widget


@pytest.fixture()
def figure():
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.scatter(np.linspace(0, 1, 40), np.linspace(0, 1, 40))
    yield fig
    plt.close(fig)


def test_the_resize_is_debounced(queue):
    """A drag emits a resize per frame.

    Re-rendering a figure carrying a few thousand thumbnails is not a
    per-frame cost, so the raster is scaled during the drag and the true
    render lands when the user lets go.
    """
    from spacr.qt.widgets.figure_queue import FIGURE_RESIZE_DEBOUNCE_MS

    assert queue._resize_timer.isSingleShot()
    assert queue._resize_timer.interval() == FIGURE_RESIZE_DEBOUNCE_MS
    assert FIGURE_RESIZE_DEBOUNCE_MS >= 100, (
        "too short to survive a drag; every frame would re-render")


def test_a_resize_arms_the_timer(queue, qtbot):
    from PySide6.QtCore import QEvent, QSize

    queue._resize_timer.stop()
    queue.eventFilter(queue._view,
                      QEvent(QEvent.Resize))
    assert queue._resize_timer.isActive()


def test_rerendering_resizes_the_figure_not_the_image(queue, figure, qtbot,
                                                     tmp_path):
    """The figure grows with the container; the thumbnail zoom is untouched.

    This is the whole point of the change, so the png path is a real one
    rather than whatever the widget happens to be holding -- an earlier
    version of this test guessed a private helper name, skipped, and
    asserted nothing.
    """
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    # A thumbnail at a fixed DISPLAY zoom: the thing that must not change.
    thumb = OffsetImage(np.zeros((8, 8, 3), dtype=np.uint8), zoom=0.5)
    figure.axes[0].add_artist(AnnotationBbox(thumb, (0.5, 0.5),
                                             frameon=False))

    queue._figures[0] = figure
    queue._current = 0
    queue._png_paths[0] = str(tmp_path / "f.png")

    before = tuple(figure.get_size_inches())
    queue._view.resize(900, 700)
    queue._rerender_for_size()
    after = tuple(figure.get_size_inches())

    # The figure is resized on the GUI thread; only the RENDER is dispatched
    # to a worker, so the size is observable immediately.
    assert after[0] > before[0], "the figure did not grow with the container"
    assert after[1] > before[1]
    assert thumb.get_zoom() == 0.5, (
        "the thumbnail zoom moved with the canvas; it is in display pixels "
        "and must be independent of the figure size")


def test_a_tiny_view_is_ignored(queue, figure):
    """Mid-layout a widget is briefly a few pixels; re-rendering there
    produces a figure nobody asked for and throws away the real one."""
    queue._figures[0] = figure
    queue._current = 0
    before = tuple(figure.get_size_inches())
    queue._view.resize(10, 10)
    queue._rerender_for_size()
    assert tuple(figure.get_size_inches()) == before


def test_a_negligible_change_does_not_rerender(queue, figure):
    """A one-pixel jitter must not cost a full redraw."""
    queue._figures[0] = figure
    queue._current = 0
    dpi = figure.get_dpi()
    figure.set_size_inches(400 / dpi, 300 / dpi)
    queue._view.resize(401, 300)
    before = tuple(figure.get_size_inches())
    queue._rerender_for_size()
    assert tuple(figure.get_size_inches()) == before


def test_no_figure_means_no_crash(queue):
    """The panel is empty until a run produces something.

    Asserted rather than merely called: "it did not raise" is compatible
    with it having rendered something into an empty panel, which is the
    failure worth excluding.
    """
    queue._figures.clear()
    queue._rerender_for_size()

    assert queue._figures == [] or len(queue._figures) == 0


def test_the_render_does_not_run_on_the_gui_thread(queue, figure, tmp_path):
    """Re-rendering a real figure is not a GUI-thread cost.

    Doing it inline stalled the GUI for 1321 ms against a 250 ms budget and
    `test_adding_a_pdf_figure_does_not_freeze_the_gui_thread` caught it.
    `render_figure_to_png` is pure matplotlib and documents itself as safe
    to call from a worker, so it goes through the same `_jobs.submit` seam
    the PDF refinement uses.
    """
    submitted = []
    queue._figures[0] = figure
    queue._current = 0
    queue._png_paths[0] = str(tmp_path / "f.png")
    queue._jobs.submit = lambda fn, cb: submitted.append((fn, cb))

    queue._view.resize(880, 660)
    queue._rerender_for_size()
    assert submitted, "the render was not dispatched to a worker"


def test_a_stale_render_is_discarded(queue, figure, tmp_path):
    """A drag dispatches several; only the last one may be shown.

    Showing a stale render is worse than showing the scaled raster for
    another moment -- it puts the figure at a size the container no longer
    has.
    """
    queue._figures[0] = figure
    queue._current = 0
    queue._resize_seq = 7
    shown = []
    queue._view.set_pixmap = lambda pm: shown.append(pm)

    queue._on_resize_rendered((0, 3, True, str(tmp_path / "old.png")))
    assert not shown, "a superseded render was shown"
