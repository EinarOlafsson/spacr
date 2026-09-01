"""The gallery does not steal the view, and both of its views zoom.

TWO DEFECTS, ONE SYMPTOM -- "the figure queue lags a bit and I can't zoom":

* every arriving figure jumped onto the screen, so a user reading figure 3 of
  a Cellpose run was thrown off it every few seconds. It is also most of the
  queue's cost while a run streams, because each arrival tore down the live
  canvas and built another whether or not anyone was looking;
* the raster view zooms on a plain wheel turn and the live matplotlib canvas
  did not. Which view a figure gets is invisible to the user -- it depends on
  whether its Figure is still within the live cap -- so scrolling zoomed or
  did nothing for reasons nothing on screen explained.
"""
from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spacr.qt.widgets.figure_queue import FigureQueue  # noqa: E402


@pytest.fixture
def queue(qapp):
    q = FigureQueue()
    yield q
    plt.close("all")


def _a_figure():
    fig = plt.figure()
    fig.add_subplot(111).plot([0, 1], [0, 1])
    return fig


def test_the_first_figure_is_shown(queue):
    """An empty gallery has nothing to be thrown off."""
    queue.add_figure(_a_figure())
    assert queue._current == 0


def test_the_view_follows_while_it_is_at_the_tail(queue):
    """Nobody has navigated, so following is what the user expects."""
    for _ in range(3):
        queue.add_figure(_a_figure())
    assert queue._current == 2


def test_a_new_figure_does_not_steal_the_view(queue):
    """The defect, stated. Reading figure 1 while a run streams must not be
    interrupted."""
    for _ in range(3):
        queue.add_figure(_a_figure())
    queue.show_index(0)

    queue.add_figure(_a_figure())

    assert queue._current == 0, "the newest figure jumped onto the screen"


def test_the_figure_still_arrives_in_the_list(queue):
    """Not shown is not the same as not there -- it must be clickable."""
    for _ in range(3):
        queue.add_figure(_a_figure())
    queue.show_index(0)
    queue.add_figure(_a_figure())

    assert queue.count() == 4
    assert queue._list.count() == 4


def test_clicking_it_still_works(queue):
    for _ in range(2):
        queue.add_figure(_a_figure())
    queue.show_index(0)
    queue.add_figure(_a_figure())
    queue.show_index(2)
    assert queue._current == 2


def test_returning_to_the_tail_resumes_following(queue):
    """Once the user is back on the newest figure they are following again."""
    for _ in range(3):
        queue.add_figure(_a_figure())
    queue.show_index(0)
    queue.add_figure(_a_figure())      # not followed
    queue.show_index(queue.count() - 1)

    queue.add_figure(_a_figure())

    assert queue._current == queue.count() - 1


class _Scroll:
    def __init__(self, axes, button, x, y):
        self.inaxes, self.button, self.xdata, self.ydata = axes, button, x, y


def test_scrolling_up_narrows_the_axes(queue):
    fig = _a_figure()
    axes = fig.axes[0]
    axes.set_xlim(0, 10)
    axes.set_ylim(0, 10)

    queue._on_canvas_scroll(_Scroll(axes, "up", 5, 5))

    left, right = axes.get_xlim()
    assert right - left < 10, "a wheel turn did not zoom in"


def test_scrolling_down_widens_the_axes(queue):
    fig = _a_figure()
    axes = fig.axes[0]
    axes.set_xlim(0, 10)

    queue._on_canvas_scroll(_Scroll(axes, "down", 5, 5))

    left, right = axes.get_xlim()
    assert right - left > 10


def test_the_point_under_the_cursor_stays_put(queue):
    """Zooming about the CENTRE would slide the panel being examined away,
    which is what makes a montage unusable."""
    fig = _a_figure()
    axes = fig.axes[0]
    axes.set_xlim(0, 10)
    axes.set_ylim(0, 10)

    queue._on_canvas_scroll(_Scroll(axes, "up", 2.0, 2.0))

    left, right = axes.get_xlim()
    # The cursor sat 20 % along the axis; it must still sit 20 % along it.
    assert (2.0 - left) / (right - left) == pytest.approx(0.2)


def test_an_inverted_axis_zooms_the_same_way(queue):
    """Every imshow panel has one, and Cellpose figures are made of them."""
    fig = _a_figure()
    axes = fig.axes[0]
    axes.set_ylim(10, 0)                      # inverted, as imshow leaves it

    queue._on_canvas_scroll(_Scroll(axes, "up", 5, 5))

    bottom, top = axes.get_ylim()
    assert bottom > top, "the inversion was lost"
    assert abs(bottom - top) < 10, "it did not zoom in"


def test_scrolling_off_the_axes_does_nothing(queue):
    """A figure's margins have no data coordinates to zoom about."""
    fig = _a_figure()
    axes = fig.axes[0]
    axes.set_xlim(0, 10)
    queue._on_canvas_scroll(_Scroll(None, "up", None, None))
    assert axes.get_xlim() == (0, 10)


def test_the_canvas_is_actually_connected():
    """A source check: every behaviour above passes with the canvas never
    wired to the handler."""
    from pathlib import Path

    import spacr.qt.widgets.figure_queue as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert 'canvas.mpl_connect("scroll_event", self._on_canvas_scroll)' in source


# ---------------------------------------------------------------------------
# Zooming a spilled figure renders its vector page finer
# ---------------------------------------------------------------------------
#
# A figure past the live cap is only a picture, and the picture is rasterised
# once at PDF_DISPLAY_MAX_PX. Zooming then magnified THAT -- so a user who
# zoomed into an old figure was looking at big pixels of a vector document,
# which is the one thing a vector page exists to prevent.


class _RecordingQueue:
    """Captures the render requests a zoom produces."""

    def __init__(self, queue, tmp_path):
        from spacr.qt.widgets.figure_queue import _sibling_pdf

        self.queue = queue
        self.requested = []
        png = tmp_path / "fig_00000.png"
        png.write_bytes(b"not really a png")
        _sibling_pdf(png).write_bytes(b"%PDF-1.4 pretend")
        queue._png_paths[0] = str(png)
        queue._current = 0
        queue._jobs = self

    def submit(self, work, _done):
        # Run the bound lambda far enough to learn the max_px it asked for.
        import spacr.qt.widgets.figure_queue as module

        seen = {}
        real = module.render_pdf_to_image
        module.render_pdf_to_image = lambda path, max_px=None: seen.setdefault(
            "max_px", max_px)
        try:
            work()
        finally:
            module.render_pdf_to_image = real
        self.requested.append(seen.get("max_px"))


@pytest.fixture
def spilled(queue, tmp_path):
    return _RecordingQueue(queue, tmp_path)


def test_zooming_in_asks_for_a_finer_page(spilled):
    from spacr.qt.widgets.figure_queue import PDF_DISPLAY_MAX_PX

    spilled.queue._pdf_render_px[0] = PDF_DISPLAY_MAX_PX
    spilled.queue._view.viewport().resize(1000, 800)

    spilled.queue._on_view_zoomed(8.0)

    assert spilled.requested, "zooming asked for no finer render at all"
    assert spilled.requested[-1] > PDF_DISPLAY_MAX_PX


def test_not_zoomed_in_asks_for_nothing(spilled):
    spilled.queue._on_view_zoomed(1.0)
    assert spilled.requested == []


def test_a_small_zoom_does_not_start_a_render(spilled):
    """Every wheel notch must not start a render, or they queue behind a user
    who is still turning the wheel."""
    from spacr.qt.widgets.figure_queue import PDF_DISPLAY_MAX_PX

    spilled.queue._pdf_render_px[0] = PDF_DISPLAY_MAX_PX
    spilled.queue._view.viewport().resize(1000, 800)
    spilled.queue._on_view_zoomed(1.1)
    assert spilled.requested == []


def test_the_request_is_capped(spilled):
    """A ceiling exists because the buffer is held for the figure on screen:
    8000 px is already ~256 MB as ARGB."""
    from spacr.qt.widgets.figure_queue import PDF_ZOOM_MAX_PX

    spilled.queue._view.viewport().resize(1000, 800)
    spilled.queue._on_view_zoomed(500.0)
    assert spilled.requested[-1] == PDF_ZOOM_MAX_PX


def test_it_stops_asking_once_it_is_as_fine_as_it_gets(spilled):
    from spacr.qt.widgets.figure_queue import PDF_ZOOM_MAX_PX

    spilled.queue._pdf_render_px[0] = PDF_ZOOM_MAX_PX
    spilled.queue._view.viewport().resize(1000, 800)
    spilled.queue._on_view_zoomed(500.0)
    assert spilled.requested == [], "it re-rendered a page already at the cap"


def test_no_vector_page_means_no_request(queue, tmp_path):
    """A figure written under the PNG preference has no page to refine."""
    png = tmp_path / "fig_00000.png"
    png.write_bytes(b"x")
    queue._png_paths[0] = str(png)
    queue._current = 0
    calls = []
    queue._jobs = type("J", (), {"submit": lambda _s, w, d: calls.append(w)})()
    queue._on_view_zoomed(8.0)
    assert calls == []


def test_the_view_is_actually_connected():
    from pathlib import Path

    import spacr.qt.widgets.figure_queue as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "self._view.zoom_changed.connect(self._on_view_zoomed)" in source
