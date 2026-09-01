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
