"""``spacr.qt.widgets.ambient``'s three refusal paths.

The ambient backdrop is decoration: the animated wallpaper behind a page, and
the fractal a screen can install in place of it. Nothing in a run depends on
it, which is exactly why its error handling matters -- a decoration that
raises takes down the screen it was decorating, and the user loses a panel of
real work over a picture.

All three of these guards had never run.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import QEvent, QObject, QRect                   # noqa: E402
from PySide6.QtWidgets import QWidget                               # noqa: E402

from spacr.qt.widgets import ambient                                # noqa: E402

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# how big a buffer the screen justifies
# ---------------------------------------------------------------------------

def test_a_screen_that_cannot_be_measured_falls_back_to_the_cap(qtbot,
                                                                monkeypatch):
    """``screen_pixels`` sizes an offscreen buffer, so it must always answer.

    A headless plugin, a screen unplugged between the call and the reply, a Qt
    wrapper released underneath -- any of them raise here, and the fallback is
    the CAP rather than zero. Zero would allocate nothing and the backdrop
    would draw into an empty buffer; the cap is the size it would have used
    for a large display, which is the safe direction to be wrong in.
    """
    widget = QWidget()
    qtbot.addWidget(widget)

    class _Exploding:
        def screen(self):
            raise RuntimeError("wrapped C/C++ object has been deleted")

    assert ambient.screen_pixels(_Exploding()) == ambient.BUFFER_MAX_PIXELS


def test_a_nonsensical_screen_size_is_not_taken_as_a_ceiling(qtbot,
                                                             monkeypatch):
    """A headless plugin can report a screen of nothing.

    Believing it would shrink every buffer to a few pixels and the backdrop
    would be a smear. Anything below one minimum edge squared is treated as no
    answer at all.
    """
    class _Tiny:
        def size(self):
            from PySide6.QtCore import QSize
            return QSize(1, 1)

        def devicePixelRatio(self):
            return 1.0

    class _Host:
        def screen(self):
            return _Tiny()

    assert ambient.screen_pixels(_Host()) == ambient.BUFFER_MAX_PIXELS


def test_a_real_screen_is_believed(qtbot):
    """Otherwise the two fallbacks above would pass on a constant."""
    widget = QWidget()
    qtbot.addWidget(widget)

    pixels = ambient.screen_pixels(widget)

    assert pixels > 0


# ---------------------------------------------------------------------------
# retiring a fractal that will not go quietly
# ---------------------------------------------------------------------------

class _StubbornFractal(QWidget):
    """A fractal whose shutdown raises, as a dead render thread's would."""

    backend_name = "stub"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.asked = 0

    def shutdown(self):
        self.asked += 1
        raise RuntimeError("the render thread is already gone")


def test_a_fractal_that_will_not_shut_down_is_still_retired(qtbot):
    """The count is what the caller uses to decide it may install a new one.

    A fractal left in the tree because its shutdown raised would keep drawing
    over the one that replaces it -- two animations on one page, both fighting
    for the same GPU.
    """
    host = QWidget()
    qtbot.addWidget(host)
    stubborn = _StubbornFractal(host)

    retired = ambient._retire_fractals_on(host)

    assert stubborn.asked == 1, "shutdown was never attempted"
    assert retired == 1, "a fractal that raised was not counted as retired"


def test_a_host_that_cannot_be_searched_retires_nothing_and_says_so(qtbot):
    """``findChildren`` raises on a host Qt has already released.

    Returning zero is the honest answer -- nothing was retired -- and it must
    not be an exception, because this runs while a screen is being torn down.
    """
    class _Gone:
        def findChildren(self, *args, **kwargs):
            raise RuntimeError("wrapped C/C++ object has been deleted")

    assert ambient._retire_fractals_on(_Gone()) == 0


def test_a_widget_that_is_not_a_fractal_is_left_alone(qtbot):
    """``backend_name`` is the marker, and an ordinary child has none."""
    host = QWidget()
    qtbot.addWidget(host)
    QWidget(host)

    assert ambient._retire_fractals_on(host) == 0


# ---------------------------------------------------------------------------
# following the host's size
# ---------------------------------------------------------------------------

def test_the_backdrop_follows_a_resize(qtbot):
    """The filter's whole job, asserted before its failure path."""
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(320, 200)
    backdrop = QWidget(host)

    watcher = ambient._FractalTracksItsHost(backdrop, host)
    watcher.eventFilter(host, QEvent(QEvent.Type.Resize))

    assert backdrop.geometry() == host.rect()


def test_a_resize_delivered_to_a_dead_backdrop_is_swallowed(qtbot):
    """Qt delivers a queued resize after the widget behind it is gone.

    Letting the RuntimeError out of an event filter propagates into Qt's own
    event dispatch, which is where an exception becomes a crash rather than a
    traceback.
    """
    host = QWidget()
    qtbot.addWidget(host)

    class _Dead:
        def setGeometry(self, _rect):
            raise RuntimeError("wrapped C/C++ object has been deleted")

    watcher = ambient._FractalTracksItsHost(_Dead(), host)

    assert watcher.eventFilter(host, QEvent(QEvent.Type.Resize)) is False


def test_an_event_that_is_not_a_resize_is_ignored(qtbot):
    """The filter must not claim events it has no opinion about."""
    host = QWidget()
    qtbot.addWidget(host)
    backdrop = QWidget(host)
    before = backdrop.geometry()

    watcher = ambient._FractalTracksItsHost(backdrop, host)
    handled = watcher.eventFilter(host, QEvent(QEvent.Type.Show))

    assert handled is False
    assert backdrop.geometry() == before
