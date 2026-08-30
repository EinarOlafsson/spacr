"""``spacr.qt.screensaver``: opening it, and the black screen it falls back to.

Three behaviours are pinned here that the module's own docstring promises and
nothing else exercised:

* the backdrop is *optional*. When one cannot be built the screensaver is a
  plain black screen -- "which is still a screensaver and still closes on a
  key" -- rather than a window that failed to open, and the failure is logged
  rather than raised;
* it opens **on the screen the parent is on**, which is what makes
  Ctrl+Shift+F on a second monitor blank the monitor the user is looking at.
  A parent that is not on a screen yet must not stop it opening;
* ``paintEvent`` paints black. With ``WA_OpaquePaintEvent`` set, whatever
  Qt last left in the backing store shows through if it does not.
"""
from __future__ import annotations

import logging

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import QEvent, Qt                              # noqa: E402
from PySide6.QtGui import QColor, QImage, QKeyEvent                # noqa: E402
from PySide6.QtWidgets import QWidget                              # noqa: E402

from spacr.qt import screensaver as module                         # noqa: E402
from spacr.qt.screensaver import Screensaver, show_screensaver     # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture
def close_after(qapp):
    """Close every screensaver a test opened, tolerantly.

    NOT ``qtbot.addWidget``: ``Screensaver`` sets WA_DeleteOnClose on
    purpose, so pytest-qt's teardown close lands on a freed C++ object and
    blames the previous test.
    """
    opened = []
    yield opened.append
    for window in opened:
        try:
            if not window.isHidden():
                window.close()
        except RuntimeError:
            pass
    qapp.processEvents()


@pytest.fixture
def no_backdrop(monkeypatch):
    """Make ``_build_the_backdrop`` fail the way a missing GL context does.

    The settings read is the first thing the builder does, so refusing it
    exercises the same ``except`` the real failures land in.
    """
    from spacr.qt import preferences

    def _refuse():
        raise RuntimeError("no fractal settings on this machine")

    monkeypatch.setattr(preferences, "get_fractal_settings", _refuse)


def _key(key=Qt.Key.Key_A):
    return QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier)


# ---------------------------------------------------------------------------
# a backdrop that cannot be built
# ---------------------------------------------------------------------------

def test_a_backdrop_that_cannot_be_built_leaves_a_screensaver_behind(
        close_after, no_backdrop, caplog):
    """It is opened from a menu action; raising there loses the click.

    The comparison is made in one test because "no backdrop" only means
    something against a run where one *was* built: the same constructor,
    with the settings readable, does add a widget to the layout.
    """
    with caplog.at_level(logging.ERROR, logger="spacr.qt.screensaver"):
        blank = Screensaver()
    close_after(blank)

    assert blank._backdrop is None
    assert blank.layout().count() == 0, "nothing may be added to the layout"
    assert any("backdrop" in record.message for record in caplog.records), (
        "the failure has to reach the log, or it is invisible")
    # ...and it is still a screensaver: any key gives the screen back.
    blank.show()
    blank.keyPressEvent(_key())
    assert not blank.isVisible()


def test_a_backdrop_that_can_be_built_is_put_in_the_layout(close_after):
    """The other side of the fallback, so "empty layout" above means something."""
    built = Screensaver()
    close_after(built)

    if built._backdrop is None:            # no GL/software renderer here
        pytest.skip("this machine cannot build a fractal widget either")
    assert built.layout().count() == 1
    assert built.layout().itemAt(0).widget() is built._backdrop


def test_with_no_backdrop_the_window_paints_black(close_after, no_backdrop):
    """WA_OpaquePaintEvent means Qt does not clear: unpainted is stale pixels.

    Rendered onto a red image, so "black" is a change this test made and not
    the colour the image already was.
    """
    blank = Screensaver()
    close_after(blank)
    blank.resize(16, 16)
    canvas = QImage(16, 16, QImage.Format.Format_RGB32)
    canvas.fill(QColor(255, 0, 0))
    assert QColor(canvas.pixel(8, 8)) == QColor(255, 0, 0)

    blank.render(canvas)

    assert QColor(canvas.pixel(8, 8)) == QColor(0, 0, 0)
    assert QColor(canvas.pixel(0, 0)) == QColor(0, 0, 0)


# ---------------------------------------------------------------------------
# stopping a backdrop that has no `pause`
# ---------------------------------------------------------------------------

def test_a_backdrop_whose_pause_is_not_callable_is_left_alone(close_after):
    """``pause`` is looked up on whatever could be built, not on a type.

    A backdrop carrying a *value* called ``pause`` must be skipped rather
    than called -- and the window must still close, which is the point.
    """
    class _Attribute:
        pause = "the render thread stopped itself"

    saver = Screensaver()
    close_after(saver)
    saver._backdrop = _Attribute()

    saver.close()

    assert not saver.isVisible()
    assert _Attribute.pause == "the render thread stopped itself"


def test_a_backdrop_that_raises_from_pause_still_closes(close_after, caplog):
    """A screensaver that refuses to close is worse than a leaked thread."""
    class _Stuck:
        def pause(self):
            raise RuntimeError("the render thread is already gone")

    saver = Screensaver()
    close_after(saver)
    saver._backdrop = _Stuck()

    with caplog.at_level(logging.DEBUG, logger="spacr.qt.screensaver"):
        saver.close()

    assert not saver.isVisible()
    assert any("pause" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# opening it
# ---------------------------------------------------------------------------

class _Recording(Screensaver):
    """A screensaver that remembers the geometry it was given.

    ``show_screensaver`` builds its own window, so the only way to see which
    geometry it chose -- before ``showFullScreen`` overwrites it with the
    same numbers on every platform, offscreen included -- is to record the
    call. ``setGeometry`` is not a C++ virtual, so this override sees the
    module's own call and nothing Qt makes internally.
    """

    calls: list = []

    def setGeometry(self, *args):
        type(self).calls.append(args)
        super().setGeometry(*args)


@pytest.fixture
def recording(monkeypatch):
    _Recording.calls = []
    monkeypatch.setattr(module, "Screensaver", _Recording)
    return _Recording


def test_it_opens_on_the_screen_the_parent_is_on(qtbot, close_after, recording):
    """Ctrl+Shift+F on the second monitor has to blank the second monitor."""
    parent = QWidget()
    qtbot.addWidget(parent)
    parent.show()
    qtbot.waitExposed(parent)

    saver = show_screensaver(parent)

    assert saver is not None
    close_after(saver)
    assert saver.isVisible()
    assert saver.hasFocus() or saver.focusWidget() is not None
    assert recording.calls == [(parent.screen().geometry(),)], (
        "the geometry must come from the parent's own screen")


def test_a_parent_not_on_a_screen_yet_still_gets_a_screensaver(close_after,
                                                               recording):
    """The guard is real: ``screen()`` is None before a window is created.

    Nothing is placed by hand in that case -- ``showFullScreen`` picks the
    screen -- but the window still opens, which is the behaviour that
    matters to whoever pressed the shortcut.
    """
    class _NotShown(QWidget):
        def screen(self):
            return None

    parent = _NotShown()

    saver = show_screensaver(parent)

    assert saver is not None
    close_after(saver)
    assert saver.isVisible()
    assert recording.calls == [], "no screen, so no geometry to take from it"


def test_with_no_parent_at_all_it_still_opens_full_screen(close_after,
                                                          recording):
    """It is also reachable with no window to open from -- a timer, a test."""
    saver = show_screensaver(None)

    assert saver is not None
    close_after(saver)
    assert saver.isVisible()
    assert saver.isFullScreen()
    assert recording.calls == []


def test_a_screensaver_that_cannot_be_opened_returns_none(monkeypatch, caplog):
    """It is started from a menu and from a timer; an exception has nowhere
    to go in either."""
    class _Refuses(Screensaver):
        def __init__(self, *args, **kwargs):
            raise RuntimeError("no window system here")

    monkeypatch.setattr(module, "Screensaver", _Refuses)

    with caplog.at_level(logging.ERROR, logger="spacr.qt.screensaver"):
        assert show_screensaver(None) is None

    assert any("screensaver" in record.message for record in caplog.records)
