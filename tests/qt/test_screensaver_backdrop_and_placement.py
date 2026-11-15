"""The screensaver: its optional backdrop, its black fallback, where it opens.

* ``Screensaver`` builds one fractal backdrop at the user's settings; when it
  cannot, it is a black window that still closes on any key or click, and the
  failure is logged. Closing pauses whatever backdrop it has, and a backdrop
  whose ``pause`` is missing, not callable or raising never stops the close;
* ``show_screensaver`` opens it full screen on the parent's own screen, still
  opens when the parent has no screen or there is no parent, and answers
  ``None`` (with a log line) when it cannot be opened at all.
"""
from __future__ import annotations

import logging

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QImage, QKeyEvent, QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QWidget  # noqa: E402

from spacr.qt import screensaver as module  # noqa: E402
from spacr.qt.screensaver import Screensaver, show_screensaver  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture
def close_after(qapp):
    """Close every window a test opened, tolerating WA_DeleteOnClose."""
    opened = []
    yield opened.append
    for window in opened:
        try:
            if not window.isHidden():
                window.close()
        except RuntimeError:
            pass
    qapp.processEvents()


def _refuse_settings(patch):
    """Make the backdrop builder fail where a missing GL context does."""
    from spacr.qt import preferences

    def _refuse():
        raise RuntimeError("no fractal settings on this machine")

    patch.setattr(preferences, "get_fractal_settings", _refuse)


def _key():
    return QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_A,
                     Qt.KeyboardModifier.NoModifier)


def _click():
    return QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(2, 2),
                       QPointF(2, 2), Qt.MouseButton.LeftButton,
                       Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier)


def test_the_screensaver_uses_its_backdrop_or_falls_back_to_black(
        subtests, close_after, monkeypatch, caplog):
    """A built backdrop fills the layout; none is a black window that still closes."""
    from spacr.qt import preferences
    from spacr.qt.widgets import fractal_travel

    with subtests.test(msg="a backdrop built from the settings fills the layout"), \
            monkeypatch.context() as patch:
        values = preferences.get_fractal_settings()
        built = []

        class _Backdrop(QWidget):
            paused = 0

            def pause(self):
                type(self).paused += 1

        def create(settings, controls):
            built.append((settings, controls))
            return _Backdrop()

        patch.setattr(fractal_travel, "create_fractal_widget", create)
        saver = Screensaver()
        close_after(saver)
        assert len(built) == 1
        settings, controls = built[0]
        assert (settings.pattern, settings.backend, settings.quality,
                settings.scale, settings.supersampling) == (
            values["pattern"], values["backend"], values["quality"],
            values["scale"], values.get("supersampling", 2))
        assert (controls.speed, controls.dream, controls.variable_speed,
                controls.speed_min, controls.speed_max, controls.speed_period,
                controls.follow_pointer, controls.pointer_size,
                controls.pointer_strength, controls.zoom_rate) == (
            values["speed"], values["dream"], values["variable_speed"],
            values["speed_min"], values["speed_max"], values["speed_period"],
            bool(values["pointer_gravity"]), values["pointer_size"],
            values["pointer_strength"], values["zoom_rate"])
        assert saver.layout().count() == 1
        assert saver.layout().itemAt(0).widget() is saver._backdrop
        assert saver.windowTitle() == "spaCR"
        assert saver.testAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        saver.show()
        saver.close()
        assert _Backdrop.paused == 1
        assert not saver.isVisible()

    with subtests.test(msg="no backdrop is logged and leaves an empty layout"), \
            monkeypatch.context() as patch:
        _refuse_settings(patch)
        caplog.clear()
        with caplog.at_level(logging.ERROR, logger="spacr.qt.screensaver"):
            blank = Screensaver()
        close_after(blank)
        assert blank._backdrop is None
        assert blank.layout().count() == 0
        assert [r.getMessage() for r in caplog.records
                if r.name == "spacr.qt.screensaver"] == [
            "could not build the screensaver backdrop"]

    for case_id, event in [("any key", _key), ("any click", _click)]:
        with subtests.test(msg=f"{case_id} gives the screen back"), \
                monkeypatch.context() as patch:
            _refuse_settings(patch)
            blank = Screensaver()
            close_after(blank)
            blank.show()
            sent = event()
            if case_id == "any key":
                blank.keyPressEvent(sent)
            else:
                blank.mousePressEvent(sent)
            assert sent.isAccepted() is True
            assert not blank.isVisible()

    with subtests.test(msg="with no backdrop the window paints black"), \
            monkeypatch.context() as patch:
        _refuse_settings(patch)
        blank = Screensaver()
        close_after(blank)
        blank.resize(16, 16)
        canvas = QImage(16, 16, QImage.Format.Format_RGB32)
        canvas.fill(QColor(255, 0, 0))
        blank.render(canvas)
        assert {QColor(canvas.pixel(x, y)).name()
                for x in (0, 8, 15) for y in (0, 8, 15)} == {"#000000"}

    class _Attribute:
        pause = "the render thread stopped itself"

    class _Stuck:
        def pause(self):
            raise RuntimeError("the render thread is already gone")

    for case_id, backdrop, logged in [
            ("a pause that is not callable is left alone", _Attribute(), []),
            ("a pause that raises is logged", _Stuck(),
             ["could not pause the screensaver backdrop"]),
            ("no pause at all", object(), [])]:
        with subtests.test(msg=f"closing: {case_id}"), \
                monkeypatch.context() as patch:
            _refuse_settings(patch)
            saver = Screensaver()
            close_after(saver)
            saver._backdrop = backdrop
            saver.show()
            caplog.clear()
            with caplog.at_level(logging.DEBUG, logger="spacr.qt.screensaver"):
                saver.close()
            assert not saver.isVisible()
            assert [r.getMessage() for r in caplog.records
                    if r.name == "spacr.qt.screensaver"] == logged
            assert _Attribute.pause == "the render thread stopped itself"


class _Recording(Screensaver):
    """A screensaver that records the geometry ``show_screensaver`` gives it."""

    calls: list = []

    def setGeometry(self, *args):  # noqa: N802 - Qt's spelling
        type(self).calls.append(args)
        super().setGeometry(*args)


def test_the_screensaver_opens_full_screen_on_the_parents_screen(
        subtests, qtbot, close_after, monkeypatch, caplog):
    """Geometry comes from the parent's screen when there is one, else none."""
    from spacr.qt.hidpi import screen_for_widget

    monkeypatch.setattr(module, "Screensaver", _Recording)

    with subtests.test(msg="on the screen the parent is on"):
        _Recording.calls = []
        parent = QWidget()
        qtbot.addWidget(parent)
        parent.show()
        qtbot.waitExposed(parent)
        saver = show_screensaver(parent)
        close_after(saver)
        assert type(saver) is _Recording
        assert saver.isVisible() and saver.isFullScreen()
        assert saver._opened_from is parent
        assert _Recording.calls == [(screen_for_widget(parent).geometry(),)]

    with subtests.test(msg="a parent not on a screen yet"), \
            monkeypatch.context() as patch:
        _Recording.calls = []
        patch.setattr("spacr.qt.hidpi.screen_for_widget", lambda widget: None)
        parent = QWidget()
        close_after(parent)
        saver = show_screensaver(parent)
        close_after(saver)
        assert type(saver) is _Recording
        assert saver.isVisible() and saver.isFullScreen()
        assert _Recording.calls == []

    with subtests.test(msg="no parent at all"):
        _Recording.calls = []
        saver = show_screensaver(None)
        close_after(saver)
        assert type(saver) is _Recording
        assert saver.isVisible() and saver.isFullScreen()
        assert saver._opened_from is None
        assert _Recording.calls == []

    with subtests.test(msg="one that cannot be opened is None"), \
            monkeypatch.context() as patch:
        class _Refuses(Screensaver):
            def __init__(self, *args, **kwargs):
                raise RuntimeError("no window system here")

        patch.setattr(module, "Screensaver", _Refuses)
        caplog.clear()
        with caplog.at_level(logging.ERROR, logger="spacr.qt.screensaver"):
            assert show_screensaver(None) is None
        assert [r.getMessage() for r in caplog.records
                if r.name == "spacr.qt.screensaver"] == [
            "could not open the screensaver"]
