"""A window frame does not ask Python event filters about its paints.

Over the animated field every widget on screen is painted on every frame,
and Qt asked each widget's Python event filters about each paint: lesson
13's Regression screen re-entered Python 233 times per 3840x2160 frame, and
during a run each re-entry waits behind the pipeline's hold on the
interpreter lock. Filters that never act on a paint now watch through the
application hub, which a paint pass never reaches. Counted, not timed.
"""
from __future__ import annotations

import sys

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QObject, QSize  # noqa: E402
from PySide6.QtGui import QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from spacr.qt import gil_priority  # noqa: E402

pytestmark = pytest.mark.qt


class _Recorder(QObject):
    def __init__(self):
        super().__init__()
        self.kinds = []

    def eventFilter(self, watched, event):                  # noqa: N802
        self.kinds.append(event.type())
        return False


@pytest.fixture()
def shown(qtbot):
    window = QWidget()
    window.resize(240, 160)
    child = QWidget(window)
    child.setGeometry(10, 10, 100, 60)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    return window, child


def test_a_watched_widget_hears_its_kinds_and_no_paint(shown):
    window, child = shown
    recorder = _Recorder()
    gil_priority._watch_object_events(child, recorder, (QEvent.Enter,))

    QApplication.sendEvent(child, QEvent(QEvent.Enter))
    QApplication.sendEvent(child, QEvent(QEvent.Leave))
    assert recorder.kinds == [QEvent.Enter]

    recorder.kinds.clear()
    window.repaint()
    assert recorder.kinds == []

    gil_priority._stop_watching_object_events(child, recorder)
    QApplication.sendEvent(child, QEvent(QEvent.Enter))
    assert recorder.kinds == []


def test_a_resize_made_inside_a_paint_is_told_afterwards(qtbot):
    window = QWidget()
    window.resize(240, 160)

    class Grows(QWidget):
        def paintEvent(self, event):                       # noqa: N802
            QPainter(self).end()
            if self.width() < 120:
                self.resize(QSize(130, 60))

    child = Grows(window)
    child.setGeometry(10, 10, 100, 60)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    recorder = _Recorder()
    gil_priority._watch_object_events(child, recorder, (QEvent.Resize,))
    child.resize(100, 60)
    recorder.kinds.clear()
    child.update()
    window.repaint()
    qtbot.waitUntil(lambda: QEvent.Resize in recorder.kinds, timeout=3000)
    assert recorder.kinds.count(QEvent.Resize) == 1


def test_a_destroyed_widget_leaves_the_hub(shown):
    window, child = shown
    recorder = _Recorder()
    gil_priority._watch_object_events(child, recorder, (QEvent.Enter,))
    hub = gil_priority._application_event_hub(QApplication.instance(),
                                              create=False)
    assert recorder in gil_priority._hub_object_watchers(hub, child)
    from shiboken6 import delete
    key = len(hub._objects)
    delete(child)
    assert len(hub._objects) == key - 1


def test_a_paint_filter_is_still_installed_on_the_widget(shown):
    window, child = shown
    recorder = _Recorder()
    gil_priority._watch_object_events(child, recorder, (QEvent.Paint,))
    window.repaint()
    assert QEvent.Paint in recorder.kinds


def test_the_regression_screen_frame_asks_no_converted_filter(qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    screen.resize(1600, 1000)
    screen.show()
    qtbot.waitExposed(screen)
    qtbot.wait(200)
    sources = {}

    def profile(frame, event, arg):
        if event == "call" and frame.f_code.co_name in ("eventFilter",
                                                        "_observe"):
            sources.setdefault(frame.f_code.co_filename.rsplit("/", 1)[-1],
                               0)
            sources[frame.f_code.co_filename.rsplit("/", 1)[-1]] += 1

    sys.setprofile(profile)
    try:
        screen.repaint()
    finally:
        sys.setprofile(None)
    quiet = ("dock.py", "collapsible_splitter.py", "section.py", "drawer.py",
             "column_aligned_row.py", "app_screen.py", "foldable.py",
             "settings_model.py", "theme.py", "dnd.py", "chaining.py",
             "glass.py")
    assert {name: n for name, n in sources.items() if name in quiet} == {}
