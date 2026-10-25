"""A field frame checks its geometry-watched widgets without building rects.

The application hub re-reads every geometry-watched widget around each
paint pass, so a move made inside a paint is still told. Each ``QRect`` or
``QPoint`` PySide hands back gives the interpreter lock up and waits to take
it back, and with a pipeline or a search thread running that was a switch
interval per widget: lesson 12's live search and lesson 56's rebuild froze
for 1-2 s in the hub's after-paint check. The check now reads ints only.
Counted, not timed.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QObject, QSize  # noqa: E402
from PySide6.QtGui import QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from spacr.qt import gil_priority  # noqa: E402

pytestmark = pytest.mark.qt

_VALUE_GETTERS = ("geometry", "pos", "size", "rect", "frameGeometry",
                  "normalGeometry", "childrenRect")


class _Recorder(QObject):
    def __init__(self):
        super().__init__()
        self.kinds = []

    def eventFilter(self, watched, event):                  # noqa: N802
        self.kinds.append(event.type())
        return False


def _count_value_reads(monkeypatch, watched):
    reads = []
    for name in _VALUE_GETTERS:
        original = getattr(QWidget, name)

        def counted(self, *args, _name=name, _original=original):
            if self in watched:
                reads.append(_name)
            return _original(self, *args)

        monkeypatch.setattr(QWidget, name, counted)
    return reads


def test_the_paint_pass_check_reads_ints_and_still_tells_a_resize(
        qtbot, monkeypatch):
    window = QWidget()
    window.resize(240, 160)

    class Grows(QWidget):
        def paintEvent(self, event):                       # noqa: N802
            QPainter(self).end()
            if self.width() < 120:
                self.resize(QSize(130, 60))

    child = Grows(window)
    child.setGeometry(10, 10, 100, 60)
    still = [QWidget(window) for _ in range(12)]
    for index, widget in enumerate(still):
        widget.setGeometry(4 * index, 100, 8, 8)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    recorder = _Recorder()
    for widget in [child, *still]:
        gil_priority._watch_object_events(widget, recorder, (QEvent.Resize,))
    hub = gil_priority._application_event_hub(QApplication.instance(),
                                              create=False)
    states = gil_priority._hub_geometry_before(hub)
    assert states
    for _widget, hidden, place in states:
        assert isinstance(hidden, bool)
        assert type(place) is tuple and all(type(v) is int for v in place)

    child.resize(100, 60)
    recorder.kinds.clear()
    reads = _count_value_reads(monkeypatch, [child, *still])
    child.update()
    window.repaint()
    qtbot.waitUntil(lambda: QEvent.Resize in recorder.kinds, timeout=3000)
    assert recorder.kinds.count(QEvent.Resize) == 1
    assert reads == []


def test_a_window_place_is_its_client_geometry_read_once_a_change(
        qtbot, monkeypatch):
    window = QWidget()
    window.setGeometry(40, 50, 300, 200)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    native = QWidget.geometry(window)
    expected = (native.x(), native.y(), native.width(), native.height())
    assert gil_priority._hub_widget_place(window) == expected
    reads = _count_value_reads(monkeypatch, [window])
    for _ in range(5):
        assert gil_priority._hub_widget_place(window) == expected
    assert reads == []
    window.resize(320, 210)
    assert gil_priority._hub_widget_place(window)[2:] == (320, 210)
    assert reads == ["geometry"]
