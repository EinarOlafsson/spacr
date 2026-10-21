"""The first fold opened on a host no longer does its whole open in one freeze.

Lesson 43 (Illumination folded into Measure) froze the GUI for 0.35-0.4 s at the first open: the page
strip was created (the host body re-parented into it), the page was built, and it was put on show, all
without a turn of the event loop. The strip, the build and the show are separate steps now, with one
pass of the event loop (user input held back) between them.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QTimer                              # noqa: E402
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget  # noqa: E402

from spacr.qt.screens import map_barcodes                      # noqa: E402


def _host() -> QWidget:
    host = QWidget()
    layout = QVBoxLayout(host)
    layout.addWidget(QWidget(host), 0)           # masthead
    layout.addWidget(QWidget(host), 1)           # body: the stretch child becomes the host page
    return host


def test_page_strip_is_created_before_the_page_is_built_and_the_loop_runs_between(qtbot=None):
    app = QApplication.instance() or QApplication([])
    host = _host()
    order = []

    def build(_window):
        order.append("build")
        return QWidget()

    opener = map_barcodes.FoldOpener(host, "illumination", build)
    QTimer.singleShot(0, lambda: order.append("event loop pass"))
    opener.open()
    assert getattr(host, "_fold_pages", None) is not None
    assert order == ["event loop pass", "build"], (
        "the strip is made, the event loop runs, and only then is the page built")
    host.deleteLater()


def test_second_fold_does_not_recreate_the_strip():
    app = QApplication.instance() or QApplication([])
    host = _host()
    first = map_barcodes.FoldOpener(host, "a", lambda _w: QWidget())
    second = map_barcodes.FoldOpener(host, "b", lambda _w: QWidget())
    first.open()
    strip = host._fold_pages
    second.open()
    assert host._fold_pages is strip
    assert strip.count() == 3          # host page, first fold, second fold
    host.deleteLater()


def test_reparent_into_the_page_host_gets_its_own_event_loop_turn():
    app = QApplication.instance() or QApplication([])
    host = _host()
    order = []
    page = QWidget()
    opener = map_barcodes.FoldOpener(host, "a", lambda _w: page)
    opener.open()                                  # creates the strip and shows the first page
    second = QWidget()
    opener2 = map_barcodes.FoldOpener(host, "b", lambda _w: second)
    QTimer.singleShot(0, lambda: order.append("event loop pass"))
    pages = host._fold_pages
    original = pages.setCurrentIndex
    pages.setCurrentIndex = lambda i: (order.append("select"), original(i))[1]
    opener2.open()
    assert order[:2] == ["event loop pass", "select"], "the reparent is followed by an event loop pass before selecting"
    assert pages.indexOf(second) >= 0
    host.deleteLater()
