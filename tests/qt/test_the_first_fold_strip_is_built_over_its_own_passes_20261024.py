"""The first fold page strip on a shown host is built over separate event-loop passes.

Lesson 81's module_ops froze in ``host_pages`` under ``FoldOpener.open``: the
strip was made, the host body moved onto it and its tab marked, all in one
pass. The strip is now made first, the body moves on the next pass, and the
tab's close mark comes on the pass after; a ``host_pages`` call made in
between finishes the same strip rather than making a second. The strip still
takes the body's visibility. Counted, not timed.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import (QApplication, QTabBar, QTabWidget,  # noqa: E402
                               QVBoxLayout, QWidget)

from spacr.qt import gil_priority, theme  # noqa: E402
from spacr.qt.screens import map_barcodes  # noqa: E402

pytestmark = pytest.mark.qt


def _host(qtbot, shown=True):
    host = QWidget()
    layout = QVBoxLayout(host)
    layout.addWidget(QWidget(host), 0)
    body = QWidget(host)
    layout.addWidget(body, 1)
    qtbot.addWidget(host)
    host.resize(400, 300)
    if shown:
        host.show()
        qtbot.waitExposed(host)
    return host, body


def _strips(host):
    return [child for child in host.findChildren(QTabWidget)
            if child.objectName() == map_barcodes.PAGES_NAME]


def test_the_strip_the_body_and_the_page_each_get_a_pass(qtbot, monkeypatch):
    host, body = _host(qtbot)
    seen = []
    real = map_barcodes._breathe

    def breathe():
        pending = getattr(host, "_fold_pages_pending", None)
        seen.append({
            "strip": pending is not None or
            getattr(host, "_fold_pages", None) is not None,
            "body_moved": body.parentWidget() is not host,
            "laid_out": getattr(host, "_fold_pages", None) is not None,
            "imports_held": gil_priority.background_imports_held(),
        })
        real()

    monkeypatch.setattr(map_barcodes, "_breathe", breathe)
    built = []

    def build(_window):
        built.append(len(seen))
        return QWidget()

    map_barcodes.FoldOpener(host, "probe", build).open()

    assert seen[0] == {"strip": True, "body_moved": False, "laid_out": False,
                       "imports_held": True}
    assert seen[1] == {"strip": True, "body_moved": True, "laid_out": True,
                       "imports_held": True}
    assert built == [2]
    assert not gil_priority.background_imports_held()
    pages = host._fold_pages
    assert pages.widget(0) is body and pages.count() == 2
    assert pages.isVisible()
    assert _strips(host) == [pages]


def test_a_host_pages_call_between_the_passes_finishes_the_same_strip(
        qtbot, monkeypatch):
    host, body = _host(qtbot)
    asked = []
    real = map_barcodes._breathe

    def breathe():
        if not asked:
            asked.append(map_barcodes.host_pages(host))
        real()

    monkeypatch.setattr(map_barcodes, "_breathe", breathe)
    map_barcodes.FoldOpener(host, "probe", lambda _w: QWidget()).open()
    assert asked[0] is host._fold_pages
    assert _strips(host) == [host._fold_pages]
    assert host._fold_pages.widget(0) is body
    assert getattr(host, "_fold_pages_pending", None) is None


def test_the_host_tab_gets_its_close_mark_hidden_a_pass_later(qtbot):
    host, _body = _host(qtbot)
    map_barcodes.FoldOpener(host, "probe", lambda _w: QWidget()).open()
    QApplication.processEvents()
    bar = host._fold_pages.tabBar()
    buttons = [bar.tabButton(0, side)
               for side in (QTabBar.RightSide, QTabBar.LeftSide)]
    buttons = [button for button in buttons if button is not None]
    assert buttons
    assert all(theme.is_close_mark(button) and button.isHidden()
               for button in buttons)


def test_a_strip_made_over_a_hidden_body_stays_hidden(qtbot):
    host, body = _host(qtbot)
    body.hide()
    pages = map_barcodes._host_pages_in_steps(host)
    assert pages is host._fold_pages and pages.widget(0) is body
    assert pages.isHidden()
