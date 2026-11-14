"""`a_popup_is_on_screen`: what the animated backdrop asks before it moves.

A menu is an activePopupWidget; a tooltip is NOT, so both questions are asked.
The helper is called from an animation tick and may never raise: an
unanswerable question is answered "no popup".
"""
from __future__ import annotations

import pytest

from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication, QMenu, QToolTip, QWidget

from spacr.qt.widgets.popup_state import a_popup_is_on_screen


def _explode():
    raise RuntimeError("the C++ object is already gone")


def test_a_popup_is_on_screen_for_a_menu_or_a_tooltip_and_never_raises(
        qtbot, monkeypatch, subtests):
    """True for a menu or a visible tooltip, False otherwise and on any failure."""
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(200, 120)
    host.show()
    qtbot.waitExposed(host)

    with subtests.test(msg="nothing showing"):
        assert a_popup_is_on_screen() is False

    with subtests.test(msg="an open menu holds the backdrop, a closed one releases it"):
        menu = QMenu(host)
        menu.addAction("something")
        assert a_popup_is_on_screen() is False
        menu.popup(host.mapToGlobal(QPoint(10, 10)))
        try:
            qtbot.waitUntil(lambda: QApplication.activePopupWidget() is not None,
                            timeout=2000)
            assert a_popup_is_on_screen() is True
        finally:
            menu.close()
        qtbot.waitUntil(lambda: QApplication.activePopupWidget() is None,
                        timeout=2000)
        assert a_popup_is_on_screen() is False

    no_menu = staticmethod(lambda: None)
    for case, popup, tooltip, expected in (
            ("a visible tooltip counts though it is no popup widget",
             no_menu, staticmethod(lambda: True), True),
            ("a tooltip that has gone", no_menu, staticmethod(lambda: False), False),
            ("a falsy tooltip answer", no_menu, staticmethod(lambda: 0), False),
            ("the popup question raises", staticmethod(_explode),
             staticmethod(lambda: True), False),
            ("the tooltip question raises", no_menu, staticmethod(_explode), False),
    ):
        with subtests.test(msg=case):
            with monkeypatch.context() as patch:
                patch.setattr(QApplication, "activePopupWidget", popup)
                patch.setattr(QToolTip, "isVisible", tooltip)
                assert a_popup_is_on_screen() is expected

    with subtests.test(msg="each call asks Qt once, per frame not per event"):
        calls = []
        with monkeypatch.context() as patch:
            patch.setattr(QApplication, "activePopupWidget",
                          staticmethod(lambda: calls.append(1)))
            patch.setattr(QToolTip, "isVisible", staticmethod(lambda: False))
            for _ in range(10):
                a_popup_is_on_screen()
        assert len(calls) == 10
