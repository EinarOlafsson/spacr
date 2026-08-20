"""Instruction 178 D — the overflow arrows come off.

    "also in spacer whenever something dosnt fit horizontally there are two
     arrows that are visable black boxes with white arrows. these are ugly and
     can be removed. the user can allways scrole and they should also be able
     to use the arrow keys when a tab is chosen and they want to go to the
     next tab, choosing the tab via mouse click or via the tab key."

THE CONDITION IS IN THE ASK. Removing a control is only safe if what it did
still works, and the ask names the two things that must: the wheel, and the
arrow keys. Both are driven here on a bar too narrow to hold its tabs, because
"the user can always scroll" is a claim and not an assumption.

The stylesheet rules for `QTabBar::scroller` stay, deliberately. A bar this
sweep does not reach still has arrows that belong to the theme rather than to
Qt's default palette — the styling is the floor and this is the improvement
on it.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

from PySide6.QtCore import QEvent, QPoint, Qt                    # noqa: E402
from PySide6.QtGui import QKeyEvent, QWheelEvent                 # noqa: E402
from PySide6.QtWidgets import (QApplication, QTabBar,            # noqa: E402
                               QTabWidget, QWidget)

from spacr.qt.theme import take_the_scroll_arrows_off            # noqa: E402


def _overflowing(qtbot, n: int = 14) -> QTabWidget:
    tabs = QTabWidget()
    qtbot.addWidget(tabs)
    for i in range(n):
        tabs.addTab(QWidget(), f"a rather long tab name {i}")
    tabs.resize(420, 200)
    tabs.show()
    QApplication.processEvents()
    return tabs


# -- the arrows go ----------------------------------------------------------

def test_the_arrows_come_off_every_bar_under_the_root(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    inner = [_overflowing(qtbot) for _ in range(3)]
    for tabs in inner:
        tabs.setParent(host)

    assert take_the_scroll_arrows_off(host) >= 3
    assert not any(t.tabBar().usesScrollButtons() for t in inner)


def test_a_bare_tab_widget_or_bar_is_handled_too(qtbot):
    tabs = _overflowing(qtbot)
    take_the_scroll_arrows_off(tabs)
    assert tabs.tabBar().usesScrollButtons() is False

    bar = QTabBar()
    qtbot.addWidget(bar)
    for i in range(10):
        bar.addTab(f"tab {i}")
    take_the_scroll_arrows_off(bar)
    assert bar.usesScrollButtons() is False


def test_the_regression_screen_has_none(qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    assert {t.tabBar().usesScrollButtons()
            for t in screen.findChildren(QTabWidget)} == {False}


# -- and the bar is still navigable -----------------------------------------

def test_the_arrow_keys_still_walk_the_bar(qtbot):
    """The first thing the ask names."""
    tabs = _overflowing(qtbot)
    take_the_scroll_arrows_off(tabs)
    bar = tabs.tabBar()
    tabs.setCurrentIndex(0)
    bar.setFocus()

    for _ in range(2):
        QApplication.sendEvent(
            bar, QKeyEvent(QEvent.KeyPress, Qt.Key_Right, Qt.NoModifier))
    QApplication.processEvents()
    assert tabs.currentIndex() == 2


def test_the_wheel_still_walks_the_bar(qtbot):
    """The second. "the user can allways scrole" is a claim, not an assumption."""
    tabs = _overflowing(qtbot)
    take_the_scroll_arrows_off(tabs)
    bar = tabs.tabBar()
    tabs.setCurrentIndex(2)

    QApplication.sendEvent(bar, QWheelEvent(
        QPoint(50, 10), bar.mapToGlobal(QPoint(50, 10)),
        QPoint(0, -120), QPoint(0, -120), Qt.NoButton, Qt.NoModifier,
        Qt.NoScrollPhase, False))
    QApplication.processEvents()
    assert tabs.currentIndex() != 2


def test_a_tab_reached_by_key_is_a_tab_the_user_can_see(qtbot):
    """Qt scrolls the bar to reveal the current tab whichever way it moves --
    which is why the arrows were a third way to do what two already do."""
    tabs = _overflowing(qtbot, n=20)
    take_the_scroll_arrows_off(tabs)
    bar = tabs.tabBar()
    tabs.setCurrentIndex(19)
    QApplication.processEvents()

    rect = bar.tabRect(19)
    assert rect.isValid()
    assert rect.right() > 0, "the last tab was left off the visible bar"


def test_the_theme_still_styles_an_arrow_that_survives():
    """The floor under the improvement: a bar this sweep misses is still
    themed rather than drawn in Qt's default palette."""
    from spacr.qt.theme import stylesheet

    sheet = stylesheet("dark")
    assert "QTabBar::scroller" in sheet
    assert "QTabBar QToolButton" in sheet
