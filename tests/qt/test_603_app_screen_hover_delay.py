"""The actual settings screen waits before changing its help strip."""
import pytest
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QLabel

from spacr.qt import preferences
from spacr.qt.screens.app_screen import AppScreen


@pytest.mark.parametrize("category", [False, True])
def test_setting_and_category_hover_wait_and_cancel(qtbot, monkeypatch, category):
    preferences._set_tooltip_delay(1)
    preferences.set_tooltips_enabled(True)
    monkeypatch.setattr(preferences, "get_tooltips_bottom_enabled", lambda: True)
    monkeypatch.setattr(preferences, "get_tooltips_box_enabled", lambda: False)
    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    screen.resize(1200, 900)
    screen.show()
    target = QLabel("Target", screen)
    target.move(10, 10)
    target.show()
    screen._hint_map[target] = "Delayed setting explanation"
    screen._html_tip_map[target] = "Delayed setting explanation"
    if category:
        target.setProperty("settingsCategory", "Files and folders")
    qtbot.wait(30)
    screen._release_the_hint()
    screen._category_hint_pinned = ""
    screen.clear_category_hint()
    strip = screen._category_hint if category else screen._hint_strip
    before = strip.text()
    try:
        screen.eventFilter(target, QEvent(QEvent.Enter))
        qtbot.wait(50)
        assert strip.text() == before
        screen.eventFilter(target, QEvent(QEvent.Leave))
        qtbot.wait(150)
        assert strip.text() == before
        screen.eventFilter(target, QEvent(QEvent.Enter))
        qtbot.wait(50)
        assert strip.text() == before
        qtbot.waitUntil(lambda: strip.text() != before, timeout=3000)
    finally:
        preferences._set_tooltip_delay(2)
