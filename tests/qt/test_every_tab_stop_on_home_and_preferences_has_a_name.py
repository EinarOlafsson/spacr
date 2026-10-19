"""Every Tab stop on Home and in Preferences announces something.

NVDA on Windows found 52 Tab stops that read nothing (2026-10-19): on Home's
release notes every second stop between releases, and in Preferences after
the General tab, after "Colour-blind mode" and between Cancel and Apply. They
were scroll areas that took focus, sliders without a name, a read-only help
view and rich-text labels whose accessible text was HTML or nothing. This walks
the Tab chain offscreen and counts the stops that would be silent.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import (QApplication, QLabel, QTabBar,  # noqa: E402
                               QTabWidget)


def _spoken(widget) -> str:
    """What a screen reader announces for ``widget`` on focus."""
    name = widget.accessibleName()
    if name:
        return name
    if widget.focusProxy() is not None:
        return "proxy"
    if isinstance(widget, QLabel):
        return ""
    if isinstance(widget, QTabBar):
        return widget.tabText(widget.currentIndex())
    for attr in ("text", "currentText"):
        read = getattr(widget, attr, None)
        if callable(read):
            text = read()
            if text:
                return str(text)
    return ""


def _silent_stops(root) -> list:
    """Every visible Tab stop under ``root`` that announces nothing."""
    QApplication.processEvents()
    silent, count = [], 0
    widget = root.nextInFocusChain()
    for _ in range(10000):
        if widget is root or widget is None:
            break
        if ((widget.focusPolicy() & Qt.TabFocus) == Qt.TabFocus
                and widget.isVisible() and widget.isEnabled()
                and widget.window() is root.window()):
            count += 1
            if not _spoken(widget):
                silent.append(f"{type(widget).__name__} "
                              f"{widget.objectName()!r}")
        widget = widget.nextInFocusChain()
    assert count, "the walk found no Tab stop at all"
    return silent


def test_every_tab_stop_on_home_has_a_name(qtbot):
    """Home, release notes included, has no silent stop."""
    from spacr.qt.widgets.home import HomePage

    page = HomePage([("mask", "Mask", "Segment cells.", "Core")],
                    lambda key: None)
    qtbot.addWidget(page)
    page.resize(1400, 900)
    page.show()
    assert _silent_stops(page) == []


def test_every_tab_stop_in_preferences_has_a_name(qtbot):
    """Every Preferences page, walked tab by tab."""
    from spacr.qt import preferences

    dialog = preferences.PreferencesDialog()
    qtbot.addWidget(dialog)
    dialog.resize(1000, 800)
    dialog.show()
    tabs = dialog.findChild(QTabWidget, "PreferencesTabs")
    assert tabs is not None and tabs.count()
    silent = {}
    for index in range(tabs.count()):
        tabs.setCurrentIndex(index)
        found = _silent_stops(dialog)
        if found:
            silent[tabs.tabText(index)] = found
    assert not silent, silent
