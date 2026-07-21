"""MainWindow + Sidebar + startup screen tests."""
from __future__ import annotations

import pytest

from PySide6.QtCore import Qt

from spacr.qt.app import APPS, MainWindow, Sidebar, _icon_for_app
from spacr.qt.screens.startup import StartupPage


def test_apps_list_shape():
    assert len(APPS) >= 10
    keys = {a[0] for a in APPS}
    for expected in ("mask", "measure", "classify", "umap", "annotate"):
        assert expected in keys
    # Every entry is (key, name, desc, section)
    for entry in APPS:
        assert len(entry) == 4
        for field in entry:
            assert isinstance(field, str) and field


def test_sidebar_emits_nav_selected(qtbot, qt_theme_applied):
    from PySide6.QtWidgets import QPushButton
    bar = Sidebar()
    qtbot.addWidget(bar)
    buttons = [b for b in bar.findChildren(QPushButton)
               if b.objectName() == "SidebarItem"]
    assert buttons, "sidebar should have per-app buttons"
    with qtbot.waitSignal(bar.nav_selected, timeout=1000) as blocker:
        buttons[1].click()   # first is Home, second is first app
    assert isinstance(blocker.args[0], str)


def test_main_window_constructs_and_switches(qtbot, qt_theme_applied):
    win = MainWindow()
    qtbot.addWidget(win)
    assert win.windowTitle() == "spaCR"
    # Switch to mask screen; stack should register a new widget.
    starting_count = win._stack.count()
    win._on_nav_selected("mask")
    assert win._stack.count() == starting_count + 1
    assert win._stack.currentWidget() is not None
    # Switching back to Home selects the startup page.
    win._on_nav_selected("__home__")
    assert win._stack.currentWidget() is win._startup


def test_startup_page_emits_tile_clicked(qtbot, qt_theme_applied):
    from spacr.qt.widgets.tile import Tile
    page = StartupPage(APPS, _icon_for_app)
    qtbot.addWidget(page)
    tiles = page.findChildren(Tile)
    assert tiles, "startup page should render tiles"
    with qtbot.waitSignal(page.tile_clicked, timeout=1000) as blocker:
        tiles[0]._button.click()
    assert isinstance(blocker.args[0], str)


def test_icon_for_app_returns_qicon():
    from PySide6.QtGui import QIcon
    icon = _icon_for_app("mask")
    assert isinstance(icon, QIcon)
