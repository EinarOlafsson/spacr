"""Actual Home tile input reaches real module screens at laptop geometry."""
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QTabWidget

from spacr.qt.app import MainWindow
from spacr.qt.widgets.home import AppTile


@pytest.mark.parametrize('key', ['mask', 'measure', 'annotate', 'classify_merged', 'toxoplasma'])
def test_visible_home_tile_opens_real_screen(qtbot, qt_theme_applied, key):
    window = MainWindow()
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    window.resize(1280, 800)
    window.show()
    home = window._startup
    candidates = [t for t in home.findChildren(AppTile) if t.property('moduleAppKey') == key]
    assert candidates, key
    tile = next((t for t in candidates if t.isVisible()), candidates[0])
    for tab in home.findChildren(QTabWidget):
        for index in range(tab.count()):
            if tab.widget(index).isAncestorOf(tile):
                tab.setCurrentIndex(index)
    qtbot.waitUntil(tile.isVisible, timeout=3000)
    qtbot.mouseClick(tile, Qt.LeftButton, pos=QPoint(tile.width() // 2, tile.height() // 2))
    qtbot.waitUntil(lambda: key in window._screens, timeout=10000)
    assert window._stack.currentWidget() is window._screens[key]
    assert window._screens[key].isVisible()
    assert window._screens[key].visibleRegion().boundingRect().width() > 300


@pytest.mark.parametrize('key', ['mask', 'analyze_plaques'])
@pytest.mark.parametrize('size', [(1024, 768), (1280, 800)])
def test_live_preview_takes_space_and_user_can_resize_settings(qtbot, qt_theme_applied, key, size):
    from spacr.qt.screens.app_screen import AppScreen
    screen = AppScreen(key)
    qtbot.addWidget(screen)
    screen.resize(*size)
    screen.show()
    qtbot.mouseClick(screen._preview_switch, Qt.LeftButton)
    split, body = screen._runtime_splitter, screen._body_splitter
    qtbot.waitUntil(lambda: body.is_collapsed('Settings'), timeout=3000)
    for name in ('Console', 'System', 'Actions'):
        assert split.is_collapsed(name)
    preview = screen._live_preview_card
    assert split.sizes()[split.indexOf(preview)] > .7 * split.height()
    assert screen.reveal_settings()
    assert not body.is_collapsed('Settings')
    before = body.sizes()
    body.moveSplitter(max(150, before[0] - 80), 1)
    qtbot.wait(20)
    assert body.sizes()[0] != before[0], [(body.widget(i).objectName(), body.widget(i).minimumWidth(), body.widget(i).minimumSizeHint().width(), body.widget(i).sizeHint().width(), body.sizes()[i]) for i in range(body.count())]
    assert preview.isVisible()
