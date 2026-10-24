"""Native recorder navigation uses the real organism guide and its assay tile."""
from pathlib import Path
from types import SimpleNamespace
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[3]), str(Path(__file__).resolve().parents[1])]
import pytest
from PySide6.QtWidgets import QApplication, QPushButton, QStackedWidget, QTabWidget, QVBoxLayout, QWidget

import capture_ui
from spacr.qt import preferences
from spacr.qt.screens.organism_screen import OrganismScreen, FOLDED_APPS


@pytest.fixture
def route(monkeypatch):
    app = QApplication.instance() or QApplication([])
    for getter in ('get_show_alpha', '_get_show_alpha_features', '_get_show_alpha_species'):
        monkeypatch.setattr(preferences, getter, lambda: False)
    for setter in ('set_show_alpha', '_set_show_alpha_features', '_set_show_alpha_species'):
        monkeypatch.setattr(preferences, setter, lambda *_: pytest.fail('recorder must not enable alpha'))
    stack = QStackedWidget()
    stack.resize(1280, 900)
    home = QWidget()
    layout = QVBoxLayout(home)
    home._tabs = QTabWidget()
    layout.addWidget(home._tabs)
    page = QWidget()
    tiles = QVBoxLayout(page)
    home._tabs.addTab(page, 'Assays')
    stack.addWidget(home)
    window = SimpleNamespace(_startup=home, _stack=stack, _screens={},
                             _sidebar=SimpleNamespace(_rows=[]))
    clicks, requested, guides = [], [], []

    def selected(key):
        requested.append(key)
        screen = QWidget()
        screen.app_key = key
        window._screens[key] = screen
        stack.addWidget(screen)
        stack.setCurrentWidget(screen)

    window._on_nav_selected = selected

    def add_home(key, callback):
        button = QPushButton(key)
        button.setProperty('moduleAppKey', key)
        button.clicked.connect(callback)
        tiles.addWidget(button)
        return button

    def open_guide():
        guide = OrganismScreen('toxoplasma', host=window)
        guides.append(guide)
        window._screens['toxoplasma'] = guide
        stack.addWidget(guide)
        stack.setCurrentWidget(guide)
        app.processEvents()

    add_home('toxoplasma', open_guide)
    for key in ('mask', 'measure'):
        add_home(key, lambda checked=False, key=key: selected(key))
    stack.show()
    app.processEvents()

    def click(button):
        assert button.isVisible() and button.isEnabled()
        clicks.append(button.property('moduleAppKey') or button.property('organismModuleKey'))
        button.click()
        app.processEvents()

    def scroll(capture, area, button):
        assert guides and area is guides[-1]._module_scroll
        assert button.property('organismModuleKey') in FOLDED_APPS
        area.ensureWidgetVisible(button)
        app.processEvents()

    human = SimpleNamespace(click=click, pause=lambda *_: app.processEvents())
    monkeypatch.setattr(capture_ui, '_h', lambda: human)
    monkeypatch.setattr(capture_ui, 'scroll_to', scroll)
    capture = lambda *_a, **_k: True
    capture.host = object()
    yield SimpleNamespace(app=app, window=window, capture=capture, clicks=clicks,
                          requested=requested, guides=guides, open_guide=open_guide)
    stack.close()
    stack.deleteLater()
    app.processEvents()


@pytest.mark.parametrize('key', FOLDED_APPS)
def test_native_assay_route_clicks_home_guide_then_real_assay_tile(route, key):
    capture_ui.open_module(route.capture, route.window, key, lambda *_: None, prefer='tile')
    assert route.clicks == ['toxoplasma', key]
    assert route.requested == [key]
    assert route.window._stack.currentWidget() is route.window._screens[key]
    assert route.window._screens[key].app_key == key
    assert all(not getter() for getter in (preferences.get_show_alpha,
        preferences._get_show_alpha_features, preferences._get_show_alpha_species))


@pytest.mark.parametrize('key', ['mask', 'measure'])
def test_native_top_level_routes_keep_their_single_home_click(route, key):
    capture_ui.open_module(route.capture, route.window, key, lambda *_: None, prefer='tile')
    assert route.clicks == [key]
    assert route.window._stack.currentWidget() is route.window._screens[key]


@pytest.mark.parametrize('unavailable', ['hidden', 'disabled', 'duplicate'])
def test_missing_public_assay_route_fails_without_offcamera_navigation(route, monkeypatch, unavailable):
    original = route.open_guide
    def open_guide():
        original()
        guide = route.guides[-1]
        tile = next(t for t in guide._tiles if t.property('organismModuleKey') == 'analyze_plaques')
        if unavailable == 'hidden': tile.hide()
        elif unavailable == 'disabled': tile.setEnabled(False)
        else:
            duplicate = QPushButton('duplicate', guide)
            duplicate.setProperty('organismModuleKey', 'analyze_plaques')
    home = route.window._startup
    button = next(b for b in home.findChildren(QPushButton) if b.property('moduleAppKey') == 'toxoplasma')
    button.clicked.disconnect()
    button.clicked.connect(open_guide)
    with pytest.raises(RuntimeError, match='unique visible enabled organism assay tile'):
        capture_ui.open_module(route.capture, route.window, 'analyze_plaques', lambda *_: None, prefer='tile')
    assert route.clicks == ['toxoplasma']
    assert route.requested == []
