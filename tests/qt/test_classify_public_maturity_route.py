"""The assessed stable Classify module has a real route with alpha hidden."""
import pytest
from PySide6.QtCore import QSettings


@pytest.fixture
def public_preferences(tmp_path, monkeypatch):
    from spacr.qt import preferences

    path = tmp_path / 'public-route.ini'
    monkeypatch.setattr(preferences, '_settings', lambda: QSettings(str(path), QSettings.IniFormat))
    preferences.set_show_alpha(False)
    preferences.set_show_beta(False)
    preferences._set_show_alpha_features(False)
    return preferences


def test_classify_runtime_stage_matches_existing_stable_assessment(public_preferences):
    import spacr.qt
    from spacr.qt import app, maturity

    spacr.qt.register_self_registering_modules()
    assert maturity.AFFIRMED['classify_merged'][0] == 'stable'
    assert 'classify_merged' not in app.APP_STAGE
    assert app.app_stage('classify_merged') == 'stable'
    assert 'classify_merged' in {row[0] for row in app.visible_apps()}
    assert all(app.app_stage(row[0]) == 'stable' for row in app.visible_apps())


def test_actual_home_classify_tile_emits_public_route(qtbot, qt_theme_applied, public_preferences):
    from spacr.qt.app import make_home_page
    from spacr.qt.widgets.home import AppTile

    page = make_home_page()
    qtbot.addWidget(page)
    tiles = [tile for tile in page.findChildren(AppTile)
             if tile.property('moduleAppKey') == 'classify_merged']
    assert tiles
    assert all(tile.stage == 'stable' for tile in tiles)
    requested = []
    page.tile_clicked.connect(requested.append)
    tiles[0].click()
    assert requested == ['classify_merged']
    assert all(tile.stage != 'alpha' for tile in page.findChildren(AppTile))


def test_actual_sidebar_keeps_classify_visible_with_alpha_hidden(qtbot, qt_theme_applied, public_preferences):
    from spacr.qt.app import Sidebar

    sidebar = Sidebar()
    qtbot.addWidget(sidebar)
    sidebar.refresh_visibility()
    buttons = [button for button in sidebar._items
               if button.property('navKey') == 'classify_merged']
    assert len(buttons) == 1
    assert not buttons[0].isHidden()
