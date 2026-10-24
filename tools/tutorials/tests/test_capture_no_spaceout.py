"""Hidden Spaceout is unavailable to tutorial recipes; normal UI stays public."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools/tutorials')]


def test_retired_capture_cli_cannot_select_hidden_mode():
    import capture_refresh

    with pytest.raises(SystemExit) as error:
        capture_refresh.build_parser().parse_args(['--module', 'home', '--spaceout-effects-only'])
    assert error.value.code == 2
    normal = capture_refresh.build_parser().parse_args(['--module', 'home', '--home-navigation-only'])
    assert normal.home_navigation_only and normal.backdrop == 'data_art_impulse_lens'
    assert not hasattr(normal, 'spaceout_effects_only')


def test_profile_cli_rejects_retired_mode_before_writing(tmp_path, monkeypatch):
    import capture_host_mode

    output = tmp_path / 'forbidden-profile.json'
    monkeypatch.setattr(sys, 'argv', ['capture_host_mode', '--prepare-profile',
                        '--font-scale', '1.0', '--out', str(output), '--spaceout'])
    with pytest.raises(SystemExit) as error:
        capture_host_mode.main()
    assert error.value.code == 2 and not output.exists()


def test_hidden_backdrop_has_no_tutorial_policy_exception():
    from capture_policy import configure_appearance, verify_appearance

    with pytest.raises(ValueError):
        configure_appearance('dark', 'data_art_spaceout_field')
    with pytest.raises(TypeError):
        configure_appearance(spaceout_effects=True)
    with pytest.raises(TypeError):
        verify_appearance(None, spaceout_effects=True)


@pytest.fixture
def normal_store(tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings
    from spacr.qt import preferences, theme

    store = QSettings(str(tmp_path / 'ordinary.ini'), QSettings.IniFormat)
    monkeypatch.setattr(preferences, '_settings', lambda: store)
    previous = theme.spaceout_enabled()
    theme.disable_spaceout()
    preferences.set_show_alpha(False)
    preferences._set_show_alpha_features(False)
    preferences._set_show_alpha_species(False)
    yield preferences
    theme.enable_spaceout() if previous else theme.disable_spaceout()


def test_normal_preferences_retains_recorded_controls_without_hidden_effects(normal_store, qtbot):
    from PySide6.QtWidgets import QComboBox, QTabWidget, QWidget
    from spacr.qt.widgets.ambient import SPACEOUT_ONLY_THEMES

    prefs = normal_store
    dialog = prefs.PreferencesDialog()
    qtbot.addWidget(dialog)
    required = ['PerformanceLevel', 'PreferencesTabGeneral', 'PreferencesTabAppearance',
                'PreferencesTabModules', 'PreferencesTabStorage', 'PaneOpacity',
                'RestoreLastSession', 'UpdateChannel', 'ShowAlphaFeatures', 'StoragePruneButton']
    assert dialog.findChild(QTabWidget, 'PreferencesTabs') is not None
    assert all(dialog.findChild(QWidget, name) is not None for name in required)
    for name in ('AmbientTheme', 'PopupBackdrop'):
        control = dialog.findChild(QComboBox, name)
        assert control is not None
        assert not set(control.itemData(i) for i in range(control.count())) & set(SPACEOUT_ONLY_THEMES)
    assert dialog.findChild(QWidget, 'SpaceoutFieldVortex') is None
    assert len(dialog._storage_page.spins['run_folders']) == 2
    assert prefs.get_show_beta() is True
    assert not prefs.get_show_alpha() and not prefs._get_show_alpha_features() and not prefs._get_show_alpha_species()
    dialog.reject()


def test_all_selected_home_module_tiles_exist_under_normal_maturity(normal_store, qtbot, monkeypatch):
    import json
    from spacr.qt import app, register_self_registering_modules
    from spacr.qt.widgets.home import AppTile

    monkeypatch.setattr(app, 'APP_STAGE', dict(app.APP_STAGE))
    register_self_registering_modules()
    home = app.make_home_page()
    qtbot.addWidget(home)
    canonical = json.loads((ROOT / 'tools/tutorials/lessons/05_home.json').read_text())
    keys = {s['visual'].removeprefix('home_module_') for s in canonical['scenes']
            if s['visual'].startswith('home_module_')}
    tiles = {str(tile.property('moduleAppKey')) for tile in home.findChildren(AppTile)}
    assert keys <= tiles, sorted(keys - tiles)
    assert keys <= {row[0] for row in app.tiled_apps(app.visible_apps())}
    assert all(any(home._tabs.widget(i).isAncestorOf(tile) for i in range(home._tabs.count()))
               for tile in home.findChildren(AppTile) if tile.property('moduleAppKey') in keys)
    assert hasattr(home, '_aside_split') and hasattr(home, '_aside') and hasattr(home, '_aside_controls')
