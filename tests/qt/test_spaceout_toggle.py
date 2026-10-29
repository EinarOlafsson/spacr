"""Spaceout toggles restore ordinary appearance and reject retired themes."""
import pytest
from PySide6.QtCore import QSettings

from spacr.qt import preferences, theme
from spacr.qt.widgets import ambient
from tests.qt.test_733_easy_commands import commands as commands
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile as private_ai_profile


@pytest.fixture
def appearance(monkeypatch, tmp_path):
    store = QSettings(str(tmp_path / 'appearance.ini'), QSettings.IniFormat)
    monkeypatch.setattr(preferences, '_settings', lambda: store)
    monkeypatch.setattr(preferences, '_SAFE_MODE', False)
    previous = theme.spaceout_enabled()
    theme.disable_spaceout()
    try:
        yield store
    finally:
        theme.enable_spaceout() if previous else theme.disable_spaceout()


def test_spaceout_card_second_activation_restores_normal_appearance(commands, appearance):
    from spacr.qt.app import MainWindow

    preferences.set_theme('light')
    preferences.set_ambient_animation('blobs')
    preferences.set_ambient_palette('ember')
    preferences.apply_preferences_to_app()
    saved = {key: appearance.value(key) for key in appearance.allKeys()}
    normal = theme.palette_for('light')
    MainWindow._open_easy_command(commands, '__easy_command__:Spaceout')
    assert theme.spaceout_enabled()
    MainWindow._open_easy_command(commands, '__easy_command__:Spaceout')
    assert not theme.spaceout_enabled()
    assert theme.palette_for('light') == normal
    assert {key: appearance.value(key) for key in appearance.allKeys()} == saved
    assert commands.refreshed == [{'spaceout'}, {'spaceout'}]



def test_retired_vortex_falls_back_without_changing_normal_choices(appearance):
    preferences.set_ambient_animation('blobs')
    preferences.set_ambient_palette('ember')
    appearance.setValue('spaceout/animation', 'data_art_spaceout_vortex')
    appearance.setValue('spaceout/animation_palette', 'mono')
    for enabled in (True, False, True):
        theme.enable_spaceout() if enabled else theme.disable_spaceout()
        assert 'data_art_spaceout_vortex' not in preferences._animation_choices()
        expected = ambient.DEFAULT_SPACEOUT_THEME if enabled else 'blobs'
        assert preferences.get_ambient_theme() == expected
    theme.disable_spaceout()
    assert preferences.get_ambient_palette() == 'ember'
