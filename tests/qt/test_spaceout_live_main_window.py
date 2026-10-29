"""Navigator Spaceout actions and Preferences change the actual shared backdrop."""
import pytest
from PySide6.QtWidgets import QComboBox, QDialogButtonBox

from spacr.qt import preferences, theme
from spacr.qt.widgets import ambient
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile as private_ai_profile


def test_navigator_toggle_and_live_preferences_select_actual_main_window_engines(
        qtbot, qapp, private_ai_profile, monkeypatch):
    from spacr.qt.app import MainWindow

    monkeypatch.setenv('SPACR_NO_GL', '1')
    previous = theme.spaceout_enabled()
    theme.disable_spaceout()
    preferences.set_easy_mode(True)
    preferences.set_ambient_animation('blobs')
    preferences.set_ambient_palette('ember')
    preferences.set_ambient_enabled(True)
    preferences.set_reduce_motion(True)
    window = MainWindow(initial_app='__home__')
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    window.resize(1100, 820)
    window.show()
    try:
        qtbot.waitUntil(lambda: window.window_backdrop() is not None, timeout=5000)
        assert window.window_backdrop().engine.name == 'blobs'
        preferences.apply_preferences_to_app()
        normal_palette = qapp.palette()
        normal_sheet = theme.window_stylesheet(qapp)
        window._open_easy_command('__easy_command__:Spaceout')
        assert theme.spaceout_enabled()
        assert theme.window_stylesheet(qapp) != normal_sheet
        assert qapp.palette() != normal_palette
        assert window.window_backdrop().theme() == ambient.DEFAULT_SPACEOUT_THEME
        assert isinstance(window.window_backdrop().engine, ambient._SpaceoutFieldEngine)
        for selected, engine_type in (
                ('aurora', ambient.AuroraEngine),
                ('data_art_spaceout_field', ambient._SpaceoutFieldEngine)):
            dialog = preferences.PreferencesDialog(window)
            qtbot.addWidget(dialog)
            combo = dialog.findChild(QComboBox, 'AmbientTheme')
            combo.setCurrentIndex(combo.findData(selected))
            dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()
            qtbot.waitUntil(lambda selected=selected: window.window_backdrop().theme() == selected, timeout=5000)
            assert isinstance(window.window_backdrop().engine, engine_type)
            assert not window.window_backdrop().grab().isNull()
        window._open_easy_command('__easy_command__:Spaceout')
        assert not theme.spaceout_enabled()
        assert theme.window_stylesheet(qapp) == normal_sheet
        assert qapp.palette() == normal_palette
        assert not preferences.sound_is_offered()
        restored = preferences.PreferencesDialog(window)
        qtbot.addWidget(restored)
        assert restored.findChild(QComboBox, 'AmbientTheme').findData('aurora') == -1
        assert restored.findChild(QComboBox, 'AmbientTheme').currentData() == 'blobs'
        assert window.window_backdrop().engine.name == 'blobs'
        assert window.window_backdrop().palette_name() == 'ember'
        for _ in range(2):
            window._open_easy_command('__easy_command__:Spaceout')
            assert theme.spaceout_enabled()
            window._open_easy_command('__easy_command__:Spaceout')
            assert not theme.spaceout_enabled()
            assert theme.spaceout_drift_seconds() == 0
            assert theme.window_stylesheet(qapp) == normal_sheet
            assert qapp.palette() == normal_palette
            assert window.window_backdrop().engine.name == 'blobs'
            assert window.window_backdrop().palette_name() == 'ember'
    finally:
        window.close()
        theme.enable_spaceout() if previous else theme.disable_spaceout()


@pytest.mark.parametrize("normal_animation", [True, False])
def test_actual_cpu_fractal_backend_switch_and_mode_off_survive_module_home_navigation(
        qtbot, private_ai_profile, monkeypatch, normal_animation):
    from spacr.qt.app import MainWindow

    monkeypatch.setenv('SPACR_NO_GL', '1')
    previous = theme.spaceout_enabled()
    theme.disable_spaceout()
    preferences.set_easy_mode(True)
    preferences.set_show_alpha(True)
    preferences.set_ambient_animation('blobs')
    preferences.set_ambient_palette('ember')
    preferences.set_ambient_enabled(normal_animation)
    preferences.set_reduce_motion(True)
    preferences.set_fractal_settings(backend='cpu', quality='balanced', scale=.05,
                                     pattern='orbit', supersampling=1)
    window = MainWindow(initial_app='__home__')
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    window.resize(1000, 760)
    window.show()
    try:
        if normal_animation:
            qtbot.waitUntil(lambda: window.window_backdrop() is not None, timeout=5000)
        window._open_easy_command('__easy_command__:Spaceout')
        preferences.set_ambient_animation(ambient.SPACEOUT_THEME)
        preferences.apply_preferences_to_app()
        fractal = window.window_backdrop()
        assert getattr(fractal, '_spaceout_built_from', None) is not None
        assert fractal.backend_name == 'cpu'
        assert not isinstance(fractal, ambient.AmbientWidget)
        window.open_module('queue')
        qtbot.waitUntil(lambda: window._stack.currentWidget() is window._screens.get('queue'), timeout=5000)
        assert window.window_backdrop() is fractal
        window._on_nav_selected('__home__')
        assert window._stack.currentWidget() is window._startup
        assert window.window_backdrop() is fractal
        window._open_easy_command('__easy_command__:Spaceout')
        assert not theme.spaceout_enabled()
        normal = window.window_backdrop()
        if normal_animation:
            assert isinstance(normal, ambient.AmbientWidget)
            assert normal.engine.name == 'blobs' and normal.palette_name() == 'ember'
        else:
            assert normal is None
        assert fractal._stopped
    finally:
        live = window.window_backdrop()
        shutdown = getattr(live, 'shutdown', None)
        if callable(shutdown):
            shutdown()
        window.close()
        theme.enable_spaceout() if previous else theme.disable_spaceout()
