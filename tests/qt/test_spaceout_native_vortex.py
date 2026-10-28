"""Spaceout toggles reversibly and its native dot vortex stays bounded and live."""
import math

import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QWidget

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


def _pixels(engine, clock, width=500, height=500):
    engine.set_time(clock)
    result = QImage(width, height, QImage.Format_RGB32)
    result.fill(0xff191919)
    painter = QPainter(result)
    engine.paint(painter, width, height)
    painter.end()
    return np.frombuffer(result.constBits(), np.uint8).reshape(height, result.bytesPerLine())[:, :width * 4].reshape(height, width, 4).copy()


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


def test_vortex_is_selectable_only_in_spaceout_without_replacing_existing_choice(appearance):
    assert 'data_art_spaceout_vortex' not in preferences._animation_choices()
    theme.enable_spaceout()
    assert 'data_art_spaceout_vortex' in preferences._animation_choices()
    assert ambient.theme_label('data_art_spaceout_vortex') == 'Spaceout'
    assert ambient.palettes_for('data_art_spaceout_vortex') == ('mono',)
    preferences.set_ambient_animation('data_art_spaceout_vortex')
    assert preferences.get_ambient_theme() == 'data_art_spaceout_vortex'
    theme.disable_spaceout()
    assert preferences.get_ambient_theme() == ambient.DEFAULT_THEME
    theme.enable_spaceout()
    assert preferences.get_ambient_theme() == 'data_art_spaceout_vortex'


def test_native_vortex_has_white_spiral_grains_and_real_deterministic_motion(qapp):
    engine = ambient.make_engine('data_art_spaceout_vortex', 'mono', '#191919', seed=7)
    first = _pixels(engine, 0)
    later = _pixels(engine, 1.35)
    clone = ambient.make_engine('data_art_spaceout_vortex', 'mono', '#191919', seed=7)
    assert np.array_equal(first, _pixels(clone, 0))
    assert np.count_nonzero(np.max(abs(first.astype(int) - later.astype(int)), axis=2) > 20) > 3000
    assert np.max(first[:, :, :3].max(axis=2) - first[:, :, :3].min(axis=2)) == 0
    bright = first[:, :, 0] > 150
    assert bright.sum() > 3000
    central = bright[150:350, 150:350].mean()
    outer = np.concatenate((bright[:80].ravel(), bright[-80:].ravel())).mean()
    assert central > outer * 3
    rows = engine.geometry(500, 500)
    assert 1200 < len(rows) < 2500
    assert all(all(math.isfinite(value) for value in row) for row in rows)


def test_vortex_density_size_and_render_memory_remain_bounded(qapp):
    engine = ambient.make_engine('data_art_spaceout_vortex', 'mono', '#191919', seed=2)
    engine.set_density(ambient.DENSITY_RANGE[0])
    sparse = len(engine.geometry(1920, 1080))
    engine.set_density(ambient.DENSITY_RANGE[1])
    dense = len(engine.geometry(1920, 1080))
    assert sparse < dense <= 48 * 96
    small = engine.geometry(500, 500)[0][2]
    engine.set_size(2)
    assert engine.geometry(500, 500)[0][2] == pytest.approx(small * 2)
    for width, height in ((1920, 1080), (3840, 2160)):
        frame = engine.shade(width, height)
        assert frame.width() * frame.height() <= engine.max_pixels
        assert frame.sizeInBytes() <= engine.max_pixels * 4


def test_actual_native_theme_timer_stops_when_hidden_or_motion_is_reduced(
        qtbot, appearance, monkeypatch):
    monkeypatch.setenv('SPACR_NO_GL', '1')
    theme.enable_spaceout()
    preferences.set_ambient_animation('data_art_spaceout_vortex')
    preferences.set_ambient_palette('mono')
    preferences.set_reduce_motion(False)
    preferences.set_ambient_enabled(True)
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(500, 500)
    backdrop = ambient.install_ambient(host, theme='data_art_spaceout_vortex', palette='mono')
    host.show()
    qtbot.waitUntil(lambda: backdrop._last_frame is not None, timeout=4000)
    assert backdrop._timer.isActive()
    assert backdrop.engine.name == 'data_art_spaceout_vortex'
    host.hide()
    assert not backdrop._timer.isActive()
    host.show()
    qtbot.waitUntil(backdrop._timer.isActive)
    preferences.set_reduce_motion(True)
    backdrop._sync_run_state()
    assert not backdrop._timer.isActive()
    host.close()


def test_actual_preferences_select_save_reload_and_apply_the_native_theme(
        qtbot, qapp, appearance, monkeypatch):
    from PySide6.QtWidgets import QComboBox, QDialogButtonBox

    monkeypatch.setenv('SPACR_NO_GL', '1')
    theme.enable_spaceout()
    preferences.set_ambient_enabled(True)
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(400, 300)
    backdrop = ambient.install_ambient(host, theme=ambient.DEFAULT_SPACEOUT_THEME, palette='spacr')
    host.show()
    dialog = preferences.PreferencesDialog()
    qtbot.addWidget(dialog)
    combo = dialog.findChild(QComboBox, 'AmbientTheme')
    index = combo.findData('data_art_spaceout_vortex')
    assert index >= 0
    combo.setCurrentIndex(index)
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()
    assert preferences.get_ambient_theme() == 'data_art_spaceout_vortex'
    preferences.apply_ambient_preferences(qapp)
    assert backdrop.theme() == 'data_art_spaceout_vortex'
    assert isinstance(backdrop.engine, ambient._SpaceoutVortexEngine)
    reopened = preferences.PreferencesDialog()
    qtbot.addWidget(reopened)
    assert reopened.findChild(QComboBox, 'AmbientTheme').currentData() == 'data_art_spaceout_vortex'
    reopened.reject()
    host.close()
