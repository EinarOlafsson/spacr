"""The new Spaceout field stays interactive through actual launcher preferences."""

import math

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, QSettings, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QWidget

from spacr.qt import preferences as prefs, theme
from spacr.qt.widgets import ambient
from spacr.qt.widgets.toggle import Toggle


@pytest.fixture
def field_store(tmp_path, monkeypatch):
    store = QSettings(str(tmp_path / "field.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: store)
    monkeypatch.setattr(prefs, "_SAFE_MODE", False)
    monkeypatch.setattr(ambient.AmbientWidget, "_start_producer", lambda self: None)
    was_spaceout = theme.spaceout_enabled()
    theme.enable_spaceout()
    yield store
    theme.enable_spaceout() if was_spaceout else theme.disable_spaceout()


def _field(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(360, 240)
    backdrop = ambient.install_ambient(host, theme="blobs", palette="ember", seed=42)
    host.show()
    backdrop.show()
    backdrop._timer.setInterval(100000)
    return host, backdrop


def test_fresh_spaceout_installs_field_with_brand_palette(field_store, qtbot):
    assert prefs.get_ambient_animation() == ambient.DEFAULT_SPACEOUT_THEME
    assert prefs.get_ambient_palette() == "spacr"
    _host, backdrop = _field(qtbot)
    assert backdrop.theme() == ambient.DEFAULT_SPACEOUT_THEME
    assert isinstance(backdrop.engine, ambient._SpaceoutFieldEngine)
    assert backdrop.engine.name == backdrop.theme()
    assert backdrop.palette_name() == "spacr"
    assert backdrop._interaction_app is not None
    assert backdrop.engine.field_effects == prefs._spaceout_field_effects()


def test_click_ripples_work_without_gravity_on_the_spaceout_field(
        field_store, qtbot):
    prefs._set_ambient_gravity_radius(0)
    prefs._set_field_popup_wave_frequency(0)
    host, backdrop = _field(qtbot)
    qtbot.mouseClick(host, Qt.LeftButton, pos=host.rect().center())
    backdrop._on_tick()
    assert backdrop.engine._popup_waves
    assert not backdrop.engine._gravity_impulses
    backdrop.set_ripples_enabled(False)
    qtbot.mouseClick(host, Qt.LeftButton, pos=host.rect().center())
    backdrop._on_tick()
    assert not backdrop.engine._popup_waves
    assert backdrop._interaction_app is not None


def test_actual_grab_during_phenomena_keeps_bounded_spring_and_release(
        field_store, qtbot):
    host, backdrop = _field(qtbot)
    for clock in range(1, 130):
        backdrop.engine.set_time(clock)
        if backdrop.engine._field_events():
            break
    assert backdrop.engine._field_events()
    qtbot.mousePress(host, Qt.LeftButton, pos=QPoint(160, 100))
    point = QPoint(200, 120)
    QApplication.sendEvent(host, QMouseEvent(
        QEvent.MouseMove, QPointF(point), QPointF(host.mapToGlobal(point)),
        Qt.NoButton, Qt.LeftButton, Qt.NoModifier))
    backdrop.advance_frame(0.25)
    assert not backdrop.grab().isNull()
    assert backdrop.engine._field_grab_held
    assert 0 < math.hypot(*backdrop.engine._field_grab_offset) <= .18
    qtbot.mouseRelease(host, Qt.LeftButton, pos=point)
    assert not backdrop.engine._field_grab_held
    for _ in range(80):
        backdrop.advance_frame(0.1)
        backdrop.grab()
        assert math.hypot(*backdrop.engine._field_grab_offset) <= .18000001
    assert backdrop.engine._field_grab_center is None


def test_preferences_save_updates_live_field_and_keeps_effects_on_rebuild(
        field_store, qtbot):
    _host, backdrop = _field(qtbot)
    dialog = prefs.PreferencesDialog()
    qtbot.addWidget(dialog)
    dialog.findChild(Toggle, "SpaceoutFieldVortex").setChecked(False)
    dialog.findChild(Toggle, "SpaceoutFieldColorWaves").setChecked(False)
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()
    assert not backdrop.engine.field_effects["vortex"]
    assert not backdrop.engine.field_effects["color_waves"]
    clock = backdrop.engine.time
    backdrop.set_theme("blobs")
    backdrop.set_theme(ambient.DEFAULT_SPACEOUT_THEME)
    assert backdrop.engine.time == clock
    assert not backdrop.engine.field_effects["vortex"]
    assert not backdrop.engine.field_effects["color_waves"]


def test_spaceout_field_cannot_be_selected_in_ordinary_spacr(field_store):
    theme.disable_spaceout()
    assert ambient.DEFAULT_SPACEOUT_THEME not in prefs._animation_choices()
    with pytest.raises(ValueError):
        prefs.set_ambient_animation(ambient.DEFAULT_SPACEOUT_THEME)
    assert prefs.get_ambient_theme() == ambient.DEFAULT_THEME
