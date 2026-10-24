"""N684 preference edges: odd density input, the ripple switch shim, a missing
Dynamic animation policy and the density slider once the user has chosen."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QComboBox, QDialogButtonBox, QSlider, QWidget

from spacr.qt import preferences as prefs
from spacr.qt.widgets import ambient

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    settings = QSettings(str(tmp_path / "ci3_n684.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: settings)
    monkeypatch.setattr(ambient.AmbientWidget, "_start_producer", lambda self: None)
    return settings


def _theme_combo(dialog):
    return dialog.findChild(QComboBox, "AmbientTheme") or next(
        box for box in dialog.findChildren(QComboBox)
        if box.findData("data_art_impulse_lens") >= 0 and box.findData("blobs") >= 0)


@pytest.mark.parametrize("raw", ["not a number", None, object()])
def test_unparseable_density_on_a_fresh_profile_stores_the_shipped_default(store, raw):
    assert not store.contains(prefs._KEY_AMBIENT_DENSITY)
    prefs.set_ambient_density(raw)
    assert store.contains(prefs._KEY_AMBIENT_DENSITY)
    default = prefs._ambient_ranges()[4][1]
    assert float(store.value(prefs._KEY_AMBIENT_DENSITY)) == pytest.approx(default)
    prefs.set_ambient_animation("data_art_impulse_lens")
    assert prefs.get_ambient_density() == pytest.approx(default)
    assert prefs.get_ambient_density("blobs") == pytest.approx(default)


@pytest.mark.parametrize("saved,switch,expected", [
    (0.0, True, 1.0),
    (0.4, True, 0.4),
    (1.7, True, 1.7),
    (0.4, False, 0.0),
])
def test_ripple_switch_restores_full_strength_only_from_zero(store, saved, switch, expected):
    prefs._set_field_ripple_intensity(saved)
    prefs._set_field_ripples_enabled(switch)
    assert prefs._field_ripple_intensity() == pytest.approx(expected)
    assert prefs._field_ripples_enabled() is (expected > 0)
    assert not store.contains(prefs._KEY_FIELD_RIPPLES)


def test_switching_ripples_on_runs_the_migration_of_an_old_off_switch(store):
    store.setValue(prefs._KEY_FIELD_RIPPLES, False)
    store.setValue(prefs._KEY_FIELD_RIPPLE_INTENSITY, 0.6)
    prefs._set_field_ripples_enabled(True)
    assert prefs._field_ripple_intensity() == pytest.approx(1.0)
    assert not store.contains(prefs._KEY_FIELD_RIPPLES)
    assert prefs._as_bool(store.value(prefs._KEY_FIELD_RIPPLES_SLIDER), False)


def test_apply_without_a_dynamic_policy_still_configures_the_fields(
        store, qtbot, qapp, monkeypatch):
    monkeypatch.setattr(ambient, "_animation_resource_policy", lambda: None)
    prefs.set_ambient_animation("data_art_impulse_lens")
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(200, 120)
    field = ambient.AmbientWidget(host, theme="data_art_impulse_lens",
                                  gravity_radius=0, popup_wave_frequency=0,
                                  blink_percent=0, density=1, seed=3)
    field._timer.stop()
    prefs._set_field_click_ripples(False)
    prefs._set_field_ripple_intensity(0.0)
    prefs._set_flow_graphics_backend("cpu")
    prefs._set_dynamic_animation_enabled(False)
    prefs.apply_ambient_preferences(qapp)
    assert field._click_ripples is False
    assert not field.engine.ripples_enabled
    prefs._set_field_click_ripples(True)
    prefs._set_field_ripple_intensity(0.5)
    prefs.apply_ambient_preferences(qapp)
    assert field._click_ripples is True
    assert field.engine.ripples_enabled


def test_density_the_user_moved_stays_put_when_the_theme_changes(
        store, qtbot, qt_theme_applied):
    prefs.set_ambient_animation("data_art_impulse_lens")
    assert not store.contains(prefs._KEY_AMBIENT_DENSITY)
    assert prefs.get_ambient_density() == pytest.approx(0.1)
    dialog = prefs.PreferencesDialog()
    qtbot.addWidget(dialog)
    density = dialog.findChild(QSlider, "AmbientDensity")
    combo = _theme_combo(dialog)
    assert density.value() == 10
    density.setValue(50)
    combo.setCurrentIndex(combo.findData("blobs"))
    assert density.value() == 50
    combo.setCurrentIndex(combo.findData("data_art_impulse_lens"))
    assert density.value() == 50
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()
    assert float(store.value(prefs._KEY_AMBIENT_DENSITY)) == pytest.approx(0.5)


def test_a_saved_density_never_follows_the_theme_in_the_dialog(
        store, qtbot, qt_theme_applied):
    prefs.set_ambient_animation("blobs")
    store.setValue(prefs._KEY_AMBIENT_DENSITY, 0.1)
    dialog = prefs.PreferencesDialog()
    qtbot.addWidget(dialog)
    density = dialog.findChild(QSlider, "AmbientDensity")
    combo = _theme_combo(dialog)
    assert density.value() == 10
    combo.setCurrentIndex(combo.findData("data_art_impulse_lens"))
    assert density.value() == 10
    combo.setCurrentIndex(combo.findData("blobs"))
    assert density.value() == 10
