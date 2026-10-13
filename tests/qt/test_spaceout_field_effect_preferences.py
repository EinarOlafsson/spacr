"""Spaceout field effects are independent, persistent and previewable."""

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QComboBox, QDialogButtonBox, QPushButton, QSlider, QWidget

from spacr.qt import preferences, theme
from spacr.qt.widgets.section import Section
from spacr.qt.widgets.toggle import Toggle

NAMES = {
    "attractors": "SpaceoutFieldAttractors",
    "relaxation": "SpaceoutFieldRelaxation",
    "elastic_release": "SpaceoutFieldElasticRelease",
    "vortex": "SpaceoutFieldVortex",
    "density_pulses": "SpaceoutFieldDensityPulses",
    "density_waves": "SpaceoutFieldDensityWaves",
    "color_waves": "SpaceoutFieldColorWaves",
    "spirals": "SpaceoutFieldSpirals",
}


@pytest.fixture
def field_store(tmp_path, monkeypatch):
    store = QSettings(str(tmp_path / "preferences.ini"), QSettings.IniFormat)
    monkeypatch.setattr(preferences, "_settings", lambda: store)
    monkeypatch.setattr(preferences, "_SAFE_MODE", False)
    was_spaceout = theme.spaceout_enabled()
    theme.disable_spaceout()
    yield store
    theme.enable_spaceout() if was_spaceout else theme.disable_spaceout()


def _controls(dialog):
    return {key: dialog.findChild(Toggle, name) for key, name in NAMES.items()}


def test_field_is_default_in_spaceout_and_controls_are_mode_scoped(
        field_store, qtbot):
    assert preferences._spaceout_field_effects() == dict.fromkeys(NAMES, True)
    ordinary = preferences.PreferencesDialog()
    qtbot.addWidget(ordinary)
    assert all(control is None for control in _controls(ordinary).values())
    ordinary.reject()

    theme.enable_spaceout()
    assert preferences.get_ambient_animation() == "data_art_spaceout_field"
    assert preferences.get_ambient_theme() == "data_art_spaceout_field"
    preferences.set_ambient_animation("fractal")
    dialog = preferences.PreferencesDialog()
    qtbot.addWidget(dialog)
    assert dialog.findChild(QWidget, "PreferencesTabSpaceoutField") is not None
    section = next(section for section in dialog.findChildren(Section)
                   if section.title().casefold() == "spacr field")
    assert not section.is_expanded()
    section.set_expanded(True)
    assert section.is_expanded()
    controls = _controls(dialog)
    assert all(control is not None and control.isChecked()
               for control in controls.values())
    assert all(control.accessibleDescription() for control in controls.values())
    dialog.reject()


def test_field_effects_cancel_save_and_reset_restore_exact_values(
        field_store, qtbot):
    theme.enable_spaceout()
    preferences.set_ambient_animation("fractal")
    dialog = preferences.PreferencesDialog()
    qtbot.addWidget(dialog)
    _controls(dialog)["vortex"].setChecked(False)
    _controls(dialog)["color_waves"].setChecked(False)
    dialog.reject()
    assert all(preferences._spaceout_field_effects().values())
    assert not any(key.startswith("spaceout/field/") for key in field_store.allKeys())

    saved = preferences.PreferencesDialog()
    qtbot.addWidget(saved)
    _controls(saved)["vortex"].setChecked(False)
    _controls(saved)["color_waves"].setChecked(False)
    saved.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()
    expected = dict.fromkeys(NAMES, True)
    expected["vortex"] = expected["color_waves"] = False
    assert preferences._spaceout_field_effects() == expected
    assert len([key for key in field_store.allKeys()
                if key.startswith("spaceout/field/")]) == 8

    theme.disable_spaceout()
    ordinary = preferences.PreferencesDialog()
    qtbot.addWidget(ordinary)
    assert all(control is None for control in _controls(ordinary).values())
    ordinary.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()
    assert preferences._spaceout_field_effects() == expected
    theme.enable_spaceout()

    reset = preferences.PreferencesDialog()
    qtbot.addWidget(reset)
    reset.findChild(QPushButton, "PreferencesReset").click()
    assert all(control.isChecked() for control in _controls(reset).values())
    assert preferences._spaceout_field_effects() == expected
    reset.reject()


def test_field_effect_apply_revert_preserves_unrelated_concurrent_key(
        field_store, qtbot):
    theme.enable_spaceout()
    preferences.set_ambient_animation("fractal")
    dialog = preferences.PreferencesDialog()
    qtbot.addWidget(dialog)
    dialog.show()
    control = _controls(dialog)["elastic_release"]
    control.setChecked(False)
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Apply).click()
    qtbot.waitUntil(lambda: dialog._apply_confirmation is not None)
    assert not preferences._spaceout_field_effects()["elastic_release"]
    field_store.setValue("unrelated/concurrent", "keep")
    question = dialog._apply_confirmation
    next(button for button in question.buttons()
         if button.text() == "Revert").click()
    qtbot.waitUntil(lambda: dialog._apply_confirmation is None)
    assert preferences._spaceout_field_effects()["elastic_release"]
    assert field_store.value("unrelated/concurrent") == "keep"
    assert not control.isChecked()
    dialog.reject()


def test_periodic_field_controls_follow_spaceout_field_and_none(field_store, qtbot):
    theme.enable_spaceout()
    preferences.set_ambient_animation("fractal")
    dialog = preferences.PreferencesDialog()
    qtbot.addWidget(dialog)
    animation = dialog.findChild(QComboBox, "AmbientTheme")
    frequency = dialog.findChild(QSlider, "FieldPopupWaveFrequency")
    assert frequency is not None
    field_index = animation.findData("data_art_spaceout_field")
    assert field_index >= 0
    animation.setCurrentIndex(field_index)
    assert frequency.isEnabled()
    animation.setCurrentIndex(animation.findData("none"))
    assert not frequency.isEnabled()
    animation.setCurrentIndex(animation.findData("data_art_impulse_lens"))
    assert frequency.isEnabled()
    dialog.reject()
