"""Flow theme palettes, persisted choices and existing preference switches."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QComboBox

from spacr.qt import night_themes as catalog
from spacr.qt import preferences as prefs
from spacr.qt import sound_synth
from spacr.qt import theme
from spacr.qt.preferences_navigation import row_field
from spacr.qt.widgets import ambient


@pytest.fixture
def private_store(monkeypatch, tmp_path):
    path = tmp_path / "flow-theme.ini"
    monkeypatch.setattr(
        prefs, "_settings", lambda: QSettings(str(path), QSettings.IniFormat))
    return prefs


def test_sixteen_flow_choices_are_separate_from_the_original_ten():
    bases = (
        "flow_cytoplasm", "flow_synapse", "flow_wind", "flow_atlas",
        "flow_helix", "flow_chromatin", "flow_nebula", "flow_silk",
    )
    assert catalog.FLOW_THEME_KEYS == tuple(
        key for base in bases for key in (base, f"{base}_mouse"))
    assert len(catalog.NIGHT_THEME_KEYS) == 10
    assert set(catalog.FLOW_THEME_KEYS).isdisjoint(catalog.NIGHT_THEME_KEYS)
    assert tuple(theme.THEMES) == tuple(prefs.PALETTE_THEMES)
    assert set(catalog.FLOW_THEME_KEYS).issubset(theme.THEMES)
    assert {token for _label, token in prefs.theme_choices()} >= set(
        catalog.FLOW_THEME_KEYS)


@pytest.mark.parametrize("key", catalog.FLOW_THEME_KEYS)
def test_flow_palette_and_preset_refer_to_real_resources(key):
    record = catalog.theme_for(key)
    assert record.key == key
    assert record.ambient == key
    assert record.sound in sound_synth.SOUND_THEMES
    assert record.ambient in ambient.AMBIENT_THEMES
    assert record.ambient_palette in ambient.palettes_for(key)
    assert theme.contrast_failures(key) == []
    assert theme.page_separation_failures(key) == []
    assert set(theme.palette_for(key)) == set(theme.palette_for("dark"))
    assert prefs.theme_description(key) == record.description


def test_each_geometry_has_one_palette_for_its_pointer_variant():
    for base in catalog.FLOW_THEME_KEYS[::2]:
        pointer = f"{base}_mouse"
        assert theme.palette_for(base) == theme.palette_for(pointer)
        assert catalog.sound_for(base) == catalog.sound_for(pointer)
        assert catalog.theme_for(pointer).label != catalog.theme_for(base).label
    assert len({theme.palette_for(base)["page"]
                for base in catalog.FLOW_THEME_KEYS[::2]}) == 8
    for role in ("success", "warning", "error"):
        assert {theme.palette_for(key)[role]
                for key in catalog.FLOW_THEME_KEYS} == {
                    theme.palette_for(catalog.NIGHT_THEME_KEYS[0])[role]}


def test_flow_palettes_stay_readable_through_spaceout_drift():
    failures = []
    theme.enable_spaceout()
    try:
        for key in catalog.FLOW_THEME_KEYS[::2]:
            for drift in theme._drift_grid():
                with theme._dressed_at(drift):
                    failures.extend((key, drift, failure)
                                    for failure in theme.contrast_failures(key))
    finally:
        theme.disable_spaceout()
    assert failures == []


@pytest.mark.parametrize("key", catalog.FLOW_THEME_KEYS)
def test_flow_choice_persists_without_turning_on_sound_or_motion(
        private_store, key):
    private_store.set_theme_choice(key)
    record = catalog.theme_for(key)
    assert private_store.get_theme_choice() == key
    assert private_store.resolve_effective_theme() == key
    assert private_store.get_ambient_animation() == record.ambient
    assert private_store.get_ambient_palette() == record.ambient_palette
    assert private_store.get_sound_theme() == record.sound
    assert private_store.get_sound_enabled() is False


def test_flow_choice_respects_disabled_backdrop_and_sound(private_store):
    private_store.set_ambient_animation("none")
    private_store.set_theme_choice("flow_synapse_mouse")
    assert private_store.get_theme_choice() == "flow_synapse_mouse"
    assert private_store.get_ambient_animation() == "none"
    assert private_store.get_ambient_enabled() is False
    assert private_store.get_sound_theme() == catalog.sound_for(
        "flow_synapse_mouse")
    assert private_store.get_sound_enabled() is False


def test_dialog_flow_choice_shows_the_preset_without_enabling_motion(
        private_store, qtbot):
    dialog = private_store.PreferencesDialog()
    qtbot.addWidget(dialog)
    theme_combo = row_field(dialog, "Theme")
    ambient_combo = dialog.findChild(QComboBox, "AmbientTheme")
    palette_combo = dialog.findChild(QComboBox, "AmbientPalette")
    assert theme_combo is not None
    assert ambient_combo is not None
    index = theme_combo.findData("flow_nebula_mouse")
    assert index >= 0
    theme_combo.setCurrentIndex(index)
    assert ambient_combo.currentData() == "flow_nebula_mouse"
    assert palette_combo.currentData() == "dusk"
    none_index = ambient_combo.findData("none")
    ambient_combo.setCurrentIndex(none_index)
    theme_combo.setCurrentIndex(theme_combo.findData("flow_silk"))
    assert ambient_combo.currentData() == "none"
