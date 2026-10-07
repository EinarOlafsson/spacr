"""The retained data-art presets keep their visual and settings contract."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QComboBox

from spacr.qt import night_themes as catalog
from spacr.qt import preferences as prefs
from spacr.qt import sound_synth, theme
from spacr.qt.preferences_navigation import row_field
from spacr.qt.widgets import ambient

SPEC = (
    ('data_art_impulse_lens', 'spaCR field', 'A crisp gravitational dot field with optional local mouse influence and expanding ripples.', 'mono'),
    ('data_art_genetic_advection', 'spaCR advection', 'Fine particles form evolving vortices and branching currents, with optional mouse gravity.', 'ocean'),
    ('data_art_fungal_growth', 'spaCR growth', 'Connected mycelial filaments grow from common origins, with wandering tips and recursively branching fronts. Older trails fade as new colonies begin.', 'deepwater'),
    ('data_art_point_atlas', 'spaCR waves', 'An edge-free landscape of round points carries wide travelling waves.', 'midnight'),
    ('data_art_tissue_facets', 'spaCR spinn', 'Fine paper facets move gently and respond locally to the mouse.', 'lowsun'),
)

RETIRED = (
    "bokeh", "resonance", "ripple", "cells", "data_art_spatial_strata",
    "data_art_molecular_helix", "data_art_sequence_matrix",
    "data_art_transcript_rain", "data_art_regulatory_circuit",
    "data_art_interference", "data_art_morphogenesis",
    "data_art_thore", "data_art_chromatin_ribbon",
)


@pytest.fixture
def private_store(monkeypatch, tmp_path):
    path = tmp_path / "data-art.ini"
    monkeypatch.setattr(prefs, "_settings", lambda: QSettings(str(path), QSettings.IniFormat))
    return prefs


def test_retained_data_art_choices_are_separate_from_ten_night_themes():
    expected = tuple(row[0] for row in SPEC)
    assert catalog.DATA_ART_THEME_KEYS == expected
    assert len(catalog.NIGHT_THEME_KEYS) == 10
    assert set(expected).isdisjoint(catalog.NIGHT_THEME_KEYS)
    assert not any(key.startswith("flow_") for key in theme.THEMES)
    assert tuple(theme.THEMES) == tuple(prefs.PALETTE_THEMES)
    assert {token for _label, token in prefs.theme_choices()} >= set(expected)
    assert len({theme.palette_for(key)["page"] for key in expected}) == len(expected)
    assert len({theme.palette_for(key)["accent"] for key in expected}) == len(expected)


@pytest.mark.parametrize("key", RETIRED)
def test_retired_art_and_classics_cannot_reenter_user_choices(private_store, key):
    assert key not in ambient.AMBIENT_THEMES
    assert key not in prefs.POPUP_BACKDROPS
    assert key not in prefs.PALETTE_THEMES
    with pytest.raises(ValueError):
        private_store.set_ambient_animation(key)
    with pytest.raises(ValueError):
        ambient.make_engine(key, "spacr", "#101418")
    settings = private_store._settings()
    settings.setValue(private_store._KEY_AMBIENT_THEME, key)
    settings.setValue(private_store._KEY_POPUP_BACKDROP, key)
    settings.sync()
    assert private_store.get_ambient_theme() == ambient.DEFAULT_THEME
    assert private_store.get_popup_backdrop() == prefs.DEFAULT_POPUP_BACKDROP
    if key.startswith("data_art_"):
        with pytest.raises(ValueError):
            private_store.set_theme_choice(key)
        with pytest.raises(KeyError):
            catalog.theme_for(key)
        settings.setValue(private_store._KEY_THEME, key)
        settings.sync()
        assert private_store.get_theme() == private_store.DEFAULT_THEME


@pytest.mark.parametrize("key,label,note,palette", SPEC)
def test_catalog_preset_names_exact_renderer_and_colour_resources(key, label, note, palette):
    record = catalog.theme_for(key)
    assert record.key == record.ambient == key
    assert record.label == label
    assert record.description == note
    assert record.ambient_palette == palette
    assert ambient.animation_label(key) == label
    assert ambient.animation_note(key) == note
    assert key in ambient.AMBIENT_THEMES
    assert palette in ambient.palettes_for(key)
    assert record.sound in sound_synth.SOUND_THEMES
    assert catalog.sound_for(key) == record.sound
    assert catalog.ambient_for(key) == (key, palette)
    assert prefs.theme_description(key) == note
    assert theme.contrast_failures(key) == []
    assert theme.page_separation_failures(key) == []
    actual = theme.palette_for(key)
    assert set(actual) == set(theme.palette_for("dark"))
    assert theme.relative_luminance(actual["page"]) < 0.12
    for role in ("success", "warning", "error"):
        assert actual[role] == theme.palette_for(catalog.NIGHT_THEME_KEYS[0])[role]


def test_data_art_palettes_remain_readable_through_spaceout_drift():
    failures = []
    theme.enable_spaceout()
    try:
        for key in catalog.DATA_ART_THEME_KEYS:
            for drift in theme._drift_grid():
                with theme._dressed_at(drift):
                    failures.extend(
                        (key, drift, failure) for failure in theme.contrast_failures(key)
                    )
                    failures.extend(
                        (key, drift, failure) for failure in theme.page_separation_failures(key)
                    )
    finally:
        theme.disable_spaceout()
    assert failures == []


@pytest.mark.parametrize("key,label,note,palette", SPEC)
def test_selection_persists_the_matching_backdrop_and_sound_without_enabling_it(
    private_store, key, label, note, palette
):
    private_store.set_theme_choice(key)
    assert private_store.get_theme_choice() == key
    assert private_store.resolve_effective_theme() == key
    assert private_store.get_ambient_animation() == key
    assert private_store.get_ambient_palette() == palette
    assert private_store.get_sound_theme() == catalog.sound_for(key)
    assert private_store.get_sound_enabled() is False


def test_motion_off_and_crash_suppression_do_not_rewrite_one_another(private_store, monkeypatch):
    private_store.set_ambient_animation("none")
    private_store.set_theme_choice("data_art_genetic_advection")
    assert private_store.get_ambient_animation() == "none"
    assert private_store.get_ambient_enabled() is False
    private_store.set_ambient_animation("blobs")
    private_store.set_ambient_enabled(False)
    private_store.set_theme_choice("data_art_impulse_lens")
    assert private_store.get_ambient_animation() == "blobs"
    assert private_store.get_ambient_enabled() is False
    private_store.set_ambient_enabled(True)
    monkeypatch.setenv("SPACR_NO_BACKDROP", "1")
    private_store.set_theme_choice("data_art_tissue_facets")
    assert private_store.get_ambient_animation() == "data_art_tissue_facets"
    assert private_store.get_ambient_enabled() is False
    monkeypatch.delenv("SPACR_NO_BACKDROP")
    assert private_store.get_ambient_enabled() is True
    assert private_store.get_sound_enabled() is False


def test_extra_performance_keeps_a_new_preset_static(private_store):
    private_store.set_spacr_mode("extra_performance")
    assert private_store.get_ambient_animation() == "none"
    private_store.set_theme_choice("data_art_fungal_growth")
    assert private_store.get_theme_choice() == "data_art_fungal_growth"
    assert private_store.get_ambient_animation() == "none"
    assert private_store.get_ambient_enabled() is False
    assert private_store.get_sound_enabled() is False


def test_plain_theme_keeps_the_data_art_resources_as_the_user_last_set_them(private_store):
    private_store.set_theme_choice("data_art_point_atlas")
    before = (
        private_store.get_ambient_animation(),
        private_store.get_ambient_palette(),
        private_store.get_sound_theme(),
    )
    private_store.set_theme_choice("dark")
    assert private_store.get_theme_choice() == "dark"
    assert (
        private_store.get_ambient_animation(),
        private_store.get_ambient_palette(),
        private_store.get_sound_theme(),
    ) == before


def test_dialog_shows_data_art_preset_and_respects_no_animation(
    private_store, qtbot, qt_theme_applied, monkeypatch
):
    monkeypatch.setattr(theme, "spaceout_enabled", lambda: True)
    dialog = private_store.PreferencesDialog()
    qtbot.addWidget(dialog)
    choice = row_field(dialog, "Theme")
    animation = dialog.findChild(QComboBox, "AmbientTheme")
    palette = dialog.findChild(QComboBox, "AmbientPalette")
    sound = dialog.findChild(QComboBox, "SoundTheme")
    assert all(widget is not None for widget in (choice, animation, palette, sound))
    assert {choice.itemData(index) for index in range(choice.count())} >= {row[0] for row in SPEC}
    choice.setCurrentIndex(choice.findData("data_art_tissue_facets"))
    assert animation.currentData() == "data_art_tissue_facets"
    assert palette.currentData() == "lowsun"
    assert sound.currentData() == "halcyon"
    assert (
        choice.itemData(choice.currentIndex(), Qt.ItemDataRole.ToolTipRole)
        == catalog.DATA_ART_THEMES["data_art_tissue_facets"].description
    )
    animation.setCurrentIndex(animation.findData("none"))
    choice.setCurrentIndex(choice.findData("data_art_impulse_lens"))
    assert animation.currentData() == "none"


def test_unknown_data_art_key_is_rejected_without_changing_a_choice(private_store):
    private_store.set_theme_choice("data_art_point_atlas")
    with pytest.raises(ValueError):
        private_store.set_theme_choice("data_art_not_an_engine")
    assert private_store.get_theme_choice() == "data_art_point_atlas"
    with pytest.raises(KeyError):
        catalog.theme_for("data_art_not_an_engine")


def test_requested_animation_order_and_default_reach_the_actual_dialog(
    private_store, qtbot, qt_theme_applied,
):
    labels = ("spaCR field", "spaCR advection", "spaCR growth", "spaCR waves",
              "spaCR blobs", "spaCR aurora")
    assert tuple(ambient.animation_label(key) for key in ambient.AMBIENT_THEMES[:6]) == labels
    assert ambient.DEFAULT_THEME == "data_art_impulse_lens"
    assert not private_store._settings().contains(private_store._KEY_AMBIENT_THEME)
    assert private_store.get_ambient_animation() == "data_art_impulse_lens"
    dialog = private_store.PreferencesDialog()
    qtbot.addWidget(dialog)
    combo = dialog.findChild(QComboBox, "AmbientTheme")
    assert tuple(combo.itemText(index) for index in range(6)) == labels
    assert combo.currentData() == "data_art_impulse_lens"
    assert combo.itemData(combo.count() - 1) == "none"
    assert private_store._ambient_gravity_radius() == pytest.approx(0.15)
