"""Twelve independent data-art presets keep their visual and settings contract."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QComboBox

from spacr.qt import night_themes as catalog
from spacr.qt import preferences as prefs
from spacr.qt import sound_synth, theme
from spacr.qt.preferences_navigation import row_field
from spacr.qt.widgets import ambient

SPEC = (
    (
        "data_art_point_atlas",
        "Spatial point atlas",
        "A finely sampled three-dimensional point landscape with depth and cursor-driven parallax.",
        "midnight",
    ),
    (
        "data_art_tissue_facets",
        "Tissue facets",
        "A crystalline tissue mosaic of shaded geometric facets, with slowly changing local relief.",
        "lowsun",
    ),
    (
        "data_art_spatial_strata",
        "Spatial strata",
        "Fine stacked topographic layers form a moving spatial relief with precise depth and contour detail.",
        "mono",
    ),
    (
        "data_art_molecular_helix",
        "Molecular helix",
        "A rotating molecular helix of shaded beads and paired bases, with perspective and depth.",
        "ocean",
    ),
    (
        "data_art_chromatin_ribbon",
        "Chromatin satin",
        "Folded satin-like chromatin ribbons carry fine fibres through soft, interwoven surfaces.",
        "dusk",
    ),
    (
        "data_art_sequence_matrix",
        "Genome mosaic",
        "A layered genome mosaic of tiny encoded tiles shifts through an architectural sequence field.",
        "fluor",
    ),
    (
        "data_art_transcript_rain",
        "Transcript rain",
        "Fine falling transcription marks stream through a layered field of genetic information.",
        "deepwater",
    ),
    (
        "data_art_regulatory_circuit",
        "Regulatory circuit",
        "An etched regulatory circuit routes pulses through precise orthogonal paths and small control nodes.",
        "ember",
    ),
    (
        "data_art_genetic_advection",
        "Genetic advection",
        "Thousands of fine genetic-flow particles move through a continuous wind-like field that bends near the cursor.",
        "ocean",
    ),
    (
        "data_art_interference",
        "Perturbation interference",
        "Smooth interference waves form a changing pearlescent field, distorted locally by the cursor.",
        "pastel",
    ),
    (
        "data_art_morphogenesis",
        "Morphogenesis",
        "A fine organic pattern of changing spots and labyrinths evokes the emergence of biological structure.",
        "borealis",
    ),
    (
        "data_art_impulse_lens",
        "Perturbation lens",
        "A precision dot lattice bends around moving impulses and the cursor, revealing local perturbation.",
        "mono",
    ),
)


@pytest.fixture
def private_store(monkeypatch, tmp_path):
    path = tmp_path / "data-art.ini"
    monkeypatch.setattr(prefs, "_settings", lambda: QSettings(str(path), QSettings.IniFormat))
    return prefs


def test_twelve_data_art_choices_are_separate_from_ten_night_themes():
    expected = tuple(row[0] for row in SPEC)
    assert catalog.DATA_ART_THEME_KEYS == expected
    assert len(catalog.NIGHT_THEME_KEYS) == 10
    assert set(expected).isdisjoint(catalog.NIGHT_THEME_KEYS)
    assert not any(key.startswith("flow_") for key in theme.THEMES)
    assert tuple(theme.THEMES) == tuple(prefs.PALETTE_THEMES)
    assert {token for _label, token in prefs.theme_choices()} >= set(expected)
    assert len({theme.palette_for(key)["page"] for key in expected}) == 12
    assert len({theme.palette_for(key)["accent"] for key in expected}) == 12


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
                        (key, drift, failure)
                        for failure in theme.page_separation_failures(key)
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
    private_store.set_theme_choice("data_art_interference")
    assert private_store.get_ambient_animation() == "blobs"
    assert private_store.get_ambient_enabled() is False
    private_store.set_ambient_enabled(True)
    monkeypatch.setenv("SPACR_NO_BACKDROP", "1")
    private_store.set_theme_choice("data_art_molecular_helix")
    assert private_store.get_ambient_animation() == "data_art_molecular_helix"
    assert private_store.get_ambient_enabled() is False
    monkeypatch.delenv("SPACR_NO_BACKDROP")
    assert private_store.get_ambient_enabled() is True
    assert private_store.get_sound_enabled() is False


def test_dialog_shows_data_art_preset_and_respects_no_animation(private_store, qtbot):
    dialog = private_store.PreferencesDialog()
    qtbot.addWidget(dialog)
    choice = row_field(dialog, "Theme")
    animation = dialog.findChild(QComboBox, "AmbientTheme")
    palette = dialog.findChild(QComboBox, "AmbientPalette")
    assert choice is not None and animation is not None and palette is not None
    assert {choice.itemData(index) for index in range(choice.count())} >= {row[0] for row in SPEC}
    choice.setCurrentIndex(choice.findData("data_art_tissue_facets"))
    assert animation.currentData() == "data_art_tissue_facets"
    assert palette.currentData() == "lowsun"
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
