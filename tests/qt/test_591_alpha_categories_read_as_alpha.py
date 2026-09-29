"""Item 591: alpha categories read as alpha, and Image Preprocessing holds the rest.

The maintainer, 2026-09-28: "the settings categories dont need to say alpha
just make them the alpha color and add an α at the end", and on Mask
generation the cloud, illumination correction, point spread function and
image enhancement settings are alpha, the last three as sub-categories of
Image Preprocessing ("Illumination Correction α", "Image Deconvolution α",
"Image Enhancement α").

Pinned here:

* no category is spelled "(Alpha)" any more; every alpha one ends with α,
  is coloured by maturity, and its heading reads "NAME α" -- never "ALPHA";
* the one alpha heading colour is in the stylesheet of both base themes;
* on Mask generation the three sub-categories sit inside Image
  Preprocessing, are hidden with Preferences -> Show alpha features off and
  shown with it on, and a value set while hidden still reaches the run;
* the gate is scoped: the same illumination settings stay ordinary settings
  on Measure and on the Illumination Correction module.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings                              # noqa: E402

from spacr import settings as S                                   # noqa: E402

ALPHA = "α"
SUBCATEGORIES = ("Illumination Correction α", "Image Deconvolution α",
                 "Image Enhancement α")
MASK_ALPHA = S.ALPHA_FEATURES[591]["module_settings"]["mask"]


@pytest.fixture
def prefs(tmp_path, monkeypatch):
    """Preferences read and written through a throwaway INI file."""
    from spacr.qt import preferences

    path = tmp_path / "alpha.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    return preferences


def _every_category_title():
    from spacr.qt.screens.settings_model import _APP_CATEGORY_SPECS

    titles = set(S.categories)
    for spec in _APP_CATEGORY_SPECS.values():
        titles.update(title for title, _tokens in spec)
    return titles


def test_no_category_says_alpha_and_every_alpha_one_ends_with_the_mark():
    from spacr.qt.screens.app_screen import settings_section_maturity

    titles = _every_category_title()
    assert not [t for t in titles if "(alpha)" in t.lower()]
    marked = [t for t in titles if t.endswith(ALPHA)]
    assert len(marked) >= 20
    for title in marked:
        assert title.endswith(" " + ALPHA), title
        assert settings_section_maturity("mask", title) == "alpha", title
    assert set(SUBCATEGORIES) <= titles
    assert "Cloud α" in titles


def test_an_alpha_heading_reads_name_and_mark(qtbot):
    from spacr.qt.widgets.section import Section

    section = Section("Confluency α")
    qtbot.addWidget(section)
    section.set_maturity("alpha")
    assert section._header.text() == "CONFLUENCY α"
    assert section.property("maturity") == "alpha"
    assert section._header.property("maturity") == "alpha"

    legacy = Section("Something (Alpha)")
    qtbot.addWidget(legacy)
    legacy.set_maturity("alpha")
    assert legacy._header.text() == "SOMETHING α"

    whole_module = Section("Input Data")
    qtbot.addWidget(whole_module)
    whole_module.set_maturity("alpha")
    assert whole_module._header.text() == "INPUT DATA α"
    assert "ALPHA" not in whole_module._header.text()


@pytest.mark.parametrize("theme_name", ["dark", "light"])
def test_one_alpha_heading_colour_in_both_themes(theme_name):
    from spacr.qt import theme

    sheet = theme.stylesheet(theme_name)
    rule = 'QToolButton#SectionHeader[maturity="alpha"] {'
    assert rule in sheet
    body = sheet.split(rule, 1)[1].split("}", 1)[0]
    assert theme.ALPHA_INK.lower() in body.lower()
    palette = theme.palette_for(theme_name)
    assert theme.contrast_ratio(theme.ALPHA_INK, palette["surface"]) >= 3.0


def test_the_mask_layout_nests_the_three_under_image_preprocessing():
    from spacr.qt.screens.settings_model import (_category_parents,
                                                 _nest_sections,
                                                 SettingsSection)

    parents = _category_parents("mask")
    for title in SUBCATEGORIES:
        assert parents[title] == "Image Preprocessing"
    assert "Illumination Correction α" not in _category_parents("measure")

    flat = [SettingsSection("Input & Metadata", [("a", None)]),
            SettingsSection("Image Preprocessing", [("b", None)]),
            SettingsSection("Illumination Correction α", [("c", None)]),
            SettingsSection("Image Deconvolution α", [("d", None)]),
            SettingsSection("Image Quality", [("e", None)])]
    nested = _nest_sections(flat, "mask")
    assert [s.title for s in nested] == ["Input & Metadata",
                                         "Image Preprocessing",
                                         "Image Quality"]
    pre = nested[1]
    assert [label for label, _w in pre.own_rows] == ["b"]
    assert [c.title for c in pre.children] == ["Illumination Correction α",
                                               "Image Deconvolution α"]
    assert pre.children[0].path == ("Image Preprocessing",
                                    "Illumination Correction α")


def test_the_registration_is_scoped_to_mask_generation():
    assert {"illumination_correction", "psf_operation",
            "enhance_clahe"} <= set(MASK_ALPHA)
    assert set(MASK_ALPHA) <= set(S.expected_types)
    assert "illumination_correction" in S._alpha_names("settings", "mask")
    assert "illumination_correction" in S._alpha_names("settings",
                                                       "timelapse")
    assert "illumination_correction" not in S._alpha_names("settings")
    assert "illumination_correction" not in S._alpha_names("settings",
                                                           "measure")
    for key in ("cloud_profile", "cloud_endpoint"):
        assert key in S._alpha_names("settings")


def _heading(screen, name):
    for section in screen.rendered_settings_sections():
        if str(section.title()).upper() == name.upper():
            return section
    raise AssertionError(f"no {name!r} card on the form")


def test_mask_generation_hides_and_shows_the_three(qtbot, prefs):
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.settings_search import ALL, install
    from spacr.qt.widget_cleanup import retire_pyqtgraph_menus
    from spacr.qt.widgets.section import _sections_below

    screen = AppScreen("mask")
    try:
        bar = install(screen) or getattr(screen, "_settings_search", None)
        if bar is not None:
            bar.set_level(ALL)
        keys = [k for k in MASK_ALPHA
                if k in screen._settings_model._widgets]
        assert "illumination_correction" in keys
        assert "enhance_clahe" in keys
        for key in keys:
            screen._open_the_heading_of(key)
        screen._refresh_alpha_visibility()

        parent = _heading(screen, "Image Preprocessing")
        below = {str(s.title()) for s in _sections_below(parent)}
        assert {t.upper() for t in SUBCATEGORIES} <= below
        assert not parent.isHidden()
        assert screen.setting_row_is_visible("normalize")
        for title in SUBCATEGORIES:
            heading = _heading(screen, title)
            assert heading.maturity() == "alpha"
            assert heading._header.text().endswith(" " + ALPHA)
            assert heading.isHidden(), title
        assert not any(screen.setting_row_is_visible(k) for k in keys)
        if bar is not None:
            assert not set(keys) & set(bar.indexed_keys())

        model = screen._settings_model
        assert model.set_value_for_key("illumination_correction", True)
        assert model.set_value_for_key("enhance_clahe", True)
        collected = model.collect()
        assert collected["illumination_correction"] is True
        assert collected["enhance_clahe"] is True

        prefs._set_show_alpha_features(True)
        screen._refresh_alpha_visibility()
        for title in SUBCATEGORIES:
            assert not _heading(screen, title).isHidden(), title
        assert screen.setting_row_is_visible("illumination_correction")
        if bar is not None:
            assert "illumination_correction" in set(bar.indexed_keys())

        prefs._set_show_alpha_features(False)
        screen._refresh_alpha_visibility()
        assert _heading(screen, "Illumination Correction α").isHidden()
        assert model.collect()["illumination_correction"] is True
    finally:
        retire_pyqtgraph_menus(screen)
        screen.close()
        screen.deleteLater()


@pytest.mark.parametrize("app_key", ["illumination"])
def test_the_same_settings_stay_ordinary_elsewhere(qtbot, prefs, app_key):
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widget_cleanup import retire_pyqtgraph_menus

    screen = AppScreen(app_key)
    try:
        screen._open_the_heading_of("illumination_correction")
        screen._refresh_alpha_visibility()
        assert prefs._is_alpha_visible() is False
        assert screen.setting_row_is_visible("illumination_correction")
    finally:
        retire_pyqtgraph_menus(screen)
        screen.close()
        screen.deleteLater()
