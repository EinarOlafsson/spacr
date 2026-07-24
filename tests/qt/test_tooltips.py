"""Tests for the standardized, type-aware setting tooltips."""
from __future__ import annotations


def test_type_hint_from_expected_types():
    from spacr.qt.screens.settings_model import _type_hint
    assert _type_hint("cell_min_area") == "integer"
    assert _type_hint("plot") == "boolean"
    assert _type_hint("compression") == "string"
    # union / optional types render readably
    h = _type_hint("cell_background")
    assert "integer" in h or "float" in h


def test_format_tooltip_shows_name_type_and_strips_old_prefix():
    from spacr.qt.screens.settings_model import format_tooltip
    tip = format_tooltip("(int) - Expected cell diameter.", "mask", "cell_diameter")
    assert "<b>Cell diameter</b>" in tip
    assert "(integer)" in tip
    assert "Expected cell diameter." in tip
    assert "(int) -" not in tip           # old inline type prefix removed
    assert 'href=' in tip                 # Docs link


def test_undescribed_setting_still_typed():
    from spacr.qt.screens.settings_model import format_tooltip
    tip = format_tooltip("", "mask", "compression")
    assert "<b>Compression</b>" in tip and "(string)" in tip


def test_plain_tooltip_typed():
    from spacr.qt.screens.settings_model import plain_tooltip
    p = plain_tooltip("Whether to plot.", "mask", "plot")
    assert p.startswith("Plot (boolean)")
    assert "Whether to plot." in p


def test_every_shown_setting_has_a_typed_tooltip(qtbot, qt_theme_applied):
    """Every widget on the mask screen gets a tooltip, and most carry a type."""
    from spacr.qt.screens.settings_model import SettingsWidgets
    m = SettingsWidgets("mask")
    m.build_sections()
    typed = 0
    for key, w in m._widgets.items():
        tip = w.toolTip()
        assert tip, f"{key} has no tooltip"
        if "<i>(" in tip:
            typed += 1
    # The large majority of mask settings are in expected_types → typed.
    assert typed > 0.5 * len(m._widgets)
