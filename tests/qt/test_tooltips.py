"""Tests for the standardized, type-aware setting tooltips."""
from __future__ import annotations


def test_every_app_api_link_targets_an_existing_module():
    """Tooltip links must follow the current app registry, not legacy launchers."""
    import importlib.util

    from spacr.qt.app import APPS
    from spacr.qt.screens.settings_model import _APP_API_MODULE

    missing = []
    for app_key, _name, _description, _section in APPS:
        target = _APP_API_MODULE.get(app_key)
        if not target:
            missing.append(f"{app_key}: no API mapping")
            continue
        module_name = "spacr." + target.replace("/", ".")
        if importlib.util.find_spec(module_name) is None:
            missing.append(f"{app_key}: {module_name} does not exist")
    assert not missing, "\n".join(missing)


def test_pipeline_app_api_links_follow_the_actual_backend():
    from spacr.cli import INTERACTIVE_ONLY
    from spacr.qt.app import APPS
    from spacr.qt.bridge import resolve_pipeline_entry
    from spacr.qt.screens.settings_model import _APP_API_MODULE

    mismatches = []
    for app_key, _name, _description, _section in APPS:
        if app_key in INTERACTIVE_ONLY:
            continue
        entry = resolve_pipeline_entry(app_key)
        real = getattr(entry, "__wrapped__", entry)
        expected = getattr(real, "__module__", "").removeprefix("spacr.")
        linked = _APP_API_MODULE[app_key].replace("/", ".")
        if linked != expected:
            mismatches.append(f"{app_key}: links {linked}, runs {expected}")
    assert not mismatches, "\n".join(mismatches)


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
    assert 'href="https://einarolafsson.github.io/spacr/api/' in tip
    assert "Open spaCR API documentation" in tip


def test_undescribed_setting_still_typed():
    from spacr.qt.screens.settings_model import format_tooltip
    tip = format_tooltip("", "mask", "compression")
    assert "<b>Compression</b>" in tip and "(string)" in tip


def test_plain_tooltip_typed():
    from spacr.qt.screens.settings_model import plain_tooltip
    p = plain_tooltip("Whether to plot.", "mask", "plot")
    assert p.startswith("Plot (boolean)")
    assert "Whether to plot." in p
    assert "API: https://einarolafsson.github.io/spacr/api/" in p


def test_every_typed_setting_has_a_written_description():
    """Every setting with a known type also has a written tooltip description —
    no setting shows only 'Name (type)' with no explanation."""
    from spacr.settings import tooltips, expected_types
    missing = sorted(set(expected_types) - set(tooltips))
    assert missing == [], f"settings with a type but no description: {missing}"


def test_no_freaction_typo_in_tooltips():
    """Guard against the 'Freaction' typo reappearing in user-facing tooltips."""
    from spacr.settings import tooltips
    offenders = [k for k, v in tooltips.items()
                 if isinstance(v, str) and "Freaction" in v]
    assert offenders == [], f"'Freaction' typo in: {offenders}"


def test_every_shown_setting_has_a_typed_tooltip(qtbot, qt_theme_applied):
    """Every widget on the mask screen gets a tooltip, and most carry a type."""
    from spacr.qt.screens.settings_model import SettingsWidgets
    m = SettingsWidgets("mask")
    m.build_sections()
    typed = 0
    for key, w in m._widgets.items():
        tip = w.toolTip()
        assert tip, f"{key} has no tooltip"
        assert "href=" in tip, f"{key} has no API documentation link"
        if "<i>(" in tip:
            typed += 1
    # The large majority of mask settings are in expected_types → typed.
    assert typed > 0.5 * len(m._widgets)
