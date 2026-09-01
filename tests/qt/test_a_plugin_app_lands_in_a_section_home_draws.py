"""Every plugin section must be one the Home screen actually draws.

Found 2026-09-01 while clearing a red test, and it was a real regression
rather than a stale assertion.

``register_app`` refuses a section it does not know. The plugin
registration loop catches that ValueError PER PLUGIN and carries on, so
a section that no longer exists does not raise anywhere a user or a
developer would see -- the app is simply dropped, and Home is missing a
tile nobody can explain.

The 2026-08-31 restructure left Core/Data/Tools/Assays, and
``_PLUGIN_SECTION_MAP`` still pointed ``results``, ``models``,
``explore`` and ``design`` at the retired names. Since
``AppContribution.section`` DEFAULTS to ``"results"``, every plugin app
that did not name a section was being thrown away. One red test was the
only sign.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt import app as app_mod


def test_every_mapped_section_is_one_home_draws():
    """THE GUARD. This is the check whose absence let the map rot."""
    unknown = {name: section
               for name, section in app_mod._PLUGIN_SECTION_MAP.items()
               if section not in app_mod.SECTION_ORDER}
    assert not unknown, (
        f"these plugin sections point at sections Home no longer draws, "
        f"so every plugin using them is dropped in silence: {unknown}")


def test_the_default_section_a_plugin_gets_is_mapped():
    """`AppContribution.section` has a default, and a plugin that never
    names one must still be registered."""
    from spacr import plugins

    contribution = plugins.AppContribution(
        key="probe", name="Probe", entrypoint="", description="", defaults="")

    assert contribution.section in app_mod._PLUGIN_SECTION_MAP, (
        f"the default section {contribution.section!r} is not mapped")
    assert (app_mod._PLUGIN_SECTION_MAP[contribution.section]
            in app_mod.SECTION_ORDER)


def test_every_section_the_plugin_layer_accepts_is_mapped():
    """A section spacr.plugins validates but this map has no key for is
    a KeyError, which the loop's `except (ValueError, TypeError)` does
    NOT catch -- so it would take every later plugin down with it."""
    from spacr import plugins

    accepted = getattr(plugins, "_SECTIONS", None)
    if not accepted:
        pytest.skip("the plugin layer does not publish its section list")

    missing = set(accepted) - set(app_mod._PLUGIN_SECTION_MAP)
    assert not missing, (
        f"spacr.plugins accepts {sorted(missing)} and app.py cannot map "
        f"them; that is a KeyError, not a skipped plugin")


def test_a_plugin_app_with_the_default_section_really_registers(monkeypatch):
    """THE BEHAVIOUR, not just the table.

    Drives `register_app` with the section a default contribution gets,
    which is what the registration loop does.
    """
    from spacr import plugins

    contribution = plugins.AppContribution(
        key="cov_plugin_section_probe", name="Probe", entrypoint="",
        description="a probe", defaults="")
    section = app_mod._PLUGIN_SECTION_MAP[contribution.section]

    try:
        app_mod.register_app(contribution.key, contribution.name,
                             contribution.description, section,
                             factory=lambda **_kwargs: None)
        row = next((r for r in app_mod.APPS if r[0] == contribution.key), None)
        assert row is not None, "the plugin app was dropped"
        assert row[3] == section
    finally:
        app_mod.unregister_app(contribution.key)


def test_a_section_home_does_not_draw_is_still_refused():
    """The refusal itself must stay -- it is what makes the map matter."""
    with pytest.raises(ValueError, match="unknown section"):
        app_mod.register_app("cov_bad_section_probe", "Probe", "a probe",
                             "Explore", factory=lambda **_kwargs: None)


def test_the_retired_names_still_load_an_existing_manifest():
    """A plugin shipped before the restructure names a section that is
    gone. Refusing it would break installs; it is aimed at where its
    built-ins actually went instead."""
    for retired in ("results", "models", "explore", "design"):
        assert retired in app_mod._PLUGIN_SECTION_MAP, (
            f"{retired!r} was dropped, so plugins declaring it now raise "
            f"KeyError instead of being placed")
        assert (app_mod._PLUGIN_SECTION_MAP[retired]
                in app_mod.SECTION_ORDER)
