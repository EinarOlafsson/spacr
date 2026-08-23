"""The documented-inert escape hatch is a closed list, and it only shrinks.

``tests/test_dead_settings.py`` states the rule: a setting spaCR declares is
a setting spaCR reads, and a key that loses its last reader is deleted. It
enforces that by scanning every module for a read -- except that a setting
whose own description ADMITS nothing reads it is waved through, so that the
text a user sees is at least honest about the control being inert.

That exemption has no floor. Any number of dead settings can accumulate
behind it, each individually excused, and nothing counts them. Ten are
behind it today. They are listed here by name, and the list may only get
shorter: adding a twelfth fails, and deleting one fails too, until somebody
edits this file -- which is the point, because both directions are
decisions worth seeing in a diff.

``complevel`` was the eleventh and is gone. It was not a superseded feature
but a MISSPELLING of ``comp_level``, which is live and is read at
``sequencing.py`` where the HDF5 store is opened -- so the panel offered two
compression-level boxes, one of which did nothing. Its own description was
wrong about the live one too, calling ``comp_level`` a function parameter
rather than the setting it is.

The other ten are real supersessions and deleting them is a user-facing
change: an old settings CSV naming one would stop loading. Recorded in
``instructions/open/237_every_core_module_driven_on_real_data.txt`` for a
decision rather than taken here.
"""
from __future__ import annotations

import pathlib

import pytest

import spacr.settings as S

#: Cached so the guard module is executed once, not once per parametrised case.
_ADMITS = None


#: Settings whose description admits nothing reads them. NOT a registry --
#: nothing consults this at runtime. It exists so the number is visible.
#:
#: Five belong to Map barcodes, superseded by ``target_sequence`` plus
#: ``offset_start`` for the read window and by ``grna_csv``/``row_csv``/
#: ``column_csv`` for the references.
INERT_SETTINGS = frozenset({
    "barcode_coordinates",
    "barcode_mapping",
    "compartments",
    "compression",
    "correlate",
    "downstream",
    "split_axis_lims",
    "upscale",
    "upscale_factor",
    "upstream",
})


def _description(key):
    """What the user is told about ``key``, from whichever table has it."""
    for table_name in ("descriptions", "tooltips"):
        table = getattr(S, table_name, None)
        if isinstance(table, dict) and key in table:
            return str(table[key])
    return ""


def _admits_it_is_dead(key):
    """Does ``key``'s own description say nothing reads it?

    The phrase list is loaded FROM the dead-settings guard rather than
    re-spelled here, so the two cannot drift into disagreeing about what
    counts as an admission. By path, because the tests directory is not an
    importable package.
    """
    import importlib.util

    global _ADMITS
    if _ADMITS is None:
        path = pathlib.Path(__file__).with_name("test_dead_settings.py")
        spec = importlib.util.spec_from_file_location("_dead_settings", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _ADMITS = module._ADMITS_IT_IS_DEAD
    return bool(_ADMITS.search(_description(key)))


def test_no_setting_has_joined_the_inert_list():
    """A new dead setting must be deleted, not excused by its own tooltip."""
    admitted = {k for k in S.expected_types if _admits_it_is_dead(k)}
    added = sorted(admitted - INERT_SETTINGS)
    assert not added, (
        f"{added} declare themselves inert. A setting nothing reads is "
        f"deleted, not documented -- see tests/test_dead_settings.py. If one "
        f"of these really must stay, add it here and say why.")


def test_the_inert_list_has_no_stale_entries():
    """A name here that is no longer inert means the list is out of date."""
    admitted = {k for k in S.expected_types if _admits_it_is_dead(k)}
    stale = sorted(INERT_SETTINGS - admitted)
    assert not stale, (
        f"{stale} are listed as inert but no longer are -- either they were "
        f"deleted (good: remove them here) or they were WIRED UP (better: "
        f"remove them here and test what they now do).")


def test_complevel_is_gone_and_comp_level_is_the_live_one():
    """The misspelling must not come back while the real setting stays."""
    assert "complevel" not in S.expected_types
    assert S.expected_types["comp_level"] is int
    assert S.expected_types["comp_type"] is str


@pytest.mark.parametrize("key", sorted(INERT_SETTINGS))
def test_an_inert_setting_still_says_so_to_the_user(key):
    """Whatever else is true of it, the description may not claim it works."""
    assert key in S.expected_types
    assert _admits_it_is_dead(key), (
        f"{key!r} is listed as inert but its description no longer says so")


# ---------------------------------------------------------------------------
# and none of them is a control the user can change
# ---------------------------------------------------------------------------

def test_no_app_offers_a_setting_nothing_reads(qapp):
    """A control that changes nothing is worse than no control at all.

    Verified before hiding rather than assumed: none of the ten has a
    `settings[...]`, `.get`, `.pop` or membership test anywhere in the
    package outside the tables that DECLARE it; there is no `**settings`
    splat in the package at all; and the two names that collide with a
    function parameter -- `compartments` in
    `submodules.generate_comparison_columns`, `compression` in
    `io.save_object_mask` -- are never passed the setting.

    Mask and Timelapse were rendering `compression`, `upscale` and
    `upscale_factor`.
    """
    from spacr.qt.screens import app_screen
    from spacr.qt.screens.settings_model import (INERT_SETTINGS_NOT_OFFERED,
                                                 _APP_HIDDEN_KEYS,
                                                 resolve_default_settings)

    offenders = {}
    for app_key in sorted(getattr(app_screen, "APP_TITLES", {})):
        try:
            offered = set(resolve_default_settings(app_key))
        except Exception:                                    # noqa: BLE001
            continue
        shown = (offered & set(INERT_SETTINGS_NOT_OFFERED)
                 - set(_APP_HIDDEN_KEYS.get(app_key, ())))
        if shown:
            offenders[app_key] = sorted(shown)

    assert not offenders, (
        f"these apps render a setting nothing reads: {offenders}")


def test_hiding_them_did_not_drop_them(qapp):
    """The key stays in the dict, or an old settings CSV stops loading.

    Hiding and dropping are different: a dropped key is absent from the
    run's settings, the pipeline falls back to ITS default, and the two can
    disagree. These stay at the value the module resolved.
    """
    from spacr.qt.screens.settings_model import resolve_default_settings

    defaults = resolve_default_settings("mask")
    for key in ("compression", "upscale", "upscale_factor"):
        assert key in defaults, f"{key} was dropped rather than hidden"


def test_the_two_lists_are_the_same_ten():
    """One list of inert settings, not two that can drift apart."""
    from spacr.qt.screens.settings_model import INERT_SETTINGS_NOT_OFFERED

    assert set(INERT_SETTINGS_NOT_OFFERED) == set(INERT_SETTINGS)
