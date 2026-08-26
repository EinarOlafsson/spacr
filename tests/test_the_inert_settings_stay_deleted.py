"""Eleven settings nothing read are deleted, and the escape hatch is shut.

``tests/test_dead_settings.py`` states the rule -- a setting spaCR declares
is a setting spaCR reads -- and enforces it by scanning the package for a
read.  A scan by NAME is what makes that rule impossible to fool, and it is
also why the rule alone cannot keep these eleven out: every one of their
names is an ordinary word that occurs elsewhere in ``spacr/`` for unrelated
reasons.  Measured: all eleven register as "live tokens" in that scan today,
with none of them declared.  So re-adding ``upscale_factor`` to
``expected_types`` would pass the absolute rule and put an inert control back
on the Mask panel.

This file is the guard the names need.  It replaced
``tests/test_the_inert_settings_list_only_shrinks.py``, which counted the
keys while they were still declared behind a documented-inert exemption; the
keys were deleted, so the ratchet had nothing left to count and the
protection went with it.

WHAT THEY WERE.  Five belong to Map barcodes, superseded by
``target_sequence`` plus ``offset_start`` for the read window and by
``grna_csv`` / ``row_csv`` / ``column_csv`` for the references.  The rest
were superseded elsewhere: ``compartments`` by the per-role object settings,
``compression`` by a storage backend spaCR stopped writing,
``split_axis_lims`` by limits you change on the plot rather than in a
settings file, and ``upscale``/``upscale_factor`` by a Cellpose-1 resize that
Cellpose 4 does itself from ``diameter``.  ``complevel`` is the eleventh and
is different in kind: not a supersession but a MISSPELLING of the live
``comp_level``, so the panel offered two HDF5 compression-level boxes and one
of them did nothing.
"""
from __future__ import annotations

import contextlib
import io
import re

import pytest

import spacr.settings as S


#: Deleted, and named here so a revert is visible in a diff. NOT a registry:
#: nothing consults this at runtime, and adding a name to it is a decision
#: rather than a fix.
DELETED_INERT_SETTINGS = (
    "barcode_coordinates",
    "barcode_mapping",
    "compartments",
    "compression",
    "complevel",
    "correlate",
    "downstream",
    "split_axis_lims",
    "upscale",
    "upscale_factor",
    "upstream",
)

#: The tables in ``spacr.settings`` that DECLARE a setting. A name in any one
#: of them is a name the application can offer.
_DECLARATION_TABLES = ("expected_types", "tooltips", "descriptions")

#: Phrases a description may use to admit the setting is inert. Kept in step
#: with ``tests/test_dead_settings.py``, which spells the same list for the
#: exemption this file makes sure stays unused.
_ADMITS_IT_IS_DEAD = re.compile(
    r"nothing (in spacr )?reads|no code (path )?in spacr reads|"
    r"never looks at|read by nothing|reads nowhere",
    re.IGNORECASE,
)


def _declaration_sites(key):
    """Every place in ``spacr.settings`` that still names ``key``."""
    sites = []
    for table_name in _DECLARATION_TABLES:
        table = getattr(S, table_name, None)
        if isinstance(table, dict) and key in table:
            sites.append(table_name)
    categories = getattr(S, "categories", None)
    if isinstance(categories, dict):
        for category, keys in categories.items():
            with contextlib.suppress(TypeError):
                if key in keys:
                    sites.append(f"categories[{category!r}]")
    return sites


def _keys_every_factory_produces():
    """Every key any ``set_*``/``get_*``/``default_*``/``deep_*`` helper emits.

    A key can be absent from all the declaration tables and still be handed
    to a run by a defaults factory, which is the half a table scan misses.
    """
    keys = set()
    with contextlib.redirect_stdout(io.StringIO()):
        for name, fn in list(vars(S).items()):
            if not callable(fn):
                continue
            if not name.startswith(("set_", "get_", "default_", "deep_")):
                continue
            try:
                produced = fn({})
            except Exception:                                    # noqa: BLE001
                try:
                    produced = fn()
                except Exception:                                # noqa: BLE001
                    continue
            if isinstance(produced, dict):
                keys.update(k for k in produced if isinstance(k, str))
    return keys


def _description(key):
    """What the user is told about ``key``, from whichever table holds it."""
    for table_name in ("descriptions", "tooltips"):
        table = getattr(S, table_name, None)
        if isinstance(table, dict) and key in table:
            return str(table[key])
    return ""


# ---------------------------------------------------------------------------
# they stay deleted
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", DELETED_INERT_SETTINGS)
def test_the_setting_is_gone_from_every_declaration_site(key):
    sites = _declaration_sites(key)
    assert not sites, (
        f"{key!r} is declared again in {sites}. Nothing reads it, so the "
        f"panel would offer a control that changes nothing. Delete it, or "
        f"wire it up and take its name out of this file.")


@pytest.mark.parametrize("key", DELETED_INERT_SETTINGS)
def test_no_defaults_factory_produces_it(key):
    """A key can reach a run without being in any declaration table."""
    assert key not in _keys_every_factory_produces(), (
        f"a defaults factory hands {key!r} to the run; nothing reads it")


def test_the_name_scan_alone_would_not_have_caught_them():
    """The premise of this file, asserted rather than asserted-about.

    ``test_dead_settings.test_no_declared_setting_is_unread`` scans the
    package for the NAME. If these names did not occur in ``spacr/`` for
    other reasons, that scan would refuse them on its own and this file
    would be redundant -- so it is worth knowing when that changes.
    """
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).with_name("test_dead_settings.py")
    spec = importlib.util.spec_from_file_location("_dead_settings_scan", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    live = module._live_tokens()
    shielded = [k for k in DELETED_INERT_SETTINGS if k in live]
    assert shielded, (
        "none of the deleted names collides with a live token any more, so "
        "the absolute rule now covers them and this file may be simplified")


# ---------------------------------------------------------------------------
# the exemption that let them accumulate stays shut
# ---------------------------------------------------------------------------

def test_no_declared_setting_documents_itself_as_inert():
    """A setting nothing reads is deleted, not excused by its own tooltip.

    The exemption had no floor: any number of dead settings could accumulate
    behind it, each individually honest and none of them counted. Ten did.
    """
    admitted = sorted(k for k in S.expected_types
                      if _ADMITS_IT_IS_DEAD.search(_description(k)))
    assert not admitted, (
        f"{admitted} declare themselves inert. Delete them -- see "
        f"tests/test_dead_settings.py for the rule.")


def test_the_admission_pattern_would_actually_match_one():
    """A regex that matches nothing passes the test above forever."""
    assert _ADMITS_IT_IS_DEAD.search(
        "Collected by the form; nothing in spaCR reads it.")
    assert not _ADMITS_IT_IS_DEAD.search(
        "The channel the cell mask is generated from.")


# ---------------------------------------------------------------------------
# the misspelling, and the live setting it shadowed
# ---------------------------------------------------------------------------

def test_complevel_is_gone_and_comp_level_is_the_live_one():
    """Two compression-level boxes, one of which did nothing."""
    assert "complevel" not in S.expected_types
    assert S.expected_types["comp_level"] is int
    assert S.expected_types["comp_type"] is str


def test_the_live_compression_settings_are_still_read():
    """Deleting the misspelling must not have taken the real pair with it."""
    from spacr.settings import set_default_generate_barecode_mapping

    produced = set_default_generate_barecode_mapping({})
    assert "comp_level" in produced
    assert "comp_type" in produced
    assert "complevel" not in produced


# ---------------------------------------------------------------------------
# and the machinery that used to hide them went too
# ---------------------------------------------------------------------------

def test_the_hiding_list_went_with_the_settings():
    """They were hidden before they were deleted. Hidden is not absent.

    A key kept in the dict but withheld from the panel is still a key the
    run receives; leaving that list behind would leave a second, silent
    place for an inert setting to live.
    """
    from spacr.qt.screens import settings_model

    assert not hasattr(settings_model, "INERT_SETTINGS_NOT_OFFERED")


def test_no_module_resolves_any_of_them_into_its_settings():
    """The user-facing half: no panel can offer what no run reads."""
    pytest.importorskip("PySide6")
    from spacr.qt.app import APPS
    from spacr.qt.screens.settings_model import resolve_default_settings

    deleted = set(DELETED_INERT_SETTINGS)
    offenders = {}
    for row in APPS:
        app_key = row[0]
        try:
            offered = set(resolve_default_settings(app_key))
        except Exception:                                        # noqa: BLE001
            continue
        found = sorted(offered & deleted)
        if found:
            offenders[app_key] = found
    assert not offenders, (
        f"these modules resolve a deleted inert setting: {offenders}")


def test_no_qt_category_layout_names_one_either():
    """A category list is a third declaration site, and a quieter one.

    ``_APP_CATEGORY_SPECS`` decides which heading a control appears under. A
    name in it that no module declares builds no widget, so the layout looks
    right and nothing complains -- which is how `upscale` and
    `upscale_factor` outlived their own deletion in Mask's and Timelapse's
    "Image Preprocessing" heading. Neither the settings scan nor the
    category-map test reaches this table: one reads
    ``spacr.settings.categories``, the other scans the package for readers.
    """
    pytest.importorskip("PySide6")
    from spacr.qt.screens.settings_model import _APP_CATEGORY_SPECS

    deleted = set(DELETED_INERT_SETTINGS)
    offenders = {}
    for app_key, sections in _APP_CATEGORY_SPECS.items():
        named = set()
        for _title, keys in sections:
            named.update(k for k in keys if isinstance(k, str))
        found = sorted(named & deleted)
        if found:
            offenders[app_key] = found
    assert not offenders, (
        f"these Qt category layouts still name a deleted setting: {offenders}")
