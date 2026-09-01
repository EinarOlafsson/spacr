"""Make Masks is filed under Tools, and no section is left empty by it.

It does not train, choose or run a segmentation model: it is hand
curation of masks that already exist. That moved it out of *Segmentation
models*, which emptied that section -- every other occupant had already
folded into a host -- and an empty section must not survive as a tab
onto a blank pane.

IT LANDED IN TOOLS, NOT DATA. This file asserted Data until 2026-09-01,
from before the restructure. The maintainer's own layout is Core 6 /
Data 6 / Tools 5 / Assays 4, and SECTION_TILE_ORDER meets it exactly
with Make Masks in Tools; moving it to Data would make that 7 and 4.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt import app as app_module                          # noqa: E402


def _row(key):
    return next((r for r in app_module.APPS if r[0] == key), None)


def test_make_masks_is_under_tools(qapp):
    row = _row("make_masks")
    assert row is not None, "make_masks lost its registry row"
    assert row[3] == app_module.SECTION_TOOLS


def test_the_declared_layout_is_the_one_that_was_asked_for(qapp):
    """Core 6 / Data 6 / Tools 5 / Assays 4, as specified.

    Counted from SECTION_TILE_ORDER rather than from what draws: two of
    those keys are folded onto host mastheads by instruction 318, so
    nineteen tiles appear rather than twenty-one. The LAYOUT is what was
    asked for; the folding is a later decision on top of it.
    """
    wanted = {app_module.SECTION_CORE: 6, app_module.SECTION_DATA: 6,
              app_module.SECTION_TOOLS: 5, app_module.SECTION_ASSAYS: 4}
    actual = {name: len(keys)
              for name, keys in app_module.SECTION_TILE_ORDER.items()}
    assert actual == wanted


def test_no_section_is_declared_but_empty(qapp):
    """A section with no apps is a tab that opens on nothing."""
    occupied = {row[3] for row in app_module.APPS}
    empty = [name for name in app_module.SECTIONS if name not in occupied]
    assert empty == [], f"these sections have no apps: {empty}"


def test_the_models_section_is_gone_and_cannot_come_back_by_accident(qapp):
    """RETIRED, not merely empty, and the difference matters.

    This used to assert SECTION_MODELS was still DECLARABLE -- in
    SECTION_ORDER -- and simply had nothing in it, so it would return
    the day an app registered there. The 2026-09-01 layout removed it
    from SECTION_ORDER, which makes `register_app` refuse it outright.

    That is the stronger guarantee and the one wanted: a section the
    maintainer asked to remove must not reappear because one app was
    filed under it. The name survives as a constant so an old plugin
    manifest still maps somewhere -- see _PLUGIN_SECTION_MAP -- rather
    than raising.
    """
    assert app_module.SECTION_MODELS not in app_module.SECTION_ORDER
    assert app_module.SECTION_MODELS not in app_module.SECTIONS

    # And registering into it is refused rather than reviving it.
    with pytest.raises(ValueError, match="unknown section"):
        app_module.register_app("cov_models_revival_probe", "Probe",
                                "a probe", app_module.SECTION_MODELS,
                                factory=lambda **_kwargs: None)

    # The constant itself survives so an old plugin manifest still maps
    # somewhere instead of raising KeyError.
    assert "models" in app_module._PLUGIN_SECTION_MAP
    assert (app_module._PLUGIN_SECTION_MAP["models"]
            in app_module.SECTION_ORDER)


def test_every_app_is_in_a_section_that_exists(qapp):
    for row in app_module.APPS:
        assert row[3] in app_module.SECTIONS, (
            f"{row[0]} is filed under {row[3]!r}, which is not a live section")


def test_make_masks_is_on_home_exactly_once(qapp):
    keys = [row[0] for row in app_module.APPS]
    assert keys.count("make_masks") == 1
