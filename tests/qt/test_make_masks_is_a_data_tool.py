"""Make Masks is filed under Data, and no section is left empty by it.

It does not train, choose or run a segmentation model: it is hand curation
of masks that already exist, which is the same kind of work as the other
tools under Data. Moving it emptied *Segmentation models* -- every other
occupant had already folded into a host -- and an empty section must not
survive as a tab onto a blank pane.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt import app as app_module                          # noqa: E402


def _row(key):
    return next((r for r in app_module.APPS if r[0] == key), None)


def test_make_masks_is_under_data(qapp):
    row = _row("make_masks")
    assert row is not None, "make_masks lost its registry row"
    assert row[3] == app_module.SECTION_DATA


def test_no_section_is_declared_but_empty(qapp):
    """A section with no apps is a tab that opens on nothing."""
    occupied = {row[3] for row in app_module.APPS}
    empty = [name for name in app_module.SECTIONS if name not in occupied]
    assert empty == [], f"these sections have no apps: {empty}"


def test_the_models_section_is_gone_because_nothing_is_in_it(qapp):
    """Not deleted -- derived. It returns the day an app registers there."""
    assert app_module.SECTION_MODELS in app_module.SECTION_ORDER
    assert app_module.SECTION_MODELS not in app_module.SECTIONS


def test_every_app_is_in_a_section_that_exists(qapp):
    for row in app_module.APPS:
        assert row[3] in app_module.SECTIONS, (
            f"{row[0]} is filed under {row[3]!r}, which is not a live section")


def test_make_masks_is_on_home_exactly_once(qapp):
    keys = [row[0] for row in app_module.APPS]
    assert keys.count("make_masks") == 1
