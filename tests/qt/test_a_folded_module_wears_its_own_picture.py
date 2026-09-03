"""A folded module's button draws the icon its key was assigned, not the one
its key happens to be spelled like.

Reported 2026-09-02: "the cellpose workbench icon should be the cellpose white
ico ni made, not the train icon."

THE CAUSE WAS A RESOLVER THAT WAS NEVER TOLD. `spacr.qt.app._ICON_OVERRIDES`
is the table for a module that BORROWS another module's picture, and
`app._icon_for_app` passes it to `iconset.app_icon`. Three other surfaces
called `iconset.app_icon` DIRECTLY, so they resolved a key by filename alone:

    spacr/qt/widgets/fold_strip.py      the fold strip button
    spacr/qt/widgets/section.py         the settings-heading mark
    spacr/qt/screens/map_barcodes.py    the same strip on that host

The Cellpose Workbench is the key `train_cellpose`, and `train_cellpose.png`
is a DUMBBELL -- the training glyph. Its override sends it to
`cellpose_masks.png`, the white cell outline. So the tile showed the outline
and the fold button showed a dumbbell, from one key, on one screen.

It was never only Cellpose: every borrowing key was wrong on those three
surfaces -- `analyze_plaques` (plaque.png), `agreement` (annotate.png),
`plate_view` (map_barcodes.png), `model_compare` (mask.png) and `model_zoo`
(download.png).

These tests are about the RESOLVER rather than about one button, because the
next module to borrow a picture will be broken by the same omission and
nobody will be looking at its icon.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


def _overrides():
    from spacr.qt.app import _ICON_OVERRIDES

    return _ICON_OVERRIDES


def test_the_workbench_key_is_pointed_at_the_cellpose_outline():
    """The specific report, asserted at the table it is decided in."""
    assert _overrides().get("train_cellpose") == "cellpose_masks.png", (
        "the Cellpose Workbench is the key `train_cellpose`; its icon is "
        "decided by this override, because `train_cellpose.png` is the "
        "dumbbell that the request asked to be rid of"
    )


def test_the_training_glyph_is_a_different_file_that_still_exists():
    """The two must not be quietly reconciled by deleting one.

    Renaming or removing `train_cellpose.png` would make the override
    unnecessary and this whole class of bug invisible -- and would also take
    away the dumbbell from anything that legitimately means "train".
    """
    from spacr.qt import iconset
    from pathlib import Path

    resources = Path(iconset.RESOURCE_DIR)
    assert (resources / "train_cellpose.png").is_file()
    assert (resources / "cellpose_masks.png").is_file()


@pytest.mark.parametrize("key", sorted(_overrides()))
def test_every_borrowed_picture_survives_the_fold_strip(key, qapp):
    """The resolver every folded button uses must honour the override.

    Asserted by PATH rather than by comparing two QIcons: a QIcon is re-inked
    per theme and two of them compare unequal even when they were drawn from
    the same file, which is exactly how this stayed hidden.
    """
    from spacr.qt import iconset
    from spacr.qt.app import _ICON_OVERRIDES

    borrowed = iconset.bundled_icon_path(key, override=_ICON_OVERRIDES[key])
    bare = iconset.bundled_icon_path(key)
    assert borrowed is not None, f"{key}'s override names a file that is gone"
    if bare is not None and bare != borrowed:
        # This key is the dangerous shape: a file matching its own name
        # exists AND it is told to borrow a different one. `train_cellpose`
        # is the one that was reported; the test names any other.
        assert borrowed.endswith(_ICON_OVERRIDES[key]), (
            f"{key} resolves to {bare} without the override and "
            f"{borrowed} with it -- every surface must pass the override"
        )


def test_no_surface_resolves_an_app_icon_without_the_override():
    """The rule, checked in the source rather than through a screenshot.

    `iconset.app_icon` takes the override as an argument and cannot look it
    up itself -- the table lives beside the app registry it describes. So a
    call with no `override=` is a call that will be wrong for any borrowing
    key, and that is the defect, whatever it looks like on the day.
    """
    from pathlib import Path

    import spacr.qt

    root = Path(spacr.qt.__file__).parent
    offenders = []
    for path in root.rglob("*.py"):
        for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1):
            if "iconset.app_icon(" not in line:
                continue
            if "override=" in line:
                continue
            offenders.append(f"{path.relative_to(root)}:{number}")
    assert not offenders, (
        "these call `iconset.app_icon` without an override, so a module that "
        "borrows another module's picture will draw the wrong one here: "
        + ", ".join(offenders)
        + ". Call `spacr.qt.app._icon_for_app` instead, which passes it."
    )
