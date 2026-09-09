"""OPS is reached from Mask Generation, as a switch in its actions row.

Asked for on 2026-09-09: "i think the OPS button should be in Mask
generation instead of align", then "add the ops button to Mask add it
beside the Live and 3D buttons in the same format".

Both halves are contracts, and they are different contracts. WHERE the
module is declared decides what the menu, the dock and the documentation
say hosts it -- that is `FOLDED_APPS`, and moving it is what takes OPS off
Align & Stitch. WHAT THE CONTROL LOOKS LIKE is the actions row: the same
`AiToggleLabel` the dimension switches and Live are, beside them, rather
than an icon on the masthead strip. A fold that mounts settings categories
belongs on that strip; OPS opens a page of its own, so it does not.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


@pytest.fixture()
def mask_screen(qtbot):
    """A real Mask Generation screen, with its actions row built."""
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen(app_key="mask")
    qtbot.addWidget(screen)
    return screen


def test_the_switch_sits_beside_live_and_the_dimension_switches(mask_screen):
    """Same widget class, same row, next to the ones it was asked beside."""
    from spacr.qt.widgets import AiToggleLabel
    from spacr.qt.screens.mask import OPS_TOGGLE_TEXT

    switch = mask_screen._ops_switch
    assert isinstance(switch, AiToggleLabel), (
        "the OPS control is not in the format the rest of the row uses")
    assert switch.text() == OPS_TOGGLE_TEXT
    assert switch.toolTip(), "the only place a user meets OPS says nothing"

    row = [label._full_text
           for label in mask_screen.findChildren(AiToggleLabel)]
    assert {"3D", "Live", OPS_TOGGLE_TEXT} <= set(row), row
    # Beside them, not merely present: the switches are found in the order
    # they were added to the row.
    assert row.index(OPS_TOGGLE_TEXT) == row.index("Live") + 1, row


def test_only_mask_generation_grows_one(qtbot):
    """The switch is Mask's, not every settings screen's."""
    from spacr.qt.screens.app_screen import AppScreen

    for key in ("measure", "timelapse", "ops"):
        screen = AppScreen(app_key=key)
        qtbot.addWidget(screen)
        assert screen._ops_switch is None, (
            f"{key} grew an OPS switch of its own")


def test_the_switch_opens_and_closes_the_ops_page(mask_screen):
    """On puts OPS on the host's page strip; off takes it off again."""
    from spacr.qt.screens import mask as mask_mod

    installed = mask_mod.ops_page(mask_screen)
    assert installed is not None, "the switch is connected to nothing"

    mask_screen._ops_switch.setChecked(True)
    page = installed.page
    assert getattr(page, "app_key", None) == "ops"
    pages = mask_screen._fold_pages
    assert [pages.tabText(i) for i in range(pages.count())][-1] == "OPS"

    mask_screen._ops_switch.setChecked(False)
    assert pages.indexOf(page) < 0, "the page stayed on the strip"
    assert installed.opener.window is page, (
        "the screen was thrown away rather than kept for the next press")

    mask_screen._ops_switch.setChecked(True)
    assert installed.page is page, "pressing again built a second OPS screen"


def test_closing_the_page_by_its_cross_puts_the_switch_back(mask_screen):
    """A page closed behind the switch's back must not leave it lit.

    The page strip carries its own close mark, which knows nothing about
    this switch -- so the switch follows `tabCloseRequested` rather than
    assuming it is the only way the page can go.
    """
    from spacr.qt.screens import mask as mask_mod

    switch = mask_screen._ops_switch
    switch.setChecked(True)
    installed = mask_mod.ops_page(mask_screen)
    pages = mask_screen._fold_pages

    pages.tabCloseRequested.emit(pages.indexOf(installed.page))

    assert not switch.isChecked(), (
        "the page is closed and the switch still says it is open")
    assert installed.page is None


def test_a_module_that_cannot_be_built_does_not_leave_the_switch_lit(
        mask_screen, monkeypatch):
    """A failed open is a switch back off, not a lit switch over nothing."""
    from spacr.qt.screens import mask as mask_mod

    installed = mask_mod.ops_page(mask_screen)
    monkeypatch.setattr(installed.opener, "open",
                        lambda: (_ for _ in ()).throw(RuntimeError("no")))

    mask_screen._ops_switch.setChecked(True)

    assert not mask_screen._ops_switch.isChecked()
    assert installed.page is None


def test_ops_is_declared_on_mask_and_no_longer_on_align():
    """The declaration moved, so every surface that reads it moved too.

    The menu bar, the dock and the API pages all ask `folded_children()`
    who hosts a folded module; none of them asks this screen's actions
    row. A switch added here while the declaration stayed on Align &
    Stitch would put the button in one place and its name in another.
    """
    from spacr.qt.app import folded_children
    from spacr.qt.screens import align, mask

    assert "ops" in mask.FOLDED_APPS
    assert "ops" in mask.PAGE_FOLDS
    assert "ops" not in mask.CATEGORY_FOLDS, (
        "a page fold has no gate and mounts no categories")
    assert not getattr(align, "FOLDED_APPS", ()), (
        "Align & Stitch still declares a fold")

    hosts = folded_children()
    assert "ops" in hosts.get("mask", ())
    assert "ops" not in hosts.get("align", ())


def test_the_button_still_says_what_it_said_on_align():
    """The name, the sentence and the alpha colour moved with the fold."""
    from spacr.qt.widgets.fold_strip import folded_modules

    name, description, stage = folded_modules()["ops"][:3]
    assert name == "OPS"
    assert "optical pooled screening" in description.lower()
    assert stage == "alpha"


def test_the_masthead_strip_carries_only_the_settings_folds(mask_screen):
    """OPS is not on the strip: the strip is for folds that ARE settings."""
    from spacr.qt.screens import mask as mask_mod

    strip = mask_mod.install_folds(mask_screen)
    assert strip is not None
    assert list(strip.keys()) == list(mask_mod.CATEGORY_FOLDS)
    assert "ops" not in strip.keys()
