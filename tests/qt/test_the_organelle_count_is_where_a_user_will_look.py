"""The number of organelles is on the panel a user actually opens.

"the number of organells is implemented in the measure modual but the
settings for organelle 1, 2, 3, 4 are still the same so something is broken
and the mask modual dosnt have the number of organells at all."

BOTH HALVES OF THAT WERE TRUE AT ONCE, and only the second half was about
the count being missing. Every module screen gets a settings strip
(``spacr.qt.settings_search.install``, hung on the window's screen stack),
and it opens on ESSENTIALS: the module's first layout group plus its
``_APP_ESSENTIAL_EXTRAS``. Measure names its ``@Mask & Channel Mapping``
group there, which is where the count is filed, so on Measure it was on
screen. Mask filed it under "Organelle Segmentation" and named neither, so
the strip hid it -- on the one panel that draws twenty-six organelle
channel boxes. A control the panel does not offer is a control that changes
nothing, whatever the rule behind it does.

So the count is filed beside the switches of the slots it governs, on each
of the three modules that has slots: before ``organelle_mask_dim`` on
Measure and before ``organelle_channel`` on Mask and Timelapse. That is
where a user looking for it looks, and it is in the first group, which is
what makes it essential without a second list to keep in step.

EVERY ASSERTION IS MADE THROUGH THE STRIP, at the level it opens on, with
the event loop running: the panel answers on the next turn of the loop.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

#: The key each module switches an organelle slot with: Mask and Timelapse
#: segment, so they ask for a channel; Measure reads masks somebody else
#: made, so it asks which plane they are on.
SWITCH = {"mask": "channel", "measure": "mask_dim", "timelapse": "channel"}

MODULES = tuple(SWITCH)


def _screen_with_the_strip(qtbot, app_key: str):
    """A module screen carrying the search strip the window gives it."""
    from spacr.qt import settings_search
    from spacr.qt.screens.app_screen import AppScreen

    # THE DEFAULT LEVEL, asserted rather than inherited: the strip remembers
    # the last level per module, and a test that ran after one which chose
    # All settings would be checking the wrong panel.
    settings_search.forget_disclosure(app_key)
    screen = AppScreen(app_key)
    qtbot.addWidget(screen)
    qtbot.wait(1)
    bar = settings_search.install(screen)
    assert bar is not None, f"{app_key} got no settings strip"
    qtbot.wait(1)
    assert bar.level() == settings_search.ESSENTIALS
    return screen, screen._settings_model, bar


def _slots_on_screen(screen, app_key: str) -> list:
    from spacr.organelle_types import ALL_ORGANELLE_ROLES

    suffix = SWITCH[app_key]
    return [role for role in ALL_ORGANELLE_ROLES
            if screen.setting_row_is_visible(f"{role}_{suffix}")]


# ---------------------------------------------------------------------------
# On the panel as it opens
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("app_key", MODULES)
def test_the_count_is_on_the_panel_the_module_opens_with(qtbot, app_key):
    """Not "somewhere under All settings" -- on the form, first look."""
    screen, _model, _bar = _screen_with_the_strip(qtbot, app_key)

    assert screen.setting_row_is_visible("number_of_organelles") is True


@pytest.mark.parametrize("app_key", MODULES)
def test_the_count_stands_immediately_before_the_slots_it_governs(app_key):
    """Filed with the switches, not under a heading of its own.

    Asked of the layout rather than of the rendered form, because this is a
    claim about where the setting BELONGS: the row order follows.
    """
    from spacr.qt.screens.settings_model import (categories_for_app,
                                                 get_categories)

    cats = categories_for_app(app_key, get_categories())
    holders = [keys for keys in cats.values()
               if "number_of_organelles" in keys]
    assert len(holders) == 1, "the count is filed in two places at once"
    keys = holders[0]
    index = keys.index("number_of_organelles")
    assert keys[index + 1] == f"organelle_{SWITCH[app_key]}"


@pytest.mark.parametrize("app_key", MODULES)
def test_the_count_is_one_of_the_settings_the_module_meets_you_with(app_key):
    """The strip hides everything else, so this is what "on the panel" is."""
    from spacr.qt.screens.settings_model import essential_keys

    assert "number_of_organelles" in essential_keys(app_key)


# ---------------------------------------------------------------------------
# And it still does what it says, with the strip filtering the form
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("app_key", MODULES)
def test_the_count_adds_and_hides_slots_through_the_strip(qtbot, app_key):
    """Driven on the panel the user has, not on an unfiltered one."""
    from spacr.organelle_types import organelle_roles

    screen, model, _bar = _screen_with_the_strip(qtbot, app_key)
    combo = model._widgets["number_of_organelles"]

    assert _slots_on_screen(screen, app_key) == list(organelle_roles(4))
    for count in (7, 2, 7):
        index = combo.findData(count)
        assert index >= 0, f"the count dropdown does not offer {count}"
        combo.setCurrentIndex(index)
        # THE PANEL ANSWERS ON THE NEXT TURN OF THE LOOP.
        qtbot.wait(1)
        assert _slots_on_screen(screen, app_key) == list(
            organelle_roles(count))
    # The count is never one of the rows a slot takes with it.
    assert screen.setting_row_is_visible("number_of_organelles") is True
