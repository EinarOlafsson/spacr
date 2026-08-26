"""The number of organelles decides how many organelle slots are on screen.

"a NUMBER OF ORGANELLES setting, and the organelle settings are generated
from it -- two gives organelle 1 and organelle 2, seven gives seven."

``spacr.organelle_types`` has said all of this correctly for some time --
``organelle_role``, ``MAX_ORGANELLES``, ``active_organelle_roles``,
``declared_organelle_roles`` -- and the panel was built from a defaults dict
that stopped at four slots, so there were only ever four controls to reveal
and the count changed nothing. Confirming the vocabulary is not confirming
the feature, so every assertion here is made by MOVING THE CONTROL, spinning
the event loop and COUNTING THE ROWS THAT ARE ON THE FORM.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

#: The key each module switches an organelle slot with: Mask segments, so it
#: asks for a channel; Measure reads masks somebody else made, so it asks
#: which plane they are on.
SWITCH = {"mask": "channel", "measure": "mask_dim", "timelapse": "channel"}


def _screen(qtbot, app_key: str):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen(app_key)
    qtbot.addWidget(screen)
    qtbot.wait(1)
    return screen, screen._settings_model


def _pick_count(qtbot, model, number: int) -> None:
    """Choose ``number`` in the count dropdown, as a user picking it does."""
    combo = model._widgets["number_of_organelles"]
    index = combo.findData(number)
    assert index >= 0, f"the count dropdown does not offer {number}"
    combo.setCurrentIndex(index)
    # THE PANEL ANSWERS ON THE NEXT TURN OF THE LOOP, not inside the setter.
    qtbot.wait(1)


def _slots_on_screen(screen, app_key: str) -> list:
    """The organelle slots whose switch row is on the form, in slot order."""
    from spacr.organelle_types import ALL_ORGANELLE_ROLES

    suffix = SWITCH[app_key]
    return [role for role in ALL_ORGANELLE_ROLES
            if screen.setting_row_is_visible(f"{role}_{suffix}")]


# ---------------------------------------------------------------------------
# Raising adds, lowering hides
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("app_key", ["measure", "mask"])
def test_seven_renders_seven_slots_and_two_renders_two(qtbot, app_key):
    from spacr.organelle_types import organelle_roles

    screen, model = _screen(qtbot, app_key)

    # The module's own number, before anything is touched.
    assert _slots_on_screen(screen, app_key) == list(organelle_roles(4))

    _pick_count(qtbot, model, 7)
    assert _slots_on_screen(screen, app_key) == list(organelle_roles(7))

    _pick_count(qtbot, model, 2)
    assert _slots_on_screen(screen, app_key) == list(organelle_roles(2))

    _pick_count(qtbot, model, 7)
    assert _slots_on_screen(screen, app_key) == list(organelle_roles(7))


def test_a_slot_the_count_reveals_brings_its_whole_settings_family(qtbot):
    """Slot five is a slot, not just a channel box.

    Mask gives every slot its own type, diameter and size bounds; revealing a
    slot has to reveal the settings that make it segmentable.
    """
    screen, model = _screen(qtbot, "mask")

    _pick_count(qtbot, model, 7)
    # A slot's own settings still wait on its channel, exactly like the four
    # that were always there -- the count says the slot EXISTS, the channel
    # says the run has it.
    assert screen.setting_row_is_visible("organellee_channel") is True
    assert screen.setting_row_is_visible("organellee_diameter") is False

    model._widgets["organellee_channel"].setText("2")
    qtbot.wait(1)
    assert screen.setting_row_is_visible("organellee_diameter") is True
    assert screen.setting_row_is_visible("organellee_type") is True


def test_zero_organelles_leaves_no_slot_on_screen(qtbot):
    """Most runs have none, and that is a number the dropdown offers."""
    screen, model = _screen(qtbot, "measure")

    _pick_count(qtbot, model, 0)
    assert _slots_on_screen(screen, "measure") == []
    # The count itself is never hidden; it belongs to no slot.
    assert screen.setting_row_is_visible("number_of_organelles") is True


def _slot_headings_on_screen(screen) -> list:
    """The numbers of the "Organelle N" headings the panel is showing."""
    numbers = set()
    for section in screen._settings_sections:
        title = str(section.property("settingsCategorySource")
                    or section.title())
        prefix = "Organelle "
        rest = title[len(prefix):] if title.startswith(prefix) else ""
        if rest.isdigit() and not section.isHidden():
            numbers.add(int(rest))
    return sorted(numbers)


def test_a_slot_the_run_does_not_have_takes_its_heading_with_it(qtbot):
    """A heading with every row hidden is a smaller wall, but a wall.

    Mask files each slot's settings under three families, so twenty-two
    unreachable slots would be sixty-six headings to scroll past.
    """
    screen, model = _screen(qtbot, "mask")

    assert _slot_headings_on_screen(screen) == [1, 2, 3, 4]
    _pick_count(qtbot, model, 7)
    assert _slot_headings_on_screen(screen) == [1, 2, 3, 4, 5, 6, 7]
    _pick_count(qtbot, model, 2)
    assert _slot_headings_on_screen(screen) == [1, 2]
    _pick_count(qtbot, model, 0)
    assert _slot_headings_on_screen(screen) == []


# ---------------------------------------------------------------------------
# Lowering hides, it does not delete
# ---------------------------------------------------------------------------

def test_a_panel_written_at_seven_keeps_seven_when_it_is_lowered_to_two(qtbot):
    from spacr.organelle_types import organelle_roles

    screen, model = _screen(qtbot, "measure")
    _pick_count(qtbot, model, 7)

    typed = {}
    for offset, role in enumerate(organelle_roles(7)):
        value = 10 + offset
        model._widgets[f"{role}_mask_dim"].setText(str(value))
        typed[f"{role}_mask_dim"] = value
    qtbot.wait(1)

    at_seven = model.collect()
    assert {k: at_seven[k] for k in typed} == typed

    _pick_count(qtbot, model, 2)
    assert len(_slots_on_screen(screen, "measure")) == 2
    at_two = model.collect()
    # HIDDEN, NOT DELETED: every one of the seven is still in the settings
    # dict the panel would write to a file.
    assert {k: at_two[k] for k in typed} == typed

    _pick_count(qtbot, model, 7)
    assert len(_slots_on_screen(screen, "measure")) == 7
    back = model.collect()
    assert {k: back[k] for k in typed} == typed


def test_a_file_written_at_seven_opens_at_two_with_all_seven_kept(qtbot):
    """The round trip through a settings dict, not through the widgets."""
    from spacr.organelle_types import organelle_roles

    screen, model = _screen(qtbot, "measure")
    _pick_count(qtbot, model, 7)
    for offset, role in enumerate(organelle_roles(7)):
        model._widgets[f"{role}_mask_dim"].setText(str(20 + offset))
    qtbot.wait(1)
    written = model.collect()

    reopened, reopened_model = _screen(qtbot, "measure")
    at_two = dict(written)
    at_two["number_of_organelles"] = 2
    reopened.apply_settings_dict(at_two)
    qtbot.wait(1)

    assert len(_slots_on_screen(reopened, "measure")) == 2
    read_back = reopened_model.collect()
    for offset, role in enumerate(organelle_roles(7)):
        assert read_back[f"{role}_mask_dim"] == 20 + offset

    _pick_count(qtbot, reopened_model, 7)
    assert len(_slots_on_screen(reopened, "measure")) == 7


def test_an_untouched_slot_above_the_count_is_not_written_out(qtbot):
    """The panel builds every slot it can name; a settings file is not the
    panel, and twenty-six slots of defaults in every CSV would bury the four
    a run uses."""
    from spacr.organelle_types import organelle_role_of, organelle_roles

    _screen_, model = _screen(qtbot, "measure")

    def slots(settings):
        return {role for role in map(organelle_role_of, settings) if role}

    assert slots(model.collect()) == set(organelle_roles(4))
    _pick_count(qtbot, model, 6)
    assert slots(model.collect()) == set(organelle_roles(6))


# ---------------------------------------------------------------------------
# The hazard the count must not trip
# ---------------------------------------------------------------------------

def test_the_panel_is_widened_without_widening_the_object_key_vocabulary():
    """`schema.ORGANELLE_ROLES` sizes per-field allocations in the measure
    loop, and widening it to twenty-six roles costs about 180 MB per field.
    The panel is a settings form, so it is widened on its own."""
    from spacr import schema
    from spacr.organelle_types import MAX_ORGANELLES
    from spacr.qt.screens.settings_model import PANEL_ORGANELLE_SLOTS

    assert PANEL_ORGANELLE_SLOTS == MAX_ORGANELLES
    assert len(schema.ORGANELLE_ROLES) == 4
