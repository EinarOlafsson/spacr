"""The number of organelles decides how many organelle slots are on screen.

"a NUMBER OF ORGANELLES setting, and the organelle settings are generated
from it -- two gives organelle 1 and organelle 2, seven gives seven."

``spacr.organelle_types`` has said all of this correctly for some time --
``organelle_role``, ``MAX_ORGANELLES``, ``active_organelle_roles``,
``declared_organelle_roles`` -- and the panel was built from a defaults dict
that stopped at four slots, so there were only ever four controls to reveal
and the count changed nothing. Confirming the vocabulary is not confirming
the feature. The optimized panel now builds only the slots the count names,
so every assertion here builds the same form shape a committed count asks the
window for and COUNTS THE ROWS ON THAT FORM.
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


def _screen(qtbot, app_key: str, current=None):
    from spacr.qt.screens.app_screen import AppScreen

    before = AppScreen.values_the_next_screen_is_built_for
    AppScreen.values_the_next_screen_is_built_for = current
    try:
        screen = AppScreen(app_key)
    finally:
        AppScreen.values_the_next_screen_is_built_for = before
    qtbot.addWidget(screen)
    qtbot.wait(1)
    return screen, screen._settings_model


def _screen_for_count(qtbot, app_key: str, number: int, current=None):
    """Build the shape a committed ``number_of_organelles`` requests."""
    values = dict(current or {})
    values["number_of_organelles"] = number
    return _screen(qtbot, app_key, values)


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

    for count in (0, 7, 2, 7):
        screen, model = _screen_for_count(qtbot, app_key, count)
        assert model.collect()["number_of_organelles"] == count
        assert _slots_on_screen(screen, app_key) == list(
            organelle_roles(count))


def test_a_committed_slot_channel_brings_its_whole_settings_family(
        qapp, qtbot):
    """Slot five is a slot, not just a channel box.

    Mask gives every slot its own type, diameter and size bounds; revealing a
    slot has to reveal the settings that make it segmentable.
    """
    from spacr.qt.app import MainWindow
    from spacr.qt.settings_search import (ALL, disclosure_for,
                                           remember_disclosure)

    previous = disclosure_for("mask")
    remember_disclosure("mask", ALL)
    window = MainWindow()
    qtbot.addWidget(window)
    try:
        window.show()
        window._on_nav_selected("mask")
        qapp.processEvents()
        screen = window._screens["mask"]
        count = screen._settings_model._widgets["number_of_organelles"]
        if hasattr(count, "set_value"):
            count.set_value(1)
        elif hasattr(count, "setCurrentText"):
            count.setCurrentText("1")
        else:
            count.setValue(1)
        qapp.processEvents()

        screen = window._screens["mask"]
        model = screen._settings_model
        # A slot's own settings still wait on its channel: the count says the
        # slot EXISTS, the channel says the run has it.
        assert screen.setting_row_is_visible("organelle_channel") is True
        assert screen.setting_row_is_visible("organelle_diameter") is False

        channel = model._widgets["organelle_channel"]
        channel.setText("2")
        channel.editingFinished.emit()
        qapp.processEvents()

        rebuilt = window._screens["mask"]
        assert rebuilt.setting_row_is_visible("organelle_diameter") is True
        assert rebuilt.setting_row_is_visible("organelle_type") is True
        assert rebuilt._settings_model.collect()["organelle_channel"] == 2
    finally:
        window.close()
        remember_disclosure("mask", previous)


def test_zero_organelles_leaves_no_slot_on_screen(qtbot):
    """Most runs have none, and that is a number the dropdown offers."""
    screen, _model = _screen_for_count(qtbot, "measure", 0)
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
    for count in (0, 2, 7):
        screen, _model = _screen_for_count(qtbot, "mask", count)
        assert _slot_headings_on_screen(screen) == list(range(1, count + 1))


# ---------------------------------------------------------------------------
# Lowering hides, it does not delete
# ---------------------------------------------------------------------------

def test_a_panel_written_at_seven_keeps_seven_when_it_is_lowered_to_two(qtbot):
    from spacr.organelle_types import organelle_roles

    _screen_, model = _screen_for_count(qtbot, "measure", 7)

    typed = {}
    for offset, role in enumerate(organelle_roles(7)):
        value = 10 + offset
        model._widgets[f"{role}_mask_dim"].setText(str(value))
        typed[f"{role}_mask_dim"] = value
    qtbot.wait(1)

    at_seven = model.collect()
    assert {k: at_seven[k] for k in typed} == typed

    at_two_values = dict(at_seven)
    at_two_values["number_of_organelles"] = 2
    at_two_screen, at_two_model = _screen_for_count(
        qtbot, "measure", 2, at_two_values)
    assert len(_slots_on_screen(at_two_screen, "measure")) == 2
    at_two = at_two_model.collect()
    # NOT BUILT, NOT DELETED: every one of the seven still rides in the
    # settings dict the replacement panel would write to a file.
    assert {k: at_two[k] for k in typed} == typed

    back_screen, back_model = _screen_for_count(qtbot, "measure", 7, at_two)
    assert len(_slots_on_screen(back_screen, "measure")) == 7
    back = back_model.collect()
    assert {k: back[k] for k in typed} == typed


def test_a_file_written_at_seven_opens_at_two_with_all_seven_kept(qtbot):
    """The round trip through a settings dict, not through the widgets."""
    from spacr.organelle_types import organelle_roles

    _screen_, model = _screen_for_count(qtbot, "measure", 7)
    for offset, role in enumerate(organelle_roles(7)):
        model._widgets[f"{role}_mask_dim"].setText(str(20 + offset))
    qtbot.wait(1)
    written = model.collect()

    at_two = dict(written)
    at_two["number_of_organelles"] = 2
    reopened, reopened_model = _screen_for_count(
        qtbot, "measure", 2, at_two)

    assert len(_slots_on_screen(reopened, "measure")) == 2
    read_back = reopened_model.collect()
    for offset, role in enumerate(organelle_roles(7)):
        assert read_back[f"{role}_mask_dim"] == 20 + offset

    back, _back_model = _screen_for_count(
        qtbot, "measure", 7, read_back)
    assert len(_slots_on_screen(back, "measure")) == 7


def test_an_untouched_slot_above_the_count_is_not_written_out(qtbot):
    """Only requested slots beyond Measure's fixed four-slot schema persist.

    Measure deliberately carries its four shipped slot keys for downstream
    table allocation even when the form builds none. Slots five onward are
    panel additions and appear in collected settings only when requested.
    """
    from spacr.organelle_types import organelle_role_of, organelle_roles

    def slots(settings):
        return {role for role in map(organelle_role_of, settings) if role}

    for count in (0, 4, 6):
        _screen_, model = _screen_for_count(qtbot, "measure", count)
        assert slots(model.collect()) == set(organelle_roles(max(4, count)))


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
