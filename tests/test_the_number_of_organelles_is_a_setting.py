"""The organelle slots are generated from a number, not fixed at four.

`number_of_organelles` decides how many organelle slots exist and every
per-slot key -- value, type, tooltip, category, widget spec -- is generated
from it. Two gives two slots and seven gives seven.

The half that is easy to get wrong is the way back down: LOWERING the number
hides slots, it does not delete them. A settings file written at seven has to
open at two without erroring, keep the five hidden slots' answers, and hand
them back unchanged when the number goes up again. Everything below is
measured off the real defaults factories and the real registries.
"""

from __future__ import annotations

import pytest

import spacr.settings as S
from spacr.organelle_types import (ALL_ORGANELLE_ROLES, MAX_ORGANELLES,
                                   active_organelle_roles,
                                   declared_organelle_roles, organelle_count,
                                   organelle_number, organelle_role,
                                   organelle_role_of, organelle_roles,
                                   organelle_slot_is_active,
                                   organelle_slot_label, primary_setting,
                                   slot_setting)


def _slots_in(settings):
    """The slot prefixes a settings dict actually carries keys for."""
    return {role for key in settings
            for role in [organelle_role_of(key)] if role}


# ---------------------------------------------------------------------------
# The vocabulary
# ---------------------------------------------------------------------------

def test_the_number_decides_how_many_slots_there_are():
    """Two gives two, seven gives seven, and the names do not shift."""
    assert organelle_roles(2) == ("organelle", "organelleb")
    assert organelle_roles(7) == (
        "organelle", "organelleb", "organellec", "organelled", "organellee",
        "organellef", "organelleg")
    # Raising the number must not RENAME an existing slot: the first two
    # entries are the same tuple whether the run has two slots or seven, so
    # a settings file's keys keep meaning what they meant.
    assert organelle_roles(7)[:2] == organelle_roles(2)
    assert organelle_roles(0) == ()


def test_a_slot_number_and_its_key_prefix_round_trip():
    for number in range(1, MAX_ORGANELLES + 1):
        role = organelle_role(number)
        assert organelle_number(role) == number
        assert organelle_slot_label(role) == f"Organelle {number}"


def test_the_cap_is_the_alphabet_and_it_says_so():
    """A slot's name IS its key prefix, and the prefixes are lettered."""
    assert organelle_role(MAX_ORGANELLES) == "organellez"
    with pytest.raises(ValueError) as excinfo:
        organelle_role(MAX_ORGANELLES + 1)
    assert str(MAX_ORGANELLES) in str(excinfo.value)


def test_a_key_is_traced_back_to_the_slot_that_owns_it():
    assert organelle_role_of("organelle_diameter") == "organelle"
    assert organelle_role_of("organelleb_diameter") == "organelleb"
    assert organelle_role_of("organellez_diameter") == "organellez"
    # Decisions ABOUT the organelles collectively belong to no slot, so a
    # panel hiding slot 5 does not hide the count or the roll-up.
    assert organelle_role_of("number_of_organelles") is None
    assert organelle_role_of("summarize_organelles_by") is None
    assert organelle_role_of("cell_channel") is None
    assert slot_setting("organelle_diameter", "organelleg") == \
        "organelleg_diameter"
    assert primary_setting("organelleg_diameter") == "organelle_diameter"


def test_an_unreadable_number_keeps_the_file_openable():
    """A typo in one number may not cost the whole settings file."""
    assert organelle_count({}) == 0
    assert organelle_count({"number_of_organelles": 7}) == 7
    assert organelle_count({"number_of_organelles": "7"}) == 7
    assert organelle_count({"number_of_organelles": None}) == 0
    assert organelle_count({"number_of_organelles": "seven"}) == 0
    assert organelle_count({
        "number_of_organelles": "seven", "organelleb_channel": 2,
    }) == 2
    # Above the cap it is clamped rather than raised, because the value came
    # off disk and the alternative is refusing to open the file.
    assert organelle_count({"number_of_organelles": 99}) == MAX_ORGANELLES


# ---------------------------------------------------------------------------
# Raising the number adds slots
# ---------------------------------------------------------------------------

def test_seven_yields_seven_slots_of_real_settings():
    """Not seven channels: seven copies of every organelle setting."""
    four = S.set_default_settings_preprocess_generate_masks(
        {"src": "/tmp/s", "number_of_organelles": 4})
    seven = S.set_default_settings_preprocess_generate_masks(
        {"src": "/tmp/s", "number_of_organelles": 7})

    assert _slots_in(four) == set(organelle_roles(4))
    assert _slots_in(seven) == set(organelle_roles(7))

    primary = {key for key in four if key.startswith("organelle_")}
    assert len(primary) > 40, "the primary slot lost its settings"
    for role in organelle_roles(7)[1:]:
        expected = {slot_setting(key, role) for key in primary}
        missing = expected - set(seven)
        assert not missing, f"{role} is missing {sorted(missing)[:5]}"


def test_measure_offers_the_same_slots_the_masks_were_made_with():
    """A run that segmented five organelles has five sets of masks."""
    settings = S.get_measure_crop_settings(
        {"src": "/tmp/s", "number_of_organelles": 5})
    for role in organelle_roles(5)[1:]:
        assert f"{role}_mask_dim" in settings
        assert f"{role}_min_size" in settings
        assert f"{role}_type" in settings
    assert f"{organelle_role(6)}_mask_dim" not in settings


# ---------------------------------------------------------------------------
# Lowering it hides slots and keeps their answers
# ---------------------------------------------------------------------------

def test_a_file_written_at_seven_opens_at_two_and_keeps_its_answers():
    """The whole point of a number worth exploring: going back down is free."""
    seven = S.set_default_settings_preprocess_generate_masks(
        {"src": "/tmp/s", "number_of_organelles": 7})
    seven["organelleg_diameter"] = 99
    seven["organelleg_type"] = "punctate"
    written = dict(seven)

    lowered = dict(written, number_of_organelles=2)
    at_two = S.set_default_settings_preprocess_generate_masks(lowered)

    # HIDDEN, NOT DELETED: the panel shows two slots, the file still has all
    # seven, and slot seven's answers are the ones the user typed.
    assert active_organelle_roles(at_two) == organelle_roles(2)
    assert _slots_in(at_two) == set(organelle_roles(7))
    assert at_two["organelleg_diameter"] == 99
    assert at_two["organelleg_type"] == "punctate"

    at_two["number_of_organelles"] = 7
    again = S.set_default_settings_preprocess_generate_masks(dict(at_two))
    assert again["organelleg_diameter"] == 99
    assert again["organelleg_type"] == "punctate"
    # Every answer the seven-slot file carried, back verbatim -- the count
    # is the only key that moved.
    shared = [key for key in written if key != "number_of_organelles"]
    assert {key: again[key] for key in shared} == \
        {key: written[key] for key in shared}
    assert again["number_of_organelles"] == 7


def test_the_hidden_slots_are_the_ones_above_the_count():
    settings = {"number_of_organelles": 2, "organelleg_channel": 3}
    assert active_organelle_roles(settings) == organelle_roles(2)
    # Declared is wider than active, and that difference is what stops the
    # defaults machinery from dropping a hidden slot on the way back out.
    assert declared_organelle_roles(settings) == organelle_roles(7)
    assert organelle_slot_is_active("organelleb_channel", settings)
    assert not organelle_slot_is_active("organellec_channel", settings)
    # A key belonging to no slot is never hidden by the count.
    assert organelle_slot_is_active("number_of_organelles", settings)
    assert organelle_slot_is_active("cell_channel", settings)


# ---------------------------------------------------------------------------
# Every slot arrives with a type, help, a heading and a widget
# ---------------------------------------------------------------------------

def test_every_declarable_slot_is_typed_tooltipped_and_categorised():
    """A generated control with no type and no help is not a control.

    Asserted over EVERY slot the count can name, not over the four that used
    to be fixed: a settings file written at seven is opened by a session set
    to two, and its seventh slot has to be readable then.
    """
    primary = sorted(key for key in S.expected_types
                     if key.startswith("organelle_"))
    assert primary, "the primary slot is not declared at all"
    filed = {key for keys in S.categories.values() for key in keys}
    primary_filed = [key for key in primary if key in filed]

    for role in ALL_ORGANELLE_ROLES[1:]:
        untyped = [key for key in primary
                   if slot_setting(key, role) not in S.expected_types]
        assert not untyped, f"{role}: untyped {untyped[:5]}"
        unhelped = [key for key in primary
                    if key in S.tooltips
                    and slot_setting(key, role) not in S.tooltips]
        assert not unhelped, f"{role}: no tooltip for {unhelped[:5]}"
        unfiled = [key for key in primary_filed
                   if slot_setting(key, role) not in filed]
        assert not unfiled, f"{role}: no category for {unfiled[:5]}"


def test_every_declarable_slot_keeps_the_closed_vocabularies():
    """A slot's method is a dropdown for slot 26 as much as for slot 1."""
    from spacr.settings_spec import convert_settings_dict_for_gui

    closed = ("type", "morphology", "method", "ridge_filter",
              "network_threshold", "ring_fill_method")
    probe = {}
    for role in ALL_ORGANELLE_ROLES:
        for suffix in closed:
            probe[f"{role}_{suffix}"] = None
    spec = convert_settings_dict_for_gui(probe)

    for role in ALL_ORGANELLE_ROLES:
        for suffix in closed:
            key = f"{role}_{suffix}"
            kind, options, _default = spec[key]
            assert kind == "combo", f"{key} fell back to {kind}"
            assert options == spec[f"organelle_{suffix}"][1], key


def test_a_seven_slot_file_validates_when_the_count_is_two():
    """`check_settings` may not report a hidden slot as an unknown key.

    That warning is what "erroring on load" would look like in the panel:
    every key of the five hidden slots reported as not found, and their
    values dropped from the parsed settings.
    """
    class _Var:
        def __init__(self, value):
            self._value = value

        def get(self):
            return self._value

    seven = S.set_default_settings_preprocess_generate_masks(
        {"src": "/tmp/s", "number_of_organelles": 7})
    seven["number_of_organelles"] = 2
    hidden = sorted(key for key in seven
                    if organelle_role_of(key) in organelle_roles(7)[2:])
    assert len(hidden) > 100, "the seven-slot file has nothing hidden in it"

    vars_dict = {key: ("label", None, _Var(str(seven[key])), None)
                 for key in hidden}
    parsed, errors = S.check_settings(vars_dict, S.expected_types)
    assert not [e for e in errors if "not found in expected types" in e]
    assert set(parsed) == set(hidden)


def test_the_count_is_offered_as_the_closed_range_it_is():
    """Every number the alphabet can name, and none it cannot.

    A free number field would accept thirty and be clamped to twenty-six on
    the way in, which is a value the user did not ask for arriving without
    being mentioned.
    """
    from spacr.settings_spec import convert_settings_dict_for_gui

    kind, options, _default = convert_settings_dict_for_gui(
        {"number_of_organelles": 7})["number_of_organelles"]
    assert kind == "combo"
    assert options == list(range(MAX_ORGANELLES + 1))


def test_the_count_control_carries_the_number_the_file_holds(qapp):
    """A seven-slot file may not open showing four.

    Driven through the panel because that is where it would go wrong: the
    widget spec's third element is a fallback, and a control that ignored
    the settings dict would silently collapse the run to the default count.
    """
    pytest.importorskip("PySide6")
    from spacr.qt.screens.settings_model import SettingsWidgets

    panel = SettingsWidgets("mask", current={"number_of_organelles": 7})
    panel.build_sections()
    assert panel._widgets["number_of_organelles"].currentText() == "7"
    assert panel.collect()["number_of_organelles"] == 7


def test_the_count_itself_is_declared_like_any_other_setting():
    assert S.expected_types["number_of_organelles"] is int
    assert "number_of_organelles" in S.tooltips
    assert S.categories["Organelle"][0] == "number_of_organelles"
    for factory in (S.set_default_settings_preprocess_generate_masks,
                    S.get_measure_crop_settings):
        assert factory({"src": "/tmp/s"})["number_of_organelles"] == 0


# ---------------------------------------------------------------------------
# Driven through the panel
# ---------------------------------------------------------------------------

def test_the_panel_builds_a_control_for_every_slot_the_number_asks_for(qapp):
    """Measured on the real widget map, not on the category table.

    A slot can be typed, tooltipped and categorised and still arrive as
    nothing: `build_sections` drops any key that produced no widget. So the
    count is read off `SettingsWidgets`'s own widget map after it has built
    the mask panel from a seven-slot settings dict.
    """
    pytest.importorskip("PySide6")
    from spacr.qt.screens.settings_model import SettingsWidgets

    def built_slots(defaults):
        panel = SettingsWidgets("mask", current=defaults)
        panel.build_sections()
        return panel, {role for key in panel._widgets
                       for role in [organelle_role_of(key)] if role}

    seven = S.set_default_settings_preprocess_generate_masks(
        {"src": "/tmp/s", "number_of_organelles": 7})
    _panel, slots = built_slots(seven)
    assert slots == set(organelle_roles(7))
    for role in organelle_roles(7):
        assert f"{role}_type" in _panel._widgets
        assert f"{role}_diameter" in _panel._widgets


def test_a_slot_the_panel_does_not_draw_still_reaches_the_run(qapp):
    """Hidden is not deleted, measured through what the panel hands back.

    `collect()` carries every default it did not render, so a slot above the
    count keeps its value whether the panel drew a control for it or not.
    This is the half of "hidden, not deleted" that the settings layer owes:
    the panel decides what to draw, and what it hands back still carries the
    hidden slots.
    """
    pytest.importorskip("PySide6")
    from spacr.qt.screens.settings_model import SettingsWidgets

    seven = S.set_default_settings_preprocess_generate_masks(
        {"src": "/tmp/s", "number_of_organelles": 7})
    seven["organelleg_diameter"] = 99
    lowered = S.set_default_settings_preprocess_generate_masks(
        dict(seven, number_of_organelles=2))

    panel = SettingsWidgets("mask", current=lowered)
    panel.build_sections()
    collected = panel.collect()

    assert collected["number_of_organelles"] == 2
    assert collected["organelleg_diameter"] == 99
    for role in organelle_roles(7):
        assert f"{role}_type" in collected
