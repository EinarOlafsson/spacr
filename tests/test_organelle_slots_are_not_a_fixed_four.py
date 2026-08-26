"""The organelle slots are generated from a number, not written out four times.

Four slots used to be spelled out by hand, and the two places that still did
it after the count became a setting are what this file pins:

  * the SUB-HEADING a slot's rows are drawn under resolved the slot through
    the schema's list of the four slots that carry a mask plane today, so a
    fifth slot was drawn under "Organellee" -- the internal lettered spelling
    that exists so object keys can round-trip through a ``prcfo`` key, and
    that the user was never meant to read;
  * the sub-heading HELP was a table of four literals, so slot five fell back
    to "Settings that control organelle 5." and slot four told the user it
    was "the last one spaCR offers" -- true while the slots were fixed and a
    lie the moment the count became a setting.

The rest of the file holds the promise the count makes: LOWERING IT HIDES
SLOTS RATHER THAN DELETING THEM, through a settings CSV on disk rather than a
dict passed hand to hand, because the file is where a user's seven answers
actually have to survive being opened by a session set to two.
"""

import pytest

from spacr.organelle_types import (ALL_ORGANELLE_ROLES, MAX_ORGANELLES,
                                   NUMBER_OF_ORGANELLES,
                                   active_organelle_roles,
                                   declared_organelle_roles, organelle_role,
                                   organelle_roles, organelle_slot_label)
from spacr.qt.screens.settings_model import (OBJECT_SUBHEADING_TOOLTIPS,
                                             SettingsSection,
                                             _object_subheading,
                                             _split_rows_by_object,
                                             keys_hidden_by_their_object,
                                             section_tooltip,
                                             section_tooltip_is_curated)


def _nested(title):
    """A sub-heading with a parent, which is what carries a path."""
    section = SettingsSection(title, [("row", None)])
    SettingsSection("Object filtration", children=(section,))
    return section


# ---------------------------------------------------------------------------
# The heading a slot's rows are drawn under
# ---------------------------------------------------------------------------

def test_a_slot_past_the_fourth_is_still_headed_by_its_number():
    """The lettered prefix is an internal name and may not reach a heading.

    ``organellee`` used to capitalise to "Organellee" because the slot was
    recognised by the schema's four-entry list rather than by its name.
    """
    for number in range(1, MAX_ORGANELLES + 1):
        role = organelle_role(number)
        assert _object_subheading(role) == f"Organelle {number}"

    # The exact regression, named: slot five is the first one the schema's
    # list does not carry.
    assert _object_subheading("organellee") == "Organelle 5"
    assert _object_subheading("organellez") == "Organelle 26"


def test_the_objects_that_are_not_slots_are_unaffected():
    assert _object_subheading("cell") == "Cell"
    assert _object_subheading("nucleus") == "Nucleus"
    assert _object_subheading("cytoplasm") == "Cytoplasm"


def test_a_seven_slot_run_draws_seven_numbered_sub_headings():
    """Through the function the panel actually splits its rows with."""
    keys = [f"{role}_min_size" for role in organelle_roles(7)]
    rows = [(f"Min size {index}", None) for index in range(len(keys))]

    _own, children = _split_rows_by_object(rows, keys)

    assert [child.title for child in children] == [
        f"Organelle {number}" for number in range(1, 8)]
    # Every row reached a heading; none was dropped on the way.
    assert sum(len(child.rows) for child in children) == len(rows)


# ---------------------------------------------------------------------------
# Every heading that can be drawn has help waiting
# ---------------------------------------------------------------------------

def test_every_slot_the_alphabet_allows_has_written_sub_heading_help():
    """A heading a lowered count can bring back has to have help when it does."""
    for role in ALL_ORGANELLE_ROLES:
        title = organelle_slot_label(role)
        assert title.upper() in OBJECT_SUBHEADING_TOOLTIPS
        section = _nested(title)
        assert section_tooltip_is_curated("mask", section)
        text = section_tooltip("mask", section)
        assert text and not text.startswith("Settings that control")


@pytest.mark.parametrize("number", [5, 7, 26])
def test_a_slot_past_the_fourth_gets_more_than_the_generic_fallback(number):
    section = _nested(f"Organelle {number}")
    text = section_tooltip("mask", section)
    assert f"organelle slot {number}" in text
    assert "own channel" in text


def test_no_slot_is_advertised_as_the_last_one():
    """Slot four stopped being the last one when the count became a setting."""
    for role in ALL_ORGANELLE_ROLES:
        text = OBJECT_SUBHEADING_TOOLTIPS[organelle_slot_label(role).upper()]
        assert "last one" not in text
        assert "the fourth organelle slot" not in text


# ---------------------------------------------------------------------------
# A CSV written at seven, opened at two
# ---------------------------------------------------------------------------

def _seven_slot_settings(src):
    settings = {"src": str(src), NUMBER_OF_ORGANELLES: 7}
    for number, role in enumerate(organelle_roles(7), start=1):
        settings[f"{role}_channel"] = number
        settings[f"{role}_diameter"] = 10 * number
        settings[f"{role}_min_size"] = 5 * number
    return settings


def _write_then_read(settings, tmp_path):
    import os

    from spacr.utils import load_settings, save_settings

    save_settings(settings, name="seven")
    return load_settings(os.path.join(str(tmp_path), "settings", "seven.csv"))


def test_a_csv_written_at_seven_loads_at_two_with_every_answer_intact(
        tmp_path):
    """The file is where the promise has to hold, not just a dict in memory."""
    import spacr.settings as S

    written = _seven_slot_settings(tmp_path)
    loaded = _write_then_read(written, tmp_path)

    lowered = dict(loaded, **{NUMBER_OF_ORGANELLES: 2})
    at_two = S.set_default_settings_preprocess_generate_masks(lowered)

    # Two on the panel, seven in the file.
    assert active_organelle_roles(at_two) == organelle_roles(2)
    assert declared_organelle_roles(at_two) == organelle_roles(7)

    # Not one of the five hidden slots lost an answer.
    for number, role in enumerate(organelle_roles(7), start=1):
        assert at_two[f"{role}_channel"] == number
        assert at_two[f"{role}_diameter"] == 10 * number
        assert at_two[f"{role}_min_size"] == 5 * number


def test_lowering_then_raising_the_count_restores_the_hidden_answers(tmp_path):
    import spacr.settings as S

    written = _seven_slot_settings(tmp_path)
    loaded = _write_then_read(written, tmp_path)

    at_two = S.set_default_settings_preprocess_generate_masks(
        dict(loaded, **{NUMBER_OF_ORGANELLES: 2}))
    back_at_seven = S.set_default_settings_preprocess_generate_masks(
        dict(at_two, **{NUMBER_OF_ORGANELLES: 7}))

    assert active_organelle_roles(back_at_seven) == organelle_roles(7)
    for number, role in enumerate(organelle_roles(7), start=1):
        assert back_at_seven[f"{role}_channel"] == number
        assert back_at_seven[f"{role}_diameter"] == 10 * number


def test_a_csv_written_at_seven_survives_a_full_round_trip_back_to_disk(
        tmp_path):
    """Opened at two and saved again, the file still carries all seven.

    The failure this rules out is silent: a session that shows two slots
    writing a file with five of them dropped, so the values are gone the next
    time the number is raised rather than merely out of sight.
    """
    import spacr.settings as S

    written = _seven_slot_settings(tmp_path)
    loaded = _write_then_read(written, tmp_path)
    at_two = S.set_default_settings_preprocess_generate_masks(
        dict(loaded, **{NUMBER_OF_ORGANELLES: 2}))

    rewritten = _write_then_read(dict(at_two, src=str(tmp_path)), tmp_path)

    for number, role in enumerate(organelle_roles(7), start=1):
        assert rewritten[f"{role}_channel"] == number
        assert rewritten[f"{role}_diameter"] == 10 * number


# ---------------------------------------------------------------------------
# Hidden, not deleted -- the same rule on the panel
# ---------------------------------------------------------------------------

def _panel_keys():
    keys = [NUMBER_OF_ORGANELLES, "cell_channel", "cell_diameter"]
    for role in organelle_roles(3):
        keys += [f"{role}_channel", f"{role}_type", f"{role}_diameter",
                 f"{role}_ridge_filter", f"{role}_log_min_sigma",
                 f"{role}_fill_holes", f"{role}_adaptive_block_size"]
    return keys


def test_a_channel_naming_no_plane_hides_its_settings_and_keeps_the_switch():
    settings = {NUMBER_OF_ORGANELLES: 2, "cell_channel": None,
                "organelle_channel": 0, "organelleb_channel": None}
    hidden = keys_hidden_by_their_object(_panel_keys(), settings)

    assert "cell_diameter" in hidden
    # The switch itself stays, or the user cannot turn the object back on.
    assert "cell_channel" not in hidden
    assert "organelleb_channel" not in hidden
    assert "organelleb_diameter" in hidden
    # The slot that does name a plane is untouched.
    assert "organelle_diameter" not in hidden


def test_hiding_a_setting_does_not_change_a_single_value():
    """`keys_hidden_by_their_object` answers a question; it erases nothing."""
    settings = {NUMBER_OF_ORGANELLES: 1, "cell_channel": None,
                "cell_diameter": 40, "organelle_channel": 0,
                "organelleb_channel": None, "organelleb_diameter": 17}
    before = dict(settings)

    hidden = keys_hidden_by_their_object(_panel_keys(), settings)

    assert "cell_diameter" in hidden and "organelleb_diameter" in hidden
    assert settings == before
    assert settings["cell_diameter"] == 40
    assert settings["organelleb_diameter"] == 17


def test_the_type_decides_which_of_a_slots_settings_appear():
    """Two slots, two types, and each shows only its own morphology's rows."""
    settings = {NUMBER_OF_ORGANELLES: 2,
                "organelle_channel": 0, "organelle_type": "punctate",
                "organelleb_channel": 1, "organelleb_type": "filamentous"}
    hidden = keys_hidden_by_their_object(_panel_keys(), settings)

    # Punctate is spots: the LoG sigmas, not the ridge filter.
    assert "organelle_log_min_sigma" not in hidden
    assert "organelle_ridge_filter" in hidden
    assert "organelle_fill_holes" in hidden

    # Filamentous is a network: the ridge filter, not the LoG sigmas.
    assert "organelleb_ridge_filter" not in hidden
    assert "organelleb_log_min_sigma" in hidden

    # A setting no morphology claims is shown for both.
    assert "organelle_adaptive_block_size" not in hidden
    assert "organelleb_adaptive_block_size" not in hidden


def test_changing_one_slots_type_moves_only_that_slots_settings():
    base = {NUMBER_OF_ORGANELLES: 2,
            "organelle_channel": 0, "organelle_type": "punctate",
            "organelleb_channel": 1, "organelleb_type": "punctate"}
    before = keys_hidden_by_their_object(_panel_keys(), base)
    after = keys_hidden_by_their_object(
        _panel_keys(), dict(base, organelleb_type="filamentous"))

    assert "organelleb_ridge_filter" in before
    assert "organelleb_ridge_filter" not in after
    # Slot one did not move.
    assert ({key for key in before if key.startswith("organelle_")}
            == {key for key in after if key.startswith("organelle_")})


def test_a_slot_above_the_count_takes_its_channel_off_the_form_too():
    settings = {NUMBER_OF_ORGANELLES: 2, "organelle_channel": 0,
                "organelleb_channel": 1, "organellec_channel": 2}
    hidden = keys_hidden_by_their_object(_panel_keys(), settings)

    assert "organellec_channel" in hidden
    assert "organellec_diameter" in hidden
    assert "organelleb_channel" not in hidden
