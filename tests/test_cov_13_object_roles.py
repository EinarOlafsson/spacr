"""Role helpers must reject a wrong name loudly and translate a slot's keys.

Every one of these guards protects a caller that would otherwise get a
plausible-looking but wrong answer: a settings key built for a role that has
none, an organelle view that silently keeps the wrong channel, an index for
something that is not a slot at all. Each is cheap to get wrong and expensive
to notice downstream, so each raises instead of returning a default.
"""
from __future__ import annotations

import pytest

from spacr import object_roles
from spacr.schema import ORGANELLE_ROLES, SEGMENTED_ROLES


def test_is_organelle_accepts_every_slot_and_refuses_everything_else():
    """The slot test answers for non-strings without raising.

    Callers pass whatever a settings dict held, which may be ``None`` or a
    number. An exception there would turn "this key is not a slot" into a
    crash in code whose whole job is to ask the question safely.
    """
    for role in ORGANELLE_ROLES:
        assert object_roles.is_organelle(role) is True
    assert object_roles.is_organelle("cell") is False
    assert object_roles.is_organelle(None) is False
    assert object_roles.is_organelle(7) is False


def test_organelle_index_names_the_bad_role_and_lists_the_real_ones():
    """A wrong slot name must say what was passed and what was expected.

    ``organelle_index`` is called with names assembled from settings keys, so
    the failure a user actually hits is a typo. The message has to carry both
    halves or the fix is a guessing game.
    """
    assert object_roles.organelle_index("organelle") == 1
    # FOUR, NOT "THE LAST ONE". This read `len(ORGANELLE_ROLES)`, which was
    # the same number while there were exactly four slots; 326 made the count
    # arbitrary and the vocabulary now runs to 702, so the assertion said
    # `organelled` was slot 702. What it is actually about is that the letter
    # names count from one, and `d` is the fourth letter whatever the ceiling.
    assert object_roles.organelle_index("organelled") == 4

    with pytest.raises(ValueError) as excinfo:
        object_roles.organelle_index("nucleus")
    message = str(excinfo.value)
    assert "'nucleus'" in message
    assert "organelleb" in message
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_organelle_label_refuses_a_non_slot_rather_than_labelling_it():
    """A label for a non-slot would read fine and be meaningless."""
    with pytest.raises(ValueError, match="not an organelle role"):
        object_roles.organelle_label("cytoplasm")


def test_role_setting_refuses_a_role_that_has_no_settings():
    """``cytoplasm`` is derived, so a ``cytoplasm_channel`` key is a fiction.

    Building the key anyway would produce a setting nothing reads and no
    error, which is how a channel silently goes unset.
    """
    assert object_roles.role_setting("organelleb", "channel") == "organelleb_channel"
    assert object_roles.role_setting("organelleb", "_channel") == "organelleb_channel"
    assert "cytoplasm" not in SEGMENTED_ROLES

    with pytest.raises(ValueError, match="'cytoplasm' is not a segmented role"):
        object_roles.role_setting("cytoplasm", "channel")


def test_organelle_settings_view_refuses_a_role_that_is_not_a_slot():
    """The legacy adapter must not invent an ``organelle_*`` view of a cell."""
    with pytest.raises(ValueError, match="unknown organelle role 'cell'"):
        object_roles.organelle_settings_view({"cell_channel": 0}, "cell")


def test_organelle_settings_view_carries_the_recorded_cellpose_channel():
    """The slot's resolved cellpose channel must reach the legacy key.

    The classical segmenter reads ``cellpose_organelle_channel``. A slot other
    than the first records its own ``cellpose_<role>_channel``, and dropping
    that during translation makes the segmenter fall back to whatever the
    first slot resolved to -- it would run, and it would segment the wrong
    plane.
    """
    settings = {
        "organellec_channel": 2,
        "organellec_diameter": 11,
        "cellpose_organellec_channel": 5,
        "cellpose_organelle_channel": 0,
    }
    view = object_roles.organelle_settings_view(settings, "organellec")

    assert view["organelle_channel"] == 2
    assert view["organelle_diameter"] == 11
    assert view["cellpose_organelle_channel"] == 5


def test_organelle_settings_view_leaves_the_legacy_channel_alone_when_unrecorded():
    """With nothing recorded for the slot the original key must survive.

    Overwriting it with ``None`` would be worse than not translating: the
    segmenter would read a missing channel where a usable one was present.
    """
    settings = {"organellec_channel": 2, "cellpose_organelle_channel": 0}
    view = object_roles.organelle_settings_view(settings, "organellec")

    assert view["cellpose_organelle_channel"] == 0


def test_organelle_settings_view_returns_a_plain_copy_for_the_legacy_slot():
    """The first slot already speaks the legacy names, so nothing is renamed."""
    settings = {"organelle_channel": 1, "unrelated": "x"}
    view = object_roles.organelle_settings_view(settings, "organelle")

    assert view == settings
    assert view is not settings
