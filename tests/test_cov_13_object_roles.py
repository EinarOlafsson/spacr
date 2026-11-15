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


