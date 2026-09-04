"""326 step 2: the lettering carries past z, and 1..26 do not move."""
import pytest

from spacr.organelle_types import (MAX_ORGANELLES, organelle_number,
                                   organelle_role, organelle_role_of)


class TestTheFirstTwentySixAreByteIdentical:
    """The "DONT BREAK ANYTHING" constraint, satisfied by construction.

    Slots 1..26 keep their exact current names, so no measurement database,
    settings CSV or saved state moves.
    """

    def test_slot_one_is_still_the_bare_word(self):
        assert organelle_role(1) == "organelle"

    @pytest.mark.parametrize("slot,expected", [
        (2, "organelleb"), (3, "organellec"), (26, "organellez")])
    def test_the_lettered_slots_are_unchanged(self, slot, expected):
        assert organelle_role(slot) == expected

    def test_organellea_is_never_minted(self):
        """Slot 1 is the bare word, so a single 'a' cannot collide with 'aa'."""
        assert "organellea" not in {organelle_role(n)
                                    for n in range(1, min(60, MAX_ORGANELLES) + 1)}


class TestTheLetteringCarries:
    """A hundred organelles must work, so the alphabet cannot be the ceiling."""

    def test_twenty_seven_carries_to_two_letters(self):
        assert organelle_role(27) == "organelleaa"

    @pytest.mark.parametrize("slot,expected", [
        (28, "organelleab"), (52, "organelleaz"), (53, "organelleba")])
    def test_the_carry_counts_in_base_twenty_six(self, slot, expected):
        assert organelle_role(slot) == expected

    def test_a_hundred_organelles_is_reachable(self):
        """The maintainer's own number: "if the user chooses 100 organelles"."""
        assert MAX_ORGANELLES >= 100
        assert organelle_role(100)

    def test_every_role_is_digit_free_and_separator_free(self):
        """The two hard constraints. A digit is ambiguous against the object
        LABEL and an underscore is the key separator."""
        for slot in range(1, min(300, MAX_ORGANELLES) + 1):
            role = organelle_role(slot)
            assert role.isalpha(), role
            assert "_" not in role


class TestTheRolesRoundTrip:
    def test_number_inverts_role_for_every_slot(self):
        for slot in range(1, min(300, MAX_ORGANELLES) + 1):
            assert organelle_number(organelle_role(slot)) == slot

    def test_no_two_slots_share_a_role(self):
        roles = [organelle_role(n) for n in range(1, min(300, MAX_ORGANELLES) + 1)]
        assert len(set(roles)) == len(roles)

    def test_a_settings_key_finds_its_carried_slot(self):
        """`organelle_role_of` matches longest-first, so a carried role must
        not be read as a shorter one."""
        assert organelle_role_of("organelleaa_channel") == "organelleaa"
        assert organelle_role_of("organelleb_channel") == "organelleb"
        assert organelle_role_of("organelle_channel") == "organelle"


class TestTheCeilingStillSpeaks:
    def test_past_the_ceiling_raises_rather_than_clamping(self):
        """A file asking for more must be TOLD which keys stopped existing."""
        with pytest.raises(ValueError):
            organelle_role(MAX_ORGANELLES + 1)

    def test_zero_and_negative_are_refused(self):
        for bad in (0, -1):
            with pytest.raises(ValueError):
                organelle_role(bad)
