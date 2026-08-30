"""The registry rule that keeps a typed object id readable back.

``spacr.schema`` validates OBJECT_TYPES where it is declared, and the comment
says why failing the import is the right answer: a typed object id concatenates
role and numeric label with no separator, so a digit in a role is AMBIGUOUS --
``cell1`` + 7 and ``cell`` + 17 are the same string -- and an underscore would
split the surrounding prcfo key.

An identity that cannot round-trip is silent corruption of every join that uses
it, which is why this is a hard failure at import rather than a warning.

WHAT IS NOT COVERED, and why. The raise itself (schema.py:287) fires only for a
role that is shipped, and OBJECT_TYPES is a literal in the same module -- so
reaching it means editing the source, not arranging an input. It is an
import-time assertion, and its false side cannot be exercised from a test
without rewriting the file under it. Recorded in instruction 310 as an argued
exclusion; what IS tested here is the rule it enforces, against the roles that
actually ship, plus the round trip that rule exists to protect.
"""
from __future__ import annotations

import pytest


def test_every_shipped_role_satisfies_the_rule():
    """The invariant, checked against what actually ships.

    The four organelle roles are where a digit is easiest to introduce -- the
    next one after ``organelled`` is naturally called ``organelle2`` -- so this
    is the test that would catch it, at the moment it is added rather than at
    the moment a join goes wrong.
    """
    from spacr.schema import KEY_SEPARATOR, OBJECT_TYPES

    assert OBJECT_TYPES, "the registry must not be empty"
    for role in OBJECT_TYPES:
        assert role, "an empty role would make a typed id start with its number"
        assert KEY_SEPARATOR not in role, (
            f"{role!r} contains {KEY_SEPARATOR!r}, which splits the prcfo key")
        assert not any(character.isdigit() for character in role), (
            f"{role!r} contains a digit; 'cell1' + 7 and 'cell' + 17 are the "
            f"same string")


def test_a_typed_id_round_trips_for_every_shipped_role():
    """Why the rule exists, demonstrated rather than asserted abstractly."""
    from spacr.schema import OBJECT_TYPES, object_type_prefix, split_object_id

    for role in OBJECT_TYPES:
        typed = f"{object_type_prefix(role)}17"
        assert split_object_id(typed) == (role, "17"), typed


def test_a_role_with_a_digit_would_make_two_ids_collide():
    """The ambiguity the rule prevents, shown with the comment's own example.

    This is what the registry check is buying, and it is worth stating as a
    test because the cost is invisible: both strings are valid, both parse,
    and they name different objects.
    """
    hypothetical_role_with_a_digit = "cell1"
    ordinary_role = "cell"

    assert hypothetical_role_with_a_digit + "7" == ordinary_role + "17"


def test_a_role_with_a_separator_would_split_the_surrounding_key():
    """The other half of the rule, on the key the id is embedded in."""
    from spacr.schema import KEY_SEPARATOR

    prcfo = KEY_SEPARATOR.join(["plate1", "r1", "c1", "1", "my_role7"])

    # The role's own underscore adds a sixth component to a five-part key.
    assert len(prcfo.split(KEY_SEPARATOR)) == 6
