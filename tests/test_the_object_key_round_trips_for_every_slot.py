"""``prcfo`` composes and parses back for every organelle slot.

Instruction 326, step 1, in its own words: "Write the round-trip property test
FIRST, against the CURRENT scheme, and watch it pass. It is the safety net, and
a net written after the change tests the change rather than the invariant."

So this asserts nothing about the scheme that will replace ``organelle`` ..
``organellez``. It pins what must remain true through that change: an object
key composed from a role and a label parses back to exactly that role and that
label. The three cases 326 names as the ones naive schemes break are here by
name -- a label that begins with a digit, the roles at the old 26/27 boundary,
and a plate whose name contains the key separator.
"""
from __future__ import annotations

import pytest

from spacr.organelle_types import MAX_ORGANELLES, organelle_role
from spacr.schema import ORGANELLE_ROLES, compose_prcfo, parse_prcfo


def _round_trip(plate, row, column, field, obj, object_type=None):
    key = compose_prcfo(plate, row, column, field, obj,
                        object_type=object_type)
    parsed = parse_prcfo(key)
    return key, parsed


@pytest.mark.parametrize("role", ORGANELLE_ROLES)
def test_every_keyable_organelle_role_round_trips(role):
    """Every role the KEY vocabulary admits survives a round trip.

    Parametrized over ``schema.ORGANELLE_ROLES`` rather than over
    ``range(MAX_ORGANELLES)`` on purpose -- see
    :func:`test_the_keyable_roles_are_fewer_than_the_slots_offered`, which
    records that those two numbers do not currently agree.
    """
    key, parsed = _round_trip("plate1", 1, 1, 2, 7, object_type=role)
    assert parsed.objectType == role, (role, key)


def test_the_keyable_roles_are_fewer_than_the_slots_offered():
    """The bound 326 has to reconcile before it can raise anything.

    ``organelle_types.MAX_ORGANELLES`` is 702 -- raised from 26 when the
    lettering learned to carry past ``z`` -- and the settings panel offers
    that many slots (``PANEL_ORGANELLE_SLOTS``), while ``schema`` keys objects
    by a CLOSED vocabulary holding four organelle roles. Slots 5 and up
    therefore cannot be written into an object key at all: ``is_object_type``
    answers False and the frame is left untyped, which reinstates exactly the
    collision the type was added to prevent -- "a nucleus labelled 1 and a
    pathogen labelled 1 in the same field were the same key".

    Pinned rather than asserted-equal so the disagreement is visible and this
    test must be updated deliberately when 326 raises the ceiling.
    """
    assert len(ORGANELLE_ROLES) == 4, ORGANELLE_ROLES
    assert MAX_ORGANELLES == 702
    assert len(ORGANELLE_ROLES) < MAX_ORGANELLES, (
        "the vocabularies now agree; update 326 and this test together")


def test_a_label_that_begins_with_a_digit_survives():
    """``cell1`` + 7 must not be confusable with ``cell`` + 17.

    This is the ambiguity the digit-free role rule exists to prevent, and it
    is the first thing an escaped-digit scheme would put at risk.
    """
    key, parsed = _round_trip("plate1", 1, 1, 2, "1x", object_type="nucleus")
    assert parsed.objectType == "nucleus", key
    assert str(parsed.objectLabel) == "1x", (key, parsed.objectLabel)


def test_the_roles_at_the_boundaries_behave():
    """The last KEYABLE role round trips; the first unkeyable one is refused.

    Both halves matter to 326. The first says the vocabulary works up to its
    edge; the second says the edge is enforced rather than silently accepted,
    which is what stops a slot-5 organelle being written as an untyped object
    that collides with a nucleus of the same label.
    """
    from spacr.schema import KeyParseError

    last = ORGANELLE_ROLES[-1]
    key, parsed = _round_trip("plate1", 1, 1, 2, 7, object_type=last)
    assert parsed.objectType == last, key

    beyond = organelle_role(len(ORGANELLE_ROLES) + 1)
    assert beyond not in ORGANELLE_ROLES, beyond
    with pytest.raises(KeyParseError):
        compose_prcfo("plate1", 1, 1, 2, 7, object_type=beyond)

    with pytest.raises(ValueError):
        organelle_role(MAX_ORGANELLES + 1)


def test_a_plate_name_containing_the_separator_round_trips():
    """``_`` is KEY_SEPARATOR, so an underscore in a plate name splits the key
    unless it is escaped. ``io.migrate_unescaped_plate_names`` exists because
    this was once wrong on disk."""
    key, parsed = _round_trip("plate_one", 1, 1, 2, 7, object_type="cell")
    assert parsed.objectType == "cell", key
    assert str(parsed.plateID) == "plate_one", (key, parsed.plateID)


def test_an_untyped_key_still_parses_untyped():
    """Every prcfo written before object types must keep meaning what it meant."""
    key, parsed = _round_trip("plate1", 1, 1, 2, 7)
    assert parsed.objectType is None, key
    assert key == "plate1_r1_c1_f2_o7", key


def test_parsing_is_idempotent():
    """A key that grows each time it passes the parser joins to nothing.

    The docstring of ``parse_prcfo`` records exactly this bug for non-numeric
    labels (``'oxy'`` -> ``'ooxy'`` -> ``'ooooxy'``), so it is pinned here.
    """
    key = compose_prcfo("plate1", 1, 1, 2, "xy")
    once = parse_prcfo(key)
    twice = parse_prcfo(once.prcfo)
    assert once.objectID == twice.objectID, (key, once.objectID, twice.objectID)
    assert once.prcfo == twice.prcfo, (once.prcfo, twice.prcfo)
