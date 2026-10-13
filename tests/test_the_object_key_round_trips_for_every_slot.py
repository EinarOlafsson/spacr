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


def test_the_keyable_roles_are_exactly_the_slots_offered():
    """The disagreement 326 was filed to end. It has ended.

    This test used to assert the OPPOSITE, and said so: ``schema`` keyed
    objects by a closed vocabulary of four organelle roles while
    ``MAX_ORGANELLES`` offered twenty-six and then 702, so slots five and up
    produced a valid settings prefix that could not be written into an object
    key. ``is_object_type`` answered False, the frame was left untyped, and
    that reinstated exactly the collision the object type was added to
    prevent -- "a nucleus labelled 1 and a pathogen labelled 1 in the same
    field were the same key".

    Its own docstring required it to be "updated deliberately when 326 raises
    the ceiling". This is that update, and it now asserts agreement rather
    than pinning a gap.
    """
    assert len(ORGANELLE_ROLES) == MAX_ORGANELLES, (
        len(ORGANELLE_ROLES), MAX_ORGANELLES)
    assert ORGANELLE_ROLES[0] == "organelle"
    assert ORGANELLE_ROLES[-1] == organelle_role(MAX_ORGANELLES)


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

    # PAST the ceiling rather than past the vocabulary -- those used to be
    # different places and are now the same one. `organelleaaa` is what slot
    # 703 would be called, and nothing may key it.
    beyond = "organelleaaa"
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


def test_the_two_statements_of_the_lettering_rule_agree():
    """`schema` restates the lettering rule instead of importing it.

    It has to: everything imports `schema`, so it must cost nothing to
    import, and `test_module_imports_with_only_the_stdlib_and_pandas` bans
    every `spacr` import inside it. That is a real constraint and not one to
    weaken for a three-line convenience.

    But two statements of one rule is precisely what `schema`'s own
    `KEY_ESCAPES` comment calls "exactly the kind of pair that drifts apart
    later", and this very pair had already drifted by 698 entries before
    today -- four hand-written roles against a minter producing 702. So the
    trade is a dependency this module cannot afford, exchanged for this
    test: every slot, both sides, must agree.
    """
    from spacr import schema as sch
    from spacr.organelle_types import MAX_ORGANELLES, organelle_role

    assert sch._MAX_ORGANELLES == MAX_ORGANELLES, (
        "the two ceilings disagree; one of them was raised alone")
    mismatched = [
        slot for slot in range(1, MAX_ORGANELLES + 1)
        if sch._organelle_role(slot) != organelle_role(slot)
    ]
    assert not mismatched, (
        f"{len(mismatched)} slots spell differently in schema and "
        f"organelle_types, first {mismatched[:5]}")


def test_every_slot_the_panel_offers_can_be_written_into_an_object_key():
    """The gap 326 exists to close, asserted from the other direction.

    An organelle whose role is not in the keying vocabulary produces an
    UNTYPED frame, which is the collision the object type was introduced to
    remove: "a nucleus labelled 1 and a pathogen labelled 1 in the same field
    were the same key". Slots 5 and up were in that state.
    """
    from spacr.schema import is_object_type, split_object_id
    from spacr.organelle_types import MAX_ORGANELLES, organelle_role

    for slot in (1, 2, 4, 5, 26, 27, 100, MAX_ORGANELLES):
        role = organelle_role(slot)
        assert is_object_type(role), f"slot {slot} ({role}) cannot be keyed"
        kind, label = split_object_id(f"{role}7")
        assert (kind, label) == (role, "7"), (slot, role, kind, label)


def test_the_vocabulary_is_still_closed():
    """Widening it to 702 roles must not make it open.

    An open vocabulary would mean every unrecognised token in the object
    slot became a type, and `plate1_r1_c1_f2_x7` -- which is not an object
    key -- would parse as object 7 of type 'x'.
    """
    from spacr.schema import is_object_type

    for made_up in ("mitochondrion", "organelle1", "organelle_b", "x",
                    "organellea", "ORGANELLEZZZZZZZ"):
        assert not is_object_type(made_up), made_up
