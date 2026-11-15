"""Filename families the inference engine has to refuse or name correctly.

``propose`` is offered to a user staring at a drop from a microscope spaCR has
never met, and its answer becomes the regex the importer runs over every file.
The cases below are the ones where a wrong answer is worse than no answer: a
family with nothing varying, a family that merged two different microscopes,
and slots whose role has to be read off the values rather than off a vendor
letter.
"""

from __future__ import annotations

from spacr.regex_infer import (
    FieldEvidence,
    _assign_roles,
    propose,
)


# ---------------------------------------------------------------------------
# Role assignment read off the values
# ---------------------------------------------------------------------------

def test_a_slot_that_holds_wells_is_named_a_well_without_any_hint():
    """Values like ``A01`` are a well whatever literal precedes them."""
    slot = FieldEvidence(index=0, values=("A01", "B02", "H12"), numeric=False)
    _assign_roles([slot])

    assert slot.role == "wellID"
    assert slot.because == "every value looks like a well"


def test_a_slot_with_one_repeated_value_is_named_the_plate():
    """A component identical in every file is the plate, not a field id."""
    slot = FieldEvidence(index=0, values=("exp", "exp"), numeric=False)
    _assign_roles([slot])

    assert slot.role == "plateID"
    assert slot.because == "the same in every file"


# ---------------------------------------------------------------------------
# Two microscopes are not one family
# ---------------------------------------------------------------------------

def test_a_family_varying_in_too_many_words_is_thrown_away():
    """A pattern that matches everything by naming nothing is not offered.

    ``ab1cd2ef3`` and ``gh1ij2kl3`` tokenise alike, so the coarse grouping
    merges them, and every one of the three word slots varies. That is two
    unrelated naming conventions, and the regex built from it would capture
    three groups that mean nothing.
    """
    assert propose(["ab1cd2ef3.tif", "gh1ij2kl3.tif"]) == []
