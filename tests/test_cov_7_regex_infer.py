"""Filename families the inference engine has to refuse or name correctly.

``propose`` is offered to a user staring at a drop from a microscope spaCR has
never met, and its answer becomes the regex the importer runs over every file.
The cases below are the ones where a wrong answer is worse than no answer: a
family with nothing varying, a family that merged two different microscopes,
and slots whose role has to be read off the values rather than off a vendor
letter.
"""

from __future__ import annotations

import pytest

from spacr import regex_infer
from spacr.regex_infer import (
    FieldEvidence,
    _assign_roles,
    _merge_wells,
    _proposal_for,
    propose,
)


# ---------------------------------------------------------------------------
# Families that cannot become a proposal
# ---------------------------------------------------------------------------

def test_no_names_at_all_is_not_a_proposal():
    """An empty family has no tokens to compare, so there is no pattern."""
    assert _proposal_for([], []) is None


def test_a_family_where_nothing_varies_is_not_a_proposal():
    """One repeated name yields a regex with no capture groups at all."""
    assert _proposal_for(["A01.tif", "A01.tif"], ["A01.tif"]) is None


def test_a_slot_whose_values_are_all_common_text_becomes_a_literal(monkeypatch):
    """A slot with nothing left after factoring contributes no group.

    ``_factor`` splits a varying column into a shared head, a shared tail and
    the remainder. When the remainder is empty for every file the column
    carries no information, and folding it into the surrounding literal is the
    only honest answer -- a capture group matching the empty string would
    appear in the proposal as a metadata field the files do not have.
    """
    monkeypatch.setattr(regex_infer, "_factor",
                        lambda values: ("head", "tail", ("", "")))

    assert _proposal_for(["ab.tif", "cd.tif"], ["ab.tif", "cd.tif"]) is None


# ---------------------------------------------------------------------------
# Literals sitting between slots
# ---------------------------------------------------------------------------

def test_a_literal_between_two_slots_is_not_mistaken_for_a_well():
    """Well merging only joins a letter slot to the digit slot beside it.

    ``1x2.tif`` puts a literal ``x`` between two numeric slots. Neither pair
    is a letter followed by digits, so both slots survive as their own groups
    and the literal stays literal.
    """
    proposal = _proposal_for(["1x2.tif", "3x4.tif"], ["1x2.tif", "3x4.tif"])

    assert proposal is not None
    for field in ("fields", "matched", "total", "unmatched", "suffix"):
        assert f":param {field}:" in (type(proposal).__doc__ or "")
    assert "wellID" not in proposal.fields
    assert proposal.matched == 2
    assert len(proposal.fields) == 2


def test_a_constant_number_is_only_absorbed_when_it_makes_a_well():
    """``ab12`` is not a well, so the ``12`` stays in the literal text.

    The absorption rule exists for a plate whose wells are all ``A01`` and
    ``B01``; letting it fire on any letter run followed by digits would name a
    two-letter channel code as a well.
    """
    names = ["ab12.tif", "cd12.tif"]
    proposal = _proposal_for(names, names)

    assert proposal is not None
    assert "wellID" not in proposal.fields
    assert "12" in proposal.pattern
    assert proposal.matched == 2


def test_a_letter_run_and_a_digit_run_that_do_not_make_a_well_stay_apart():
    """The check that keeps ``L01`` and ``C02`` out of the well group.

    ``_merge_wells`` folds an adjacent letter slot and digit slot into one --
    ``A01`` tokenises as two runs and neither of them is a well. But a
    two-letter code beside a three-digit number joins to ``ab123``, which is
    not a well, and merging it would name a channel or a cycle as the plate
    position. Every downstream grouping -- normalisation, the plate heatmap,
    the well-level regression -- then aggregates by something that is not a
    well.

    Driven at the function rather than through a filename, because the
    tokeniser decides whether these two slots are ever adjacent, and this is
    about what happens once they are.
    """
    pieces = [
        {"slot": FieldEvidence(index=0, values=("ab", "cd"), numeric=False)},
        {"slot": FieldEvidence(index=1, values=("123", "456"), numeric=True)},
    ]

    _merge_wells(pieces)

    assert len(pieces) == 2, "two slots that are not a well were merged"
    assert not pieces[0]["slot"].role, "the letter run was labelled a well"
    assert pieces[1]["slot"].values == ("123", "456")


def test_a_letter_and_two_digits_are_folded_into_one_well_slot():
    """The path the refusal above is defined against.

    Without it, "did not merge" would pass on an implementation that never
    merges anything.
    """
    pieces = [
        {"slot": FieldEvidence(index=0, values=("A", "B"), numeric=False)},
        {"slot": FieldEvidence(index=1, values=("01", "02"), numeric=True)},
    ]

    _merge_wells(pieces)

    assert len(pieces) == 1
    merged = pieces[0]["slot"]
    for field in ("before", "role", "fixed_tail", "because"):
        assert f":ivar {field}:" in (type(merged).__doc__ or "")
    assert merged.role == "wellID"
    assert merged.values == ("A01", "B02")
    assert "read as a well" in merged.because


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


def test_a_role_is_never_handed_out_twice():
    """Two groups with one name is a pattern Python refuses to compile."""
    first = FieldEvidence(index=0, values=("exp", "exp"), numeric=False)
    second = FieldEvidence(index=1, values=("run", "run"), numeric=False)
    _assign_roles([first, second])

    assert first.role == "plateID"
    assert second.role != "plateID"


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


def test_two_adjacent_slots_are_only_merged_when_they_read_as_a_well():
    """A number followed by a word is two fields, not one well.

    ``1a.tif`` puts a numeric slot directly against a text slot with no
    literal between them. Merging that pair would produce a ``wellID`` group
    capturing ``1a``, which matches neither of the two well conventions spaCR
    reads.
    """
    names = ["1a.tif", "2b.tif"]
    proposal = _proposal_for(names, names)

    assert proposal is not None
    assert "wellID" not in proposal.fields
    assert len(proposal.fields) == 2
    assert proposal.matched == 2
