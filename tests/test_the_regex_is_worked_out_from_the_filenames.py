"""Instruction 137 B — the program proposes the regex.

    "for the import data modulation the user should be able to drag and drop
     images into a table and the program based on those chooses desires the
     regex to use for detecting what all files are."

`spacr.utils._get_regex` is a four-branch lookup of which three are FIXED
PATTERNS ('auto' is not automatic; it is a fourth hard-coded regex) and the
fourth means typing a Python regex with the right named groups by hand. A
microscope spaCR has not met is a wall before anything can be imported.

The property these tests hold above all the others: IT NEVER PICKS SILENTLY.
The failure being fixed is a regex that looked right and grouped the files
wrong, so a proposal carries how many files it matches, how many distinct
values each group takes, WHICH files it leaves out, and why each role was
guessed. A guess presented without its reason is indistinguishable from a
fact.
"""
from __future__ import annotations

import re

import pytest

from spacr.regex_infer import (KNOWN_ROLES, propose, rename_preview,
                               structure, tokenise)


def _cellvoyager(wells=("A01", "A02", "B01"), fields=(1, 2), channels=(1, 2, 3)):
    return [f"plate1_{w}_T0001F{f:03d}L01A01Z01C{c:02d}.tif"
            for w in wells for f in fields for c in channels]


def _cq1(wells=("A01", "B02"), fields=(1, 2), channels=(1, 2, 3)):
    return [f"W{w}F{f:03d}T0001Z01C{c}.tif"
            for w in wells for f in fields for c in channels]


def _unseen():
    return [f"exp7-P{p}-{r}{c:02d}-s{s}-ch{ch}.png"
            for p in (1, 2) for r in ("A", "B") for c in (1, 2)
            for s in (1, 2) for ch in (1, 2)]


# -- it matches what it is given --------------------------------------------

@pytest.mark.parametrize("names", [_cellvoyager(), _cq1(), _unseen()])
def test_the_best_proposal_matches_every_file_it_was_shown(names):
    best = propose(names)[0]
    assert best.matched == len(names)
    assert best.unmatched == ()
    assert best.coverage == 1.0


@pytest.mark.parametrize("names,expected", [
    (_cellvoyager(), {"wellID", "fieldID", "chanID"}),
    (_cq1(), {"wellID", "fieldID", "chanID"}),
    (_unseen(), {"plateID", "wellID", "fieldID", "chanID"}),
])
def test_the_groups_are_named_for_what_they_hold(names, expected):
    """A group name spaCR's importer does not read is a silent import of nothing."""
    best = propose(names)[0]
    assert expected <= set(best.fields)
    for name in best.fields:
        assert name in KNOWN_ROLES or name.startswith("group")


def test_a_well_number_that_never_changes_is_still_part_of_the_well():
    """Two wells, one column: the digits fold into the literal and A is not a well."""
    best = propose(_cellvoyager(wells=("A01", "B01")))[0]
    assert "wellID" in best.fields
    assert set(best.fields["wellID"].values) == {"A01", "B01"}


def test_a_literal_word_beats_a_value_count_for_naming_a_channel():
    """`-ch1` IS the channel; a plate code with two values only looks like one."""
    best = propose(_unseen())[0]
    assert best.fields["chanID"].distinct == 2
    assert "ch" in best.fields["chanID"].because
    assert best.fields["plateID"].distinct == 2


def test_a_separator_stops_a_word_from_claiming_the_slot_after_it():
    """`plate1_` ends in "plate" and the slot after it is the WELL."""
    best = propose(_cellvoyager())[0]
    assert "wellID" in best.fields
    assert best.fields.get("plateID") is None or (
        best.fields["plateID"].values[0] != "A01")


# -- and it says what it does not match --------------------------------------

def test_a_file_that_does_not_match_is_named_never_dropped():
    """412 files appearing without comment is how half a plate goes missing."""
    names = _cellvoyager() + ["notes.txt", "README"]
    best = propose(names)[0]

    assert best.matched == len(names) - 2
    assert set(best.unmatched) == {"notes.txt", "README"}
    assert "NOT matched" in best.evidence()
    assert "notes.txt" in best.evidence()


def test_two_microscopes_get_two_proposals_and_neither_is_picked_silently():
    both = _cellvoyager() + _cq1()
    proposals = propose(both)

    assert len(proposals) >= 2
    # Neither covers everything, and that is exactly what the user must see.
    assert proposals[0].matched < len(both)
    assert proposals[0].unmatched


def test_a_family_that_merged_two_unrelated_microscopes_is_refused():
    """A regex that matches everything and means nothing is worse than none."""
    for proposal in propose(_cellvoyager() + _cq1()):
        literal_groups = [f for f in proposal.fields.values() if not f.numeric]
        assert len(literal_groups) <= 2


def test_the_evidence_names_the_counts_and_the_reason_for_every_guess():
    best = propose(_unseen())[0]
    text = best.evidence()

    assert "32 of 32 files match (100%)" in text
    for name, info in best.fields.items():
        assert name in text
        if info.because:
            assert info.because in text


# -- the preview shows and does not write ------------------------------------

def test_the_preview_says_what_each_file_would_become(tmp_path):
    names = _cellvoyager()
    best = propose(names)[0]
    preview = rename_preview(best, names)

    assert len(preview) == len(names)
    assert all(row["matched"] for row in preview)
    assert preview[0]["values"]["wellID"] == "A01"
    # NOTHING IS WRITTEN. The old `_run_test_mode` copies real files to answer
    # this question; the point is answering it before any commitment.
    assert list(tmp_path.iterdir()) == []


def test_an_unmatched_file_is_in_the_preview_saying_so():
    names = _cellvoyager() + ["stray.tif"]
    preview = rename_preview(propose(names)[0], names)

    stray = [row for row in preview if row["old"] == "stray.tif"]
    assert len(stray) == 1 and stray[0]["matched"] is False


def test_the_user_renames_a_group_without_touching_the_regex():
    """137 C: nobody types a group name; the dropdown is what makes that safe."""
    names = _cq1()
    best = propose(names)[0]
    preview = rename_preview(best, names, roles={"fieldID": "sliceID"})

    assert "sliceID" in preview[0]["values"]
    assert "fieldID" not in preview[0]["values"]


def test_the_folder_tree_is_the_spacr_structure_with_counts():
    names = _cellvoyager()
    tree = structure(rename_preview(propose(names)[0], names))

    assert sum(tree.values()) == len(names)
    # plate / well / field / channel, in that order, and only the parts present.
    assert all(len(folder.split("/")) <= 4 for folder in tree)


# -- the pieces --------------------------------------------------------------

def test_a_name_splits_into_runs_of_digits_and_everything_else():
    assert tokenise("W A01F003.tif") == ["W A", "01", "F", "003", ".tif"]


def test_nothing_dropped_gives_nothing_proposed():
    assert propose([]) == []
    assert propose(["", "   "]) == []


def test_one_file_alone_has_nothing_to_compare_and_says_so_by_proposing_nothing():
    assert propose(["plate1_A01_T0001F001L01A01Z01C01.tif"]) == []


def test_only_the_basename_is_read():
    """A user dropping a folder must not get their home directory in the regex."""
    names = [f"/home/someone/Data 2026/{n}" for n in _cq1()]
    best = propose(names)[0]

    assert "home" not in best.pattern
    assert best.matched == len(names)


def test_every_proposal_compiles():
    for names in (_cellvoyager(), _cq1(), _unseen(), _cellvoyager() + _cq1()):
        for proposal in propose(names):
            assert re.compile(proposal.pattern) is not None
