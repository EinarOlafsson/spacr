"""One column arriving from two tables was kept twice and never checked.

Joining cell to cytoplasm gives back ``plateID`` and ``plateID_cytoplasm``,
``prcf`` and ``prcf_cytoplasm`` -- the same plate and the same field, written
by two stages of one run. Nothing compared them. The merged frame carried
both, twice as wide as it needed to be, and a downstream reader picking
``plateID_cytoplasm`` over ``plateID`` had no way to know whether it
mattered.

It matters when they disagree. Both tables read their identity off the same
image, so a mismatch means the two rows describe DIFFERENT objects under one
identity -- and averaging across that produces a number for a cell that does
not exist.

The distinction this module pins: identity must agree, measurements must
not. Cell ``area`` and cytoplasm ``area`` are supposed to differ, and
flagging that would bury the real conflicts under noise.
"""

import numpy as np
import pandas as pd
import pytest

from spacr.merge_tables import (MUST_AGREE, ColumnConflict, MergeError,
                                reconcile_duplicates)

SUFFIX = "_cytoplasm"


def merged(**columns):
    return pd.DataFrame({"prcfo": ["a", "b", "c"], **columns})


# ---------------------------------------------------------------------------
# agreeing duplicates collapse
# ---------------------------------------------------------------------------

def test_an_identical_pair_leaves_one_column():
    out = reconcile_duplicates(
        merged(plateID=["p1"] * 3, plateID_cytoplasm=["p1"] * 3), SUFFIX)
    assert "plateID" in out.columns
    assert "plateID_cytoplasm" not in out.columns
    assert list(out["plateID"]) == ["p1"] * 3


def test_the_column_that_survives_is_the_left_hand_one():
    """The cell table is the anchor, so its spelling is the one kept."""
    out = reconcile_duplicates(
        merged(prcf=["x", "y", "z"], prcf_cytoplasm=["x", "y", "z"]), SUFFIX)
    assert list(out.columns) == ["prcfo", "prcf"]


def test_two_missing_values_are_agreement_not_a_conflict():
    """NaN != NaN is right for arithmetic and wrong for this question."""
    out = reconcile_duplicates(
        merged(timeID=[1.0, np.nan, 3.0], timeID_cytoplasm=[1.0, np.nan, 3.0]),
        SUFFIX)
    assert "timeID_cytoplasm" not in out.columns


def test_float_noise_is_not_a_disagreement():
    out = reconcile_duplicates(
        merged(object_label=[1.0, 2.0, 3.0],
               object_label_cytoplasm=[1.0, 2.0 + 1e-13, 3.0]), SUFFIX)
    assert "object_label_cytoplasm" not in out.columns


def test_several_pairs_collapse_in_one_pass():
    out = reconcile_duplicates(
        merged(plateID=["p1"] * 3, plateID_cytoplasm=["p1"] * 3,
               rowID=["r1"] * 3, rowID_cytoplasm=["r1"] * 3,
               fieldID=["f1"] * 3, fieldID_cytoplasm=["f1"] * 3), SUFFIX)
    assert list(out.columns) == ["prcfo", "plateID", "rowID", "fieldID"]


# ---------------------------------------------------------------------------
# disagreeing identity is reported
# ---------------------------------------------------------------------------

def test_a_disagreeing_identity_column_warns_and_keeps_both(capsys):
    out = reconcile_duplicates(
        merged(rowID=["r1", "r1", "r1"], rowID_cytoplasm=["r1", "r1", "r9"]),
        SUFFIX, left_name="cell", right_name="cytoplasm")

    printed = capsys.readouterr().out
    assert "rowID" in printed
    assert "cell" in printed and "cytoplasm" in printed
    assert "rowID_cytoplasm" in out.columns, (
        "a column that could not be reconciled must not be silently dropped")


def test_the_warning_says_how_many_and_names_an_example(capsys):
    reconcile_duplicates(
        merged(fieldID=["f1", "f1", "f1"],
               fieldID_cytoplasm=["f1", "f2", "f3"]), SUFFIX)
    printed = capsys.readouterr().out
    assert "2 of 3" in printed
    assert "prcfo" in printed
    assert "b" in printed and "c" in printed


def test_raise_stops_the_analysis_instead():
    with pytest.raises(ColumnConflict) as raised:
        reconcile_duplicates(
            merged(plateID=["p1", "p1", "p1"],
                   plateID_cytoplasm=["p1", "p1", "p2"]),
            SUFFIX, on_conflict="raise")
    assert "plateID" in str(raised.value)


def test_an_unknown_policy_is_refused_before_anything_is_dropped():
    frame = merged(plateID=["p1"] * 3, plateID_cytoplasm=["p1"] * 3)
    with pytest.raises(MergeError):
        reconcile_duplicates(frame, SUFFIX, on_conflict="ignore")
    assert "plateID_cytoplasm" in frame.columns


# ---------------------------------------------------------------------------
# measurements that share a name are not conflicts
# ---------------------------------------------------------------------------

def test_two_different_measurements_under_one_name_are_both_kept(capsys):
    """The false positive that would make the warning worthless.

    A cell's area and its cytoplasm's area are different numbers by
    definition. If that reported as a conflict, every real conflict would
    arrive buried in one per measured column.
    """
    out = reconcile_duplicates(
        merged(area=[100.0, 200.0, 300.0],
               area_cytoplasm=[60.0, 120.0, 180.0]), SUFFIX)

    assert "area" in out.columns and "area_cytoplasm" in out.columns
    assert capsys.readouterr().out == ""


def test_a_measurement_that_happens_to_match_still_collapses():
    """Identical is identical: one copy is enough whatever the column is."""
    out = reconcile_duplicates(
        merged(solidity=[0.9, 0.9, 0.9], solidity_cytoplasm=[0.9, 0.9, 0.9]),
        SUFFIX)
    assert "solidity_cytoplasm" not in out.columns


def test_must_agree_holds_the_identity_and_not_the_measurements():
    for identity in ("plateID", "rowID", "columnID", "fieldID", "prcf",
                     "prcfo", "object_label", "cell_id"):
        assert identity in MUST_AGREE
    for measurement in ("area", "solidity", "perimeter", "mean_intensity"):
        assert measurement not in MUST_AGREE


# ---------------------------------------------------------------------------
# nothing to do
# ---------------------------------------------------------------------------

def test_a_frame_with_no_duplicates_comes_back_unchanged():
    frame = merged(plateID=["p1"] * 3, cell_area=[1.0, 2.0, 3.0])
    out = reconcile_duplicates(frame, SUFFIX)
    pd.testing.assert_frame_equal(out, frame)


def test_an_empty_suffix_is_a_no_op_rather_than_matching_everything():
    """`endswith('')` is True for every column in the frame."""
    frame = merged(plateID=["p1"] * 3)
    pd.testing.assert_frame_equal(reconcile_duplicates(frame, ""), frame)


def test_a_column_merely_ending_in_the_suffix_is_left_alone():
    """`count_cytoplasm` has no `count` beside it to be compared against."""
    frame = merged(count_cytoplasm=[1, 2, 3])
    out = reconcile_duplicates(frame, SUFFIX)
    assert "count_cytoplasm" in out.columns


# ---------------------------------------------------------------------------
# through the real reader
# ---------------------------------------------------------------------------

def test_read_and_join_tables_takes_the_policy():
    """The setting has to reach the join, not just exist on the helper."""
    import inspect

    from spacr.io import _read_and_join_tables

    signature = inspect.signature(_read_and_join_tables)
    assert "duplicate_column_policy" in signature.parameters
    assert signature.parameters["duplicate_column_policy"].default == "warn"
