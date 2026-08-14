"""Annotation mode overwrote location_column and the user could not get out.

Issues #91, #92 and #93 -- one user, one afternoon, one plate. They are not
three bugs; they are one defect walked through in sequence, and the third is
the user following the advice the second one gave.

STEP 1. `generate_ml_scores` set `settings['location_column'] =
settings['annotation_column']` on the annotation path, but left
`positive_control` and `negative_control` as the plate column names they
default to. The run then looked for 'c1' inside a column holding 1.0 and 2.0
and found nothing:

    no rows matched negative_control='c1' and positive_control='c2' in
    column 'cc2d1a_recruitment' ... contains: '1.0', '2.0', 'nan'

STEP 2. That message says "set positive_control and negative_control to
values that appear there". The user did, and switched dataset_mode back to
'metadata'. It still failed -- now with a pandas KeyError -- because the
assignment in step 1 had MUTATED THE SETTINGS DICT and nothing put it back.
`location_column` still named the annotation column, which is not in the
measurement frame.

A user who tried annotation mode once could not return to metadata mode by
changing the mode. They had to know an invisible write had happened.
"""

import inspect

import pandas as pd
import pytest

from spacr.ml import _resolve_controls, generate_ml_scores, ml_analysis


def matcher(series, control):
    """The same shape as `ml_analysis`'s nested `_match_control_values`."""
    values = control if isinstance(control, (list, tuple, set)) else [control]
    text = series.astype(str).str.strip()
    return text.isin([str(v).strip() for v in values])


# ---------------------------------------------------------------------------
# the settings dict is not written to
# ---------------------------------------------------------------------------

def test_the_annotation_path_no_longer_writes_location_column():
    """The mutation is gone from the source.

    Asserted against the source because reaching that line needs a whole
    database; the assignment is one line and its absence is the fix.
    """
    # COMMENTS STRIPPED FIRST. The fix's own comment quotes the line it
    # removed, which is the right thing for a reader and would make a naive
    # substring check pass forever.
    source = inspect.getsource(generate_ml_scores)
    code = "\n".join(line.split("#", 1)[0] for line in source.splitlines())

    assert "settings['location_column'] =" not in code, (
        "the caller's settings are being mutated again")
    assert "_label_column" in code


def test_the_training_column_is_derived_not_taken_from_the_mutated_setting():
    source = inspect.getsource(generate_ml_scores)
    code = "\n".join(line.split("#", 1)[0] for line in source.splitlines())
    assert "_label_column or settings['location_column']" in code


# ---------------------------------------------------------------------------
# the classes come from the annotation column
# ---------------------------------------------------------------------------

def test_well_names_against_an_annotation_column_use_the_columns_classes():
    """Issues #91 and #92, exactly.

    'c1' and 'c2' are plate columns; the annotation column holds 1.0 and
    2.0. Neither control appears, and the column says unambiguously what its
    two classes are.
    """
    df = pd.DataFrame({"cc2d1a_recruitment": [1.0, 2.0, 1.0, 2.0]})
    negative, positive, derived = _resolve_controls(
        df, "cc2d1a_recruitment", "c1", "c2", matcher)

    assert derived is True
    assert (negative, positive) == (1.0, 2.0)


def test_it_says_what_it_derived_and_why(capsys):
    """Deriving silently would be its own version of this bug."""
    df = pd.DataFrame({"ann": [1.0, 2.0]})
    _resolve_controls(df, "ann", "c1", "c2", matcher)

    printed = capsys.readouterr().out
    assert "two classes" in printed
    assert "1.0" in printed and "2.0" in printed
    assert "c1" in printed and "c2" in printed


def test_controls_that_do_match_are_never_touched():
    """An explicit choice is always honoured."""
    df = pd.DataFrame({"columnID": ["c1", "c2", "c1"]})
    negative, positive, derived = _resolve_controls(
        df, "columnID", "c1", "c2", matcher)

    assert derived is False
    assert (negative, positive) == ("c1", "c2")


def test_a_deliberate_two_of_five_subset_is_honoured():
    """Five plate columns, two named: that is a choice, not a mistake."""
    df = pd.DataFrame({"columnID": ["c1", "c2", "c3", "c4", "c5"]})
    negative, positive, derived = _resolve_controls(
        df, "columnID", "c1", "c3", matcher)

    assert derived is False
    assert (negative, positive) == ("c1", "c3")


def test_three_classes_are_not_guessed_at():
    """With more than two the user has to say which two, and the refusal
    downstream lists what is there."""
    df = pd.DataFrame({"ann": [1.0, 2.0, 3.0]})
    negative, positive, derived = _resolve_controls(
        df, "ann", "c1", "c2", matcher)

    assert derived is False
    assert (negative, positive) == ("c1", "c2")


def test_one_class_is_not_guessed_at():
    df = pd.DataFrame({"ann": [1.0, 1.0]})
    _, _, derived = _resolve_controls(df, "ann", "c1", "c2", matcher)
    assert derived is False


def test_a_missing_column_is_left_for_the_caller_to_report():
    df = pd.DataFrame({"columnID": ["c1", "c2"]})
    negative, positive, derived = _resolve_controls(
        df, "not_here", "c1", "c2", matcher)
    assert (negative, positive, derived) == ("c1", "c2", False)


def test_a_duplicated_column_is_left_for_the_caller_to_report():
    """`df[name]` is a DataFrame then, and every matcher fails against it."""
    df = pd.DataFrame([[1.0, 2.0]], columns=["ann", "ann"])
    _, _, derived = _resolve_controls(df, "ann", "c1", "c2", matcher)
    assert derived is False


# ---------------------------------------------------------------------------
# the poisoned state is named, not raised through
# ---------------------------------------------------------------------------

def test_a_location_column_absent_from_the_table_names_the_cause():
    """Issue #93 raised a pandas KeyError three frames down that said only
    "None of [Index([...])] are in the [columns]"."""
    df = pd.DataFrame({"columnID": ["c1", "c2"], "feature": [1.0, 2.0],
                       "prcfo": ["a", "b"]})

    with pytest.raises(ValueError) as raised:
        ml_analysis(df, 1, "cc2d1a_recruitment", "c2", "c1")

    message = str(raised.value)
    assert "cc2d1a_recruitment" in message
    assert "not a column of the measurement table" in message
    # the columns that ARE there, so a typo is self-correcting
    assert "columnID" in message
    # and the sentence that would have saved the reporter an afternoon
    assert "annotation mode" in message
    assert "location_column back" in message


def test_it_is_a_value_error_not_a_key_error():
    """A KeyError from inside pandas is not a message to a user."""
    df = pd.DataFrame({"columnID": ["c1"], "prcfo": ["a"]})
    with pytest.raises(ValueError):
        ml_analysis(df, 1, "missing_column", "c2", "c1")
