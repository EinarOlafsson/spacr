"""`ml_analysis` crashed inside sklearn when no control matched.

When neither `positive_control` nor `negative_control` matches anything in
`location_column`, both subsets are empty, so the combined frame is empty and
the failure surfaces three frames down::

    ValueError: With n_samples=0, test_size=0.2 and train_size=None, the
    resulting train set will be empty. Adjust any of the aforementioned
    parameters.

That names neither the setting that is wrong nor the value it should have
had. It was auto-filed to the spaCR tracker TEN TIMES in one day -- issues
#79, #80, #81, #84 through #90, all fingerprint 500e6c -- from a real run.

The defaults are `location_column='columnID'`, `positive_control='c2'`,
`negative_control='c1'`. A plate whose columns are named '1' and '2', or
whose controls live in a different column, matches nothing.

There is a `verbose` branch that prints "samples: 0", but verbose is False on
every shipped path.
"""

import numpy as np
import pandas as pd
import pytest

from spacr.ml import ml_analysis


def plate(column_values, n=10):
    rng = np.random.default_rng(0)
    rows = []
    for i, value in enumerate(column_values):
        for j in range(n):
            rows.append({
                "columnID": value,
                "prcfo": f"p1_r1_{value}_f1_o{i}{j}",
                "feat_a": float(rng.normal()),
                "feat_b": float(rng.normal()),
            })
    return pd.DataFrame(rows)


def test_no_control_match_names_the_column_and_its_values():
    """The message has to say what to change, not that something is empty."""
    with pytest.raises(ValueError) as excinfo:
        ml_analysis(plate(["1", "2"]), verbose=False)

    message = str(excinfo.value)
    assert "columnID" in message, "the column is not named"
    assert "'1'" in message and "'2'" in message, (
        "the values actually present are not shown, so the user cannot see "
        "what to set the controls to")
    assert "c1" in message and "c2" in message, (
        "the values looked for are not shown")
    assert "n_samples=0" not in message, "the sklearn error still leaks out"


def test_only_the_missing_control_is_named():
    """Half a match is the more confusing case: one control is fine."""
    with pytest.raises(ValueError) as excinfo:
        ml_analysis(plate(["c1", "9"]), verbose=False)

    # The DIAGNOSIS line only. The advice line below it legitimately names
    # both settings ("set positive_control and negative_control to values
    # that appear there"), which is fine -- what must not happen is the
    # diagnosis blaming a control that matched.
    diagnosis = str(excinfo.value).split("\n")[0]
    assert "positive_control='c2'" in diagnosis
    assert "negative_control" not in diagnosis, (
        "the negative control matched, so naming it in the diagnosis sends "
        "the user to the wrong setting")


def test_a_plate_whose_controls_match_still_runs_past_the_guard():
    """The guard must not fire on a workable plate."""
    frame = plate(["c1", "c2"], n=40)
    try:
        ml_analysis(frame, verbose=False)
    except ValueError as exc:
        assert "nothing to train on" not in str(exc), (
            "the guard fired on a plate whose controls do match")
    except Exception:
        pass  # any later failure is not this guard's business


def test_the_guard_reports_a_large_column_without_dumping_it():
    """A 384-well plate has many values; the message stays readable."""
    with pytest.raises(ValueError) as excinfo:
        ml_analysis(plate([str(i) for i in range(1, 40)], n=2), verbose=False)

    message = str(excinfo.value)
    assert "distinct values" in message, "the value list was not summarised"
    assert len(message.split("\n")[1]) < 220, "the value line is unreadable"


def test_a_duplicated_location_column_gets_its_own_diagnosis():
    """`df[name]` is a DataFrame when the name appears twice, so every
    matching strategy fails against it.

    Named separately because the fix is to the TABLE, not to the control
    values: no amount of correcting positive_control helps, so sending the
    user to those settings would waste their time.
    """
    frame = plate(["c1", "c2"])
    frame["dup"] = frame["columnID"]
    frame.columns = ["columnID" if c == "dup" else c for c in frame.columns]

    with pytest.raises(ValueError) as excinfo:
        ml_analysis(frame, verbose=False)

    message = str(excinfo.value)
    assert "2 columns named 'columnID'" in message
    assert "positive_control" not in message, (
        "this sends the user to a setting that cannot fix it")
