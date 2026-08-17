"""`c1,c2` is two reference controls, not one with a comma in its name.

Reported 2026-08-17, from a real run that had just succeeded with
`batch_control_values = None` and failed the moment it was set:

    ValueError: Only 0 total reference-control row(s) matched ['c1,c2'];
    need at least 3.

The whole string had been wrapped into a ONE-ELEMENT list and matched against
well names, which of course matched nothing.

It is the widget underneath: `list_shape_for` gives a key a list editor only
when its declared type admits NOTHING BUT a list, and `batch_control_values`
admits `str, int, float, list, tuple, None`, so it gets a plain text box. The
same panel shows `filter_value` and `control_wells` directly above it holding
`['c1', 'c2', 'c3']` -- the same well names, correctly split -- which is why
typing a comma here is the natural thing to do.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.batch_correction import _as_controls


# --------------------------------------------------------------------------- #
#  The parse
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("given,expected", [
    ("c1,c2", ["c1", "c2"]),                 # exactly what was reported
    (["c1,c2"], ["c1", "c2"]),               # after a settings-CSV round trip
    (" c1 , c2 ", ["c1", "c2"]),             # typed with spaces
    ("c1", ["c1"]),                          # a single value still works
    (["c1", "c2"], ["c1", "c2"]),            # a real list is untouched
    (["c1", "c2,c3"], ["c1", "c2", "c3"]),   # mixed
    (None, []),
    ("", []),
])
def test_a_comma_separates_controls(given, expected):
    assert _as_controls(given) == expected


def test_a_number_is_not_split():
    """A gene id used as a control is a number, and numbers have no commas
    to split on -- but they must survive the string branch untouched."""
    assert _as_controls(233460) == [233460]
    assert _as_controls([233460, 239740]) == [233460, 239740]


# --------------------------------------------------------------------------- #
#  End to end: the run that failed now runs
# --------------------------------------------------------------------------- #

def _frame(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "pred": rng.normal(0, 1, n),
        "plateID": ["plate1"] * (n // 2) + ["plate2"] * (n - n // 2),
        "columnID": [f"c{1 + i % 6}" for i in range(n)],
    })


def test_the_reported_run_no_longer_fails():
    """`c1,c2` matched 0 rows and raised; it must now match the wells in
    columns c1 and c2 and correct against them."""
    from spacr.batch_correction import correct_batch_effects

    frame = _frame()
    corrected, report = correct_batch_effects(
        frame[["pred"]],
        frame["plateID"],
        method="control_center",
        control=frame["columnID"],
        control_values="c1,c2",
        min_samples=3,
    )

    assert report.controls == int(frame["columnID"].isin(["c1", "c2"]).sum())
    assert report.controls > 0
    assert len(corrected) == len(frame)


def test_a_control_name_that_matches_nothing_says_what_the_column_holds():
    """"matched nothing" with no sight of the column sends the user to the
    wrong place. The commonest cause is a name that is not one of the values
    the column actually has."""
    from spacr.batch_correction import correct_batch_effects

    frame = _frame()
    with pytest.raises(ValueError) as caught:
        correct_batch_effects(
            frame[["pred"]],
            frame["plateID"],
            method="control_center",
            control=frame["columnID"],
            control_values="zz9",
            min_samples=3,
        )

    message = str(caught.value)
    assert "zz9" in message
    assert "The column holds" in message
    assert "c1" in message


def test_the_column_listing_is_bounded():
    """A plate column has 24 values; a 384-well layout has more, and an error
    that prints hundreds is one nobody reads."""
    from spacr.batch_correction import correct_batch_effects

    rng = np.random.default_rng(1)
    n = 400
    frame = pd.DataFrame({
        "pred": rng.normal(0, 1, n),
        # TWO batches. With one, correction short-circuits as a no-op ("Only
        # 1 batch was present") and never reaches the control check at all --
        # so a single-plate fixture tests nothing here, which is what a first
        # version of this did.
        "plateID": ["plate1"] * (n // 2) + ["plate2"] * (n - n // 2),
        "columnID": [f"c{i}" for i in range(n)]})

    with pytest.raises(ValueError) as caught:
        correct_batch_effects(
            frame[["pred"]], frame["plateID"],
            method="control_center", control=frame["columnID"],
            control_values="nope", min_samples=3)

    assert "and more" in str(caught.value)
