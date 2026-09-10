"""What a cross-fit refuses before it trains, and why each refusal is worth it.

The classifier is fitted on BAG labels -- a well is a target or it is not -- and
its probabilities decide which cells a user then looks at. Every guard here
stops a model that would look trained and mean nothing.

The leakage guard is the one with real teeth: a feature carrying a well id
lets the model memorise which wells were targets, and its held-out score would
then be near perfect on a screen with no signal at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _frame(n_wells=10, n_per_well=8, n_plates=1, n_targets=5, seed=0):
    """A design that clears the four-target/four-control floor.

    That floor is checked before anything here, so a smaller fixture never
    reaches the guards these tests are about.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for plate in range(n_plates):
        for well in range(n_wells):
            for _ in range(n_per_well):
                rows.append({
                    "plateID": f"p{plate + 1}",
                    "rowID": "r1",
                    "columnID": f"c{well + 1}",
                    "area": rng.normal(100.0, 10.0),
                    "perimeter": rng.normal(40.0, 4.0),
                    "target_well": well < n_targets,
                })
    return pd.DataFrame(rows)


def test_too_few_independent_wells_are_refused_with_both_counts():
    """The floor that guards everything below it, and names what it found.

    Four targets and four controls is the minimum an honest held-out estimate
    needs. The message carries BOTH counts, because "you need four of each"
    without saying which side is short leaves the user counting wells.
    """
    from spacr.hit_attribution import (InsufficientDesignError,
                                       crossfit_candidate_probabilities)

    frame = _frame(n_wells=6, n_targets=1)

    with pytest.raises(InsufficientDesignError) as excinfo:
        crossfit_candidate_probabilities(
            frame, feature_columns=["area", "perimeter"])

    message = str(excinfo.value)
    assert "four independent target" in message
    assert "have 1 and 5" in message


def test_a_well_whose_target_status_disagrees_is_refused():
    """One well cannot be both a target and a control.

    It means two rows of the same well were labelled differently, which makes
    every bag label downstream ambiguous -- so it is refused rather than
    resolved by majority.
    """
    from spacr.hit_attribution import (HitAttributionError,
                                       crossfit_candidate_probabilities)

    frame = _frame()
    frame.loc[frame.index[0], "target_well"] = not frame.loc[frame.index[0],
                                                             "target_well"]

    with pytest.raises(HitAttributionError, match="disagrees within a well"):
        crossfit_candidate_probabilities(
            frame, feature_columns=["area", "perimeter"])


def test_a_feature_that_carries_an_identifier_is_refused():
    """The leakage guard, naming the offending columns.

    A model given the well id memorises which wells were targets and scores
    near perfectly on a screen with no signal. Naming the columns is what
    lets the user drop the right ones rather than guessing.
    """
    from spacr.hit_attribution import (HitAttributionError,
                                       crossfit_candidate_probabilities)

    frame = _frame()
    frame["columnID_numeric"] = 1.0

    with pytest.raises(HitAttributionError) as excinfo:
        crossfit_candidate_probabilities(
            frame, feature_columns=["area", "columnID_numeric"])

    assert "leak" in str(excinfo.value)
    assert "columnID_numeric" in str(excinfo.value)


def test_a_feature_that_is_entirely_missing_is_refused_by_name():
    """The all-NaN guard.

    Median-filling an all-NaN column produces a constant, and a constant
    feature contributes nothing while looking like a feature -- so the model
    would be trained on fewer measurements than the report says it used.
    """
    from spacr.hit_attribution import (HitAttributionError,
                                       crossfit_candidate_probabilities)

    frame = _frame()
    frame["never_measured"] = np.nan

    with pytest.raises(HitAttributionError) as excinfo:
        crossfit_candidate_probabilities(
            frame, feature_columns=["area", "never_measured"])

    assert "entirely missing" in str(excinfo.value)
    assert "never_measured" in str(excinfo.value)


def test_a_workable_design_returns_a_probability_per_row():
    """The path all the guards protect, so they are visibly the exceptions."""
    from spacr.hit_attribution import crossfit_candidate_probabilities

    frame = _frame(n_wells=10, n_per_well=10)

    out, features, level, _notes = crossfit_candidate_probabilities(
        frame, feature_columns=["area", "perimeter"], n_splits=3)

    assert len(out) == len(frame)
    assert set(features) == {"area", "perimeter"}
    assert level in ("well", "plate")


def test_a_screen_on_one_plate_splits_by_well_not_plate():
    """The ``prefer_plate and plate_count >= 4`` condition.

    Splitting by plate needs four plates to hold one out meaningfully. A
    single-plate screen must fall back to wells rather than producing one
    group and hitting the refusal above.
    """
    from spacr.hit_attribution import crossfit_candidate_probabilities

    frame = _frame(n_wells=10, n_per_well=10, n_plates=1)

    _out, _features, level, _notes = crossfit_candidate_probabilities(
        frame, feature_columns=["area", "perimeter"], n_splits=3)

    assert level == "well"
