"""Translating batch settings into call keywords, and the no-op correction.

``correction_kwargs`` emits exactly six keys and deliberately leaves the
combat-only ones out, so the result is safe to splat into signatures that do
not accept them. Its uncovered arc is the ordinary one: a control column and
values the user actually chose, which must NOT be replaced by the caller's
defaults.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_the_six_keys_are_emitted_and_no_more():
    """The contract the docstring states, which callers splat on.

    A seventh key would be a TypeError at a call site that does not accept it,
    which is why the combat-only ones are excluded.
    """
    from spacr.batch_correction import correction_kwargs

    out = correction_kwargs({})

    assert set(out) == {
        "batch_correction", "batch_column", "batch_control_column",
        "batch_control_values", "batch_min_samples", "batch_missing_control",
    }


@pytest.mark.parametrize("settings, expected_column, expected_values", [
    pytest.param(
        {}, "condition", ["nc"],
        id="test_defaults_fill_a_control_column_and_values_the_user_left_unset",
    ),
    pytest.param(
        {"batch_control_column": "well_type",
         "batch_control_values": ["untreated", "dmso"]},
        "well_type", ["untreated", "dmso"],
        id="test_a_chosen_control_column_and_values_survive_the_defaults",
    ),
])
def test_control_choices_keep_their_original_defaults_and_overrides(
        settings, expected_column, expected_values):
    """Keep each original control choice paired with the same column and ordered values."""
    from spacr.batch_correction import correction_kwargs

    out = correction_kwargs(settings, default_control_column="condition",
                            default_control_values=["nc"])
    assert out["batch_control_column"] == expected_column
    assert out["batch_control_values"] == expected_values


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_a_blank_control_value_falls_back_to_the_default(blank):
    """The other half of the same guard: a cleared box is not a choice.

    A settings CSV round trip writes "" for a cleared field, and treating that
    as "correct against the empty set" would leave every batch uncorrected
    while reporting that it was.
    """
    from spacr.batch_correction import correction_kwargs

    out = correction_kwargs({"batch_control_values": blank},
                            default_control_values=["nc"])

    assert out["batch_control_values"] == ["nc"]


# ---------------------------------------------------------------------------
# correct_batch_effects — the no-op, with and without a warning
# ---------------------------------------------------------------------------

def _features_and_batch(n_batches):
    """``(features, batch)`` -- the two arguments the function actually takes."""
    rng = np.random.default_rng(0)
    rows, batches = [], []
    for index in range(n_batches):
        for _ in range(6):
            rows.append({"feature_a": rng.normal(), "feature_b": rng.normal()})
            batches.append(f"plate{index + 1}")
    return pd.DataFrame(rows), pd.Series(batches, name="plateID")


def test_correction_turned_off_is_a_no_op_and_says_nothing():
    """Arc 702 -> 706: ``method == "none"``, so no warning is appended.

    "Correction was a no-op" is true and useless when the user asked for no
    correction. The warning exists for the OTHER no-op -- a method was chosen
    and there was only one batch to correct across -- and conflating them
    would train the reader to skip both.
    """
    from spacr.batch_correction import correct_batch_effects

    features, batch = _features_and_batch(3)
    out, report = correct_batch_effects(features, batch, method="none")

    assert out.shape == features.shape
    assert not [w for w in report.warnings if "no-op" in w]


def test_a_single_batch_is_a_no_op_that_says_so():
    """The taken side: a method WAS chosen and could not do anything.

    Silence here is the dangerous one -- the user believes their data was
    corrected across plates when there was only one plate.
    """
    from spacr.batch_correction import correct_batch_effects

    features, batch = _features_and_batch(1)
    _out, report = correct_batch_effects(features, batch,
                                         method="zscore")

    assert [w for w in report.warnings if "no-op" in w]
