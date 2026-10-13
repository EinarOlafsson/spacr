"""Reliability bins, and the label the calibration table refuses to score.

The docstring records the bug the refusal fixed: a label outside the
probability columns matches no class, so every ``observed_frequency`` read 0.0
and the curve rendered as CATASTROPHICALLY MISCALIBRATED rather than as an
error. A reader would have concluded the model was broken when the labelling
was.

That is the shape worth testing: not a crash, but a plausible wrong picture.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _probabilities(n=40, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    raw = rng.random((n, n_classes))
    return raw / raw.sum(axis=1, keepdims=True)


def test_a_well_formed_family_bins_every_class():
    """The baseline the refusals below are measured against."""
    from spacr.classifier_evaluation import calibration_table

    probs = _probabilities()
    y = np.arange(len(probs)) % probs.shape[1]

    table = calibration_table(y, probs)

    assert isinstance(table, pd.DataFrame)
    assert not table.empty


@pytest.mark.parametrize("stray", [3, 7, -1])
def test_a_label_outside_the_probability_columns_is_refused(stray):
    """The raise, and the message names the offending indices.

    Naming them is the whole value: with three columns and a stray 7, the user
    needs to know it is the LABELS that are wrong, not the model.
    """
    from spacr.classifier_evaluation import calibration_table

    probs = _probabilities(n_classes=3)
    y = np.arange(len(probs)) % 3
    y[0] = stray

    with pytest.raises(ValueError) as excinfo:
        calibration_table(y, probs)

    message = str(excinfo.value)
    assert str(stray) in message
    assert "3 probability columns" in message


def test_class_names_must_match_the_probability_columns():
    """The second refusal, which protects the legend rather than the curve.

    Three columns labelled with two names would draw two curves and silently
    drop the third class.
    """
    from spacr.classifier_evaluation import calibration_table

    probs = _probabilities(n_classes=3)
    y = np.arange(len(probs)) % 3

    with pytest.raises(ValueError, match="equal length"):
        calibration_table(y, probs, classes=["a", "b"])


def test_mismatched_lengths_are_refused_before_anything_else():
    """The first guard: a row count that does not line up is not binnable."""
    from spacr.classifier_evaluation import calibration_table

    probs = _probabilities(n=40, n_classes=3)

    with pytest.raises(ValueError, match="equal length"):
        calibration_table(np.zeros(39, dtype=int), probs)


def test_an_empty_family_is_not_refused_for_a_stray_label():
    """The ``if len(y) and n_classes:`` guard around the stray check.

    Nothing to check means nothing to refuse. An empty result is a legitimate
    outcome -- a fold with no held-out rows -- and raising there would turn it
    into an error the caller cannot act on.
    """
    from spacr.classifier_evaluation import calibration_table

    table = calibration_table(np.zeros(0, dtype=int),
                              np.zeros((0, 3), dtype=float))

    assert isinstance(table, pd.DataFrame)


def test_named_classes_reach_the_table():
    """The names the caller supplied are what the legend will read."""
    from spacr.classifier_evaluation import calibration_table

    probs = _probabilities(n_classes=2)
    y = np.arange(len(probs)) % 2

    table = calibration_table(y, probs, classes=["negative", "positive"])

    joined = " ".join(str(v) for v in table.to_numpy().ravel())
    assert "negative" in joined and "positive" in joined
