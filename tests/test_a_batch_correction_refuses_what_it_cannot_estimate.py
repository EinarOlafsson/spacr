"""Batch correction refuses the inputs it cannot estimate a batch from.

Every refusal here is a number that would otherwise be produced and
believed: a plate label that is missing, a column that is text, a batch
with no reference controls in it. The messages name the column, because
the commonest cause is a name that does not appear in the data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.batch_correction import (METHODS, _prior_width,
                                    correct_batch_effects,
                                    correct_from_metadata)


@pytest.fixture()
def frame():
    rows = 12
    rng = np.random.default_rng(0)
    features = pd.DataFrame({
        "area": rng.normal(100.0, 5.0, rows),
        "intensity": rng.normal(20.0, 2.0, rows),
    })
    batch = pd.Series(["p1"] * 6 + ["p2"] * 6, name="plateID")
    return features, batch


def test_a_single_estimate_has_no_spread_to_learn_a_prior_from():
    assert _prior_width(np.array([3.0])) == 0.0, (
        "one feature gives an undefined variance, not a NaN prior")
    assert _prior_width(np.array([3.0, 3.0])) == 0.0, (
        "estimates that coincide leave no shrinkage room")


def _a_row_with_no_plate_label(features, batch):
    batch = batch.copy()
    batch.iloc[3] = np.nan
    return features, batch, {"method": "center", "batch_column": "plateID"}


def _a_text_column_among_the_features(features, batch):
    features = features.copy()
    features["well"] = ["A1"] * len(features)
    return features, batch, {"method": "center"}


def _a_control_name_matching_nothing(features, batch):
    control = pd.Series(["negative"] * len(features))
    return features, batch, {"method": "control_center", "control": control,
                             "control_values": "neg", "min_samples": 2}


def _an_unknown_method(features, batch):
    return features, batch, {"method": "quantile"}


def _an_unknown_missing_control_policy(features, batch):
    return features, batch, {"method": "center", "missing_control": "ignore"}


@pytest.mark.parametrize("setup, fragments", [
    pytest.param(_a_row_with_no_plate_label,
                 ["plateID is missing for 1 feature row(s)"],
                 id="row_with_no_plate_label_is_not_guessed_at"),
    pytest.param(_a_text_column_among_the_features,
                 ["non-numeric values in: ['well']"],
                 id="text_column_in_the_feature_matrix_is_named"),
    pytest.param(_a_control_name_matching_nothing,
                 ["The column holds ['negative']"],
                 id="control_name_matching_nothing_reports_the_column"),
    pytest.param(_an_unknown_method, ["quantile", str(METHODS)],
                 id="unknown_method_names_the_ones_that_exist"),
    pytest.param(_an_unknown_missing_control_policy,
                 ["must be 'error' or 'skip'"],
                 id="unknown_missing_control_policy"),
])
def test_a_correction_that_cannot_be_estimated_is_refused_with_the_reason(
        frame, setup, fragments):
    features, batch, kwargs = setup(*frame)

    with pytest.raises(ValueError) as caught:
        correct_batch_effects(features, batch, **kwargs)

    for fragment in fragments:
        assert fragment in str(caught.value)


def _controls_thin_in_the_second_batch(rows):
    """Three reference controls in the first batch, one in the second."""
    half = rows // 2
    return pd.Series(["neg"] * 3 + ["treat"] * (half - 3)
                     + ["neg"] + ["treat"] * (rows - half - 1))


def test_a_batch_with_no_reference_controls_stops_the_run_by_default(frame):
    features, batch = frame
    control = _controls_thin_in_the_second_batch(len(features))

    with pytest.raises(ValueError) as caught:
        correct_batch_effects(features, batch, method="control_center",
                              control=control, control_values="neg",
                              min_samples=2)

    assert "No usable reference controls" in str(caught.value)
    assert "['p2']" in str(caught.value), "the batch that lacks them is named"


def test_skip_leaves_the_uncorrectable_batch_alone_and_says_so(frame):
    features, batch = frame
    control = _controls_thin_in_the_second_batch(len(features))

    corrected, report = correct_batch_effects(
        features, batch, method="control_center", control=control,
        control_values="neg", min_samples=2, missing_control="skip")

    assert any("were unchanged" in warning for warning in report.warnings)
    pd.testing.assert_frame_equal(corrected.loc[batch == "p2"],
                                  features.loc[batch == "p2"])
    assert not corrected.loc[batch == "p1"].equals(
        features.loc[batch == "p1"]), "the batch that had controls was shifted"
    assert report.controls == 4, "every matched control row is counted"


def test_metadata_without_the_batch_column_names_the_column_it_wanted(frame):
    features, _batch = frame
    metadata = pd.DataFrame({"wellID": ["A1"] * len(features)})

    with pytest.raises(ValueError) as caught:
        correct_from_metadata(features, metadata,
                              batch_correction="center",
                              batch_column="plateID")

    assert "batch_column='plateID'" in str(caught.value)
    assert "absent" in str(caught.value)
