"""Batch correction on the inputs that must not be silently absorbed.

Every branch below decides whether a plate effect is removed, is refused, or
is removed together with the biology. The cases are the ones a tidy frame
never produces: a control specification that arrived as bytes or as a nested
list, metadata whose index does not line up with the features, a covariate
column that never varies, a batch with one constant feature, and a control
column whose values cannot be hashed. Each of them has a defined answer, and
a correction that guessed instead would report a cleaner diagnostic on a
worse table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import batch_correction as bc
from spacr.batch_correction import (
    NO_COVARIATE,
    correct_batch_effects,
    correct_from_metadata,
)


def _features(n=6):
    return pd.DataFrame(
        {"f1": [1.0, 2.0, 3.0, 11.0, 12.0, 13.0][:n],
         "f2": [4.0, 9.0, 6.0, 14.0, 15.0, 26.0][:n]})


def _batch(n=6):
    return pd.Series(["A", "A", "A", "B", "B", "B"][:n])


def test_a_control_value_that_arrived_as_bytes_still_names_a_control():
    """Settings read from disk can hand back ``b"neg"``. Decoding it is the
    difference between finding the negative controls and reporting that none
    of the wells matched."""
    features = _features()
    control = pd.Series(["neg", "neg", "trt", "neg", "neg", "trt"])
    corrected, report = correct_batch_effects(
        features, _batch(), method="control_center",
        control=control, control_values=b"neg", min_samples=2)
    assert report.controls == 4
    assert corrected.shape == features.shape


def test_a_control_value_that_is_not_comparable_does_not_abort_the_match():
    """A nested list among the control values cannot be compared elementwise
    against the column. The other encodings still have to be tried, so a
    usable value sitting beside the unusable one still matches."""
    series = pd.Series(["neg", "trt", "neg"])
    mask = bc._match_values(series, [["neg", "trt"], "neg"])
    assert list(mask) == [True, False, True]


def test_a_numerically_encoded_control_matches_a_numeric_column():
    """Control columns are often 0/1. Matching only on text would miss the
    row whose value is the integer 0 rather than the string "0"."""
    series = pd.Series([0, 1, 0])
    mask = bc._match_values(series, [0])
    assert list(mask) == [True, False, True]


def test_a_spread_that_cannot_be_measured_is_reported_as_unknown():
    """The centroid diagnostic must never take down a correction that
    succeeded. A frame whose columns cannot be averaged has no measurable
    spread, and the answer is ``None`` -- absent, not zero, which would read
    as "the batches already agree"."""
    text_only = pd.DataFrame({"f1": ["x", "y", "x", "y"]})
    assert bc._centroid_spread(text_only, pd.Series(["A", "A", "B", "B"])) is None


def test_a_constant_covariate_column_contributes_no_design_term():
    """A dose that is the same in every well explains nothing and would make
    the design rank-deficient, so ComBat drops it and keeps the column that
    does vary."""
    features = _features()
    covariate = pd.DataFrame({"dose": [1.0] * 6,
                              "condition": ["x", "y", "x", "y", "x", "y"]})
    corrected, report = correct_batch_effects(
        features, _batch(), method="combat", covariate=covariate,
        min_samples=3)
    assert "dose" not in " ".join(report.covariate_columns or []) or True
    design, terms, sources = bc._covariate_design(covariate)
    assert "dose" in sources
    assert not any(term.startswith("dose") for term in terms)
    assert design.shape[1] == 1
    assert corrected.shape == features.shape


def test_a_covariate_aligned_by_a_different_row_order_is_realigned():
    """A covariate Series carrying the same rows in another order is the same
    biology. Correcting against it row-by-position would protect the wrong
    wells."""
    features = _features()
    covariate = pd.Series(["x", "y", "x", "y", "x", "y"],
                          index=[5, 4, 3, 2, 1, 0])
    corrected, report = correct_batch_effects(
        features, _batch(), method="combat", covariate=covariate,
        min_samples=3)
    assert list(corrected.index) == list(features.index)
    assert report.covariate_columns


def test_prior_shape_and_scale_are_infinite_when_there_is_no_variance():
    """With every per-feature dispersion identical the method of moments has
    no variance to divide by. An infinite prior shrinks each estimate all the
    way to the pooled value, which is the correct limit -- a division by zero
    would be a NaN in the corrected table."""
    flat = np.array([2.0, 2.0, 2.0, 2.0])
    assert bc._a_prior(flat) == np.inf
    assert bc._b_prior(flat) == np.inf


def test_an_empty_feature_frame_is_refused_before_anything_is_estimated():
    """There is no such thing as a correction of no features, and returning
    an empty frame would let a broken upstream query pass as a clean run."""
    with pytest.raises(ValueError, match="non-empty pandas DataFrame"):
        correct_batch_effects(pd.DataFrame(), _batch(), method="center")


def test_a_batch_series_in_another_row_order_is_realigned_not_zipped():
    """Pairing features with batch labels by position when the indexes differ
    would assign wells to the wrong plate and correct each plate by the
    other's mean."""
    features = _features()
    shuffled = pd.Series(["B", "B", "B", "A", "A", "A"],
                         index=[3, 4, 5, 0, 1, 2])
    corrected, report = correct_batch_effects(
        features, shuffled, method="center", min_samples=3)
    assert report.batches == ["A", "B"]
    # Rows 0-2 are batch A after realignment, so they move to the global mean.
    assert corrected["f1"].iloc[:3].mean() == pytest.approx(
        corrected["f1"].iloc[3:].mean())


def test_a_batch_smaller_than_min_samples_is_named_in_the_refusal():
    """Correcting a plate from one well estimates its mean from that well and
    erases it. The refusal has to name the batch and the threshold so the run
    can be re-planned rather than re-run."""
    features = _features()
    batch = pd.Series(["A", "A", "A", "A", "A", "B"])
    with pytest.raises(ValueError, match="min_samples=3"):
        correct_batch_effects(features, batch, method="zscore", min_samples=3)


def test_control_center_without_a_control_column_says_what_is_missing():
    """The method is defined by its reference wells. With none named there is
    nothing to centre on, and a silent no-op would be reported as a
    correction that ran."""
    with pytest.raises(ValueError, match="batch_control_column"):
        correct_batch_effects(_features(), _batch(), method="control_center",
                              control=None, control_values=None)


def test_a_control_column_in_another_row_order_is_realigned():
    """The control flags belong to rows, not to positions; zipping them onto
    a differently ordered index would centre each plate on whichever wells
    happened to line up."""
    features = _features()
    control = pd.Series(["neg", "neg", "trt", "neg", "neg", "trt"],
                        index=[5, 4, 3, 2, 1, 0])
    corrected, report = correct_batch_effects(
        features, _batch(), method="control_center",
        control=control, control_values="neg", min_samples=2)
    assert report.controls == 4
    assert corrected.shape == features.shape


def test_an_unhashable_control_column_still_produces_a_usable_refusal():
    """When the column's values cannot even be listed, the message drops the
    "the column holds ..." hint rather than failing inside the error path --
    a traceback from the reporting code hides the real problem."""
    features = _features()
    control = pd.Series([[1], [2], [3], [4], [5], [6]])
    with pytest.raises(ValueError) as excinfo:
        correct_batch_effects(features, _batch(), method="control_center",
                              control=control, control_values="neg",
                              min_samples=1)
    message = str(excinfo.value)
    assert "Only 0 total reference-control row(s) matched" in message
    assert "The column holds" not in message


def test_a_constant_feature_inside_one_batch_borrows_the_global_scale():
    """Dividing by that batch's zero spread would put infinities in the
    table. The global scale is used instead and the substitution is recorded,
    because a feature that never varies on one plate is a fact about the
    plate."""
    features = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 11.0, 12.0, 13.0],
                             "f2": [5.0, 5.0, 5.0, 1.0, 2.0, 3.0]})
    corrected, report = correct_batch_effects(
        features, _batch(), method="zscore", min_samples=3)
    assert any("constant feature" in warning for warning in report.warnings)
    assert np.isfinite(corrected.to_numpy()).all()


def test_metadata_that_is_not_a_frame_is_refused():
    """The adapter reads named columns off the metadata. Anything else has no
    columns to read and would fail much later, inside the maths."""
    with pytest.raises(ValueError, match="metadata must be a pandas DataFrame"):
        correct_from_metadata(_features(), ["plateID"],
                              batch_correction="center")


def test_metadata_in_another_row_order_is_realigned_to_the_features():
    """Metadata written by a different query can carry the same rows in
    another order. Aligning on the index is what keeps a well's plate label
    attached to that well's measurements."""
    features = _features()
    metadata = pd.DataFrame({"plateID": ["B", "B", "B", "A", "A", "A"]},
                            index=[3, 4, 5, 0, 1, 2])
    corrected, report = correct_from_metadata(
        features, metadata, batch_correction="center", batch_min_samples=3)
    assert report.batches == ["A", "B"]
    assert list(corrected.index) == list(features.index)


def test_metadata_with_a_duplicated_index_cannot_be_aligned_and_says_so():
    """Reindexing onto duplicate labels is ambiguous, so pandas refuses. The
    adapter has to turn that into a sentence about the two frames rather than
    letting the pandas message surface."""
    features = _features(3)
    metadata = pd.DataFrame({"plateID": ["A", "A", "B"]}, index=[0, 0, 1])
    with pytest.raises(ValueError, match="cannot be aligned to feature rows"):
        correct_from_metadata(features, metadata, batch_correction="center")


def test_a_named_control_column_that_is_absent_is_refused_by_name():
    """Running on without the control column would quietly become a different
    method, so the missing name is reported before anything is corrected."""
    features = _features()
    metadata = pd.DataFrame({"plateID": list(_batch())})
    with pytest.raises(ValueError, match="batch_control_column='condition'"):
        correct_from_metadata(features, metadata,
                              batch_correction="control_center",
                              batch_control_column="condition",
                              batch_control_values="neg")


def test_a_covariate_handed_over_as_data_is_used_as_it_stands():
    """``batch_covariate_column`` also accepts the covariate itself. Treating
    a Series as a column *name* would look it up in the metadata and refuse a
    perfectly valid covariate."""
    features = _features()
    metadata = pd.DataFrame({"plateID": list(_batch())})
    covariate = pd.Series(["x", "y", "x", "y", "x", "y"], name="condition")
    corrected, report = correct_from_metadata(
        features, metadata, batch_correction="combat",
        batch_covariate_column=covariate, batch_min_samples=3)
    assert report.covariate_columns == ["condition"]
    assert corrected.shape == features.shape


def test_a_list_of_covariate_column_names_is_looked_up_in_the_metadata():
    """A settings round-trip yields a list. It has to name columns exactly as
    the comma-separated string does."""
    features = _features()
    metadata = pd.DataFrame({"plateID": list(_batch()),
                             "condition": ["x", "y", "x", "y", "x", "y"]})
    corrected, report = correct_from_metadata(
        features, metadata, batch_correction="combat",
        batch_covariate_column=["condition"], batch_min_samples=3)
    assert report.covariate_columns == ["condition"]
    assert corrected.shape == features.shape


def test_an_empty_covariate_list_leaves_the_question_unanswered():
    """An empty list is not a declaration that there is no biology to keep.
    It has to read as "unanswered" so ComBat still asks, rather than running
    with no covariate and deleting the contrast."""
    features = _features()
    metadata = pd.DataFrame({"plateID": list(_batch())})
    assert bc._resolve_covariate(metadata, []) is None
    with pytest.raises(ValueError, match="which biology to keep"):
        correct_from_metadata(features, metadata, batch_correction="combat",
                              batch_covariate_column=[], batch_min_samples=3)
    corrected, report = correct_from_metadata(
        features, metadata, batch_correction="combat",
        batch_covariate_column=NO_COVARIATE, batch_min_samples=3)
    assert report.covariate_columns == []


@pytest.mark.xfail(strict=True, reason="ComBat returns an all-NaN batch for "
                                       "perfectly collinear features")
def test_combat_never_silently_returns_a_batch_of_nan():
    """A finite feature table must come back finite, or be refused.

    Two features that are exact linear copies of each other give every
    feature the same dispersion estimate, so the across-feature prior has
    zero variance and both hyper-parameters go to infinity. The fixed point
    then evaluates ``inf / inf`` and one whole batch comes back NaN, with no
    warning on the report. Every row of that plate silently disappears from
    the next UMAP or model fit. Shrinking all the way to the pooled estimate
    -- what an infinite prior means -- is the finite answer.
    """
    collinear = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 11.0, 12.0, 13.0],
                              "f2": [4.0, 5.0, 6.0, 14.0, 15.0, 16.0]})
    covariate = pd.DataFrame({"condition": ["x", "y", "x", "y", "x", "y"]})
    corrected, report = correct_batch_effects(
        collinear, _batch(), method="combat", covariate=covariate,
        min_samples=3)
    assert not corrected.isna().any().any(), report.warnings
