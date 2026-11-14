"""Outlier detection on the degenerate columns and well tables real plates have.

A tied column must score zero rather than divide by a zero spread, a column
with nothing finite must be named in the notes, partly measured objects and
wells must be told why they were not scored, and a covariance that cannot be
inverted is refused with the way out.  Every scan's caveats end with the one
sentence that is always true -- flagged is not deleted -- so a report always
carries its caveat block.  A result names its wells by the well key it was
given: plain names for one key column, tuples for several, nothing when the
key columns are gone.

One general test per unit, each walking a case table with pytest's
``subtests`` fixture.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets.outlier_model import (
    METHOD_IQR,
    METHOD_MAD,
    METHOD_MAHALANOBIS,
    OutlierError,
    OutlierResult,
    OutlierSpec,
    detect_outliers,
)


def _plate(n, seed, features):
    """``n`` objects over four wells of one plate, lognormal features."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "plateID": "p1", "rowID": "r1",
        "columnID": [f"c{i % 4 + 1}" for i in range(n)],
        "fieldID": "f1", "object_label": range(n)})
    for feature in features:
        frame[feature] = rng.lognormal(0, 0.2, n)
    return frame


def _well_with_a_missing_feature():
    rng = np.random.default_rng(5)
    rows = []
    for well in range(8):
        for _ in range(25):
            rows.append({
                "plateID": "p1", "rowID": "A", "columnID": str(well),
                "a": float(rng.normal()),
                "b": np.nan if well == 3 else float(rng.normal()),
            })
    return pd.DataFrame(rows)


def _two_normal(seed, missing_first_b=False):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"a": rng.normal(size=40), "b": rng.normal(size=40)})
    if missing_first_b:
        frame.loc[0, "b"] = np.nan
    return frame


def test_detect_outliers_scores_notes_and_caveats_degenerate_input(
        monkeypatch, subtests):
    """Each degenerate input has a stated answer, and it reaches the result."""
    def check_tied(result):
        assert result.n_flagged == 0
        np.testing.assert_array_equal(result.scores, np.zeros(20))

    def check_empty_feature(result):
        assert any("'empty'" in note and "no finite value" in note
                   for note in result.notes), result.notes

    def check_partly_scored(result):
        assert any("scored on the features they have" in note
                   for note in result.notes), result.notes

    def check_missing_feature_reason(result):
        assert "needs a finite value" in result.reasons[0]
        assert not any("needs a finite value" in reason
                       for reason in result.reasons[1:])

    def check_well_without_median(result):
        reasons = list(result.wells["well_outlier_reason"])
        assert any("no median for every feature" in reason
                   for reason in reasons), reasons

    cases = [
        ("a tied column under the fence rule scores zero",
         pd.DataFrame({"v": [7.0] * 20}),
         dict(features=("v",), method=METHOD_IQR, per_well=False),
         check_tied),
        ("a feature with no finite value is noted by name",
         pd.DataFrame({"v": [1.0, 2.0, 3.0, 40.0], "empty": [np.nan] * 4}),
         dict(features=("v", "empty"), per_well=False), check_empty_feature),
        ("objects scored on only some features are noted",
         pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                       "b": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0]}),
         dict(features=("a", "b"), per_well=False), check_partly_scored),
        ("an object missing a feature is told why",
         _two_normal(1, missing_first_b=True),
         dict(features=("a", "b"), method=METHOD_MAHALANOBIS, per_well=False),
         check_missing_feature_reason),
        ("a well with no median for every feature is told why",
         _well_with_a_missing_feature(),
         dict(features=("a", "b"), method=METHOD_MAHALANOBIS, per_well=True),
         check_well_without_median),
    ]
    for case_id, frame, spec, check in cases:
        with subtests.test(msg=case_id):
            check(detect_outliers(frame, OutlierSpec(**spec)))

    with subtests.test(msg="a singular covariance is refused with the way out"):
        import sklearn.covariance as cov

        class _Singular:
            def __init__(self, **_kwargs):
                pass

            def fit(self, _matrix):
                raise np.linalg.LinAlgError("singular matrix")

        with monkeypatch.context() as patch:
            patch.setattr(cov, "MinCovDet", _Singular)
            with pytest.raises(OutlierError) as excinfo:
                detect_outliers(_two_normal(0), OutlierSpec(
                    features=("a", "b"), method=METHOD_MAHALANOBIS,
                    per_well=False))
        assert "collinear" in str(excinfo.value)
        assert "PCA" in str(excinfo.value)

    scans = [
        ("mahalanobis over two features", _plate(120, 0, ["cell_area",
                                                          "cell_perimeter"]),
         dict(features=("cell_area", "cell_perimeter"),
              method=METHOD_MAHALANOBIS)),
        ("one feature, default wells", _plate(80, 1, ["cell_area"]),
         dict(features=("cell_area",))),
        ("one feature, every well big enough", _plate(80, 1, ["cell_area"]),
         dict(features=("cell_area",), min_well_objects=1)),
        ("one feature, no well big enough", _plate(80, 1, ["cell_area"]),
         dict(features=("cell_area",), min_well_objects=500)),
    ]
    for case_id, frame, spec in scans:
        with subtests.test(msg=f"every report has caveats: {case_id}"):
            result = detect_outliers(frame, OutlierSpec(**spec))
            caveats = result.caveats()
            assert caveats
            assert "Flagged is not deleted" in caveats[-1]
            assert "  ! " in result.report()


def _result(**kwargs):
    base = dict(method=METHOD_MAD, features=("v",),
                scores=np.zeros(1), flags=np.zeros(1, dtype=bool),
                reasons=("",), threshold=3.5, n_rows_in=1, n_scored=1)
    base.update(kwargs)
    return OutlierResult(**base)


def test_outlier_result_names_wells_by_the_key_it_was_given(subtests):
    """One key column gives names, several give tuples, none give nothing."""
    cases = [
        ("no well pass", {}, (), ()),
        ("a one-column key yields plain names",
         dict(wells=pd.DataFrame({"prc": ["p1_r1_c1", "p1_r1_c2"],
                                  "well_outlier": [True, False],
                                  "well_scored": [True, False]}),
              well_keys=("prc",)),
         ("p1_r1_c1",), ("p1_r1_c2",)),
        ("a multi-column key yields tuples",
         dict(wells=pd.DataFrame({"plateID": ["p1"], "rowID": ["A"],
                                  "columnID": ["1"], "well_outlier": [True],
                                  "well_scored": [False]}),
              well_keys=("plateID", "rowID", "columnID")),
         (("p1", "A", "1"),), (("p1", "A", "1"),)),
        ("a well table that lost its key columns",
         dict(wells=pd.DataFrame({"well_outlier": [True],
                                  "well_scored": [False]}),
              well_keys=("prc",)),
         (), ()),
    ]
    for case_id, kwargs, flagged, unscored in cases:
        with subtests.test(msg=case_id):
            result = _result(**kwargs)
            assert result.flagged_wells() == flagged
            assert result.unscored_wells() == unscored
