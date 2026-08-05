"""Robust outlier detection: hand arithmetic, planted defects, and refusals.

Nothing here is a smoke test. Every number asserted is either worked out in a
comment above the assertion or planted in the data by the fixture that built
it, so a failure says which part of the maths moved rather than "something
changed".

The four things worth pinning, and why
--------------------------------------

**The MAD scale.** ``[1, 2, 3, 4, 100]`` has median 3 and deviations
``[2, 1, 0, 1, 97]``, whose median is 1. So MAD = 1, the robust sigma is
1.4826, and the modified z of 100 is ``97 / 1.4826 = 65.4``. That arithmetic
is short enough to check by eye and it is the whole method.

**MAD == 0.** Six zeros and four positive values put the median and the MAD
both at zero. Divided by, that scores the entire tail infinity — the failure
the fallback exists to prevent — so the test asserts the fallback fired, that
it is noted, and that exactly the one extreme value is flagged.

**The planted well.** 40 wells x 60 objects from a seeded lognormal, one well
multiplied by 1.2. The well pass names exactly that well at a robust score of
8.0 against a next-highest of 2.5, while the object pass flags 24 objects over
the whole plate of which **one** is in the planted well — and two innocent
wells contain three flagged objects each. That asymmetry is the entire
argument for having a well pass at all, so it is pinned rather than described.

**The off-axis points.** Five points placed inside both marginal ranges but
across the correlation axis of a tight 2-D cloud. Per-feature MAD flags none
of them; MCD Mahalanobis flags all five, at squared distances of 84 to 236
against a chi-square threshold of 13.8.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets.outlier_model import (
    DEFAULT_ALPHA, DEFAULT_IQR_C, DEFAULT_MAD_K, MAD_TO_SIGMA,
    MCD_MIN_OBJECTS_PER_FEATURE, MEAN_AD_TO_SIGMA, METHOD_IQR, METHOD_MAD,
    METHOD_MAHALANOBIS, MIN_WELLS_TO_SCORE, OBJECT_COLUMNS, TRANSFORM_LOG10,
    TRANSFORM_NONE, OutlierError, OutlierSpec, candidate_features,
    detect_outliers, median_absolute_deviation, robust_scale, tukey_fences,
    well_key_columns,
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

#: The hand vector. median 3, deviations [2, 1, 0, 1, 97], MAD 1.
HAND = [1.0, 2.0, 3.0, 4.0, 100.0]

#: Nine values whose quartiles land exactly on data points under numpy's
#: linear interpolation: (n-1)*0.25 = 2 and (n-1)*0.75 = 6.
QUARTILE_VECTOR = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 100.0]


def planted_plate(seed: int = 13, bad_well: int = 13, shift: float = 1.2,
                  n_wells: int = 40, per_well: int = 60) -> pd.DataFrame:
    """40 wells x 60 objects of clean lognormal, with ONE well shifted.

    ``cell_area`` is multiplied by ``shift`` in well ``bad_well`` and nowhere
    else; ``cell_perimeter`` is never touched, so a rule that names it has
    found noise.

    The shift is deliberately modest — 1.2x on a distribution whose objects
    have a robust SD of 0.20 around a median of 1.0, so it is 0.9 within-well
    SDs. That is nearly invisible object by object (an object would have to
    clear 3.5 SD, and the shift buys it only 0.9 of them) and enormous well by
    well (the median of 60 objects has a sampling spread of 0.16 SD, so the
    same shift is 5.6 of those). Those two numbers are the reason this module
    runs two passes.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for well in range(n_wells):
        factor = shift if well == bad_well else 1.0
        area = factor * rng.lognormal(0.0, 0.2, per_well)
        perimeter = rng.lognormal(0.0, 0.2, per_well)
        for i in range(per_well):
            rows.append(("p1", f"r{well // 10 + 1}", f"c{well % 10 + 1}",
                         "f1", i, area[i], perimeter[i]))
    return pd.DataFrame(rows, columns=[
        "plateID", "rowID", "columnID", "fieldID", "object_label",
        "cell_area", "cell_perimeter"])


#: Where :func:`planted_plate` puts the bad well: 13 // 10 + 1 = 2,
#: 13 % 10 + 1 = 4.
PLANTED_WELL = ("p1", "r2", "c4")


def correlated_cloud(n: int = 600, seed: int = 4
                     ) -> tuple[pd.DataFrame, np.ndarray]:
    """A tight 2-D correlated cloud plus five points across its axis.

    ``f_b = f_a + N(0, 0.2)``, so the cloud is a narrow ridge. The planted
    points sit at ``(a, -a)`` for |a| around 1 — comfortably inside both
    marginal ranges (the cloud spans about -3.2 to 2.8 on each axis) and
    perpendicular to the ridge, which is exactly the configuration a
    per-feature test cannot see.

    :returns: ``(frame, planted_positions)``.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 1.0, n)
    b = a + rng.normal(0.0, 0.2, n)
    planted = np.array([[1.5, -1.5], [-1.5, 1.5], [1.2, -1.0],
                        [-1.0, 1.2], [0.9, -0.9]])
    stacked = np.vstack([np.column_stack([a, b]), planted])
    frame = pd.DataFrame({
        "plateID": "p1", "rowID": "r1", "columnID": "c1",
        "f_a": stacked[:, 0], "f_b": stacked[:, 1]})
    return frame, np.arange(n, n + len(planted))


# ---------------------------------------------------------------------------
# 1. The MAD, on a vector small enough to check by eye
# ---------------------------------------------------------------------------

def test_the_mad_is_the_median_of_the_absolute_deviations():
    # median([1,2,3,4,100]) = 3; |x - 3| = [2,1,0,1,97]; median of those = 1.
    assert median_absolute_deviation(HAND) == 1.0


def test_the_robust_scale_is_the_mad_times_the_gaussian_constant():
    centre, scale, note = robust_scale(HAND)
    assert centre == 3.0
    # 1.4826 is 1 / Phi^-1(0.75): it makes the MAD estimate sigma, which is
    # what lets k be read as "3.5 SD" without an SD being computed.
    assert scale == pytest.approx(1.0 * MAD_TO_SIGMA)
    assert note == ""


def test_the_modified_z_score_is_hand_computable():
    frame = pd.DataFrame({"v": HAND})
    result = detect_outliers(frame, OutlierSpec(features=("v",),
                                                per_well=False))
    # |100 - 3| / (1.4826 * 1) = 97 / 1.4826022185 = 65.4255
    assert result.scores[4] == pytest.approx(97.0 / MAD_TO_SIGMA)
    assert result.scores[4] == pytest.approx(65.42551, abs=1e-4)
    # |1 - 3| / 1.4826 = 1.34898, and |4 - 3| / 1.4826 = 0.674490.
    assert result.scores[0] == pytest.approx(2.0 / MAD_TO_SIGMA)
    assert result.scores[3] == pytest.approx(1.0 / MAD_TO_SIGMA)
    # Only the 100 clears k = 3.5.
    assert list(result.flags) == [False, False, False, False, True]
    assert result.threshold == DEFAULT_MAD_K


def test_the_flag_reason_names_the_feature_and_the_side():
    frame = pd.DataFrame({"v": HAND})
    result = detect_outliers(frame, OutlierSpec(features=("v",),
                                                per_well=False))
    assert "v high" in result.reasons[4]
    assert "65.4" in result.reasons[4]
    assert result.reasons[0] == ""


def test_the_mad_rule_is_not_dragged_by_the_outlier_it_is_looking_for():
    """The whole reason for not using a z-score, in one assertion.

    ``[1, 2, 3, 4, 100]`` has mean 22 and sample SD ``sqrt(7610/4) = 43.62``,
    so the classical ``|x - mean| / sd`` of the 100 is ``78 / 43.62 = 1.79`` —
    under *any* usual cut-off, because the point being tested is most of what
    built the SD. The robust score of the same point is 65.
    """
    values = np.array(HAND)
    assert values.std(ddof=1) == pytest.approx(43.6177, abs=1e-3)
    classical = abs(values[4] - values.mean()) / values.std(ddof=1)
    assert classical == pytest.approx(1.7883, abs=1e-3)
    result = detect_outliers(pd.DataFrame({"v": HAND}),
                             OutlierSpec(features=("v",), per_well=False))
    assert result.scores[4] > 60.0


# ---------------------------------------------------------------------------
# 2. MAD == 0 — the degenerate case that must not flag the whole tail
# ---------------------------------------------------------------------------

#: Thirteen tied zeros and a three-value tail. The median is 0 and every
#: absolute deviation is ``[0]*13 + [0.5, 1.5, 6.0]``, whose median is also 0 —
#: so MAD = 0 and the naive modified z of the last three is infinity.
#:
#: Thirteen rather than eight, so that Q3 lands inside the tied block too
#: ((16-1) * 0.75 = 11.25, an index where the value is still 0) and the same
#: vector is degenerate under both univariate rules.
TIED = [0.0] * 13 + [0.5, 1.5, 6.0]


def test_a_zero_mad_falls_back_instead_of_dividing_by_zero():
    assert median_absolute_deviation(TIED) == 0.0
    centre, scale, note = robust_scale(TIED)
    assert centre == 0.0
    assert note == "mad-zero"
    # mean(|x - 0|) = (0.5 + 1.5 + 6) / 16 = 0.5; times sqrt(pi/2) = 1.2533.
    assert scale == pytest.approx(0.5 * MEAN_AD_TO_SIGMA)
    assert scale == pytest.approx(0.6266571, abs=1e-6)


def test_a_zero_mad_flags_the_extreme_value_and_not_the_whole_tail():
    result = detect_outliers(pd.DataFrame({"v": TIED}),
                             OutlierSpec(features=("v",), per_well=False))
    # 6.0 / 0.6266571 = 9.575 > 3.5; 1.5 / 0.6266571 = 2.394 < 3.5.
    assert result.scores[15] == pytest.approx(9.57461, abs=1e-4)
    assert result.scores[14] == pytest.approx(2.39365, abs=1e-4)
    assert result.scores[13] == pytest.approx(0.79788, abs=1e-4)
    assert result.n_flagged == 1                 # NOT the three-value tail
    assert bool(result.flags[15])
    assert not result.flags[:15].any()
    assert np.isfinite(result.scores).all()      # no inf from a zero divisor


def test_the_zero_mad_fallback_is_said_out_loud():
    result = detect_outliers(pd.DataFrame({"v": TIED}),
                             OutlierSpec(features=("v",), per_well=False))
    assert any("MAD of 'v' is zero" in note for note in result.notes)
    assert any("mean absolute deviation" in c for c in result.caveats())


def test_a_constant_feature_flags_nothing_and_does_not_raise():
    result = detect_outliers(pd.DataFrame({"v": [7.0] * 20}),
                             OutlierSpec(features=("v",), per_well=False))
    assert result.n_flagged == 0
    assert (result.scores == 0.0).all()
    assert any("one value" in note for note in result.notes)


# ---------------------------------------------------------------------------
# 3. IQR fences on a vector with known quartiles
# ---------------------------------------------------------------------------

def test_the_tukey_fences_are_hand_computable():
    # n = 9, so numpy's linear interpolation puts Q1 at index (9-1)*0.25 = 2
    # and Q3 at index 6 exactly: Q1 = 3, Q3 = 7, IQR = 4.
    q1, q3, low, high, note = tukey_fences(QUARTILE_VECTOR, DEFAULT_IQR_C)
    assert (q1, q3) == (3.0, 7.0)
    # 3 - 1.5*4 = -3 and 7 + 1.5*4 = 13.
    assert (low, high) == (-3.0, 13.0)
    assert note == ""


def test_the_iqr_rule_flags_only_what_is_outside_the_fence():
    frame = pd.DataFrame({"v": QUARTILE_VECTOR})
    result = detect_outliers(
        frame, OutlierSpec(features=("v",), method=METHOD_IQR,
                           per_well=False))
    assert result.fences["v"] == (-3.0, 13.0)
    # Score is "IQRs past the nearer quartile": (100 - 7) / 4 = 23.25.
    assert result.scores[8] == pytest.approx(23.25)
    # (3 - 1) / 4 = 0.5 for the smallest value, which is inside the fence.
    assert result.scores[0] == pytest.approx(0.5)
    assert list(result.flags) == [False] * 8 + [True]
    assert result.threshold == DEFAULT_IQR_C


def test_a_zero_iqr_rebuilds_the_fence_instead_of_collapsing_it():
    q1, q3, low, high, note = tukey_fences(TIED, DEFAULT_IQR_C)
    # Q1 = Q3 = 0 on the raw data, so the fence would be [0, 0] and every
    # positive value "outside" it. Rebuilt from the robust sigma instead:
    # sigma = 0.6266571, quartiles at +/- 0.6745 sigma = +/- 0.42267, and an
    # IQR of 1.34898 sigma = 0.845347, so the fence is
    # 0.42267 + 1.5 * 0.845347 = 1.69070 either side.
    assert note == "iqr-zero"
    assert q3 == pytest.approx(0.422674, abs=1e-6)
    assert high == pytest.approx(1.690695, abs=1e-6)
    assert low == pytest.approx(-1.690695, abs=1e-6)
    result = detect_outliers(
        pd.DataFrame({"v": TIED}),
        OutlierSpec(features=("v",), method=METHOD_IQR, per_well=False))
    assert result.n_flagged == 1        # the 6.0, NOT the whole tail
    assert bool(result.flags[15])
    assert any("IQR of 'v' is zero" in note for note in result.notes)


def test_the_symmetric_fence_flags_a_skewed_right_tail_by_construction():
    """The asymmetry problem, demonstrated on data with nothing wrong with it.

    A clean lognormal(0, 1) sample has no outliers at all, and Tukey's
    symmetric fence flags several percent of it on the right and essentially
    nothing on the left. The log10 transform is the offered fix and it works.
    """
    rng = np.random.default_rng(2)
    values = rng.lognormal(0.0, 1.0, 4000)
    frame = pd.DataFrame({"v": values})
    raw = detect_outliers(frame, OutlierSpec(features=("v",),
                                             method=METHOD_IQR,
                                             per_well=False))
    logged = detect_outliers(frame, OutlierSpec(features=("v",),
                                                method=METHOD_IQR,
                                                transform=TRANSFORM_LOG10,
                                                per_well=False))
    flagged = raw.flags
    assert raw.flagged_share > 0.03           # several percent, of clean data
    assert (values[flagged] > np.median(values)).all()   # all on the right
    assert logged.flagged_share < raw.flagged_share / 3
    assert any("symmetric in the quartiles" in c for c in raw.caveats())
    assert not any("symmetric in the quartiles" in c for c in logged.caveats())


def test_the_log_transform_refuses_non_positive_values():
    frame = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0, -4.0]})
    with pytest.raises(OutlierError) as excinfo:
        detect_outliers(frame, OutlierSpec(features=("a",),
                                           transform=TRANSFORM_LOG10,
                                           per_well=False))
    message = str(excinfo.value)
    assert "log10" in message
    assert "a (2)" in message               # names the feature and the count
    assert "pseudocount" in message         # and why it was not invented


def test_the_log_transform_is_never_implicit():
    assert OutlierSpec().transform == TRANSFORM_NONE
    frame = pd.DataFrame({"v": [1.0, 10.0, 100.0, 1000.0, 10000.0]})
    result = detect_outliers(frame, OutlierSpec(features=("v",),
                                                per_well=False))
    assert result.transform == TRANSFORM_NONE
    assert result.centres["v"] == 100.0      # the raw median, not log10 of it


# ---------------------------------------------------------------------------
# 4. The planted well — the asymmetry the well pass exists for
# ---------------------------------------------------------------------------

def test_the_well_pass_names_exactly_the_planted_well():
    frame = planted_plate()
    result = detect_outliers(frame, OutlierSpec(
        features=("cell_area", "cell_perimeter")))
    assert result.well_keys == ("plateID", "rowID", "columnID")
    assert result.n_wells == 40
    assert result.n_wells_scored == 40
    assert result.flagged_wells() == (PLANTED_WELL,)

    wells = result.well_frame()
    planted = wells[(wells["rowID"] == "r2") & (wells["columnID"] == "c4")]
    # The planted well's robust score across wells, and the runner-up.
    assert float(planted["well_outlier_score"].iloc[0]) == pytest.approx(
        7.979, abs=1e-3)
    assert np.sort(wells["well_outlier_score"].to_numpy())[-2] == \
        pytest.approx(2.455, abs=1e-3)
    assert "cell_area" in planted["well_outlier_reason"].iloc[0]
    assert "cell_perimeter" not in planted["well_outlier_reason"].iloc[0]


def test_the_object_pass_on_the_same_data_does_not_name_that_well():
    """The asymmetry, pinned. Both numbers come from the same seeded frame.

    24 objects are flagged over the whole plate and exactly ONE of them is in
    the planted well — while two entirely innocent wells contain three flagged
    objects each. Ranking wells by flagged-object count therefore puts the bad
    well *below* two good ones, which is precisely why "the well contains many
    flagged objects" is not the well test.
    """
    frame = planted_plate()
    result = detect_outliers(frame, OutlierSpec(
        features=("cell_area", "cell_perimeter")))
    assert result.n_flagged == 24
    wells = result.well_frame()
    planted = wells[(wells["rowID"] == "r2") & (wells["columnID"] == "c4")]
    assert int(planted["n_flagged_objects"].iloc[0]) == 1
    assert int(planted["n_objects"].iloc[0]) == 60
    assert float(planted["flagged_share"].iloc[0]) == pytest.approx(1 / 60)
    # Two clean wells beat it on the object-flag signal.
    assert int(wells["n_flagged_objects"].max()) == 3
    louder = wells[wells["n_flagged_objects"] > 1]
    assert len(louder) == 4
    assert not ((louder["rowID"] == "r2") & (louder["columnID"] == "c4")).any()
    # And none of those noisier wells is flagged by the well rule.
    assert (louder["well_outlier"] == False).all()  # noqa: E712 - array compare


def test_both_well_signals_are_reported_because_they_find_different_things():
    frame = planted_plate()
    result = detect_outliers(frame, OutlierSpec(features=("cell_area",)))
    wells = result.well_frame()
    for column in ("n_objects", "n_scored_objects", "n_flagged_objects",
                   "flagged_share", "well_outlier_score", "well_outlier",
                   "well_outlier_reason", "well_scored", "cell_area_median"):
        assert column in wells.columns


def test_a_well_below_the_minimum_is_reported_not_scored_and_not_dropped():
    frame = planted_plate()
    # Keep only 3 objects of one clean well; everything else untouched.
    thin = (frame["rowID"] == "r1") & (frame["columnID"] == "c1")
    keep = ~thin | (frame.groupby(["rowID", "columnID"]).cumcount() < 3)
    frame = frame[keep].reset_index(drop=True)
    result = detect_outliers(frame, OutlierSpec(features=("cell_area",)))
    wells = result.well_frame()
    small = wells[(wells["rowID"] == "r1") & (wells["columnID"] == "c1")]
    assert len(small) == 1                       # present, never dropped
    assert int(small["n_objects"].iloc[0]) == 3
    assert not bool(small["well_scored"].iloc[0])
    assert np.isnan(float(small["well_outlier_score"].iloc[0]))
    assert "not scored" in small["well_outlier_reason"].iloc[0]
    assert "3 object" in small["well_outlier_reason"].iloc[0]
    assert result.n_wells == 40 and result.n_wells_scored == 39
    assert ("p1", "r1", "c1") in result.unscored_wells()
    assert any("not scored" in c for c in result.caveats())


def test_too_few_wells_means_no_well_is_scored_rather_than_a_guess():
    frame = planted_plate(n_wells=4, per_well=60, bad_well=1)
    result = detect_outliers(frame, OutlierSpec(features=("cell_area",)))
    assert result.n_wells == 4
    assert result.n_wells_scored == 0
    assert result.flagged_wells() == ()
    assert any(f"{MIN_WELLS_TO_SCORE}" in note for note in result.notes)
    assert len(result.unscored_wells()) == 4


def test_the_well_pass_runs_whichever_rule_was_chosen():
    """"The same robust rules across wells" means the same rules, not the MAD.

    The Tukey and the MCD passes both have to reach the well medians, or a
    user who picked a method for the objects would silently get a different
    one for the wells.
    """
    # 24 wells rather than 12: the well-level MCD is fitted on a *subset* of
    # the wells, so twelve of them is the "unstable until n >= 2p" regime the
    # module docstring warns about and a poor place to assert from.
    frame = planted_plate(seed=3, n_wells=24, per_well=40, bad_well=3)
    features = ("cell_area", "cell_perimeter")

    tukey = detect_outliers(frame, OutlierSpec(features=features,
                                               method=METHOD_IQR))
    assert ("p1", "r1", "c4") in tukey.flagged_wells()
    reason = tukey.well_frame().loc[
        tukey.well_frame()["columnID"] == "c4", "well_outlier_reason"].iloc[0]
    assert "IQR" in reason

    mcd = detect_outliers(frame, OutlierSpec(features=features,
                                             method=METHOD_MAHALANOBIS))
    assert ("p1", "r1", "c4") in mcd.flagged_wells()
    wells = mcd.well_frame()
    planted = wells[wells["columnID"] == "c4"]
    # chi2.ppf(0.999, 2) = 13.8155, and the planted well sits at 32 — well
    # past it, and the only well that is.
    assert float(planted["well_outlier_score"].iloc[0]) == pytest.approx(
        31.95, abs=0.05)
    assert mcd.threshold == pytest.approx(13.8155, abs=1e-3)
    assert "13.8" in planted["well_outlier_reason"].iloc[0]
    assert any("wells: MCD fitted" in note for note in mcd.notes)
    assert any("complete wells" in note for note in mcd.notes)


def test_too_few_wells_for_an_mcd_is_noted_rather_than_guessed():
    rng = np.random.default_rng(0)
    frame = planted_plate(n_wells=6, per_well=40, bad_well=3)
    frame["cell_x"] = rng.normal(size=len(frame))
    frame["cell_y"] = rng.normal(size=len(frame))
    result = detect_outliers(frame, OutlierSpec(
        features=("cell_area", "cell_perimeter", "cell_x", "cell_y"),
        method=METHOD_MAHALANOBIS))
    # 6 scored wells for 4 features is under n >= 2p = 8.
    assert result.n_wells_scored == 0
    assert result.n_wells == 6                       # still reported
    assert any("across-well MCD did not run" in note for note in result.notes)
    assert all("too few wells" in reason for reason in
               result.well_frame()["well_outlier_reason"])


def test_the_well_pass_can_be_turned_off():
    frame = planted_plate(n_wells=6, per_well=30)
    result = detect_outliers(frame, OutlierSpec(features=("cell_area",),
                                                per_well=False))
    assert result.well_keys == ()
    assert result.n_wells == 0
    assert not result.has_wells
    assert any("per object" in c for c in result.caveats())


# ---------------------------------------------------------------------------
# 5. Multivariate — what a per-feature test cannot see
# ---------------------------------------------------------------------------

def test_per_feature_mad_misses_points_that_are_off_the_correlation_axis():
    frame, planted = correlated_cloud()
    # Every planted coordinate is inside the cloud's own marginal range, so
    # there is nothing for a one-column rule to notice.
    for column in ("f_a", "f_b"):
        values = frame[column].to_numpy()
        clean = values[:planted[0]]
        assert values[planted].min() > clean.min()
        assert values[planted].max() < clean.max()
    result = detect_outliers(frame, OutlierSpec(
        features=("f_a", "f_b"), method=METHOD_MAD, per_well=False))
    assert result.n_flagged == 0
    assert not result.flags[planted].any()


def test_mcd_mahalanobis_catches_every_one_of_them():
    frame, planted = correlated_cloud()
    result = detect_outliers(frame, OutlierSpec(
        features=("f_a", "f_b"), method=METHOD_MAHALANOBIS, per_well=False,
        seed=0))
    # chi2.ppf(0.999, 2) = -2 * ln(0.001) = 13.8155.
    assert result.threshold == pytest.approx(-2.0 * np.log(0.001), abs=1e-9)
    assert result.flags[planted].all()
    assert result.scores[planted].min() > 80.0
    # The planted five, plus a couple of the 600 clean points at the stated
    # per-object rate; nothing like a "contamination fraction" of the data.
    assert result.n_flagged == 7
    assert "Mahalanobis" in result.reasons[planted[0]]


def test_the_chi_square_threshold_holds_roughly_the_stated_false_positive_rate():
    """Clean data only. alpha = 0.001 should flag about one object in a
    thousand — generously bounded, because the reweighted MCD's finite-sample
    scale makes the realised rate a small multiple of the nominal one."""
    rng = np.random.default_rng(0)
    n = 2000
    a = rng.normal(0.0, 1.0, n)
    b = a + rng.normal(0.0, 0.2, n)
    frame = pd.DataFrame({"f_a": a, "f_b": b})
    result = detect_outliers(frame, OutlierSpec(
        features=("f_a", "f_b"), method=METHOD_MAHALANOBIS, per_well=False,
        seed=0))
    assert result.n_flagged == 4
    assert result.flagged_share < 0.01           # ten times nominal, no more
    assert result.flagged_share > 0.0


def test_the_multiple_testing_arithmetic_is_stated():
    frame, _ = correlated_cloud(n=300)
    result = detect_outliers(frame, OutlierSpec(
        features=("f_a", "f_b"), method=METHOD_MAHALANOBIS, per_well=False))
    text = " ".join(result.caveats())
    assert "200,000 objects would produce roughly 200 false flags" in text
    assert "Bonferroni" in text
    assert f"α = {DEFAULT_ALPHA:g}" in text


def test_mcd_is_reproducible_given_a_seed():
    frame, _ = correlated_cloud(n=300)
    spec = OutlierSpec(features=("f_a", "f_b"),
                       method=METHOD_MAHALANOBIS, per_well=False, seed=7)
    first = detect_outliers(frame, spec)
    second = detect_outliers(frame, spec)
    assert np.array_equal(first.flags, second.flags)
    np.testing.assert_allclose(first.scores, second.scores)


def test_mcd_refuses_when_there_are_too_few_objects_for_the_features():
    # 3 features, 3 objects: n <= p, so there is no covariance at all.
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 1.0, 5.0],
                          "c": [3.0, 5.0, 1.0]})
    with pytest.raises(OutlierError) as excinfo:
        detect_outliers(frame, OutlierSpec(
            features=("a", "b", "c"), method=METHOD_MAHALANOBIS,
            per_well=False))
    message = str(excinfo.value)
    assert "more objects than features" in message
    assert "pca_model" in message               # names the way out


def test_mcd_refuses_below_two_objects_per_feature_and_points_at_pca():
    rng = np.random.default_rng(1)
    # 4 features, 5 objects: n > p but n < 2p = 8.
    frame = pd.DataFrame(rng.normal(size=(5, 4)),
                         columns=["a", "b", "c", "d"])
    with pytest.raises(OutlierError) as excinfo:
        detect_outliers(frame, OutlierSpec(
            features=("a", "b", "c", "d"), method=METHOD_MAHALANOBIS,
            per_well=False))
    message = str(excinfo.value)
    assert f"n = {MCD_MIN_OBJECTS_PER_FEATURE}p = 8" in message
    assert "pca_model" in message


def test_the_multivariate_method_refuses_a_single_feature():
    frame = pd.DataFrame({"a": np.arange(50.0)})
    with pytest.raises(OutlierError) as excinfo:
        detect_outliers(frame, OutlierSpec(features=("a",),
                                           method=METHOD_MAHALANOBIS,
                                           per_well=False))
    assert METHOD_MAD in str(excinfo.value)


# ---------------------------------------------------------------------------
# 6. Nothing is dropped, nothing is overwritten
# ---------------------------------------------------------------------------

def test_the_frame_is_never_shortened_and_no_input_column_is_modified():
    frame = planted_plate(n_wells=8, per_well=25)
    before = frame.copy(deep=True)
    result = detect_outliers(frame, OutlierSpec(features=("cell_area",)))
    out = result.object_frame(frame)

    assert len(out) == len(frame) == 200
    pd.testing.assert_frame_equal(frame, before)             # input untouched
    pd.testing.assert_frame_equal(out[list(before.columns)], before)
    for name in OBJECT_COLUMNS.values():
        assert name in out.columns
    assert out["outlier"].dtype == bool
    assert out["outlier_method"].iloc[0] == "mad(k=3.5)"
    assert out["outlier"].sum() == result.n_flagged


def test_an_existing_column_is_suffixed_rather_than_overwritten():
    frame = planted_plate(n_wells=6, per_well=25)
    frame["outlier"] = "mine"
    result = detect_outliers(frame, OutlierSpec(features=("cell_area",),
                                                per_well=False))
    assert result.column_names["outlier"] == "outlier_2"
    out = result.object_frame(frame)
    assert (out["outlier"] == "mine").all()      # the user's column survives
    assert out["outlier_2"].dtype == bool
    assert any("outlier_2" in note for note in result.notes)


def test_filtered_is_the_only_thing_that_removes_rows_and_says_so():
    frame = pd.DataFrame({"v": HAND})
    result = detect_outliers(frame, OutlierSpec(features=("v",),
                                                per_well=False))
    kept = result.filtered(frame)
    assert len(kept) == 4
    assert 100.0 not in set(kept["v"])
    assert len(frame) == 5                        # the original is intact
    assert "choosing to delete" in result.filtered.__doc__


def test_unscorable_objects_are_neither_flagged_nor_dropped():
    frame = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, np.nan, np.inf, 100.0]})
    result = detect_outliers(frame, OutlierSpec(features=("v",),
                                                per_well=False))
    assert result.n_rows_in == 7
    assert result.n_scored == 5
    assert result.n_not_scored == 2
    assert result.n_non_finite == 1               # the inf; the NaN was one
    assert not result.flags[4] and not result.flags[5]
    assert np.isnan(result.scores[4]) and np.isnan(result.scores[5])
    assert "not scored" in result.reasons[4]
    assert len(result.object_frame(frame)) == 7
    assert len(result.filtered(frame)) == 6       # only the 100 goes


def test_writing_flags_onto_a_different_frame_is_refused():
    frame = pd.DataFrame({"v": HAND})
    result = detect_outliers(frame, OutlierSpec(features=("v",),
                                                per_well=False))
    with pytest.raises(OutlierError) as excinfo:
        result.object_frame(frame.head(3))
    assert "positional" in str(excinfo.value)
    with pytest.raises(OutlierError):
        result.filtered(frame.head(3))


# ---------------------------------------------------------------------------
# 7. Well identity
# ---------------------------------------------------------------------------

def test_the_canonical_trio_is_detected_first():
    frame = planted_plate(n_wells=6, per_well=25)
    frame["prc"] = "p1_r1_c1"
    assert well_key_columns(frame) == ("plateID", "rowID", "columnID")


@pytest.mark.parametrize("columns, expected", [
    (["prc"], ("prc",)),
    (["plateID", "well"], ("plateID", "well")),
    (["plate", "row_name", "column_name"], ("plate", "row_name",
                                            "column_name")),
    (["well"], ("well",)),
])
def test_every_spelling_spacr_writes_is_accepted(columns, expected):
    frame = pd.DataFrame({name: ["a"] * 4 for name in columns})
    frame["v"] = [1.0, 2.0, 3.0, 4.0]
    assert well_key_columns(frame) == expected


def test_no_well_columns_is_an_actionable_refusal():
    frame = pd.DataFrame({"alpha": np.arange(50.0), "beta": np.arange(50.0)})
    with pytest.raises(OutlierError) as excinfo:
        detect_outliers(frame, OutlierSpec(features=("alpha", "beta")))
    message = str(excinfo.value)
    assert "alpha" in message and "beta" in message      # names what IS there
    assert "plateID+rowID+columnID" in message           # and what it wanted
    assert "per_well=False" in message                   # and the way out


def test_explicit_well_keys_win_and_a_wrong_one_is_named():
    frame = planted_plate(n_wells=6, per_well=25)
    result = detect_outliers(frame, OutlierSpec(
        features=("cell_area",), well_keys=("plateID", "rowID")))
    assert result.well_keys == ("plateID", "rowID")
    with pytest.raises(OutlierError) as excinfo:
        detect_outliers(frame, OutlierSpec(features=("cell_area",),
                                           well_keys=("wellID",)))
    assert "wellID" in str(excinfo.value)


def test_a_well_whose_key_is_missing_gets_its_own_row_rather_than_vanishing():
    frame = planted_plate(n_wells=6, per_well=25)
    frame.loc[frame.index[:25], "columnID"] = np.nan
    result = detect_outliers(frame, OutlierSpec(features=("cell_area",)))
    assert int(result.well_frame()["n_objects"].sum()) == len(frame)


# ---------------------------------------------------------------------------
# 8. The spec
# ---------------------------------------------------------------------------

def test_the_spec_refuses_an_unknown_method_at_construction():
    with pytest.raises(OutlierError) as excinfo:
        OutlierSpec(method="zscore")
    message = str(excinfo.value)
    assert "zscore" in message
    assert METHOD_MAD in message and METHOD_IQR in message


@pytest.mark.parametrize("kwargs, needle", [
    ({"transform": "ln"}, "unknown transform"),
    ({"k": 0}, "must be positive"),
    ({"c": -1}, "must be positive"),
    ({"alpha": 0.0}, "(0, 1)"),
    ({"alpha": 1.0}, "(0, 1)"),
    ({"min_well_objects": 0}, "at least 1"),
    ({"support_fraction": 0.0}, "(0, 1]"),
    ({"support_fraction": 1.5}, "(0, 1]"),
])
def test_every_bad_parameter_is_refused_with_its_own_sentence(kwargs, needle):
    with pytest.raises(OutlierError) as excinfo:
        OutlierSpec(**kwargs)
    assert needle in str(excinfo.value)


def test_the_spec_round_trips_through_json():
    spec = OutlierSpec(features=("a", "b"), method=METHOD_MAHALANOBIS,
                       alpha=0.005, transform=TRANSFORM_LOG10,
                       well_keys=("prc",), min_well_objects=5, seed=9)
    assert OutlierSpec.from_json(spec.to_json()) == spec
    assert OutlierSpec.from_dict(spec.to_dict()) == spec
    # Unknown keys are ignored and missing ones default, so a spec written by
    # another build still opens.
    payload = json.loads(spec.to_json())
    payload["invented_by_a_later_build"] = True
    del payload["k"]
    assert OutlierSpec.from_dict(payload).method == METHOD_MAHALANOBIS
    assert OutlierSpec.from_dict(payload).k == DEFAULT_MAD_K


def test_the_spec_is_frozen_and_edits_return_copies():
    spec = OutlierSpec()
    assert spec.with_method(METHOD_IQR).method == METHOD_IQR
    assert spec.method == METHOD_MAD
    assert spec.with_features(["a"]).features == ("a",)
    assert spec.with_transform(TRANSFORM_LOG10).transform == TRANSFORM_LOG10
    assert spec.with_well_keys(["prc"]).well_keys == ("prc",)
    with pytest.raises(Exception):
        spec.k = 9.0


def test_describe_says_the_rule_in_one_line():
    assert "modified z > 3.5" in OutlierSpec().describe()
    assert "IQR" in OutlierSpec(method=METHOD_IQR).describe()
    assert "α" in OutlierSpec(method=METHOD_MAHALANOBIS).describe()
    assert "objects only" in OutlierSpec(per_well=False).describe()


def test_the_threshold_depends_on_the_dimension_for_the_multivariate_rule():
    spec = OutlierSpec(method=METHOD_MAHALANOBIS)
    # chi2.ppf(0.999, 1) = 10.828, chi2.ppf(0.999, 2) = 13.816.
    assert spec.threshold(1) == pytest.approx(10.8276, abs=1e-3)
    assert spec.threshold(2) == pytest.approx(13.8155, abs=1e-3)
    assert spec.threshold(2) > spec.threshold(1)
    assert OutlierSpec().threshold(5) == DEFAULT_MAD_K   # k, whatever p is


# ---------------------------------------------------------------------------
# 9. Feature selection and the report
# ---------------------------------------------------------------------------

def test_candidate_features_reuses_the_one_column_classifier():
    frame = planted_plate(n_wells=6, per_well=25)
    features = candidate_features(frame)
    assert "cell_area" in features and "cell_perimeter" in features
    # Keys identify rather than describe, and are not offered.
    for name in ("plateID", "rowID", "columnID", "fieldID", "object_label"):
        assert name not in features
    assert list(features) == sorted(features)


def test_a_table_with_no_continuous_column_is_refused():
    frame = pd.DataFrame({"plateID": ["p1"] * 4, "rowID": ["r1"] * 4,
                          "columnID": ["c1"] * 4, "gene": list("abcd")})
    with pytest.raises(OutlierError) as excinfo:
        detect_outliers(frame)
    assert "no continuous columns" in str(excinfo.value)


def test_an_empty_table_is_refused_rather_than_returned_empty():
    with pytest.raises(OutlierError) as excinfo:
        detect_outliers(pd.DataFrame({"v": []}))
    assert "no objects" in str(excinfo.value)


def test_a_feature_that_is_not_a_column_is_named():
    frame = pd.DataFrame({"v": np.arange(50.0)})
    with pytest.raises(OutlierError) as excinfo:
        detect_outliers(frame, OutlierSpec(features=("nope",),
                                           per_well=False))
    assert "nope" in str(excinfo.value)


def test_the_report_carries_the_counts_the_thresholds_and_the_caveats():
    frame = planted_plate()
    result = detect_outliers(frame, OutlierSpec(features=("cell_area",)))
    text = result.report()
    assert "2,400 objects" in text
    assert "cell_area: median" in text
    assert "40 found" in text
    assert "1 flagged" in text
    assert "p1+r2+c4" in text
    assert "Flagged is not deleted" in text
    assert result.headline().endswith(".")
    assert len(result) == 2400                    # the input count, always


def test_the_headline_explains_why_a_quiet_object_pass_and_a_loud_well_pass():
    result = detect_outliers(planted_plate(),
                             OutlierSpec(features=("cell_area",)))
    assert "1 of 40 scored wells" in result.headline()
