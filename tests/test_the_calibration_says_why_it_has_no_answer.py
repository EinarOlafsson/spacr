"""What the fraction-threshold sweep reports when it cannot fit.

A calibration that quietly returns a number it did not measure is worse than
one that refuses, because the number is used: ``fraction_threshold`` decides
how many gRNAs each well is credited with and therefore what every downstream
coefficient is regressed on. So every way the fit can fail has to come back
named -- a candidate that could not be fitted at all, a fit that was refused
for having too few wells, a comparison with no slope on one side -- and the
caller has to be able to tell those three apart.
"""
import numpy as np
import pytest

from spacr.fraction_calibration import (compare_normalisations,
                                        sweep_fraction_threshold)

from tests.test_the_threshold_is_measured_per_screen import PC, _screen


def test_a_candidate_that_cannot_be_fitted_carries_its_error_not_a_slope():
    """With no pure positive-control wells the mixture line has one end.

    The row still exists -- the threshold and the guides-per-well are facts
    about the counts and do not depend on the imaging -- but it says ``error``
    instead of inventing a slope.
    """
    counts, features, wells, _pure_pc, pure_nc = _screen()

    result = sweep_fraction_threshold(
        counts, features, wells, positive_guide=PC,
        pure_pc_wells=[], pure_nc_wells=pure_nc, candidates=(0.02,))

    row = result["candidates"][0]
    assert "pure wells" in row["error"]
    assert "slope" not in row
    assert "median_absolute_disagreement" not in row
    assert row["threshold"] == 0.02
    assert row["wells"] == 0
    assert result["chosen"] is None


def test_a_screen_with_no_cells_reports_no_training_wells():
    """``training_wells_in_fit`` counts the wells the classifier was fitted on.

    With no cells there are no wells to count, and the answer is 0 rather than
    a call into the training-well matcher with an empty label list.
    """
    counts, _features, _wells, _pure_pc, _pure_nc = _screen()

    result = sweep_fraction_threshold(
        counts.iloc[:0], np.zeros((0, 1)), [], positive_guide=PC,
        pure_pc_wells=[], pure_nc_wells=[], candidates=(0.02,))

    assert result["training_wells_in_fit"] == 0
    assert result["chosen"] is None


def test_a_fit_with_no_per_well_pairs_reports_an_unknown_disagreement(
        monkeypatch):
    """The choice is made on ``median |imaging - sequencing|`` per well.

    A fit that reports no per-well pairs has not supplied that quantity, and
    it comes back NaN -- not 0.0, which would read as perfect agreement and
    would win the sweep outright.
    """
    from spacr import annotation_validation

    def _fit_without_pairs(*_args, **_kwargs):
        return {"wells": 12, "slope": 1.0, "intercept": 0.0,
                "median_absolute_residual": 0.0, "per_well": {},
                "reading": "a fit that reported no per-well pairs"}

    monkeypatch.setattr(annotation_validation, "mixed_ratio_calibration",
                        _fit_without_pairs)
    counts, features, wells, pure_pc, pure_nc = _screen()

    result = sweep_fraction_threshold(
        counts, features, wells, positive_guide=PC,
        pure_pc_wells=pure_pc, pure_nc_wells=pure_nc, candidates=(0.02,))

    row = result["candidates"][0]
    assert np.isnan(row["median_absolute_disagreement"])
    # An unmeasurable disagreement cannot win the sweep, and it must not
    # crash it either: NaN loses every comparison, which once left the
    # tie-break with nothing to take a minimum of.
    assert result["chosen"] is None
    assert "measurable disagreement" in result["reason"]


def test_comparing_the_two_fraction_definitions_needs_two_slopes():
    """No slope on one side means no ratio, and the reason travels with it."""
    counts, features, wells, _pure_pc, pure_nc = _screen()

    out = compare_normalisations(
        counts, features, wells, threshold=0.02, positive_guide=PC,
        pure_pc_wells=[], pure_nc_wells=pure_nc)

    assert out["ratio"] is None
    assert out["more_consistent"] is None
    assert "no ratio" in out["reading"]
    assert "pure wells" in out["reading"]
    assert out["threshold"] == 0.02


def test_a_refused_fit_is_reported_as_refused_not_as_an_error():
    """Too few wells is a refusal, not a failure to fit.

    The candidate fitted perfectly well; the sweep declined to choose it. The
    two are different answers and ``compare_normalisations`` keeps them apart,
    because "the fit broke" and "there was not enough data to trust it" send
    the reader to different places.
    """
    counts, features, wells, pure_pc, pure_nc = _screen()

    out = compare_normalisations(
        counts, features, wells, threshold=0.02, positive_guide=PC,
        pure_pc_wells=pure_pc, pure_nc_wells=pure_nc, minimum_wells=999)

    for name in ("raw", "normalised"):
        assert "error" not in out[name]
        assert "999 control wells" in out[name]["refused"]
        assert out[name]["slope"] == pytest.approx(out[name]["slope"])
    # Both sides still fitted, so the ratio is real even though neither
    # threshold was chosen.
    assert out["ratio"] is not None


def test_neither_fraction_definition_wins_when_the_gap_is_unmeasured(
        monkeypatch):
    """``more_consistent`` names the definition whose two measurements agree.

    With no per-well pairs there is no such measurement on either side. NaN
    loses every comparison, so a plain ``min`` would name whichever definition
    happened to be enumerated first and read as a result.
    """
    from spacr import annotation_validation

    def _fit_without_pairs(*_args, **_kwargs):
        return {"wells": 12, "slope": 1.0, "intercept": 0.0,
                "median_absolute_residual": 0.0, "per_well": {},
                "reading": "a fit that reported no per-well pairs"}

    monkeypatch.setattr(annotation_validation, "mixed_ratio_calibration",
                        _fit_without_pairs)
    counts, features, wells, pure_pc, pure_nc = _screen()

    out = compare_normalisations(
        counts, features, wells, threshold=0.02, positive_guide=PC,
        pure_pc_wells=pure_pc, pure_nc_wells=pure_nc)

    assert out["ratio"] == pytest.approx(1.0)
    assert out["more_consistent"] is None
    assert "leave imaging and sequencing closer" not in out["reading"]
