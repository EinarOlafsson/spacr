"""`fraction_threshold` is measured on the control wells, not assumed.

The default 0.02 is a constant applied to every screen, and it decides how
many gRNAs each well is credited with -- which is what every coefficient
downstream is regressed on. A control well answers the same question twice:
sequencing says the positive control is a fraction of its reads, imaging says
a proportion of its cells look like it. Sweeping the threshold and refitting
one against the other turns the constant into a measurement of this screen.

The choice is made on the SPREAD around the line rather than the slope,
because the slope is penetrance times fraction bias and neither is separable
from a single control; a threshold that leaves spurious barcodes in the
denominator scatters the wells by however much each one is contaminated.
"""
import numpy as np
import pandas as pd
import pytest

from spacr.fraction_calibration import (
    DEFAULT_THRESHOLD_CANDIDATES,
    describe,
    reported_control_share,
    sweep_fraction_threshold,
    well_fractions,
)

PC = "pc_guide"
NC = "nc_guide"
#: One spurious barcode per well, at these shares in turn. Every one of them
#: is above 0.015 and below 0.02, so the smallest candidate threshold that
#: removes them all is 0.02 -- which is the answer the sweep has to find.
CONTAMINATION = (0.016, 0.017, 0.018)
CELLS_PER_WELL = 5


def _screen(contamination=CONTAMINATION):
    """A control plate whose true PC proportion per well is known exactly.

    Column 1 is pure negative control, column 2 pure positive control and
    column 3 a ratio series. Each well also carries one spurious barcode, the
    thing `fraction_threshold` exists to remove.
    """
    rows = []
    features = []
    wells = []
    pure_pc, pure_nc = [], []
    plan = []
    for index in range(6):
        plan.append((f"plate1_r{index + 1}_c1", 0.0))
        plan.append((f"plate1_r{index + 1}_c2", 1.0))
        plan.append((f"plate1_r{index + 1}_c3", 0.1 + 0.15 * index))
    for index, (well, proportion) in enumerate(plan):
        junk = contamination[index % len(contamination)]
        total = 100000
        rows.append({"prc": well, "grna": PC,
                     "count": proportion * (1 - junk) * total})
        rows.append({"prc": well, "grna": NC,
                     "count": (1 - proportion) * (1 - junk) * total})
        rows.append({"prc": well, "grna": f"junk{index}",
                     "count": junk * total})
        if well.endswith("_c2"):
            pure_pc.append(well)
        if well.endswith("_c1"):
            pure_nc.append(well)
        # One feature, so a well's mean lands exactly on the mixture line
        # between the two references and the imaging proportion is the
        # planted one.
        for _ in range(CELLS_PER_WELL):
            features.append([proportion * 5.0])
            wells.append(well)
    return (pd.DataFrame(rows), np.asarray(features, dtype=float), wells,
            pure_pc, pure_nc)


def _sweep(**kwargs):
    counts, features, wells, pure_pc, pure_nc = _screen()
    return sweep_fraction_threshold(
        counts, features, wells, positive_guide=PC,
        pure_pc_wells=pure_pc, pure_nc_wells=pure_nc, **kwargs)


def test_the_threshold_that_removes_the_spurious_barcodes_is_chosen():
    """0.02 is not assumed here: the largest planted contaminant is 0.018, and
    the sweep has to find the smallest candidate above it from the data."""
    result = _sweep()

    assert result["chosen"] == 0.02
    assert "most consistent" in result["reason"]
    assert result["fit"]["wells"] == 18


def test_the_chosen_threshold_beats_keeping_everything():
    """The claim the sweep rests on: while the spurious barcodes are in the
    denominator sequencing under-reports the control, and removing them makes
    the two measurements agree."""
    result = _sweep()
    at = {row["threshold"]: row for row in result["candidates"]}

    assert (at[0.02]["median_absolute_disagreement"]
            < at[0.0]["median_absolute_disagreement"])
    assert abs(at[0.02]["slope"] - 1.0) < abs(at[0.0]["slope"] - 1.0)
    # Sequencing under-reports the control while the junk is in the
    # denominator, which is the direction contamination can only push.
    assert at[0.0]["slope"] > 1.0


def test_the_scatter_around_the_line_cannot_see_a_systematic_deflation():
    """Why the choice is made on the disagreement and not on the residual:
    fractions that are all three per cent too low sit exactly on a line, so
    the fit looks perfect while both measurements disagree."""
    result = _sweep()
    keeping_everything = result["candidates"][0]

    assert keeping_everything["threshold"] == 0.0
    # A perfect-looking fit: every well sits on the fitted line.
    assert keeping_everything["median_absolute_residual"] == pytest.approx(
        0.0, abs=1e-9)
    # And the two measurements are nowhere near each other.
    assert keeping_everything["median_absolute_disagreement"] > 0.008


def test_a_higher_threshold_only_discards_more_data():
    """Ties go to the smallest threshold: once the spurious barcodes are gone
    a higher cut changes nothing except how much real data it destroys."""
    result = _sweep()
    at = {row["threshold"]: row for row in result["candidates"]}

    assert at[0.05]["guides_per_well"] <= at[0.02]["guides_per_well"]
    assert result["chosen"] <= 0.05


def test_how_many_guides_a_well_had_is_reported_at_every_candidate():
    """A control well has two guides competing for its reads and a screen well
    has hundreds, so the threshold this measures is the one that makes the
    CONTROL columns consistent. The count is reported rather than assumed
    away."""
    result = _sweep()

    for row in result["candidates"]:
        assert row["guides_per_well"] > 0
    assert (result["candidates"][0]["guides_per_well"]
            > result["candidates"][-1]["guides_per_well"])


def test_too_few_wells_is_refused_rather_than_reported():
    """A median over a handful of wells is decided by which wells happened to
    be usable, and the sweep exists to compare fits."""
    result = _sweep(minimum_wells=50)

    assert result["chosen"] is None
    assert "50" in result["reason"]
    assert describe(result).startswith("fraction_threshold not measured")


def test_the_classifier_correction_needs_a_confusion_matrix():
    """Sensitivity and specificity are two quantities and an accuracy is one;
    on a well where the control is a minority the accuracy is dominated by the
    majority class."""
    with pytest.raises(
        ValueError,
        match="requires both sensitivity and specificity",
    ):
        _sweep(sensitivity=0.96)
    with pytest.raises(
        ValueError,
        match="requires both sensitivity and specificity",
    ):
        _sweep(specificity=0.98)


def test_a_classifier_no_better_than_chance_is_refused():
    """`se + sp - 1` is the denominator of the correction; at or below zero it
    would invert the estimate and report it as a measurement."""
    with pytest.raises(ValueError, match="no better than chance"):
        _sweep(sensitivity=0.5, specificity=0.5)


def test_the_correction_rescales_the_slope_and_says_by_how_much():
    """Rogan--Gladen is an affine map of the imaging side, so it moves the
    line rather than widening it, and it inflates the variance by the square
    of the same denominator."""
    result = _sweep(sensitivity=0.9604, specificity=0.9812)
    row = result["fit"]

    denominator = 0.9604 + 0.9812 - 1.0
    assert result["corrected"] is True
    assert row["corrected_slope"] == pytest.approx(row["slope"] / denominator)
    assert row["variance_inflation"] == pytest.approx(1.0 / denominator ** 2)
    assert row["variance_inflation"] > 1.0


def test_the_wells_that_trained_the_classifier_are_counted_not_assumed():
    """A classifier scored on the wells it was fitted to agrees with itself,
    so how much of the fit is in-sample has to be visible."""
    result = _sweep()

    # Columns one and two are the training columns by default: twelve of the
    # eighteen wells here.
    assert result["training_wells_in_fit"] == 12
    assert "trained the classifier" in describe(result)


def test_a_share_is_measured_against_what_survived_or_against_everything():
    """`normalise_fraction` on and off differ whenever the threshold removes
    anything, and by more the more it removed."""
    counts = pd.DataFrame([
        {"prc": "plate1_r1_c1", "grna": "a", "count": 60},
        {"prc": "plate1_r1_c1", "grna": "b", "count": 39},
        {"prc": "plate1_r1_c1", "grna": "junk", "count": 1},
    ])

    raw = well_fractions(counts, threshold=0.02, normalise=False)
    assert sorted(raw["fraction"].round(4)) == [0.39, 0.60]

    scaled = well_fractions(counts, threshold=0.02, normalise=True)
    assert scaled["fraction"].sum() == pytest.approx(1.0)
    assert set(scaled["grna"]) == {"a", "b"}


def test_a_well_with_no_reads_at_all_is_not_a_measurement():
    """Dividing by a zero total would make a column of NaN that reads like a
    fraction."""
    counts = pd.DataFrame([
        {"prc": "plate1_r1_c1", "grna": "a", "count": 0},
        {"prc": "plate1_r1_c2", "grna": "a", "count": 10},
    ])

    fractions = well_fractions(counts, threshold=0.0)

    assert list(fractions["prc"]) == ["plate1_r1_c2"]


def test_the_well_is_built_from_the_plate_keys_when_it_has_no_prc():
    """The sweep reads the file the sequencing step wrote, not a prepared
    version of it."""
    counts = pd.DataFrame([
        {"plateID": "plate1", "rowID": "r1", "columnID": "c3",
         "grna": "a", "count": 10},
    ])

    fractions = well_fractions(counts)

    assert list(fractions["prc"]) == ["plate1_r1_c3"]

    with pytest.raises(KeyError, match="missing too"):
        well_fractions(pd.DataFrame([{"grna": "a", "count": 10}]))


def test_a_control_the_threshold_removed_is_reported_as_absent():
    """Zero is the measurement: the threshold decided that guide was not
    there. A missing entry would leave the well out of the fit instead."""
    counts = pd.DataFrame([
        {"prc": "w1", "grna": PC, "count": 1},
        {"prc": "w1", "grna": NC, "count": 999},
    ])

    fractions = well_fractions(counts, threshold=0.02, normalise=True)

    assert reported_control_share(fractions, PC) == {"w1": 0.0}


def test_the_grid_can_return_keeping_everything():
    """"Keep every read" is a real answer and the sweep has to be able to
    give it."""
    assert DEFAULT_THRESHOLD_CANDIDATES[0] == 0.0

    counts, features, wells, pure_pc, pure_nc = _screen(contamination=(0.0,))
    result = sweep_fraction_threshold(
        counts, features, wells, positive_guide=PC,
        pure_pc_wells=pure_pc, pure_nc_wells=pure_nc,
        candidates=(0.0, 0.02, 0.05))

    assert result["chosen"] == 0.0
