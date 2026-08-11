"""`level='plate'` was a documented value that fell through to object pooling.

The setting's tooltip offers three levels -- "'object' pools every parasite
into one bar per condition, 'well' averages the per-well proportions and
draws SD whiskers, 'plate' does the same across plates". The code checked
``level in ['well', 'plateID']``.

So 'plate', the spelling the tooltip uses, matched neither and fell to the
else: every object in one bar per condition, no per-plate averaging, no SD
whiskers. A different figure answering a different question, with nothing on
screen to say the requested level had been ignored.

An unrecognised level now raises instead of pooling, because falling back to
'object' is precisely what made the typo invisible.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from spacr.plot import plot_proportion_stacked_bars


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    rows = []
    for plate in ("p1", "p2", "p3"):
        for well in range(1, 4):
            for _ in range(200):
                rows.append({"condition": rng.choice(["nc", "pc"]),
                             "cls": int(rng.random() < 0.5),
                             "prc": f"{plate}_r1_c{well}", "plateID": plate})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("level", ["object", "well", "plate", "plateID"])
def test_every_documented_level_runs(level, frame):
    results, _pairwise, _fig = plot_proportion_stacked_bars(
        {"verbose": False}, frame, "condition", bin_column="cls", level=level)
    assert not results.empty


@pytest.mark.parametrize("level", ["PLATE", " plate ", "Well"])
def test_the_level_is_read_case_and_space_tolerantly(level, frame):
    """A settings CSV round-trips whatever the user typed."""
    results, _pairwise, _fig = plot_proportion_stacked_bars(
        {"verbose": False}, frame, "condition", bin_column="cls", level=level)
    assert not results.empty


def test_plate_aggregates_across_plates_not_across_objects(frame):
    """The regression, told apart from 'it ran'.

    Object pooling draws one bar per condition with no spread; plate level
    averages three per-plate proportions and has SD whiskers. The figure is
    what differs, so the figure is what this reads.
    """
    _r, _p, fig_object = plot_proportion_stacked_bars(
        {"verbose": False}, frame, "condition", bin_column="cls",
        level="object")
    _r, _p, fig_plate = plot_proportion_stacked_bars(
        {"verbose": False}, frame, "condition", bin_column="cls",
        level="plate")

    def has_error_bars(fig):
        return any(len(ax.containers) and any(
            getattr(c, "has_yerr", False) for c in ax.containers)
            for ax in fig.axes)

    assert not has_error_bars(fig_object), "object level should have no spread"
    assert has_error_bars(fig_plate), (
        "plate level drew no SD whiskers, so it pooled objects instead of "
        "averaging plates")


@pytest.mark.parametrize("level", ["welll", "plates", "field", ""])
def test_an_unrecognised_level_is_refused_by_name(level, frame):
    """Silently pooling is what made 'plate' invisible for so long."""
    if level == "":
        pytest.skip("empty falls back to 'object' by documented default")
    with pytest.raises(ValueError, match="is not one of"):
        plot_proportion_stacked_bars({"verbose": False}, frame, "condition",
                                     bin_column="cls", level=level)


def test_the_chi_squared_does_not_follow_the_level(frame):
    """DOCUMENTS A DEFECT THAT IS STILL OPEN, so it cannot be forgotten.

    The tooltip says "the reported statistics always treat the well as the
    unit of replication". They do not: the chi-squared is computed on object
    counts before the level branch is reached, so it is byte-identical at
    every level. Pinned as the current behaviour rather than asserted as
    correct -- see instruction 77. When the statistic is moved to the
    declared level, this test fails and should be rewritten, not deleted.
    """
    stats = []
    for level in ("object", "well", "plate"):
        results, _pairwise, _fig = plot_proportion_stacked_bars(
            {"verbose": False}, frame, "condition", bin_column="cls",
            level=level)
        stats.append(float(results["chi_squared_stat"].iloc[0]))

    assert len(set(stats)) == 1, (
        "the chi-squared now varies with level -- if that was deliberate, "
        "instruction 77's item is done and this test needs rewriting")


def test_plate_level_needs_a_plate_column_and_says_which(frame):
    """'plateID' used to group by `prc`, so it never needed a plate column.

    That is the defect from the other side: a plate-level request that
    averaged WELLS ran happily on a table with no plates in it. Now it
    refuses, and names the column it wanted rather than raising KeyError
    from inside a groupby.
    """
    no_plates = frame.drop(columns=["plateID"])
    with pytest.raises(ValueError, match="groups by 'plateID'"):
        plot_proportion_stacked_bars({"verbose": False}, no_plates,
                                     "condition", bin_column="cls",
                                     level="plate")
