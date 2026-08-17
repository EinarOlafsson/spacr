"""An effect size with an unstated reference is not interpretable.

Asked for on 2026-08-16: "the user should be able to determine the
intercept". Measured before building it (recorded in instruction 124): moving
the model's reference level shifts the intercept by 0.806 and changes NOT ONE
guide or gene coefficient, because every guide and gene gets its own
coefficient -- they appear only in interaction with a continuous term, so
patsy uses full coding and drops no level.

So the literal control would be a control that changes nothing anybody
reports. What the request is about is the sentence beside it: the guide
coefficients are slopes referenced to ZERO -- "no dose-response" -- and a
reader of a screen figure assumes they are differences from the non-targeting
controls. This says which, and lets the user move it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import baseline


def _frame(control_effect=0.0, n_controls=24, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_controls):
        rows.append({"feature": f"fraction:grna[000000_{i}]",
                     "coefficient": control_effect + rng.normal(0, .2),
                     "p_value": rng.uniform(), "condition": "nc"})
    for i in range(200):
        rows.append({"feature": f"fraction:grna[{411000 + i}_1]",
                     "coefficient": rng.normal(0, .5),
                     "p_value": rng.uniform(), "condition": "other"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  It always says something
# --------------------------------------------------------------------------- #

def test_the_default_baseline_says_what_it_is():
    """Today's default is not "no baseline", it is zero -- and a figure that
    does not say so lets the reader supply their own assumption."""
    chosen = baseline.resolve(_frame())

    assert chosen.kind == baseline.ZERO
    assert "zero" in chosen.sentence
    assert "dose-response" in chosen.sentence


def test_a_request_that_cannot_be_honoured_still_carries_a_sentence():
    """A figure with no baseline sentence at all is the state this exists to
    end, so a failed request comes back as zero WITH its reason rather than
    as an exception the caller may swallow."""
    frame = _frame().drop(columns=["condition"])
    chosen = baseline.resolve(frame, baseline.CONTROLS)

    assert chosen.kind == baseline.ZERO
    assert chosen.sentence
    assert "condition" in chosen.reason


def test_it_never_raises():
    for kind in (baseline.ZERO, baseline.CONTROLS, baseline.NAMED, "banana",
                 None, ""):
        assert baseline.resolve(_frame(), kind).sentence


def test_the_intercept_is_described_rather_than_offered():
    """It is an arbitrary corner of the plate, not a condition anybody chose,
    and moving it changes no effect -- so the figure says what it is instead
    of offering a control that does nothing."""
    sentence = baseline.describe_intercept()

    assert "zero guide fraction" in sentence
    assert "no guide or gene effect" in sentence


# --------------------------------------------------------------------------- #
#  Measuring from the controls
# --------------------------------------------------------------------------- #

def test_the_controls_become_the_zero():
    frame = _frame(control_effect=1.5)
    chosen = baseline.resolve(frame, baseline.CONTROLS)
    shifted = baseline.apply(frame, chosen)

    controls = shifted.loc[shifted["condition"] == "nc", "coefficient"]
    assert abs(float(controls.median())) < 1e-9


def test_it_says_how_many_controls_it_used():
    """"median of 24 control coefficients" is checkable; "controls" is not."""
    chosen = baseline.resolve(_frame(), baseline.CONTROLS)

    assert chosen.n == 24
    assert "24" in chosen.sentence


def test_the_median_not_the_mean():
    """THE REAL FAILURE. `000000_22` -- a non-targeting control -- is the
    strongest effect in this screen at +4.37. A mean baseline would take that
    one guide and shift EVERY effect in the screen by it."""
    frame = _frame(control_effect=0.0)
    frame.loc[frame.index[0], "coefficient"] = 4.37

    chosen = baseline.resolve(frame, baseline.CONTROLS)

    assert abs(chosen.shift) < 0.15, (
        f"one outlying control moved the baseline to {chosen.shift:.3f}; the "
        f"mean of this column is {frame.loc[frame.condition == 'nc', 'coefficient'].mean():.3f}")


def test_too_few_controls_is_refused_with_a_number():
    """A baseline placed on one coefficient is that coefficient's noise,
    applied to the whole screen."""
    chosen = baseline.resolve(_frame(n_controls=1), baseline.CONTROLS)

    assert chosen.kind == baseline.ZERO
    assert "1 non-targeting control" in chosen.reason


def test_a_table_with_no_controls_is_not_a_table_that_cannot_say():
    """Different failures: one has no idea which rows are controls, the other
    knows and there are none. Collapsing them offers a control-based baseline
    on a table that has no idea."""
    frame = _frame()
    frame["condition"] = "other"
    knows = baseline.resolve(frame, baseline.CONTROLS)

    cannot = baseline.resolve(frame.drop(columns=["condition"]),
                              baseline.CONTROLS)

    assert "control coefficient" in knows.reason
    assert "not knowable" in cannot.reason


# --------------------------------------------------------------------------- #
#  Measuring from a named gene
# --------------------------------------------------------------------------- #

def test_a_named_gene_becomes_the_zero():
    frame = _frame()
    frame.loc[frame.index[30], "coefficient"] = 2.0
    name = frame.loc[frame.index[30], "feature"].split("[")[1].rstrip("]")

    chosen = baseline.resolve(frame, baseline.NAMED, name=name)

    assert chosen.kind == baseline.NAMED
    assert chosen.shift == pytest.approx(2.0)
    assert name in chosen.sentence


def test_a_name_that_matches_nothing_says_so():
    chosen = baseline.resolve(_frame(), baseline.NAMED, name="not_a_gene")

    assert chosen.kind == baseline.ZERO
    assert "not_a_gene" in chosen.reason


# --------------------------------------------------------------------------- #
#  What applying it must not do
# --------------------------------------------------------------------------- #

def test_the_run_s_own_table_is_not_shifted_under_it():
    """The caller is a figure and the table is the run's results. Shifting in
    place moves the numbers under the coefficient table, the export and every
    other panel -- each of which then disagrees with its own caption."""
    frame = _frame(control_effect=1.5)
    before = frame["coefficient"].copy()

    baseline.apply(frame, baseline.resolve(frame, baseline.CONTROLS))

    pd.testing.assert_series_equal(frame["coefficient"], before)


def test_the_p_values_are_left_alone():
    """A location shift changes what each effect is measured FROM, not how
    precisely it was estimated. The stars still test against zero, which is
    exactly why the sentence has to be in the caption."""
    frame = _frame(control_effect=1.5)
    chosen = baseline.resolve(frame, baseline.CONTROLS)
    shifted = baseline.apply(frame, chosen)

    pd.testing.assert_series_equal(shifted["p_value"], frame["p_value"])


def test_the_zero_baseline_changes_nothing():
    frame = _frame()
    chosen = baseline.resolve(frame, baseline.ZERO)

    assert chosen.moves is False
    assert baseline.apply(frame, chosen) is frame


def test_the_spacing_between_effects_survives():
    """A baseline moves where zero is. Two guides 0.4 apart are still 0.4
    apart afterwards -- if they were not, this would be a rescaling wearing a
    baseline's name."""
    frame = _frame(control_effect=1.5)
    chosen = baseline.resolve(frame, baseline.CONTROLS)
    shifted = baseline.apply(frame, chosen)

    gaps_before = frame["coefficient"].diff().dropna()
    gaps_after = shifted["coefficient"].diff().dropna()
    np.testing.assert_allclose(gaps_before, gaps_after)
