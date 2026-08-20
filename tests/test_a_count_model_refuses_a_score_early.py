"""A Poisson model told to fit a score says so before it does the work.

Caught in the maintainer's own log on 2026-08-19:

    ValueError: horseshoe (a sparse Poisson GLM over well counts) requires
    integer count data; use a continuous response model for fractional values.

    run closed [failed] in 19.2s

The refusal was correct and the message was good. What was wrong is WHEN: it
came from inside the fit, after both input CSVs were read, the QC tables
written and the diagnostic plots drawn. `process_scores` already carried a
comment saying a continuous score hands a Poisson model a fraction "which
_validate_poisson_response refuses - loudly, but at the very end", so the
combination was known about and still allowed to cost the run.
"""
import numpy as np
import pandas as pd
import pytest


def _scores(values, n_per_well=4):
    """One row per object, several objects per well -- what the run reads."""
    rows = []
    for well, value in enumerate(values):
        for _ in range(n_per_well):
            rows.append({
                "pred": value,
                "plateID": "plate1", "rowID": "r1",
                "columnID": f"c{well + 1}",
                "prc": f"plate1_r1_c{well + 1}",
            })
    return pd.DataFrame(rows)


# HORSESHOE ONLY. `_fit_horseshoe_poisson` validates the response strictly
# and is what raised in the reported run. The statsmodels Poisson path
# tolerates a fractional response (quasi-likelihood), and tests that fit
# `poisson` on a score pass today -- so guarding it here would replace one
# late error with a new early one for a model that works.
# BOTH MODELS. The maintainer reported horseshoe first and poisson right
# after; the earlier narrowing to horseshoe was based on a test file whose
# every parametrisation already fails at HEAD, so it proved nothing.
@pytest.mark.parametrize("model", ["poisson", "horseshoe"])
def test_a_fractional_response_is_refused_before_the_fit(model):
    from spacr.ml import process_scores

    frame = _scores(np.linspace(0.1, 0.9, 12))

    with pytest.raises(ValueError) as excinfo:
        process_scores(frame, "pred", plate="plate1", min_cell_count=1,
                       regression_type=model)

    message = str(excinfo.value)
    assert model in message
    assert "pred" in message
    # IT SAYS WHAT TO DO, both ways out -- the failing run's message named
    # only "use a continuous response model", without naming one.
    assert "ols" in message or "mixed" in message
    # AND IT NAMES THE CAUSE. The message this replaced said only "requires
    # integer count data", which is true and leaves the reader to work out
    # why their counts are fractional: the model sums a per-cell 0/1 LABEL
    # to get the well's positive count, and a classification score is a
    # probability -- 152 cells at ~0.14 sum to 21.68, which counts nothing.
    assert "0/1" in message
    assert "positive COUNT" in message


def test_the_refusal_names_the_offending_number():
    """A message that says "fractional" without a value leaves the reader
    guessing which column and how far off it is."""
    from spacr.ml import process_scores

    # 0.3 x 4 = 1.2 per well. NOT 0.25: four of those sum to exactly 1.0,
    # which is an integer and correctly passes -- the guard judges the well
    # SUM, which is what these models fit.
    frame = _scores([0.3] * 8)

    with pytest.raises(ValueError) as excinfo:
        process_scores(frame, "pred", plate="plate1", min_cell_count=1,
                       regression_type="horseshoe")

    assert "1.2" in str(excinfo.value)    # the per-well sum, 4 x 0.3


def test_a_real_count_still_fits():
    """The guard must not refuse the data these models are FOR."""
    from spacr.ml import process_scores

    rng = np.random.default_rng(0)
    frame = _scores(rng.integers(0, 5, 12).astype(float))

    out, name = process_scores(frame, "pred", plate="plate1",
                               min_cell_count=1, regression_type="horseshoe")

    assert name == "pred"
    assert len(out)


def test_a_continuous_model_is_untouched_by_the_guard():
    """The same fractional response is exactly right for ols."""
    from spacr.ml import process_scores

    frame = _scores(np.linspace(0.1, 0.9, 12))

    out, name = process_scores(frame, "pred", plate="plate1",
                               min_cell_count=1, regression_type="ols")

    assert name == "pred"
    assert len(out)
