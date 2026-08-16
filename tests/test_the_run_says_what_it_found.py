"""A finished run ends with a paragraph, not just a table.

Asked for on 2026-08-16: "id also like a little written summary at the end in
the console saying what is significant and so on".

PROSE, NOT A TABLE DUMP. The table is already on screen and already in a CSV.
What is missing is the paragraph a person would write after looking at it:
how many hypotheses were tested, how many survived and under WHICH rule,
which genes, whether the assay worked, and whether the test was conservative
or anti-conservative -- because on a real screen the calibration is routinely
off, and which way it is off decides whether the hit count is an over- or an
undercount.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from spacr.figures.summary import summarise


def _results(n=400, seed=0, with_q=True, controls=True):
    rng = np.random.default_rng(seed)
    effect = rng.normal(0, .3, n)
    effect[:10] += rng.choice([-3.0, 3.0], 10)
    p = rng.uniform(size=n)
    p[:10] = rng.uniform(1e-12, 1e-6, 10)
    frame = pd.DataFrame({
        "feature": [f"fraction:grna[{i // 4}_{i % 4}]" for i in range(n)],
        "coefficient": effect,
        "p_value": p,
        "grna": [f"{i // 4}_{i % 4}" for i in range(n)],
        "gene": [None] * n,
    })
    if with_q:
        frame["q_value"] = np.minimum(frame["p_value"] * 3, 1.0)
    if controls:
        frame["condition"] = list(rng.choice(["nc", "pc", "other"], n,
                                             p=[.06, .02, .92]))
    intercept = {"feature": "Intercept", "coefficient": .19,
                 "p_value": 3e-46, "grna": None, "gene": None}
    if with_q:
        intercept["q_value"] = np.nan
    if controls:
        intercept["condition"] = "other"
    return pd.concat([pd.DataFrame([intercept]), frame], ignore_index=True)


def test_it_says_how_many_were_tested_and_excluded():
    text = summarise(_results())

    assert "400 coefficients were tested" in text
    assert "nuisance" in text and "excluded" in text


def test_it_names_the_correction_it_counted_under():
    """"54 significant" means nothing without the rule that called them."""
    assert "BH q" in summarise(_results())


def test_an_uncorrected_run_says_so_loudly():
    """A q-value column absent is not the same as a correction applied."""
    text = summarise(_results(with_q=False))

    assert "NO correction was applied" in text


def test_it_names_the_hits():
    """A count without names is not something anyone can act on."""
    text = summarise(_results())

    assert "Strongest:" in text
    assert text.count("(") >= 3, text


def test_it_separates_significant_from_worth_following_up():
    """With a thousand wells a trivial effect is significant. The effect-size
    cut is what separates detectable from worth following up, and both
    numbers belong in the sentence."""
    text = summarise(_results())

    assert "effect-size cut" in text
    assert "smaller than the spread of the guides that target nothing" in text


def test_it_reports_the_assay_window():
    """A screen whose controls do not separate has not measured anything,
    however many hits the correction reports."""
    text = summarise(_results())

    assert "Assay window" in text
    assert "σ of the negative spread" in text


def test_no_positive_controls_is_said_rather_than_skipped():
    frame = _results()
    frame.loc[frame.condition == "pc", "condition"] = "other"

    text = summarise(frame)

    assert "nothing to check the assay window against" in text


def test_a_conservative_test_is_called_conservative():
    """The direction changes the reading. Deflation means the hit count is an
    UNDERCOUNT, which is the opposite of the usual worry."""
    frame = _results(n=600, seed=3)
    # p-values skewed high: deflated
    frame["p_value"] = np.clip(frame["p_value"] ** 0.35, 1e-12, 1)
    frame.loc[0, "p_value"] = 3e-46

    text = summarise(frame)

    assert "λ" in text
    assert "conservative" in text.lower() or "deflated" in text.lower(), text


def test_an_inflated_test_is_called_an_upper_bound():
    frame = _results(n=600, seed=4)
    frame["p_value"] = np.clip(frame["p_value"] ** 3.0, 1e-300, 1)

    text = summarise(frame)

    assert "inflated" in text or "upper bound" in text, text


def test_nothing_to_summarise_returns_nothing():
    """The caller says "no results" itself. A paragraph about an absence is
    worse than silence."""
    assert summarise(pd.DataFrame()) == ""
    assert summarise(None) == ""


def test_a_backend_with_no_p_value_says_what_it_ranks_by():
    frame = _results(with_q=False).drop(columns=["p_value"])
    frame["selection_frequency"] = np.linspace(0, 1, len(frame))

    text = summarise(frame)

    assert "selection frequency" in text


def test_it_comes_from_the_same_cut_the_panel_draws():
    """A summary that recomputed the threshold could disagree with the
    picture beside it, and a reader would not know which was wrong."""
    from spacr.figures.panels import control_threshold

    frame = _results()
    rule, cut = control_threshold(frame)

    text = summarise(frame)
    assert rule in text
    assert f"{cut:.3g}" in text
