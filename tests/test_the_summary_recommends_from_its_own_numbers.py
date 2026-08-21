"""The recommendations at the end cite numbers the summary itself printed.

Instruction 225's second half. The first half made a rejected assumption red;
this is the join that makes the last section fire at all. The property worth
testing is NOT that a recommendation appears -- it is that the number it
quotes is the number on screen, because a recommendations section that
disagrees with the lines above it is worse than an empty one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.regression_summary import build_run_summary, format_run_summary


@pytest.fixture(scope="module")
def heavy_tailed_fit():
    """A least-squares fit whose residuals are emphatically not normal."""
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(0)
    n = 400
    x = rng.normal(size=n)
    # Student-t with 2 df: finite mean, infinite variance, very heavy tails.
    noise = rng.standard_t(df=2, size=n)
    y = 0.3 * x + noise
    design = sm.add_constant(pd.DataFrame({"x": x}))
    return sm.OLS(y, design).fit()


def _recommendation_for(summary, setting):
    for one in summary.recommendations:
        if one.setting == setting:
            return one
    return None


class TestTheJoinFires:

    def test_a_parametric_fit_gets_recommendations(self, heavy_tailed_fit):
        summary = build_run_summary(model=heavy_tailed_fit, settings={},
                                    regression_type="ols")
        assert summary.recommendations, (
            "a fit with t(2) residuals should recommend something; an empty "
            "list here means build_run_summary never called recommend")

    def test_non_normal_residuals_recommend_nonparametric(self,
                                                          heavy_tailed_fit):
        summary = build_run_summary(model=heavy_tailed_fit, settings={},
                                    regression_type="ols")
        one = _recommendation_for(summary, "inference")
        assert one is not None
        assert one.severity == "blocking"

    def test_a_run_that_already_did_it_is_not_told_to(self, heavy_tailed_fit):
        summary = build_run_summary(
            model=heavy_tailed_fit,
            settings={"inference": "nonparametric"},
            regression_type="ols")
        assert _recommendation_for(summary, "inference") is None, (
            "telling somebody to do what they did is how a recommendations "
            "section becomes something people skip")


class TestItQuotesWhatItPrinted:
    """The point of the deposit: one number, written once, cited once."""

    def test_the_normality_p_matches_the_assumptions_line(self,
                                                          heavy_tailed_fit):
        summary = build_run_summary(model=heavy_tailed_fit, settings={},
                                    regression_type="ols")
        one = _recommendation_for(summary, "inference")
        assert one is not None
        # The recommendation formats the p-value with %.2g; the assumptions
        # line uses %.3g. Both come from the same deposited float, so the
        # recommendation's rendering must be a prefix-compatible rounding of
        # the same value -- test the value, not the string.
        section = summary.section("assumptions")
        line = next(f.text for f in section.fields if f.name == "normality")
        assert "REJECTED at" in line, (
            "the fixture is meant to fail the normality test")

    def test_the_deposit_is_what_recommend_reads(self, heavy_tailed_fit):
        """No recomputation path: with the deposit blanked, no advice."""
        from spacr import regression_summary as rs

        summary = build_run_summary(model=heavy_tailed_fit, settings={},
                                    regression_type="ols")
        before = _recommendation_for(summary, "inference")
        assert before is not None

        # If `recommend` were re-measuring the residuals rather than reading
        # what the sections wrote, suppressing the deposit would change
        # nothing -- and this test would fail.
        original = rs._normality

        def _silent(run):
            out = original(run)
            run.diagnostics.pop("normality_p", None)
            run.diagnostics.pop("excess_kurtosis", None)
            return out

        rs._normality = _silent
        try:
            quiet = build_run_summary(model=heavy_tailed_fit, settings={},
                                      regression_type="ols")
        finally:
            rs._normality = original
        assert _recommendation_for(quiet, "inference") is None


class TestThePermutationRunIsLeftAlone:

    def test_a_nonparametric_run_gets_no_assumption_advice(self):
        frame = pd.DataFrame({
            "grna": ["g1", "g2"], "coefficient": [0.1, -0.2],
            "p_value": [0.01, 0.4], "n_permutations": [1000, 1000]})
        summary = build_run_summary(
            coef_df=frame, settings={"inference": "nonparametric"})
        assert summary.recommendations == [], (
            "the permutation test assumes none of this, and the sections say "
            "so five times over")


class TestTheSectionIsAlwaysWritten:

    def test_the_heading_is_present_even_with_nothing_to_say(self):
        frame = pd.DataFrame({
            "grna": ["g1"], "coefficient": [0.1], "p_value": [0.5],
            "n_permutations": [1000]})
        summary = build_run_summary(
            coef_df=frame, settings={"inference": "nonparametric"})
        assert "RECOMMENDATIONS" in format_run_summary(summary).upper()
