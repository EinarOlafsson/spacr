"""The Scale-location tab answers its own question, and names the fix.

Instruction 128 M:

    "HOMOGENEITY OF VARIANCE -- scale-location is the plot for it, and it is
     there, but the QUESTION it answers is not asked anywhere: is the residual
     spread constant across the fitted range? A trend in that plot is
     heteroscedasticity, and OLS standard errors are wrong when it is present
     -- which changes every p-value in the table."

The tab shipped the picture. A reader who cannot separate a flat cloud from a
funnel by eye over 300 points -- which is most readers -- had no way to learn
that every standard error in the Summary tab, and so every p-value in the
coefficient table and every height on the volcano, was optimistic.

NOT ONE STATISTIC IS RE-DERIVED HERE. `spacr.regression_qc` already computes
them for the PDF the run writes beside the results, and it computes TWO
because one is not enough: a rank correlation is exactly zero for a SYMMETRIC
funnel, which is what a mis-specified link produces and which is common in
this pipeline. The live tab reads that module's own dict through
`draw_panel`, so the tab and the PDF cannot describe one fit two ways. There
is a test below that asserts exactly that, because it is the only property
that makes the number on screen worth trusting.

AND IT NAMES THE FIX. "the standard errors are optimistic" with no remedy is
a warning a reader can only act on by guessing. `cov_type='HC3'` is a
sandwich estimator valid under heteroscedasticity and is already a spaCR
setting for ols, wls, glm, poisson, quasi_binomial, logit and probit.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
sm = pytest.importorskip("statsmodels.api")

pytestmark = pytest.mark.qt

WELLS = 300


def _design(seed: int = 7):
    rng = np.random.default_rng(seed)
    design = pd.DataFrame({"Intercept": np.ones(WELLS),
                           "a": rng.random(WELLS),
                           "b": rng.random(WELLS)})
    return design, rng


def _clean():
    """Constant spread: the residual SD does not depend on the fitted value."""
    design, rng = _design()
    return design, 0.3 + 1.2 * design["a"] + rng.normal(scale=0.2, size=WELLS)


def _megaphone():
    """The spread grows with the fit -- the classic heteroscedastic screen."""
    design, rng = _design()
    return design, 0.3 + 1.2 * design["a"] + rng.normal(
        scale=0.05 + 1.5 * design["a"], size=WELLS)


def _funnel():
    """Wide at both ends, narrow in the middle. Spearman's rho is ~0 on this
    and a panel that stopped at rho would print "no trend in spread" over
    it -- a confident wrong answer, which is why regression_qc runs a
    Brown-Forsythe test as well."""
    design, rng = _design()
    return design, 0.3 + 1.2 * design["a"] + rng.normal(
        scale=0.05 + 2.0 * np.abs(design["a"] - 0.5), size=WELLS)


@pytest.fixture()
def panel(qtbot):
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    widget = RegressionResultsPanel()
    qtbot.addWidget(widget)
    return widget


def _fit(maker, **kwargs):
    design, y = maker()
    return sm.OLS(y, design).fit(**kwargs)


# --------------------------------------------------------------------------- #
#  The verdict is on the tab, beside the picture
# --------------------------------------------------------------------------- #

def test_a_constant_spread_is_said_out_loud(panel):
    """"nothing is wrong" is a finding too, and a blank space under a plot is
    not the same claim."""
    panel.set_diagnostics(_fit(_clean), regression_type="ols")

    verdict = panel.homogeneity_verdict()
    assert "CONSTANT SPREAD" in verdict, verdict
    assert "are the right ones" in verdict, verdict


def test_a_growing_spread_says_the_standard_errors_are_optimistic(panel):
    """The sentence that changes what a reader does: it is not "the plot
    slopes", it is "every p-value in the table is smaller than it should
    be"."""
    panel.set_diagnostics(_fit(_megaphone), regression_type="ols")

    verdict = panel.homogeneity_verdict()
    assert "SPREAD GROWS WITH THE FITTED VALUE" in verdict, verdict
    assert "OPTIMISTIC" in verdict, verdict
    assert "every p-value in the coefficient table" in verdict, verdict


def test_the_verdict_names_hc3_as_the_fix(panel):
    """A warning with no remedy is a warning a reader acts on by guessing.
    cov_type is already a spaCR setting -- see spacr/settings.py:3383 -- so
    this is a re-fit, not a feature request."""
    panel.set_diagnostics(_fit(_megaphone), regression_type="ols")

    verdict = panel.homogeneity_verdict()
    assert "cov_type='HC3'" in verdict, verdict
    assert "sandwich" in verdict, verdict
    assert "leaving the coefficients exactly where they are" in verdict, (
        "the verdict does not say what HC3 changes and what it does not")


def test_a_constant_spread_is_not_told_to_re_fit(panel):
    """Advice to reach for a robust estimator when nothing is wrong is how a
    panel teaches its reader to ignore it."""
    panel.set_diagnostics(_fit(_clean), regression_type="ols")

    assert "HC3" not in panel.homogeneity_verdict()


def test_the_verdict_is_visible_on_the_tab_not_only_readable_in_python(panel):
    """It is a label under the plot rather than the plot's status line: a
    status line is overwritten by whatever was last clicked, and a verdict
    that vanishes when the user interacts with the panel is one they have not
    read."""
    panel.set_diagnostics(_fit(_megaphone), regression_type="ols")
    panel.show()
    panel.tabs.setCurrentIndex(panel.tabs.indexOf(panel._scale_location_tab))

    assert panel.homogeneity.text() == panel.homogeneity_verdict()
    assert panel.homogeneity.isVisible(), (
        "the verdict is not on screen when the Scale-location tab is open")


def test_the_scale_location_tab_is_still_called_that(panel):
    """The verdict was added by wrapping the plot in a container, and a tab
    that lost its name -- or that showed the verdict INSTEAD of the picture --
    would be a worse tab than the one it replaced."""
    names = [panel.tabs.tabText(i) for i in range(panel.tabs.count())]
    assert "Scale-location" in names, names

    panel.show()
    panel.tabs.setCurrentIndex(panel.tabs.indexOf(panel._scale_location_tab))

    assert panel.scale_location.isVisible(), "the plot itself is gone"
    assert panel.homogeneity.isVisible(), "the verdict is not beside it"


# --------------------------------------------------------------------------- #
#  The numbers are regression_qc's, not this panel's
# --------------------------------------------------------------------------- #

def test_the_numbers_are_the_saved_panel_s_numbers_exactly(panel):
    """The whole reason `judge_homogeneity` draws into a throwaway axes
    instead of computing a rank correlation here. Two implementations of
    "is the spread constant" disagree within a week, and the reader would have
    no way to tell which of the tab and the PDF beside it was wrong."""
    from matplotlib.figure import Figure
    from spacr.regression_qc import context_from_model, draw_panel

    model = _fit(_megaphone)
    panel.set_diagnostics(model, regression_type="ols")

    ctx = context_from_model(model, regression_type="ols")
    saved = draw_panel("scale_location", ctx, Figure().add_subplot(111))

    assert panel.homogeneity_stats() == saved, (
        panel.homogeneity_stats(), saved)


def test_the_numbers_are_quoted_beside_the_verdict(panel):
    """A verdict with no numbers under it cannot be checked against the PDF,
    and both of them exist so that it can be."""
    panel.set_diagnostics(_fit(_megaphone), regression_type="ols")
    stats = panel.homogeneity_stats()

    verdict = panel.homogeneity_verdict()
    assert f"Spearman rho = {stats['spearman_rho']:+.2f}" in verdict, verdict
    assert f"max/min quartile SD = {stats['quartile_sd_ratio']:.2f}" in verdict
    assert f"over {stats['n_points']} wells" in verdict, verdict
    assert "regression_qc/scale_location" in verdict, (
        "the verdict does not say where the same numbers are on disk")


def test_a_symmetric_funnel_is_caught_where_a_rank_correlation_is_not(panel):
    """The case that makes two statistics necessary. Spread is large at both
    ends and small in the middle, so Spearman's rho is ~0; a panel that
    stopped there would print "no trend in spread" over a fit whose standard
    errors are wrong everywhere."""
    panel.set_diagnostics(_fit(_funnel), regression_type="ols")

    stats = panel.homogeneity_stats()
    assert abs(stats["spearman_rho"]) < 0.3, (
        "the fixture is no longer a symmetric funnel")
    assert stats["levene_p"] < 0.01, stats
    verdict = panel.homogeneity_verdict()
    assert "SPREAD IS NOT CONSTANT" in verdict, verdict
    assert "funnel rather than a slope" in verdict, verdict
    assert "cov_type='HC3'" in verdict, verdict


# --------------------------------------------------------------------------- #
#  A fit that already did the right thing is not told to do it again
# --------------------------------------------------------------------------- #

def test_a_fit_already_using_hc3_is_told_its_errors_are_already_robust(panel):
    """The picture is identical -- HC3 changes the standard errors, not the
    residuals -- and it means something different. Telling this reader their
    p-values are optimistic would be false, and telling them to re-fit with
    HC3 would be advice to repeat what they did."""
    panel.set_diagnostics(_fit(_megaphone, cov_type="HC3"),
                          regression_type="ols")

    verdict = panel.homogeneity_verdict()
    assert "SPREAD GROWS WITH THE FITTED VALUE" in verdict, verdict
    assert "ALREADY robust" in verdict, verdict
    assert "OPTIMISTIC" not in verdict, verdict
    assert "re-fit with cov_type" not in verdict, verdict


# --------------------------------------------------------------------------- #
#  A fit that cannot be judged says why, rather than looking broken
# --------------------------------------------------------------------------- #

def test_no_model_at_all_says_what_it_is_waiting_for(panel):
    said = panel.homogeneity_verdict()

    assert "only a run in this session" in said, said
    assert panel.homogeneity_stats() == {}


def test_a_quantile_fit_says_it_has_no_error_scale(panel):
    """Quantile regression estimates the median of the response, not its
    mean, so it has no error-variance parameter and no standardised residual
    to judge. "this fit cannot answer that" and "this tab is broken" must not
    look the same."""
    design, y = _clean()

    panel.set_diagnostics(sm.QuantReg(y, design).fit(q=0.5),
                          regression_type="quantile")

    said = panel.homogeneity_verdict()
    assert said.startswith("No constant-spread verdict:"), said
    assert "quantile" in said.lower(), said
    assert panel.homogeneity_stats() == {}


def test_a_penalised_fit_says_why_it_cannot_be_judged(panel):
    """sklearn's Lasso keeps neither design nor response, so nothing can be
    recomputed from it."""
    Lasso = pytest.importorskip("sklearn.linear_model").Lasso
    design, y = _clean()

    assert panel.set_diagnostics(
        Lasso(alpha=0.1).fit(design.to_numpy(), y.to_numpy())) is False

    said = panel.homogeneity_verdict()
    assert "does not keep the design matrix" in said, said


def test_a_new_table_takes_the_old_fit_s_verdict_away(panel):
    """A sentence about the previous fit sitting under an empty plot is the
    worst of both: authoritative, and about nothing."""
    panel.set_diagnostics(_fit(_megaphone), regression_type="ols")
    assert "SPREAD GROWS" in panel.homogeneity_verdict()

    panel.set_frame(pd.DataFrame({"feature": ["fraction:grna[a_1]"],
                                  "coefficient": [0.5],
                                  "p_value": [0.01]}))

    assert "SPREAD GROWS" not in panel.homogeneity_verdict()
    assert panel.homogeneity_stats() == {}
    assert "only a run in this session" in panel.homogeneity_verdict()


def test_the_tab_tooltip_carries_the_verdict_too(panel):
    """The tab bar is where a reader decides which tab is worth opening."""
    panel.set_diagnostics(_fit(_megaphone), regression_type="ols")

    index = panel.tabs.indexOf(panel._scale_location_tab)

    assert panel.tabs.tabToolTip(index) == panel.homogeneity_verdict()


def test_asking_again_re_reads_the_fit_the_tabs_were_drawn_from(panel):
    """`judge_homogeneity()` with no argument re-judges the stored context, so
    a caller does not have to keep its own copy of it to ask twice -- and two
    copies of a QC context is how the tab and the PDF start to disagree."""
    panel.set_diagnostics(_fit(_megaphone), regression_type="ols")
    first = panel.homogeneity_stats()

    assert panel.judge_homogeneity() == panel.homogeneity_verdict()
    assert panel.homogeneity_stats() == first


def test_asking_with_no_fit_at_all_is_a_sentence_not_a_crash(panel):
    assert panel.judge_homogeneity() == panel.NO_HOMOGENEITY_VERDICT


def test_a_context_the_panel_cannot_read_says_so_instead_of_raising(panel):
    """NEVER RAISES INTO THE GUI. The diagnostics module is free to change
    what a context carries; a panel that answered that with a traceback would
    take the window down over a tab the user may not even have opened."""
    said = panel.judge_homogeneity(object())

    assert said.startswith("No constant-spread verdict:"), said
    assert "could not be computed" in said, said
    assert "AttributeError" in said, said
    assert panel.homogeneity_stats() == {}


def test_a_verdict_this_panel_has_no_sentence_for_is_still_shown(panel,
                                                                monkeypatch):
    """regression_qc owns the verdict strings and may grow one. A panel that
    silently printed nothing for an unrecognised verdict would turn a new
    finding into an empty tab -- so the raw verdict and its numbers are shown,
    with the fix, and the reader is told the wording is missing."""
    import spacr.regression_qc as rq

    real = rq.draw_panel

    def invented(name, ctx, ax):
        stats = real(name, ctx, ax)
        stats["verdict"] = "the spread is stepwise across plates"
        return stats

    monkeypatch.setattr(rq, "draw_panel", invented)

    panel.set_diagnostics(_fit(_megaphone), regression_type="ols")

    said = panel.homogeneity_verdict()
    assert "the spread is stepwise across plates" in said, said
    assert "has no sentence for" in said, said
    assert "cov_type='HC3'" in said, said
    assert "Brown-Forsythe p" in said, said
