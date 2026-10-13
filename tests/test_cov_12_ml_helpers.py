"""spacr.ml's small helpers on the inputs that make them defend themselves.

Each of these sits between a fit and the hit list, so what they do when the
input is not what the happy path assumed decides whether a run reports a wrong
number or refuses. They are exercised directly here because the conditions --
a results object with no inference, a bootstrap where every resample is
one-class, a design that is not dummy-coded -- are ones a whole-pipeline test
cannot arrange.
"""
from __future__ import annotations

import os
import types

import numpy as np
import pandas as pd
import pytest
from sklearn.svm import LinearSVC

from spacr import ml


# ---------------------------------------------------------------------------
# what a mixed fit is going to cost
# ---------------------------------------------------------------------------

def test_the_gpu_alternative_is_offered_when_it_is_usable(monkeypatch, capsys):
    """With the torch backend enabled the notice names it and the setting.

    The statsmodels mixed fit prints nothing for the tens of minutes it runs;
    the one thing worth saying first is that the same estimates are available
    in seconds and how to ask for them.
    """
    monkeypatch.setattr(ml, 'backend_status',
                        lambda name, kind: {'enabled': True, 'reason': ''})
    ml._say_what_a_mixed_fit_will_cost(ml.DEFAULT_REGRESSION_BACKEND,
                                       df=pd.DataFrame({'a': [1, 2, 3]}))

    printed = capsys.readouterr().out
    assert 'on 3 wells' in printed
    assert "regression_backend='torch'" in printed


def test_a_backend_probe_that_raises_leaves_the_notice_without_an_offer(
        monkeypatch, capsys):
    """If the backend cannot be interrogated, the cost is still announced.

    The notice exists so a user does not think the run has hung. Losing it
    because an optional backend's probe raised would cost the whole message
    for the sake of one sentence in it.
    """
    def boom(name, kind):
        raise RuntimeError('backend table is unreadable')

    monkeypatch.setattr(ml, 'backend_status', boom)
    ml._say_what_a_mixed_fit_will_cost(ml.DEFAULT_REGRESSION_BACKEND, df=None)

    printed = capsys.readouterr().out
    assert 'This is the slow one' in printed
    assert 'GPU' not in printed


def test_a_probability_left_unset_is_not_checked_against_the_unit_interval():
    """``None`` means "not answered" and passes, unlike a number out of range.

    Every settings dialog can hand this a None for a box nobody filled;
    refusing that would stop runs over a setting the defaults will supply.
    """
    ml._reject_impossible_probabilities({'fdr_alpha': None})

    with pytest.raises(ValueError, match='outside 0 and 1'):
        ml._reject_impossible_probabilities({'fdr_alpha': 1.05})


# ---------------------------------------------------------------------------
# noticing which figures a helper wrote
# ---------------------------------------------------------------------------

def test_a_directory_named_like_a_figure_is_not_stamped(tmp_path):
    """Only real files are stamped, so a folder called ``x.png`` is skipped.

    The stamps decide which figures get copied into the run folder; treating a
    directory as a figure would make the copy fail for the whole run.
    """
    (tmp_path / 'plate.png').mkdir()
    real = tmp_path / 'sweep.pdf'
    real.write_bytes(b'%PDF-1.4')

    stamps = ml._figure_stamps([str(tmp_path)])
    assert list(stamps) == [str(real)]


def test_a_figure_that_cannot_be_stat_ed_is_skipped_not_fatal(monkeypatch,
                                                              tmp_path):
    """An entry whose stat fails drops out rather than ending the sweep.

    A file being written by another process, or on a mount that went away, must
    cost that one figure and not the run's whole figure collection.
    """
    good = tmp_path / 'good.png'
    good.write_bytes(b'\x89PNG')

    class Hostile:
        name = 'vanished.png'
        path = str(tmp_path / 'vanished.png')

        def is_file(self):
            return True

        def stat(self):
            raise OSError('file went away')

    real_scandir = os.scandir

    def scandir(folder):
        return [Hostile()] + list(real_scandir(folder))

    monkeypatch.setattr(ml.os, 'scandir', scandir)
    stamps = ml._figure_stamps([str(tmp_path)])
    assert list(stamps) == [str(good)]


def test_a_folder_that_cannot_be_listed_is_skipped(tmp_path):
    """A missing figure folder contributes no stamps and no exception.

    The screen-level folders are derived from settings, so one of them not
    existing is ordinary rather than a failure.
    """
    assert ml._figure_stamps([str(tmp_path / 'not-here')]) == {}


# ---------------------------------------------------------------------------
# p-values for results objects that do not carry them
# ---------------------------------------------------------------------------

def test_a_results_object_with_only_standard_errors_gets_a_wald_p_value():
    """Coefficients over standard errors become a two-sided normal p-value.

    The horseshoe fit reports standard errors from a Laplace approximation and
    no test; computing the Wald p-value here is what keeps it in the same hit
    table as every other family.
    """
    model = types.SimpleNamespace(bse=np.array([0.5, 0.0, 1.0]))
    out = ml._statsmodels_p_values(model, np.array([1.0, 2.0, 0.0]))

    assert out.shape == (3,)
    assert out[0] == pytest.approx(2.0 * (1.0 - ml.st.norm.cdf(2.0)))
    # A zero standard error has no test; z is forced to 0, so p is 1.
    assert out[1] == pytest.approx(1.0)
    assert out[2] == pytest.approx(1.0)


def test_a_results_object_with_no_inference_at_all_is_refused():
    """Neither pvalues nor bse means no p-value can be attached, and it says so.

    Inventing a column of ones here would put a whole fit into the hit table
    with p-values nothing computed.
    """
    with pytest.raises(ValueError, match='neither .pvalues nor .bse'):
        ml._statsmodels_p_values(types.SimpleNamespace(),
                                 np.array([1.0, 2.0]))


def test_a_bootstrap_where_every_resample_is_one_class_refuses_to_report():
    """With no fittable resample the bootstrap raises instead of returning 1.0.

    A standard deviation over zero draws would make every p-value exactly 1.0,
    which reads as a clean screen with no hits rather than as inference that
    never happened.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 3))
    y = np.zeros(12)                      # one class, so no resample can fit
    model = LinearSVC()

    with pytest.raises(RuntimeError, match='bootstrap resamples'):
        ml._bootstrap_wald_p_values(model, X, y, n_boot=5, random_state=1)


# ---------------------------------------------------------------------------
# blank settings and absorbed designs
# ---------------------------------------------------------------------------

def test_a_setting_whose_emptiness_cannot_be_tested_counts_as_answered():
    """A value whose ``!=`` is not a truth value is treated as supplied.

    An array in a settings cell is not a threshold, but guessing "blank" for it
    would let a policed setting through unchecked; the honest default is to
    treat it as answered and let the setting's own validation refuse it.
    """
    assert ml._left_blank(None) is True
    assert ml._left_blank('   ') is True
    assert ml._left_blank(float('nan')) is True
    assert ml._left_blank(np.array([1.0, 2.0])) is False


def test_a_design_row_in_two_levels_of_one_factor_cannot_be_absorbed():
    """Overlapping indicator columns are refused, not silently argmax-ed.

    Absorption reads the level back out of the dummies; a row in two levels
    means the columns are not a factor, and projecting them out anyway would
    fit a different model than the one reported.
    """
    design = pd.DataFrame({
        'Intercept': [1.0, 1.0],
        'rowID[T.B]': [1.0, 0.0],
        'rowID[T.C]': [1.0, 1.0],
    })
    with pytest.raises(ValueError, match='more than one'):
        ml._absorbed_factor_codes(design)


# ---------------------------------------------------------------------------
# the absorbed results object
# ---------------------------------------------------------------------------

def absorbed_results():
    params = pd.Series({'Intercept': 1.5, 'fraction': -0.25})
    bse = pd.Series({'Intercept': 0.5, 'fraction': 0.05})
    pvalues = pd.Series({'Intercept': 0.003, 'fraction': 1e-5})
    resid = np.array([0.1, -0.1, 0.05, -0.05])
    fitted = np.array([1.0, 1.2, 1.4, 1.6])
    return ml._AbsorbedLeastSquaresResults(
        params=params, bse=bse, pvalues=pvalues, resid=resid, fitted=fitted,
        scale=0.01, df_model=2, df_resid=1, nobs=4,
        model=None, converged=True, absorbed=('rowID', 'columnID'),
        rsquared=0.87)


def test_an_absorbed_fit_returns_its_own_fitted_values_but_predicts_nothing():
    """In-sample values come back; a new design is refused with the reason.

    The absorbed levels were never estimated, so a prediction for an unseen row
    would be made from an intercept that does not include that row's plate.
    """
    results = absorbed_results()
    assert results.predict() is results.fittedvalues

    with pytest.raises(ValueError) as caught:
        results.predict(np.ones((2, 2)))
    assert 'rowID, columnID' in str(caught.value)
    assert "regression_backend='statsmodels'" in str(caught.value)


def test_an_absorbed_fit_still_writes_a_model_summary():
    """The summary names the absorbed factors and one row per coefficient.

    ``_write_model_summary`` is what a reader opens six months later; an
    absorbed fit with no summary would be the only family in the run with no
    record of what was fitted.
    """
    text = absorbed_results().summary().as_text()

    assert 'Absorbed least squares' in text
    assert 'absorbed factors      rowID, columnID' in text
    assert 'observations          4' in text
    assert 'demeaning converged   True' in text
    assert 'Intercept' in text and 'fraction' in text
    assert text.count('\n') >= 10


def test_the_absorbed_summary_prints_as_the_text_it_reports():
    """``str()`` of the summary is the summary, as statsmodels' is.

    Callers print the object directly; a default repr there would put an
    object address into the run's log where the fit should be.
    """
    summary = absorbed_results().summary()
    assert str(summary) == summary.as_text()
