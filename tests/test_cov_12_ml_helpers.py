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

def case_the_gpu_alternative_is_offered_when_it_is_usable(monkeypatch, capsys):
    """With the torch backend enabled the notice names it and the setting.

    The statsmodels mixed fit prints nothing for the tens of minutes it runs;
    the one thing worth saying first is that the same estimates are available
    in seconds and how to ask for them.
    """
    _seen = []
    monkeypatch.setattr(ml, 'backend_status',
                        lambda name, kind: {'enabled': True, 'reason': ''})
    ml._say_what_a_mixed_fit_will_cost(ml.DEFAULT_REGRESSION_BACKEND,
                                       df=pd.DataFrame({'a': [1, 2, 3]}))

    printed = capsys.readouterr().out
    _seen.append(('on 3 wells' if 'on 3 wells' in (printed) else ("missing", 'on 3 wells')))
    _seen.append(("regression_backend='torch'" if "regression_backend='torch'" in (printed) else ("missing", "regression_backend='torch'")))
    return _seen


def case_a_backend_probe_that_raises_leaves_the_notice_without_an_offer(
        monkeypatch, capsys):
    """If the backend cannot be interrogated, the cost is still announced.

    The notice exists so a user does not think the run has hung. Losing it
    because an optional backend's probe raised would cost the whole message
    for the sake of one sentence in it.
    """
    _seen = []
    def boom(name, kind):
        raise RuntimeError('backend table is unreadable')

    monkeypatch.setattr(ml, 'backend_status', boom)
    ml._say_what_a_mixed_fit_will_cost(ml.DEFAULT_REGRESSION_BACKEND, df=None)

    printed = capsys.readouterr().out
    _seen.append(('This is the slow one' if 'This is the slow one' in (printed) else ("missing", 'This is the slow one')))
    _seen.append((("absent", 'GPU') if 'GPU' not in (printed) else ("present", 'GPU')))
    return _seen


def case_a_probability_left_unset_is_not_checked_against_the_unit_interval():
    """``None`` means "not answered" and passes, unlike a number out of range.

    Every settings dialog can hand this a None for a box nobody filled;
    refusing that would stop runs over a setting the defaults will supply.
    """
    _seen = []
    ml._reject_impossible_probabilities({'fdr_alpha': None})

    with pytest.raises(ValueError, match='outside 0 and 1') as _exc1:
        ml._reject_impossible_probabilities({'fdr_alpha': 1.05})
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


# ---------------------------------------------------------------------------
# noticing which figures a helper wrote
# ---------------------------------------------------------------------------

def case_a_directory_named_like_a_figure_is_not_stamped(tmp_path):
    """Only real files are stamped, so a folder called ``x.png`` is skipped.

    The stamps decide which figures get copied into the run folder; treating a
    directory as a figure would make the copy fail for the whole run.
    """
    _seen = []
    (tmp_path / 'plate.png').mkdir()
    real = tmp_path / 'sweep.pdf'
    real.write_bytes(b'%PDF-1.4')

    stamps = ml._figure_stamps([str(tmp_path)])
    _seen.append(bool(
        list(stamps) == [str(real)]))
    return _seen


def case_a_figure_that_cannot_be_stat_ed_is_skipped_not_fatal(monkeypatch,
                                                              tmp_path):
    """An entry whose stat fails drops out rather than ending the sweep.

    A file being written by another process, or on a mount that went away, must
    cost that one figure and not the run's whole figure collection.
    """
    _seen = []
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
    _seen.append(bool(
        list(stamps) == [str(good)]))
    return _seen


def case_a_folder_that_cannot_be_listed_is_skipped(tmp_path):
    """A missing figure folder contributes no stamps and no exception.

    The screen-level folders are derived from settings, so one of them not
    existing is ordinary rather than a failure.
    """
    _seen = []
    _seen.append(keep(ml._figure_stamps([str(tmp_path / 'not-here')])))
    return _seen


# ---------------------------------------------------------------------------
# p-values for results objects that do not carry them
# ---------------------------------------------------------------------------

def case_a_results_object_with_only_standard_errors_gets_a_wald_p_value():
    """Coefficients over standard errors become a two-sided normal p-value.

    The horseshoe fit reports standard errors from a Laplace approximation and
    no test; computing the Wald p-value here is what keeps it in the same hit
    table as every other family.
    """
    _seen = []
    model = types.SimpleNamespace(bse=np.array([0.5, 0.0, 1.0]))
    out = ml._statsmodels_p_values(model, np.array([1.0, 2.0, 0.0]))

    _seen.append(keep(out.shape))
    _seen.append(bool(
        out[0] == pytest.approx(2.0 * (1.0 - ml.st.norm.cdf(2.0)))))
    # A zero standard error has no test; z is forced to 0, so p is 1.
    _seen.append(bool(
        out[1] == pytest.approx(1.0)))
    _seen.append(bool(
        out[2] == pytest.approx(1.0)))
    return _seen


def case_a_results_object_with_no_inference_at_all_is_refused():
    """Neither pvalues nor bse means no p-value can be attached, and it says so.

    Inventing a column of ones here would put a whole fit into the hit table
    with p-values nothing computed.
    """
    _seen = []
    with pytest.raises(ValueError, match='neither .pvalues nor .bse') as _exc1:
        ml._statsmodels_p_values(types.SimpleNamespace(),
                                 np.array([1.0, 2.0]))
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_a_bootstrap_where_every_resample_is_one_class_refuses_to_report():
    """With no fittable resample the bootstrap raises instead of returning 1.0.

    A standard deviation over zero draws would make every p-value exactly 1.0,
    which reads as a clean screen with no hits rather than as inference that
    never happened.
    """
    _seen = []
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 3))
    y = np.zeros(12)                      # one class, so no resample can fit
    model = LinearSVC()

    with pytest.raises(RuntimeError, match='bootstrap resamples') as _exc1:
        ml._bootstrap_wald_p_values(model, X, y, n_boot=5, random_state=1)
    _seen.append(('RuntimeError' if isinstance(_exc1.value, RuntimeError) else type(_exc1.value).__name__))
    return _seen


# ---------------------------------------------------------------------------
# blank settings and absorbed designs
# ---------------------------------------------------------------------------

def case_a_setting_whose_emptiness_cannot_be_tested_counts_as_answered():
    """A value whose ``!=`` is not a truth value is treated as supplied.

    An array in a settings cell is not a threshold, but guessing "blank" for it
    would let a policed setting through unchecked; the honest default is to
    treat it as answered and let the setting's own validation refuse it.
    """
    _seen = []
    _seen.append(bool(
        ml._left_blank(None) is True))
    _seen.append(bool(
        ml._left_blank('   ') is True))
    _seen.append(bool(
        ml._left_blank(float('nan')) is True))
    _seen.append(bool(
        ml._left_blank(np.array([1.0, 2.0])) is False))
    return _seen


def case_a_design_row_in_two_levels_of_one_factor_cannot_be_absorbed():
    """Overlapping indicator columns are refused, not silently argmax-ed.

    Absorption reads the level back out of the dummies; a row in two levels
    means the columns are not a factor, and projecting them out anyway would
    fit a different model than the one reported.
    """
    _seen = []
    design = pd.DataFrame({
        'Intercept': [1.0, 1.0],
        'rowID[T.B]': [1.0, 0.0],
        'rowID[T.C]': [1.0, 1.0],
    })
    with pytest.raises(ValueError, match='more than one') as _exc1:
        ml._absorbed_factor_codes(design)
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


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


def case_an_absorbed_fit_returns_its_own_fitted_values_but_predicts_nothing():
    """In-sample values come back; a new design is refused with the reason.

    The absorbed levels were never estimated, so a prediction for an unseen row
    would be made from an intercept that does not include that row's plate.
    """
    _seen = []
    results = absorbed_results()
    _seen.append(bool(
        results.predict() is results.fittedvalues))

    with pytest.raises(ValueError) as caught:
        results.predict(np.ones((2, 2)))
    _seen.append(('ValueError' if isinstance(caught.value, ValueError) else type(caught.value).__name__))
    _seen.append(('rowID, columnID' if 'rowID, columnID' in (str(caught.value)) else ("missing", 'rowID, columnID')))
    _seen.append(("regression_backend='statsmodels'" if "regression_backend='statsmodels'" in (str(caught.value)) else ("missing", "regression_backend='statsmodels'")))
    return _seen


def case_an_absorbed_fit_still_writes_a_model_summary():
    """The summary names the absorbed factors and one row per coefficient.

    ``_write_model_summary`` is what a reader opens six months later; an
    absorbed fit with no summary would be the only family in the run with no
    record of what was fitted.
    """
    _seen = []
    text = absorbed_results().summary().as_text()

    _seen.append(('Absorbed least squares' if 'Absorbed least squares' in (text) else ("missing", 'Absorbed least squares')))
    _seen.append(('absorbed factors      rowID, columnID' if 'absorbed factors      rowID, columnID' in (text) else ("missing", 'absorbed factors      rowID, columnID')))
    _seen.append(('observations          4' if 'observations          4' in (text) else ("missing", 'observations          4')))
    _seen.append(('demeaning converged   True' if 'demeaning converged   True' in (text) else ("missing", 'demeaning converged   True')))
    _seen.append(bool(
        'Intercept' in text and 'fraction' in text))
    _seen.append(bool(
        text.count('\n') >= 10))
    return _seen


def case_the_absorbed_summary_prints_as_the_text_it_reports():
    """``str()`` of the summary is the summary, as statsmodels' is.

    Callers print the object directly; a default repr there would put an
    object address into the run's log where the fit should be.
    """
    _seen = []
    summary = absorbed_results().summary()
    _seen.append(bool(
        str(summary) == summary.as_text()))
    return _seen


from tests.observed_case import keep, observe  # noqa: E402

@pytest.mark.parametrize(("case", "expected"), [
    pytest.param(case_a_directory_named_like_a_figure_is_not_stamped, [True], id="a_directory_named_like_a_figure_is_not_stamped"),
    pytest.param(case_a_figure_that_cannot_be_stat_ed_is_skipped_not_fatal, [True], id="a_figure_that_cannot_be_stat_ed_is_skipped_not_fatal"),
    pytest.param(case_a_folder_that_cannot_be_listed_is_skipped, [{}], id="a_folder_that_cannot_be_listed_is_skipped"),
    pytest.param(case_the_gpu_alternative_is_offered_when_it_is_usable, ['on 3 wells', "regression_backend='torch'"], id="the_gpu_alternative_is_offered_when_it_is_usable"),
    pytest.param(case_a_backend_probe_that_raises_leaves_the_notice_without_an_offer, ['This is the slow one', ("absent", 'GPU')], id="a_backend_probe_that_raises_leaves_the_notice_without_an_offer"),
    pytest.param(case_a_probability_left_unset_is_not_checked_against_the_unit_interval, ['ValueError'], id="a_probability_left_unset_is_not_checked_against_the_unit_interval"),
    pytest.param(case_a_results_object_with_only_standard_errors_gets_a_wald_p_value, [(3,), True, True, True], id="a_results_object_with_only_standard_errors_gets_a_wald_p_value"),
    pytest.param(case_a_results_object_with_no_inference_at_all_is_refused, ['ValueError'], id="a_results_object_with_no_inference_at_all_is_refused"),
    pytest.param(case_a_bootstrap_where_every_resample_is_one_class_refuses_to_report, ['RuntimeError'], id="a_bootstrap_where_every_resample_is_one_class_refuses_to_report"),
    pytest.param(case_a_setting_whose_emptiness_cannot_be_tested_counts_as_answered, [True, True, True, True], id="a_setting_whose_emptiness_cannot_be_tested_counts_as_answered"),
    pytest.param(case_a_design_row_in_two_levels_of_one_factor_cannot_be_absorbed, ['ValueError'], id="a_design_row_in_two_levels_of_one_factor_cannot_be_absorbed"),
    pytest.param(case_an_absorbed_fit_returns_its_own_fitted_values_but_predicts_nothing, [True, 'ValueError', 'rowID, columnID', "regression_backend='statsmodels'"], id="an_absorbed_fit_returns_its_own_fitted_values_but_predicts_nothing"),
    pytest.param(case_an_absorbed_fit_still_writes_a_model_summary, ['Absorbed least squares', 'absorbed factors      rowID, columnID', 'observations          4', 'demeaning converged   True', True, True], id="an_absorbed_fit_still_writes_a_model_summary"),
    pytest.param(case_the_absorbed_summary_prints_as_the_text_it_reports, [True], id="the_absorbed_summary_prints_as_the_text_it_reports"),
])
def test_12_ml_helpers_contracts(request, case, expected):
    """One contract per row: the case records what it checked, the row states it."""
    observed = observe(case, request)
    assert observed == expected
