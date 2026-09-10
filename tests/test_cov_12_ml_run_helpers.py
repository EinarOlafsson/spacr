"""Run-level helpers in spacr.ml: what they do with a design they cannot read.

Between the settings and the fit sit a dozen small decisions -- which inference
to use, which nuisance columns exist, whether the score and count tables even
describe the same wells. Getting one of them wrong produces a number rather than
an error, so each is driven here on the input that makes it choose.
"""
from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from spacr import ml, schema


# ---------------------------------------------------------------------------
# how well the fit went
# ---------------------------------------------------------------------------

def test_a_gaussian_fit_with_no_residuals_reports_no_r_squared():
    """A results object missing resid_response says so instead of raising.

    The note is printed after a completed fit; an exception here would lose
    the fit to a line of console decoration.
    """
    model = types.SimpleNamespace(family=sm.families.Gaussian())
    assert ml.fit_quality_note(model) == 'R²: not available for this fit'


def test_a_constant_response_has_no_variance_to_explain():
    """With zero total sum of squares R² is undefined and named as such.

    Dividing by it would give inf or NaN, and "R²: nan" reads as a failed fit
    rather than as a response that never varied.
    """
    model = types.SimpleNamespace(
        family=sm.families.Gaussian(),
        resid_response=np.zeros(4),
        fittedvalues=np.full(4, 0.3))
    assert ml.fit_quality_note(model) == \
        'R²: not available for this fit (the response is constant)'


def test_a_glm_with_no_usable_null_falls_back_and_then_gives_up():
    """A None llnull is stepped over, and a zero null deviance ends the note.

    Dividing by a zero null log-likelihood would report a McFadden R² of inf;
    saying the number is unavailable is the only honest answer.
    """
    model = types.SimpleNamespace(family=sm.families.Binomial(),
                                  llnull=None, llf=-10.0, null_deviance=0.0)
    assert ml.fit_quality_note(model) == \
        "McFadden's R²: not available for this fit"


# ---------------------------------------------------------------------------
# the console summary
# ---------------------------------------------------------------------------

def test_a_summary_with_no_coefficient_table_is_printed_whole():
    """Without a "coef / std err" header there is nothing to trim, so it all goes.

    The compact form exists to keep a hundreds-of-rows table off the terminal;
    a summary with no such table must not be truncated at a guess.
    """
    text = 'Some other model\n================\nconverged: True'
    model = types.SimpleNamespace(summary=lambda: text)
    assert ml.summary_for_console(model) == text


# ---------------------------------------------------------------------------
# well keys
# ---------------------------------------------------------------------------

def test_a_well_key_with_an_empty_row_or_column_is_refused():
    """An empty component identifies no well, so it cannot be a prc.

    Grouping on it would merge every well of the plate into one row and report
    the plate mean as a well measurement.
    """
    with pytest.raises(schema.KeyParseError) as caught:
        ml._split_prc('plate1__c1')
    assert 'every well of' in str(caught.value)

    with pytest.raises(schema.KeyParseError):
        ml._split_prc('plate1_r1_')


# ---------------------------------------------------------------------------
# choosing the inference automatically
# ---------------------------------------------------------------------------

def test_auto_will_not_choose_a_permutation_test_on_per_object_rows():
    """Per-object rows send auto to the simultaneous model, with the reason.

    The permutation test needs one row per well; choosing it for object rows
    would pick a mode that refuses the moment it is used.
    """
    mode, reason = ml.resolve_auto_inference(
        pd.DataFrame({'prc': ['p_r1_c1'], 'grna': ['g1']}),
        {'inference': 'auto', 'analysis_unit': 'object', 'agg_type': 'mean'})
    assert mode == 'regression'
    assert 'one per OBJECT' in reason


def test_a_design_that_cannot_be_measured_gets_the_test_valid_at_any_width():
    """Missing well or guide columns fall back to the permutation test.

    The permutation test stays valid however many guides there are, so it is
    the safe answer when the counts behind the choice cannot be taken.
    """
    mode, reason = ml.resolve_auto_inference(
        pd.DataFrame({'something_else': [1, 2]}),
        {'inference': 'auto', 'analysis_unit': 'well', 'agg_type': 'mean'})
    assert mode == 'guide_permutation'
    assert 'could not be measured' in reason


# ---------------------------------------------------------------------------
# the paired inputs
# ---------------------------------------------------------------------------

def test_paired_data_that_is_not_a_list_of_rows_is_refused():
    """A scalar or a mapping in paired_data is rejected by shape, with a reason.

    A hand-edited settings CSV is the way this arrives; iterating a string here
    would silently pair its characters.
    """
    with pytest.raises(ValueError, match='must be a list of score/count rows'):
        ml.normalize_regression_input_pairs({'paired_data': 'scores.csv'})

    with pytest.raises(ValueError, match=r'paired_data\[1\] must be a mapping'):
        ml.normalize_regression_input_pairs(
            {'paired_data': [{'score': 's.csv', 'count': 'c.csv'},
                             'not a row']})


def test_a_legacy_run_with_no_score_list_is_refused_by_name():
    """Migrating an empty score_data leaves a pair with no score, and it says so.

    The legacy lists are zipped positionally; a missing side would otherwise
    reach the fit as a None path and fail much later.
    """
    with pytest.raises(ValueError, match='at least one score CSV'):
        ml.normalize_regression_input_pairs(
            {'score_data': None, 'count_data': 'counts.csv'})


# ---------------------------------------------------------------------------
# score / count pairing
# ---------------------------------------------------------------------------

def test_tables_with_no_well_column_are_reported_as_sharing_no_plate():
    """A frame with no prc column contributes no plates and no wells.

    The message a user reads names the plates on each side; guessing them from
    a column that is not there would print plate names nothing supports.
    """
    counts = pd.DataFrame({'grna': ['g1'], 'count': [10]})
    scores = pd.DataFrame({'prc': ['plate1_r1_c1'], 'pathogen_rate': [0.5]})
    merged = pd.DataFrame({'prc': []})

    with pytest.raises(ValueError) as caught:
        ml._check_score_count_pairing(counts, scores, merged)
    message = str(caught.value)
    assert 'no well in common' in message
    assert "count wells:   0 on plates []" in message


# ---------------------------------------------------------------------------
# nuisance columns that cannot be used
# ---------------------------------------------------------------------------

def test_a_nuisance_column_collinear_with_the_block_is_dropped_and_said(
        monkeypatch, capsys):
    """A rank-deficient nuisance design drops columns rather than failing the run.

    Row and column are DEFAULTS now; a layout where plate position determines
    the block must not be unrunnable because of a setting nobody chose.
    """
    import spacr.guide_permutation as gp

    def rank_deficient(data, block, columns):
        raise ValueError('the nuisance design is rank deficient')

    monkeypatch.setattr(gp, '_nuisance_design', rank_deficient)
    frame = pd.DataFrame({'plateID': ['p1'] * 4,
                          'rowID': ['r1', 'r1', 'r2', 'r2'],
                          'columnID': ['c1', 'c2', 'c1', 'c2']})

    usable = ml._usable_nuisance_columns(
        frame, {'guide_nuisance_columns': ['rowID', 'columnID']})

    assert usable == []
    printed = capsys.readouterr().out
    assert "'columnID' is collinear" in printed
    assert "'rowID' is collinear" in printed


def test_a_nuisance_design_that_fails_for_another_reason_keeps_its_columns(
        monkeypatch):
    """Only rank deficiency drops a column; any other error stops the pruning.

    An absent block column raises from the same helper, and treating that as
    collinearity threw away nuisance columns that were perfectly good.
    """
    import spacr.guide_permutation as gp

    def other_failure(data, block, columns):
        raise TypeError('the block column is not categorical')

    monkeypatch.setattr(gp, '_nuisance_design', other_failure)
    frame = pd.DataFrame({'rowID': ['r1', 'r2'], 'columnID': ['c1', 'c2']})

    assert ml._usable_nuisance_columns(
        frame, {'guide_nuisance_columns': ['rowID', 'columnID']}) == [
            'rowID', 'columnID']


def test_nuisance_columns_the_table_does_not_have_are_named_out_loud(capsys):
    """Missing columns are reported, because the shuffle then does not remove them.

    A user who believes position was removed and reads a p-value computed
    without removing it has been told something false by omission.
    """
    frame = pd.DataFrame({'plateID': ['p1', 'p1']})
    assert ml._usable_nuisance_columns(
        frame, {'guide_nuisance_columns': ['rowID', 'columnID']}) == []
    printed = capsys.readouterr().out
    assert 'named 2 column(s) this table does not have' in printed
    assert 'rowID, columnID' in printed
