"""What a run says about itself when a courtesy step fails or a design is thin.

None of these change what is fitted. They decide what a user is told: that the
design is saturated, that the shuffle they are about to trust is questionable,
that the results folder had to move aside, that a figure could not be drawn.
Each is exercised on the input that makes it speak, because the failure mode of
a courtesy is silence and silence looks exactly like success.
"""
from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from spacr import ml


def long_guide_frame(row_effect=0.0, seed=0):
    """One row per well/guide, with the response constant within a well."""
    rng = np.random.default_rng(seed)
    offsets = {'r1': -row_effect, 'r2': 0.0, 'r3': row_effect}
    records = []
    for plate in ('plate1', 'plate2'):
        for row in ('r1', 'r2', 'r3'):
            for column in ('c1', 'c2', 'c3'):
                well = f'{plate}_{row}_{column}'
                score = offsets[row] + float(rng.normal(0, 0.05))
                for guide in ('g1', 'g2', 'g3'):
                    records.append({
                        'plateID': plate, 'rowID': row, 'columnID': column,
                        'prc': well, 'grna': guide,
                        'fraction': float(rng.uniform(0.05, 0.6)),
                        'score': score,
                    })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# is the fit identifiable at all
# ---------------------------------------------------------------------------

def case_a_table_with_no_well_or_guide_column_gets_no_saturation_warning():
    """A design that cannot be counted produces no warning, not a crash.

    This runs before the fit purely to inform; a KeyError here would stop a run
    the fit itself could have completed.
    """
    _seen = []
    _seen.append(bool(
        ml._identifiability_warning(
        pd.DataFrame({'something': [1, 2]}), {}) is None))
    _seen.append(bool(
        ml._identifiability_warning(
        pd.DataFrame({'prc': ['a', 'b']}), {}) is None))
    return _seen


# ---------------------------------------------------------------------------
# is the shuffle defensible
# ---------------------------------------------------------------------------

def case_a_clean_design_is_reported_as_exchangeable(capsys):
    """With nothing found, the numbers behind that verdict are still printed.

    "Nothing found" has to carry the Durbin-Watson, the well count and the block
    count, or it cannot be told from "the check did not run".
    """
    _seen = []
    frame = long_guide_frame(row_effect=0.0, seed=0)
    report = ml._report_exchangeability(
        frame, 'score',
        {'guide_permutation_block': 'plateID',
         'guide_nuisance_columns': ['rowID', 'columnID']}, None)

    _seen.append(bool(
        report is not None))
    _seen.append(keep(report['n']))
    _seen.append(keep(report['blocks']))
    printed = capsys.readouterr().out
    _seen.append(('Exchangeability: nothing found' if 'Exchangeability: nothing found' in (printed) else ("missing", 'Exchangeability: nothing found')))
    _seen.append(('over 18 well(s) in 2 block(s)' if 'over 18 well(s) in 2 block(s)' in (printed) else ("missing", 'over 18 well(s) in 2 block(s)')))
    return _seen


def case_position_that_explains_the_residual_makes_the_shuffle_questionable(
        capsys):
    """A row effect left in the residual is named, with what to do about it.

    The within-block shuffle treats the residual as noise; structure still in it
    is counted as evidence, and that is where false positives come from.
    """
    _seen = []
    frame = long_guide_frame(row_effect=3.0, seed=2)
    report = ml._report_exchangeability(
        frame, 'score',
        {'guide_permutation_block': 'plateID',
         'guide_nuisance_columns': []}, None)

    printed = capsys.readouterr().out
    _seen.append(('the within-block shuffle is questionable' if 'the within-block shuffle is questionable' in (printed) else ("missing", 'the within-block shuffle is questionable')))
    _seen.append(('rowID explains' if 'rowID explains' in (printed) else ("missing", 'rowID explains')))
    _seen.append(('-> Add rowID to guide_nuisance_columns' if '-> Add rowID to guide_nuisance_columns' in (printed) else ("missing", '-> Add rowID to guide_nuisance_columns')))
    _seen.append(bool(
        report['position']['rowID']['p_value'] < 0.01))
    return _seen


def case_a_check_that_cannot_run_costs_the_report_and_nothing_else():
    """A frame the preparation refuses returns None rather than failing the run.

    The exchangeability report is a courtesy; it must never be the reason a run
    that produced results fails to report them.
    """
    _seen = []
    _seen.append(bool(
        ml._report_exchangeability(
        pd.DataFrame({'prc': ['a'], 'score': [0.1]}), 'score',
        {'guide_permutation_block': 'plateID'}, None) is None))
    return _seen


# ---------------------------------------------------------------------------
# where the results go
# ---------------------------------------------------------------------------

def case_folders_that_cannot_be_listed_are_treated_as_taken(monkeypatch,
                                                            tmp_path):
    """An unreadable candidate is stepped over, and the search still terminates.

    A filesystem that keeps answering "yes, that exists" must not spin; after
    the limit the run takes the last name rather than looping.
    """
    _seen = []
    monkeypatch.setattr(ml.os.path, 'isdir', lambda path: True)

    def unreadable(path):
        raise PermissionError('not allowed to list this folder')

    monkeypatch.setattr(ml.os, 'listdir', unreadable)

    out = ml._next_results_folder(str(tmp_path), 'regression', limit=3)
    _seen.append(bool(
        out == str(tmp_path / 'regression') + '_3'))
    return _seen


def case_an_empty_results_folder_is_used_rather_than_stranded(tmp_path):
    """A folder that exists but holds nothing is not stepped past.

    Somebody made it and did not fill it; skipping it would strand that
    directory forever and start numbering at _1 for no reason.
    """
    _seen = []
    (tmp_path / 'regression').mkdir()
    _seen.append(bool(
        ml._next_results_folder(str(tmp_path), 'regression') == \
        str(tmp_path / 'regression')))
    return _seen


# ---------------------------------------------------------------------------
# controls that resolve to nothing
# ---------------------------------------------------------------------------

def case_controls_that_resolve_to_no_spec_select_no_rows():
    """A control list that names nothing selects an empty frame, not everything.

    An empty selection is what makes the effect-size cut say "no controls";
    falling through to the whole table would calibrate the cut on the screen.
    """
    _seen = []
    frame = pd.DataFrame({'grna': ['g1_1', 'g2_1'],
                          'gene': ['g1', 'g2'],
                          'coefficient': [0.4, -0.2]})
    empty = ml._level_control_rows(frame, 'grna', [''])
    _seen.append(keep(len(empty)))
    _seen.append(bool(
        list(empty.columns) == list(frame.columns)))

    _seen.append(keep(len(ml._level_control_rows(frame, 'grna', []))))
    return _seen


# ---------------------------------------------------------------------------
# where the annotation cache lives
# ---------------------------------------------------------------------------

def case_the_annotation_cache_follows_the_first_source_folder():
    """A list of sources caches beside the first, and no source caches nowhere.

    Every run of the same screen must find the same cache, so it is keyed on
    the folder rather than on the order the caller happened to pass.
    """
    _seen = []
    _seen.append(bool(
        ml._annotation_cache({'src': ['/data/plate1', '/data/plate2']}) == \
        ml.os.path.join('/data/plate1', 'annotation_cache')))
    _seen.append(bool(
        ml._annotation_cache({'src': []}) is None))
    _seen.append(bool(
        ml._annotation_cache({}) is None))
    return _seen


# ---------------------------------------------------------------------------
# hits for a backend with no p-value
# ---------------------------------------------------------------------------

def case_a_penalised_fit_without_its_bootstrap_cannot_call_hits(capsys):
    """Lasso ranks by selection frequency, so the bootstrap has to be passed in.

    Without it there is no p-value to correct, and silently correcting the
    conservative placeholder would produce a hit list from nothing.
    """
    _seen = []
    coef_df = pd.DataFrame({
        'feature': ['fraction:grna[g1_1]'],
        'coefficient': [0.4], 'p_value': [0.01],
        'grna': ['g1_1'], 'condition': ['other'],
    })
    with pytest.raises(ValueError, match='needs perform_regression') as _exc1:
        ml._call_level_hits(coef_df, 'grna', {'nontargeting_control_grnas': None}, 'lasso',
                            pd.DataFrame(), 'score')
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    _seen.append(('no control gRNAs were named' if 'no control gRNAs were named' in (capsys.readouterr().out) else ("missing", 'no control gRNAs were named')))
    return _seen


# ---------------------------------------------------------------------------
# recording what a stage cost
# ---------------------------------------------------------------------------

def case_a_stage_that_cannot_be_measured_is_still_remembered(monkeypatch):
    """With the resource recorder unavailable the stage name is kept in settings.

    The stage is what the failure report names; losing it would leave a
    traceback with nothing saying how far the run got.
    """
    _seen = []
    import spacr.fit_resources as fit_resources

    def boom(settings, name):
        raise RuntimeError('psutil is not installed')

    monkeypatch.setattr(fit_resources, 'record_stage', boom)
    settings = {}
    _seen.append(keep(ml._stage(settings, 'fitting')))
    _seen.append(keep(settings['_regression_stage']))
    return _seen


def case_a_settings_object_that_cannot_be_written_to_costs_only_the_stage(
        monkeypatch):
    """A non-mapping settings object is left alone and an empty record returned.

    Recording where a run got to must never be the thing that ends it.
    """
    _seen = []
    import spacr.fit_resources as fit_resources

    def boom(settings, name):
        raise RuntimeError('psutil is not installed')

    monkeypatch.setattr(fit_resources, 'record_stage', boom)
    _seen.append(keep(ml._stage(object(), 'fitting')))
    return _seen


def case_a_settings_object_that_is_not_a_mapping_still_reports_the_failure(
        monkeypatch, capsys):
    """perform_regression re-raises after reporting, whatever settings is.

    The reporter adds to a failure and must never replace one, so a caller that
    handles a specific exception type keeps seeing it.
    """
    _seen = []
    def boom(settings):
        raise KeyError('score_data')

    monkeypatch.setattr(ml, '_perform_regression', boom)

    with pytest.raises(KeyError, match='score_data') as _exc1:
        ml.perform_regression(object())
    _seen.append(('KeyError' if isinstance(_exc1.value, KeyError) else type(_exc1.value).__name__))
    _seen.append(('THE REGRESSION FAILED' if 'THE REGRESSION FAILED' in (capsys.readouterr().out) else ("missing", 'THE REGRESSION FAILED')))
    return _seen


def case_a_machine_that_can_take_no_reading_says_so_in_the_cost_file(
        monkeypatch, tmp_path):
    """With no peak reading the cost file records that, rather than zero.

    'not measured' is not zero -- psutil absent, or no CUDA tensor allocated --
    and a file of zeros would read as a fit that used no memory.
    """
    _seen = []
    import spacr.fit_resources as fit_resources

    monkeypatch.setattr(fit_resources, 'describe_resources',
                        lambda settings: 'stage  seconds  MB\nfit  1.0  10')
    monkeypatch.setattr(fit_resources, 'peak', lambda settings: {})

    path = ml._write_fit_resources({'res_folder': str(tmp_path)}, {})

    _seen.append(bool(
        path == str(tmp_path / ml.FIT_RESOURCES_FILENAME)))
    text = (tmp_path / ml.FIT_RESOURCES_FILENAME).read_text(encoding="utf-8")
    _seen.append(('No reading could be taken on this machine.' if 'No reading could be taken on this machine.' in (text) else ("missing", 'No reading could be taken on this machine.')))
    _seen.append(('WHAT THIS FIT COST' if 'WHAT THIS FIT COST' in (text) else ("missing", 'WHAT THIS FIT COST')))
    return _seen


# ---------------------------------------------------------------------------
# small reporting helpers
# ---------------------------------------------------------------------------

def case_normality_needs_three_finite_values_and_says_when_it_has_fewer():
    """Fewer than three finite values cannot be tested, and it is reported.

    scipy's own error names an internal argument; this says what the response
    was missing.
    """
    _seen = []
    _seen.append(bool(
        ml.check_normality([1.0, float('nan')], 'score',
                              verbose=True) is False))
    return _seen


def case_a_response_panel_that_raises_is_reported_and_the_run_goes_on(
        monkeypatch, capsys):
    """A rendering failure is printed and the fit is unaffected.

    The panel is a diagnostic; taking the regression down with it would trade
    the expensive half for the cheap half.
    """
    _seen = []
    def boom(*args, **kwargs):
        raise RuntimeError('the scene could not be built')

    monkeypatch.setattr(ml, '_draw_response_panel_in_pyqtgraph', boom)
    frame = pd.DataFrame({'pathogen_rate': [0.1, 0.5, 0.9]})

    ml._show_response_distribution(frame, 'pathogen_rate',
                                   {'plot': True, 'transform': 'none'})

    printed = capsys.readouterr().out
    _seen.append(('the response distribution panel could not be drawn' if 'the response distribution panel could not be drawn' in (printed) else ("missing", 'the response distribution panel could not be drawn')))
    _seen.append(('the run is unaffected' if 'the run is unaffected' in (printed) else ("missing", 'the run is unaffected')))
    return _seen


def case_the_mixed_coefficient_table_drops_the_variance_components():
    """'Group Var' is a variance, not an effect, so it gets no volcano row.

    Left in, it puts a point on the plot that no gene owns and a NaN p-value in
    the hit table's multiple-testing family.
    """
    _seen = []
    model = types.SimpleNamespace(
        params=pd.Series({'fraction:grna[g1_1]': 0.4,
                          'fraction:grna[g2_1]': -0.2,
                          'Group Var': 0.03}),
        pvalues=np.array([0.001, 0.02, float('nan')]))
    design = pd.DataFrame({'fraction:grna[g1_1]': [1.0, 0.0],
                           'fraction:grna[g2_1]': [0.0, 1.0]})

    coef_df = ml.process_model_coefficients(
        model, 'mixed', design, np.array([0.4, 0.2]), nc=None, pc=None,
        controls=None)

    _seen.append(keep(list(coef_df['feature'])))
    _seen.append((("absent", 'Group Var') if 'Group Var' not in (list(coef_df['feature'])) else ("present", 'Group Var')))
    return _seen


def case_an_automatic_glm_family_that_lands_on_poisson_checks_the_counts(
        monkeypatch):
    """A response auto-selected as Poisson is validated before it is fitted.

    ``regression_type='glm'`` picks the family from the response, so the count
    check has to run on that branch too; otherwise a fractional response reaches
    glum as a Poisson and the refusal arrives from inside the solver, naming a
    model the user never chose.
    """
    _seen = []
    import statsmodels.api as sm

    monkeypatch.setattr(ml, 'pick_glm_family_and_link',
                        lambda *a, **k: sm.families.Poisson(
                            link=sm.families.links.Log()))
    design = pd.DataFrame({'Intercept': np.ones(12),
                           'fraction': np.linspace(0.1, 0.6, 12)})
    fractions = np.linspace(0.1, 0.9, 12)

    with pytest.raises(ValueError, match='requires integer count data') as _exc1:
        ml._fit_glum_glm(design, fractions, 'glm',
                         exposure=np.full(12, 100.0))
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_an_explainer_that_will_not_build_yields_nothing(monkeypatch):
    """When every SHAP explainer refuses, the generator produces no explainer.

    Yielding one that cannot run would put the failure inside the caller's loop
    with no way to fall back to the next candidate.
    """
    _seen = []
    import shap

    def boom(*args, **kwargs):
        raise TypeError('the passed model is not callable')

    monkeypatch.setattr(shap, 'Explainer', boom)
    monkeypatch.setattr(shap, 'TreeExplainer', boom)

    model = types.SimpleNamespace(predict=lambda X: np.zeros(len(X)))
    frame = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})

    _seen.append(keep(list(ml._shap_explainers(model, frame))))
    return _seen


from tests.observed_case import keep, observe  # noqa: E402

@pytest.mark.parametrize(("case", "expected"), [
    pytest.param(case_a_clean_design_is_reported_as_exchangeable, [True, 18, 2, 'Exchangeability: nothing found', 'over 18 well(s) in 2 block(s)'], id="a_clean_design_is_reported_as_exchangeable"),
    pytest.param(case_position_that_explains_the_residual_makes_the_shuffle_questionable, ['the within-block shuffle is questionable', 'rowID explains', '-> Add rowID to guide_nuisance_columns', True], id="position_that_explains_the_residual_makes_the_shuffle_questionable"),
    pytest.param(case_a_check_that_cannot_run_costs_the_report_and_nothing_else, [True], id="a_check_that_cannot_run_costs_the_report_and_nothing_else"),
    pytest.param(case_a_stage_that_cannot_be_measured_is_still_remembered, [{}, 'fitting'], id="a_stage_that_cannot_be_measured_is_still_remembered"),
    pytest.param(case_a_settings_object_that_cannot_be_written_to_costs_only_the_stage, [{}], id="a_settings_object_that_cannot_be_written_to_costs_only_the_stage"),
    pytest.param(case_a_settings_object_that_is_not_a_mapping_still_reports_the_failure, ['KeyError', 'THE REGRESSION FAILED'], id="a_settings_object_that_is_not_a_mapping_still_reports_the_failure"),
    pytest.param(case_normality_needs_three_finite_values_and_says_when_it_has_fewer, [True], id="normality_needs_three_finite_values_and_says_when_it_has_fewer"),
    pytest.param(case_the_mixed_coefficient_table_drops_the_variance_components, [['fraction:grna[g1_1]', 'fraction:grna[g2_1]'], ("absent", 'Group Var')], id="the_mixed_coefficient_table_drops_the_variance_components"),
    pytest.param(case_an_automatic_glm_family_that_lands_on_poisson_checks_the_counts, ['ValueError'], id="an_automatic_glm_family_that_lands_on_poisson_checks_the_counts"),
    pytest.param(case_a_table_with_no_well_or_guide_column_gets_no_saturation_warning, [True, True], id="a_table_with_no_well_or_guide_column_gets_no_saturation_warning"),
    pytest.param(case_folders_that_cannot_be_listed_are_treated_as_taken, [True], id="folders_that_cannot_be_listed_are_treated_as_taken"),
    pytest.param(case_an_empty_results_folder_is_used_rather_than_stranded, [True], id="an_empty_results_folder_is_used_rather_than_stranded"),
    pytest.param(case_controls_that_resolve_to_no_spec_select_no_rows, [0, True, 0], id="controls_that_resolve_to_no_spec_select_no_rows"),
    pytest.param(case_the_annotation_cache_follows_the_first_source_folder, [True, True, True], id="the_annotation_cache_follows_the_first_source_folder"),
    pytest.param(case_a_penalised_fit_without_its_bootstrap_cannot_call_hits, ['ValueError', 'no control gRNAs were named'], id="a_penalised_fit_without_its_bootstrap_cannot_call_hits"),
    pytest.param(case_a_machine_that_can_take_no_reading_says_so_in_the_cost_file, [True, 'No reading could be taken on this machine.', 'WHAT THIS FIT COST'], id="a_machine_that_can_take_no_reading_says_so_in_the_cost_file"),
    pytest.param(case_a_response_panel_that_raises_is_reported_and_the_run_goes_on, ['the response distribution panel could not be drawn', 'the run is unaffected'], id="a_response_panel_that_raises_is_reported_and_the_run_goes_on"),
    pytest.param(case_an_explainer_that_will_not_build_yields_nothing, [[]], id="an_explainer_that_will_not_build_yields_nothing"),
])
def test_12_ml_reporting_contracts(request, case, expected):
    """One contract per row: the case records what it checked, the row states it."""
    observed = observe(case, request)
    assert observed == expected
