"""The edges of :mod:`spacr.ml`: what each step does when it is told less.

Every test here drives one path through the regression module that an
ordinary run reaches only with a particular shape of input -- an IPython that
is mid-import, a glob that matched no count file, a design whose covariate has
two values, a group lasso that ran out of sweeps, a screen with no plate
column, a permutation asked for a threshold of zero wells, a fit whose
coefficients carry no p-value at all.

Those are states real runs arrive in, and each of them used to be the
difference between a number and a lost run.  The file is ordered the way the
module is: import, the small settings helpers, the model dispatcher, the
reporting helpers, the run itself, then the scoring and drawing tails.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from spacr import ml  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


# ===========================================================================
# import time: the two decisions the module makes before anything is called
# ===========================================================================

def _load_ml_afresh(name='spacr._ml_import_probe'):
    """Execute ``spacr/ml.py`` again under another module name.

    The two arcs below are import-time ones, so the only way to drive them is
    to run the module body again with the environment they answer to.  A
    second module object is used rather than ``importlib.reload`` so the
    ``spacr.ml`` every other test holds is left exactly as it was; coverage
    still attributes the lines to ``spacr/ml.py`` because that is the file
    executed.
    """
    spec = importlib.util.spec_from_file_location(name, ml.__file__)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def test_an_ipython_that_cannot_be_imported_leaves_a_display_that_does_nothing(
        monkeypatch):
    """``display`` stays callable when IPython is mid-init.

    spaCR imports :mod:`spacr.ml` from Qt worker threads while a notebook
    front end may be part-way through importing IPython itself.  The fallback
    is what stops that race from turning an import into an exception, and it
    has to accept the same arguments the real one does.
    """
    # None in sys.modules is exactly what a half-finished import looks like
    # to the import system: `from IPython.display import display` raises.
    monkeypatch.setitem(sys.modules, 'IPython.display', None)

    module = _load_ml_afresh()

    assert module.display is not ml.display, (
        'the module bound the real IPython display, so the fallback that '
        'makes the import survivable never ran')
    assert module.display('a frame', extra=1) is None


def test_a_machine_with_nowhere_to_draw_is_demoted_to_agg(monkeypatch):
    """No DISPLAY and no Windows/macOS GUI means the Agg backend, once.

    Importing the module on a headless Linux box with an interactive backend
    still selected dies inside Matplotlib the first time a run draws, hours
    in.  The demotion is conditional because doing it unconditionally is what
    used to kill inline plotting in notebooks.
    """
    chosen = []
    monkeypatch.setattr(matplotlib, 'use', lambda *a, **k: chosen.append(a))
    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.delenv('DISPLAY', raising=False)

    _load_ml_afresh('spacr._ml_backend_probe')
    assert chosen == [('Agg',)]

    # ... and with somewhere to draw, the user's backend is left alone.
    chosen.clear()
    monkeypatch.setenv('DISPLAY', ':0')
    _load_ml_afresh('spacr._ml_backend_probe')
    assert chosen == []


# ===========================================================================
# the settings helpers
# ===========================================================================

def test_a_glob_that_matched_no_count_file_is_refused_by_name(tmp_path):
    """An empty generator of paths is 'nothing to read', not an empty frame.

    ``paths`` is truthy the moment it is an iterator, so the guard at the top
    cannot see that a caller's glob matched nothing; the second check is the
    one that catches it, and without it the run continued on an empty
    concatenation.
    """
    one = tmp_path / 'counts.csv'
    pd.DataFrame({'grna': ['g1'], 'count': [3]}).to_csv(one, index=False)

    with pytest.raises(ValueError, match='no table was given to read'):
        ml._concat_named_csvs(iter([]))

    # The same call shape with a match reads the file, so the refusal above
    # is about the emptiness and not about the iterator.
    assert len(ml._concat_named_csvs(iter([str(one)]))) == 1


def test_a_calibration_whose_module_is_missing_keeps_the_threshold_given(
        monkeypatch, capsys):
    """An unimportable sweep is an answer, not a reason to stop the run.

    ``fraction_calibration`` is optional at this point in its life; a run that
    ticked the box and cannot have it is owed the sentence saying so and the
    threshold it already had.
    """
    monkeypatch.setitem(sys.modules, 'spacr.fraction_calibration', None)

    assert ml._calibrated_fraction_threshold({'fraction_threshold': 0.01}) is None
    assert 'calibration is unavailable' in capsys.readouterr().out

    # With the module importable the same settings produce a number, so the
    # None above is the missing module and nothing else.
    monkeypatch.undo()
    import spacr.fraction_calibration as calibration

    monkeypatch.setattr(ml, '_calibration_inputs', lambda settings: {})
    monkeypatch.setattr(calibration, 'sweep_fraction_threshold',
                        lambda **kwargs: {'chosen': 0.02, 'candidates': []})
    assert ml._calibrated_fraction_threshold({}) == pytest.approx(0.02)


def test_a_column_that_is_already_categorical_is_left_as_it_is():
    """``check_and_clean_data`` re-categorises only what is not categorical.

    Converting an existing categorical again drops the categories nobody in
    this frame uses, and those are what keep two plates' factor levels lined
    up when a well is filtered out.
    """
    frame = pd.DataFrame({
        'fraction': [0.4, 0.6, 0.5, 0.5],
        'predictions': [0.1, 0.9, 0.2, 0.8],
        'grna': pd.Categorical(['g1', 'g2', 'g1', 'g2'],
                               categories=['g1', 'g2', 'g3']),
        'gene': ['geneA', 'geneB', 'geneA', 'geneB'],
        'plateID': ['plate1'] * 4,
        'rowID': ['r1', 'r1', 'r2', 'r2'],
        'columnID': ['c1', 'c2', 'c1', 'c2'],
        'prc': ['plate1_r1_c1', 'plate1_r1_c2',
                'plate1_r2_c1', 'plate1_r2_c2'],
    })

    cleaned = ml.check_and_clean_data(frame, 'predictions')

    # untouched: the unused 'g3' level survived the pass
    assert list(cleaned['grna'].cat.categories) == ['g1', 'g2', 'g3']
    # converted: 'gene' arrived as object and left as a categorical
    assert isinstance(cleaned['gene'].dtype, pd.CategoricalDtype)


# ===========================================================================
# the model dispatcher
# ===========================================================================

def _tiny_design(n=60, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, n)
    X = pd.DataFrame({'Intercept': 1.0, 'x': x})
    y = pd.Series(0.5 + 2.0 * x + rng.normal(0, 0.2, n))
    return X, y


def test_a_transform_that_is_the_link_does_not_get_a_second_one(capsys):
    """``glm_force_identity`` fits Gaussian/identity and says which response.

    The caller has already put the response on the transformed scale, so a
    family chosen from the transformed numbers would apply the link twice and
    every coefficient downstream would be on a scale nothing else reads.
    """
    X, y = _tiny_design()

    model = ml.regression_model(X, y, regression_type='glm',
                                glm_force_identity=True,
                                response_name='logit(pred)')

    import statsmodels.api as sm

    assert isinstance(model.family, sm.families.Gaussian)
    assert isinstance(model.family.link, sm.families.links.Identity)
    assert float(model.params['x']) == pytest.approx(2.0, abs=0.15)
    assert 'Identity link for logit(pred)' in capsys.readouterr().out


def test_a_covariate_with_two_values_is_not_given_a_spline_basis():
    """An indicator has nothing to bend through, so it is left alone.

    Expanding it would spend knots and degrees of freedom on a column with
    one step in it, and the guide coefficients that share the design pay for
    them.
    """
    rng = np.random.default_rng(4)
    n = 80
    continuous = rng.uniform(0, 1, n)
    indicator = np.repeat([0.0, 1.0], n // 2)
    X = pd.DataFrame({
        'Intercept': 1.0,
        'batch': indicator,                 # two values: not a covariate
        'cell_count': continuous,           # many values: a covariate
        'fraction:grna[T.geneA_1]': rng.uniform(0, 0.5, n),
    })
    y = pd.Series(continuous ** 2 + 0.3 * indicator + rng.normal(0, 0.05, n))

    model = ml.regression_model(X, y, regression_type='spline',
                                spline_knots=4, spline_degree=3)

    names = [str(name) for name in model.params.index]
    assert 'batch' in names, 'the indicator was expanded or dropped'
    assert 'cell_count' not in names, 'the covariate was not given a basis'
    assert any(name.startswith('cell_count') and name != 'cell_count'
               for name in names)
    # the guide column is never touched, whatever else the design carries
    assert 'fraction:grna[T.geneA_1]' in names


def test_a_group_lasso_that_ran_out_of_sweeps_says_the_answer_is_provisional(
        monkeypatch, capsys):
    """Non-convergence is reported, and the result says so on the object.

    The last iterate of a coordinate descent is not the solution, and a
    selection read off it is a claim about which genes survived a penalty
    that was never actually applied to convergence.
    """
    from spacr import group_lasso as group_lasso_module

    columns = ['Intercept', 'fraction:grna[T.geneA_1]',
               'fraction:grna[T.geneA_2]', 'fraction:grna[T.geneB_1]']
    rng = np.random.default_rng(5)
    X = pd.DataFrame(rng.uniform(0, 1, (40, len(columns))), columns=columns)
    X['Intercept'] = 1.0
    y = pd.Series(rng.normal(0, 1, 40))

    def _unconverged(design, response, labels, **kwargs):
        return np.array([0.0, 0.4, 0.0, -0.2]), 0.1, False

    monkeypatch.setattr(group_lasso_module, 'fit', _unconverged)

    model = ml.regression_model(X, y, regression_type='group_lasso',
                                group_lasso_lambda=0.01)

    printed = capsys.readouterr().out
    assert 'did not reach its tolerance' in printed
    assert str(group_lasso_module.MAX_ITERATIONS) in printed
    assert model.converged is False
    # the coefficients are still handed back, which is why the warning has to
    # be printed rather than the fit refused
    assert float(model.coef_[1]) == pytest.approx(0.4)


# ===========================================================================
# the reporting helpers around a fit
# ===========================================================================

def test_a_frame_with_no_well_columns_still_gets_its_qc_report(monkeypatch):
    """No plate/row/column/prc means no labels, not no report.

    The per-well metadata only ever labels a point; a design built from a
    frame that carries none of those columns still deserves its residual and
    calibration panels.
    """
    import spacr.regression_qc as regression_qc

    seen = {}

    def _report(model, X, y, dst, *, metadata=None, **kwargs):
        seen['metadata'] = metadata
        return {'verdict': 'ok'}

    monkeypatch.setattr(regression_qc, 'regression_qc_report', _report)

    X, y = _tiny_design(n=20)
    model = ml.regression_model(X, y, regression_type='ols')

    bare = pd.DataFrame({'value': np.arange(20.0)})
    assert ml._write_regression_qc(model, X, y, bare, 'unused') == {
        'verdict': 'ok'}
    assert seen['metadata'] is None

    # the same call on a frame that names its wells does carry the labels
    labelled = pd.DataFrame({'prc': [f'plate1_r1_c{i}' for i in range(20)]})
    ml._write_regression_qc(model, X, y, labelled, 'unused')
    assert list(seen['metadata'].columns) == ['prc']


def test_a_plate_panel_with_nowhere_to_save_is_still_shown(tmp_path):
    """``dst=None`` draws the plates and writes nothing.

    ``regression`` calls this once for the run folder and once for a caller
    that only wants the figure on screen; the second must not invent a file
    next to the working directory.
    """
    frame = pd.DataFrame({
        'prc': [f'plate1_r{r}_c{c}' for r in range(1, 4) for c in range(1, 5)],
        'value': np.linspace(0.1, 1.2, 12),
    })

    assert ml._show_plates(frame, 'value', None) is True
    assert not list(tmp_path.iterdir())

    assert ml._show_plates(frame, 'value', str(tmp_path)) is True
    assert (tmp_path / 'plate_heatmap_value.pdf').exists()


def test_a_fit_that_reports_no_likelihood_says_so_rather_than_guessing():
    """No ``llnull`` and no ``null_deviance`` is 'not available', not a crash.

    ``spacr.power_model`` and the group-lasso result stand in for a fitted
    model everywhere else in the run, and neither carries a likelihood; the
    console line has to survive them.
    """
    class Bare:
        family = None
        llf = 1.0                      # present, but nothing to compare it to

    assert ml.fit_quality_note(Bare()) == (
        "McFadden's R²: not available for this fit")

    # a fit that does carry both gets the number, so the sentence above is
    # about the missing null and not about the helper refusing everything
    class Full:
        family = None
        llf = -10.0
        llnull = -20.0

    assert "0.5" in ml.fit_quality_note(Full())


# ===========================================================================
# what the run decides before it fits
# ===========================================================================

def _long_guide_frame(seed=0):
    """One row per (well, guide), the response constant within a well."""
    rng = np.random.default_rng(seed)
    records = []
    for plate in ('plate1', 'plate2'):
        for row in ('r1', 'r2', 'r3'):
            for column in ('c1', 'c2', 'c3'):
                well = f'{plate}_{row}_{column}'
                score = float(rng.normal(0, 0.05))
                for guide in ('g1', 'g2', 'g3'):
                    records.append({
                        'plateID': plate, 'rowID': row, 'columnID': column,
                        'prc': well, 'grna': guide,
                        'fraction': float(rng.uniform(0.05, 0.6)),
                        'score': score,
                    })
    return pd.DataFrame(records)


def test_auto_counts_no_block_terms_when_the_table_has_no_plate_column():
    """A screen with no plateID has no block fixed effects to pay for.

    The margin is computed from the parameter count, so counting a block that
    is not in the design would push a perfectly identifiable screen onto the
    permutation test.
    """
    frame = _long_guide_frame()
    settings = {'guide_permutation_block': 'plateID'}

    mode, reason = ml.resolve_auto_inference(
        frame.drop(columns=['plateID']), settings)
    assert mode == 'regression'
    assert '0 block terms' in reason

    # with the column present the two plates cost one block term, which is
    # what makes the count above a measurement rather than a constant
    _mode, with_blocks = ml.resolve_auto_inference(frame, settings)
    assert '1 block terms' in with_blocks


def test_a_saturation_warning_counts_only_the_blocks_the_table_has():
    """The warning's parameter count matches the design that will be fitted.

    A warning that overstates the width would fire on designs that are fine,
    and a user who has seen one of those stops reading the next.
    """
    wells = [f'plate1_r1_c{i}' for i in range(1, 5)]
    frame = pd.DataFrame({
        'prc': [well for well in wells for _ in range(4)],
        'grna': [f'g{i}' for _ in wells for i in range(1, 5)],
    })

    warning = ml._identifiability_warning(
        frame, {'guide_permutation_block': 'plateID'})

    assert warning is not None
    assert '4 analysed wells are being used to estimate 5 parameters' in warning
    assert '(4 grnas + intercept + 0 block terms)' in warning


def test_settings_that_name_no_input_at_all_are_refused_before_any_read():
    """Empty score/count lists are a settings error, not an empty regression.

    The migration branch produces no pairs at all here, so nothing is printed
    about legacy lists and the refusal is the only output.
    """
    with pytest.raises(ValueError, match='at least one score CSV'):
        ml.normalize_regression_input_pairs({'score_data': [],
                                             'count_data': []})


def test_a_questionable_shuffle_with_no_remedy_still_names_its_findings(
        monkeypatch, capsys):
    """A finding is printed even when there is nothing to suggest doing.

    The remedy line is a suggestion; the findings are the measurement, and a
    verdict that carried no suggestion used to be the one case where the user
    saw neither.
    """
    import spacr.permutation_qc as permutation_qc

    monkeypatch.setattr(
        permutation_qc, 'exchangeability_verdict',
        lambda report: {'ok': False, 'findings': ['columnID explains 12%'],
                        'remedy': ''})

    report = ml._report_exchangeability(
        _long_guide_frame(), 'score',
        {'guide_permutation_block': 'plateID',
         'guide_nuisance_columns': []}, None)

    printed = capsys.readouterr().out
    assert report is not None and report['blocks'] == 2
    assert 'the within-block shuffle is questionable' in printed
    assert 'columnID explains 12%' in printed
    assert '->' not in printed, 'a remedy was invented where there was none'


# ===========================================================================
# the permutation branch's own settings
# ===========================================================================

def test_a_permutation_asked_for_zero_wells_is_refused_by_name(tmp_path):
    """``guide_min_wells=0`` is refused, and a bare int is still accepted.

    The panel writes a single number as often as a list, so the scalar has to
    be widened before it is validated or the refusal never fires.
    """
    frame = _long_guide_frame()

    with pytest.raises(ValueError, match='positive integers'):
        ml._run_guide_permutation_analysis(
            frame, 'score', str(tmp_path), {'guide_min_wells': 0})


def test_a_dependent_variable_that_is_not_a_column_names_what_is(tmp_path):
    """The refusal lists the columns the merged table actually has.

    ``dependent_variable`` is a free-text setting and a typo in it is the
    commonest way to reach this; a bare KeyError names the typo and not the
    alternatives.
    """
    frame = _long_guide_frame()

    with pytest.raises(ValueError) as caught:
        ml._run_guide_permutation_analysis(
            frame, 'nonesuch', str(tmp_path),
            {'guide_min_wells': [1], 'guide_primary_min_wells': 1})

    message = str(caught.value)
    assert "['nonesuch']" in message
    assert 'score' in message


# ===========================================================================
# calling the hits
# ===========================================================================

def _coef_frame(p_values):
    return pd.DataFrame({
        'feature': ['fraction:grna[T.geneA_1]', 'fraction:grna[T.geneB_1]'],
        'coefficient': [0.8, -0.4],
        'p_value': p_values,
    })


def test_a_fit_whose_coefficients_carry_no_p_value_corrects_nothing(capsys):
    """No testable coefficient means no correction and an empty hit list.

    The mixed fit's BLUPs and variance components arrive with a NaN p-value by
    construction; running BH over an empty family raised rather than reporting
    a screen with nothing in it.
    """
    settings = {'controls': None, 'multiple_testing_method': 'fdr_bh',
                'fdr_alpha': 0.05}

    coef_df, significant, threshold, rule = ml._call_level_hits(
        _coef_frame([np.nan, np.nan]), 'grna', settings, 'mixed',
        pd.DataFrame(), 'score')

    assert coef_df['q_value'].isna().all()
    assert len(significant) == 0
    assert threshold == 0 and rule == 'no effect-size cut'
    assert 'across 0 tested coefficients' in capsys.readouterr().out

    # the same table with p-values does get corrected, so the emptiness above
    # is the missing p-values and not the helper declining to correct
    corrected, hits, _t, _r = ml._call_level_hits(
        _coef_frame([0.001, 0.9]), 'grna', settings, 'ols',
        pd.DataFrame(), 'score')
    assert corrected['q_value'].notna().all()
    assert list(hits['feature']) == ['fraction:grna[T.geneA_1]']


def test_a_stage_whose_announcement_cannot_be_printed_still_records_it():
    """A stage name that cannot be formatted costs the line, not the reading.

    ``_stage`` exists so a long silent step can be told from a hung one; the
    resource reading it returns is read by the run summary afterwards, and
    losing that to a formatting error would lose the cost of the whole fit.
    """
    class Unprintable:
        def __format__(self, spec):
            raise RuntimeError('this name cannot be rendered')

        def __str__(self):
            return 'reading the input tables'

    settings = {}
    assert isinstance(ml._stage(settings, Unprintable()), dict)

    # a name that can be printed announces itself, which is what makes the
    # silence above a caught failure rather than a step that never ran
    import io
    import contextlib

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        ml._stage(settings, 'fitting the model')
    assert 'Regression: fitting the model' in buffer.getvalue()


# ===========================================================================
# the cell-count sweep and the threshold sweep
# ===========================================================================

def _score_csv(path, n_wells=4, n_cells=30, seed=7):
    rng = np.random.default_rng(seed)
    records = []
    for well in range(n_wells):
        for _ in range(n_cells):
            records.append({
                'plateID': 'plate1', 'rowID': 'r1',
                'columnID': f'c{well + 1}', 'fieldID': 'f1',
                'pred': float(rng.normal(0.5, 0.1)),
            })
    pd.DataFrame(records).to_csv(path, index=False)
    return str(path)


def test_a_cell_count_sweep_that_drew_nothing_reports_no_minimum(
        tmp_path, monkeypatch, capsys):
    """No figure means no answer, rather than a number nobody can check.

    The elbow is read off the curve, so a run with nowhere to draw it has not
    measured a minimum cell count -- and returning one anyway would put an
    unaudited threshold into ``process_scores``.
    """
    settings = {
        'score_data': _score_csv(tmp_path / 'scores.csv'),
        'dependent_variable': 'pred',
        'tolerance': 2,
        'min_cell_count': None,
    }

    monkeypatch.setattr(ml, '_draw_the_cell_count_sweep',
                        lambda summary, mark, path: None)
    assert ml.minimum_cell_simulation(
        dict(settings), num_repeats=2, increment=10,
        dst=str(tmp_path / 'out')) is None
    assert 'Saved' not in capsys.readouterr().out

    # with a figure the same screen answers with its elbow, so the None above
    # is the missing drawing and not the sweep declining to measure
    written = tmp_path / 'out' / 'cell_min_threshold.pdf'
    monkeypatch.setattr(ml, '_draw_the_cell_count_sweep',
                        lambda summary, mark, path: str(written))
    answer = ml.minimum_cell_simulation(
        dict(settings), num_repeats=2, increment=10,
        dst=str(tmp_path / 'out'))
    assert answer is not None and float(answer) >= 2
    assert f'Saved {written}' in capsys.readouterr().out


def test_a_sweep_with_no_pick_of_its_own_claims_nothing_about_the_threshold(
        monkeypatch, capsys):
    """Nothing is printed when the sweep did not reach a number.

    The line exists to say where the threshold in force came from; with only
    one of the two answers in hand there is no comparison to report, and a
    half-filled sentence is worse than none.
    """
    monkeypatch.setattr(ml, '_graph_sequencing_stats', lambda settings: None)
    ml._draw_the_threshold_sweep({'fraction_threshold': 0.0168}, None)
    assert capsys.readouterr().out == ''

    # with both answers in hand the run names each of them and says which is
    # in force -- the branch the silence above is the other side of
    monkeypatch.setattr(ml, '_graph_sequencing_stats', lambda settings: 0.03)
    ml._draw_the_threshold_sweep({'fraction_threshold': 0.0168}, None,
                                 measured=True)
    printed = capsys.readouterr().out
    assert 'the control-well calibration measured 0.0168' in printed
    assert "sweep's own pick on this screen is 0.03" in printed


# ===========================================================================
# aggregating the scores
# ===========================================================================

def test_a_frame_that_names_its_wells_is_not_re_split_from_its_prcfo_key():
    """Present plate/row/column columns are used as they are.

    ``prcfo`` is rebuilt by several writers and a scores CSV can carry a stale
    one; re-deriving the well from it would silently move every object to a
    well the columns disagree with.
    """
    frame = pd.DataFrame({
        'prcfo': ['stale_r9_c9_f1_o1'] * 4,
        'plateID': ['plate1'] * 4,
        'rowID': ['r1', 'r1', 'r2', 'r2'],
        'columnID': ['c1', 'c2', 'c1', 'c2'],
        'pred': [0.2, 0.4, 0.6, 0.8],
    })

    scored, name = ml.process_scores(frame, 'pred', None, min_cell_count=1,
                                     agg_type='mean')

    assert name == 'pred'
    assert sorted(scored['prc']) == ['plate1_r1_c1', 'plate1_r1_c2',
                                     'plate1_r2_c1', 'plate1_r2_c2']


def test_a_count_model_handed_no_rows_at_all_is_not_told_its_counts_are_wrong(
        ):
    """An empty table has no fractional sum, so nothing is refused.

    Every object can be dropped upstream -- a filter, a control block, an
    outlier pass -- and the count-model guard reads the per-well sums; over no
    wells it has nothing to judge and must not speak.
    """
    empty = pd.DataFrame({
        'prcfo': pd.Series([], dtype=str), 'prc': pd.Series([], dtype=str),
        'plateID': pd.Series([], dtype=str), 'rowID': pd.Series([], dtype=str),
        'columnID': pd.Series([], dtype=str), 'pred': pd.Series([], dtype=float),
    })

    scored, name = ml.process_scores(empty, 'pred', None, min_cell_count=0,
                                     agg_type='mean', regression_type='poisson')
    assert len(scored) == 0
    assert list(scored.columns) == ['prc', 'pred', 'cell_count']
    assert name == 'pred'

    # the same call over real scores does refuse them, which is what makes the
    # silence above the emptiness rather than a guard that never fires
    scores = pd.DataFrame({
        'prcfo': ['plate1_r1_c1_f1_o1', 'plate1_r1_c1_f1_o2'],
        'plateID': ['plate1'] * 2, 'rowID': ['r1'] * 2,
        'columnID': ['c1'] * 2, 'pred': [0.14, 0.21],
    })
    with pytest.raises(ValueError, match='models the well'):
        ml.process_scores(scores, 'pred', None, min_cell_count=0,
                          agg_type='mean', regression_type='poisson')


# ===========================================================================
# explaining and drawing
# ===========================================================================

def test_an_explainer_that_will_not_run_is_followed_by_the_next_one(
        monkeypatch):
    """A failing SHAP explainer costs its attempt, not the explanation.

    ``shap`` picks a fast path per model type and the fast path is the one
    that raises on an unfamiliar estimator; falling through to the
    model-agnostic explainer is the whole point of trying more than one.
    """
    def _refuses(sample):
        raise RuntimeError('this explainer cannot read the model')

    def _works(sample):
        return np.zeros((len(sample), 2))

    monkeypatch.setattr(ml, '_shap_explainers',
                        lambda model, X_train: [(_refuses, 'tree'),
                                                (_works, 'kernel')])

    values, note = ml._shap_values(object(), pd.DataFrame({'a': [1.0, 2.0]}),
                                   pd.DataFrame({'a': [1.0, 2.0]}))

    assert note == 'kernel'
    assert values.shape == (2, 2)

    # when every attempt fails the last failure is named, so the fallthrough
    # above cannot hide a model nothing can explain
    monkeypatch.setattr(ml, '_shap_explainers',
                        lambda model, X_train: [(_refuses, 'tree')])
    with pytest.raises(RuntimeError, match='cannot read the model'):
        ml._shap_values(object(), pd.DataFrame({'a': [1.0]}),
                        pd.DataFrame({'a': [1.0]}))


def test_a_plot_that_exported_nothing_is_not_announced_to_the_gallery(
        tmp_path, monkeypatch):
    """No file written means no tile claimed and no path returned.

    ``publish_file`` puts a tile in the gallery; announcing one for a file
    that was never written gives the user a broken thumbnail and a run that
    looks as if it drew a figure it did not.
    """
    import spacr.figure_sink as figure_sink

    published = []
    monkeypatch.setattr(figure_sink, 'publish_file',
                        lambda path, title=None: published.append((path, title)))

    class Plot:
        def __init__(self, result):
            self.result = result
            self.deleted = False

        def export(self, target):
            return self.result

        def deleteLater(self):
            self.deleted = True

    nothing = Plot('')
    assert not ml.write_plot(nothing, str(tmp_path / 'sweep.pdf'), 'Sweep')
    assert nothing.deleted
    assert published == []

    target = str(tmp_path / 'written.pdf')
    something = Plot(target)
    assert ml.write_plot(something, target, 'Sweep') == target
    assert published == [(target, 'Sweep')]


def test_a_sweep_that_fails_mid_draw_takes_its_scene_with_it(monkeypatch):
    """The Qt scene is deleted before the failure is re-raised.

    A ``FastPlot`` left alive after an exception keeps its scene and its
    OpenGL context; a run that draws a panel per plate leaks one per failure
    and eventually cannot draw at all.
    """
    import spacr.qt.widgets.fast_plots as fast_plots

    deleted = []

    class Exploding:
        def __init__(self, **kwargs):
            pass

        def resize(self, *args):
            pass

        def add_curve(self, *args, **kwargs):
            raise RuntimeError('the curve could not be built')

        def deleteLater(self):
            deleted.append(True)

    monkeypatch.setattr(fast_plots, 'FastPlot', Exploding)

    summary = pd.DataFrame({'sample_size': [2.0, 12.0],
                            'smoothed_mean_abs_diff': [0.2, 0.1],
                            'std_abs_diff': [0.05, 0.02]})

    with pytest.raises(RuntimeError, match='could not be built'):
        ml._draw_the_cell_count_sweep(summary, 5.0, 'unused.pdf')
    assert deleted == [True]


# ===========================================================================
# the run itself, end to end
#
# The synthetic screen and its settings builders are the ones
# tests/test_cov_ml_perform_regression.py already defines, so this file and
# that one describe the same run rather than two screens that can drift.
# ===========================================================================

from tests.test_cov_ml_perform_regression import (  # noqa: E402
    base_settings, parametric_settings, write_counts, write_metadata,
    write_scores)


@pytest.fixture
def screen(tmp_path):
    """One plate: a per-object score CSV, a per-well count CSV, metadata."""
    scores = tmp_path / 'scores'
    counts = tmp_path / 'counts'
    scores.mkdir()
    counts.mkdir()
    return {
        'root': tmp_path,
        'score': write_scores(scores / 'xgb_scores.csv'),
        'count': write_counts(counts / 'counts.csv'),
        'meta': write_metadata(tmp_path / 'TGME49_Summary.csv'),
    }


def test_a_count_table_in_the_score_slot_is_named_as_swapped_inputs(screen):
    """The refusal says the two inputs look swapped, not merely 'not found'.

    A count table has grna and count columns and no score column, which is
    exactly what a swapped pair looks like; the alternative message is a
    column dump the user has to interpret themselves.
    """
    settings = base_settings(screen)
    settings['score_data'] = [screen['count']]
    settings['paired_data'] = [{'score': screen['count'],
                                'count': screen['count']}]

    with pytest.raises(ValueError) as caught:
        ml.perform_regression(settings)

    message = str(caught.value)
    assert 'shape of a COUNT file' in message
    assert "['count', 'grna']" in message


def test_an_analysis_mode_nobody_implements_is_refused_before_the_fit(screen):
    """``analysis_mode`` is closed, and the refusal names both members.

    A settings CSV can carry any string; a mode nothing dispatches on used to
    fall through to the regression branch and produce a parametric run under
    a name that promised something else.
    """
    # `inference` normally selects the mode; 'auto' is the one value that
    # leaves an explicitly supplied analysis_mode standing to be validated.
    with pytest.raises(ValueError, match="Unsupported analysis_mode"):
        ml.perform_regression(base_settings(screen, inference='auto',
                                            analysis_mode='sideways'))


def test_batch_correction_without_its_response_column_says_which_setting(
        screen):
    """The refusal names the response and offers batch_correction=none.

    ``pathogen_nucleus_shortest_distance`` is exempted from the earlier
    dependent-variable check, so this is the one path where a correction is
    asked for on a column the score table does not have.
    """
    settings = base_settings(
        screen, dependent_variable='pathogen_nucleus_shortest_distance',
        batch_correction='center')

    with pytest.raises(ValueError) as caught:
        ml.perform_regression(settings)

    message = str(caught.value)
    assert 'Batch correction cannot run' in message
    assert 'batch_correction=none' in message


def test_the_control_blocks_reach_the_count_filter_once_each(screen,
                                                            monkeypatch):
    """A well named as a control block is removed, and never twice.

    ``filter_value`` and the three control-block settings can name the same
    well; adding it twice would print the removal twice and make the run's own
    exclusion count disagree with the wells that left.
    """
    seen = {}
    real_process_reads = ml.process_reads

    def spy(csv_path, fraction_threshold, plate, filter_column=None,
            filter_value=None, record=None):
        seen['filter_value'] = list(filter_value)
        return real_process_reads(csv_path, fraction_threshold, plate,
                                  filter_column=filter_column,
                                  filter_value=filter_value, record=record)

    monkeypatch.setattr(ml, 'process_reads', spy)

    output = ml.perform_regression(parametric_settings(
        screen, filter_column='columnID', filter_value=['c1'],
        negative_control_wells=['c1'], positive_control_wells=['c2']))

    # 'c1' was already listed and is not repeated; 'c2' is added by the block
    assert seen['filter_value'] == ['c1', 'c2']
    assert len(output['results']) > 0


def test_control_blocks_that_cannot_be_resolved_cost_only_the_exclusion(
        screen, monkeypatch):
    """A well_spec failure leaves the run with the wells the user typed.

    Resolving a plate-map block is a convenience on top of ``filter_value``;
    a failure in it must not take down a run whose exclusions were already
    stated explicitly.
    """
    import spacr.well_spec as well_spec

    def boom(settings):
        raise RuntimeError('the plate layout could not be read')

    monkeypatch.setattr(well_spec, 'control_block_wells', boom)

    seen = {}
    real_process_reads = ml.process_reads

    def spy(csv_path, fraction_threshold, plate, filter_column=None,
            filter_value=None, record=None):
        seen['filter_value'] = list(filter_value)
        return real_process_reads(csv_path, fraction_threshold, plate,
                                  filter_column=filter_column,
                                  filter_value=filter_value, record=record)

    monkeypatch.setattr(ml, 'process_reads', spy)

    output = ml.perform_regression(parametric_settings(
        screen, filter_column='columnID', filter_value=['c1'],
        negative_control_wells=['c2']))

    assert seen['filter_value'] == ['c1']
    assert len(output['results']) > 0


def test_an_outlier_filter_with_no_column_to_read_says_so_before_the_fractions(
        screen, capsys):
    """A criterion that found no column is reported, not silently skipped.

    The fractions below are computed on what survived the filter, so a filter
    the user switched on that removed nothing has to be told apart from one
    that ran and found nothing.
    """
    ml.perform_regression(parametric_settings(screen,
                                              cell_area_outlier_mads=5.0))

    printed = capsys.readouterr().out
    assert 'Outliers removed before annotation:' in printed
    assert 'cell area: this table has no cell area column' in printed


def test_an_outlier_filter_that_raised_says_the_counts_are_unfiltered(
        screen, monkeypatch, capsys):
    """A failed filter is announced, because the numbers below it change.

    Silence here means the run continues on unfiltered objects with every
    downstream fraction computed from them and nothing saying so.
    """
    import spacr.outlier_filter as outlier_filter

    def boom(frame, settings=None):
        raise RuntimeError('the MAD estimate could not be formed')

    monkeypatch.setattr(outlier_filter, 'apply', boom)

    output = ml.perform_regression(parametric_settings(
        screen, cell_area_outlier_mads=5.0))

    printed = capsys.readouterr().out
    assert '[outliers] the pre-annotation filter did not run' in printed
    assert 'the counts below are unfiltered' in printed
    assert len(output['results']) > 0


def test_a_measured_threshold_replaces_the_one_the_settings_carried(
        screen, monkeypatch):
    """A calibrated cut-off is used and recorded as automatic.

    The point of the calibration is that the number in force came from the
    control wells rather than from the box; a run that measured one and went
    on using the typed value would be reporting the wrong provenance.
    """
    monkeypatch.setattr(ml, '_calibrated_fraction_threshold',
                        lambda settings: 0.0123)

    output = ml.perform_regression(parametric_settings(
        screen, calibrate_fraction_threshold=True))
    assert output['settings']['fraction_threshold'] == pytest.approx(0.0123)
    assert ml._AUTOMATIC_SETTINGS['fraction_threshold'] == pytest.approx(0.0123)

    # a calibration that cannot answer leaves the typed threshold in force,
    # which is the branch the replacement above is the other side of
    ml._AUTOMATIC_SETTINGS.pop('fraction_threshold', None)
    monkeypatch.setattr(ml, '_calibrated_fraction_threshold',
                        lambda settings: None)
    output = ml.perform_regression(parametric_settings(
        screen, calibrate_fraction_threshold=True, fraction_threshold=0.005))
    assert output['settings']['fraction_threshold'] == pytest.approx(0.005)
    assert 'fraction_threshold' not in ml._AUTOMATIC_SETTINGS


def test_a_permutation_run_with_no_model_named_says_nothing_about_one(
        screen, capsys):
    """With regression_type unset there is no name to tell the user is unread.

    The sentence exists to stop a second run under a different model; with no
    model named there is nothing to stop and the paragraph would be noise.
    """
    settings = base_settings(screen)
    settings['regression_type'] = None

    output = ml.perform_regression(settings)

    printed = capsys.readouterr().out
    assert 'is not read' not in printed
    assert len(output['primary']) > 0

    # naming one does produce the paragraph, so the silence above is the
    # missing name rather than a run that never reached the line
    named = base_settings(screen, regression_type='mixed')
    ml.perform_regression(named)
    assert "regression_type='mixed' is not read" in capsys.readouterr().out


def test_a_permutation_run_without_the_summary_module_keeps_its_results(
        screen, monkeypatch):
    """An unimportable summary writer costs the summary and nothing else.

    The permutation path has no statsmodels summary to fall back on, so this
    is the only summary that run produces -- and losing an analysis over a
    reporting import is the trade nobody would make.
    """
    monkeypatch.setitem(sys.modules, 'spacr.regression_summary', None)

    output = ml.perform_regression(base_settings(screen))
    assert len(output['primary']) > 0
    assert not os.path.exists(os.path.join(output['res_folder'],
                                           ml.SUMMARY_FILENAME))

    # with the module importable the same run writes it, so the absence above
    # is the failed import and not a mode that never summarises
    monkeypatch.undo()
    output = ml.perform_regression(base_settings(screen))
    assert os.path.exists(os.path.join(output['res_folder'],
                                       ml.SUMMARY_FILENAME))


def test_a_fitted_run_without_the_summary_module_keeps_its_coefficients(
        screen, monkeypatch):
    """The parametric path survives the same missing module.

    ``ridge`` writes no statsmodels summary of its own, so the file is
    present exactly when spaCR's own summary could be written.
    """
    monkeypatch.setitem(sys.modules, 'spacr.regression_summary', None)

    output = ml.perform_regression(parametric_settings(
        screen, regression_type='ridge'))
    assert len(output['results']) > 0
    assert not os.path.exists(os.path.join(output['res_folder'],
                                           ml.SUMMARY_FILENAME))

    monkeypatch.undo()
    output = ml.perform_regression(parametric_settings(
        screen, regression_type='ridge'))
    assert os.path.exists(os.path.join(output['res_folder'],
                                       ml.SUMMARY_FILENAME))


def test_metadata_that_will_not_merge_is_named_and_the_run_goes_on(
        screen, monkeypatch, capsys):
    """A failing metadata merge names the file and leaves the results alone.

    The merge decorates a finished fit with gene names; a malformed summary
    table must not take down the run that produced the coefficients.
    """
    import spacr.utils as utils

    def boom(results, metadata, name=None):
        raise RuntimeError('the summary table has no Gene ID column')

    monkeypatch.setattr(utils, 'merge_regression_res_with_metadata', boom)

    output = ml.perform_regression(parametric_settings(screen))

    printed = capsys.readouterr().out
    assert 'Could not merge metadata from' in printed
    assert 'has no Gene ID column' in printed
    assert len(output['results']) > 0


def test_a_count_table_that_names_its_plate_and_row_in_one_column_agrees(
        screen, tmp_path):
    """A ``plate_row`` count table fits the same screen as a ``rowID`` one.

    The rowID repair reduces a composite '<plate>_<row>' to the row; a table
    that carries no rowID at all has nothing to reduce, and running the repair
    on it used to be an IndexError on an empty frame.
    """
    with_rows = ml.perform_regression(parametric_settings(screen))

    composite = pd.read_csv(screen['count'])
    composite['plate_row'] = (composite['plateID'].astype(str) + '_'
                              + composite['rowID'].astype(str))
    composite = composite.drop(columns=['plateID', 'rowID'])
    folded = tmp_path / 'folded_counts.csv'
    composite.to_csv(folded, index=False)

    other = dict(screen, count=str(folded))
    without_rows = ml.perform_regression(parametric_settings(other))

    assert len(without_rows['results']) == len(with_rows['results']) > 0
    np.testing.assert_allclose(
        without_rows['results']['coefficient'].to_numpy(dtype=float),
        with_rows['results']['coefficient'].to_numpy(dtype=float),
        rtol=1e-9, atol=1e-12)


def test_an_annotation_that_has_something_to_say_says_it_once(screen,
                                                              monkeypatch,
                                                              capsys):
    """The annotation's note is printed for the results table only.

    Four tables are annotated from the same source; printing the note for
    each would repeat the same sentence four times per run, which is how a
    line that matters stops being read.
    """
    import spacr.annotation as annotation

    monkeypatch.setattr(
        annotation, 'annotate_with',
        lambda frame, source, cache_dir=None, quiet=False:
            (frame, '3 of 4 genes matched TGGT1 identifiers'))
    monkeypatch.setattr(annotation, 'supplementary',
                        lambda features, path=None: None)

    output = ml.perform_regression(parametric_settings(
        screen, annotation_source='toxoplasma'))

    printed = capsys.readouterr().out
    assert printed.count('Annotation: 3 of 4 genes matched') == 1
    assert len(output['results']) > 0


def test_an_annotation_that_changed_the_row_count_is_refused(screen,
                                                             monkeypatch):
    """A merge that fans a table out is a wrong key, and is stopped.

    The coefficient table's contract is one row per coefficient; a
    many-to-many join against an annotation source would put every hit into
    results_significant.csv more than once, and nothing downstream would say
    so.
    """
    import spacr.annotation as annotation

    monkeypatch.setattr(
        annotation, 'annotate_with',
        lambda frame, source, cache_dir=None, quiet=False:
            (pd.concat([frame, frame], ignore_index=True), ''))
    monkeypatch.setattr(annotation, 'supplementary',
                        lambda features, path=None: None)

    with pytest.raises(ValueError) as caught:
        ml.perform_regression(parametric_settings(
            screen, annotation_source='toxoplasma'))

    assert 'annotation changed results from' in str(caught.value)


def test_a_control_left_unnamed_is_not_offered_to_the_concordance_report(
        screen, monkeypatch):
    """An empty control box names no control, rather than a control called ''.

    ``concordance_report`` colours the guides of each named control; a key of
    '' would claim every unlabelled guide belongs to a control condition.
    """
    import spacr.guide_concordance as guide_concordance

    seen = {}
    real = guide_concordance.concordance_report

    def spy(coef_df, alpha=0.05, controls=None):
        seen['controls'] = dict(controls or {})
        return real(coef_df, alpha=alpha, controls=controls)

    monkeypatch.setattr(guide_concordance, 'concordance_report', spy)

    ml.perform_regression(parametric_settings(screen, positive_control=None))
    assert seen['controls'] == {'233460': 'negative'}

    ml.perform_regression(parametric_settings(screen))
    assert seen['controls'] == {'239740': 'positive', '233460': 'negative'}


def test_the_legacy_volcano_is_announced_only_when_a_file_was_written(
        screen, monkeypatch, capsys):
    """A run claims the volcano it wrote, and warns when it wrote none.

    A stale file from an earlier run sits at the same path, so announcing one
    unconditionally would credit this run with a figure it did not make.
    """
    import spacr.toxo as toxo

    def writes_nothing(frame, metadata_path, **kwargs):
        return ['TGGT1_239740']

    monkeypatch.setattr(toxo, 'custom_volcano_plot', writes_nothing)

    ml.perform_regression(parametric_settings(
        screen, toxo=True, legacy_volcano=True,
        metadata_files=[screen['meta']]))

    printed = capsys.readouterr().out
    assert 'WARNING: the legacy volcano was requested but no file was' in printed
    # one curated table is not two, so the extra reports are skipped -- and
    # the gene list they would have used is still non-empty
    assert 'Skipping the phenotype and transcription reports' in printed
    assert 'No gene_list produced' not in printed

    def writes_one(frame, metadata_path, save_path=None, **kwargs):
        open(save_path, 'w').write('%PDF-1.4\n')
        return []

    monkeypatch.setattr(toxo, 'custom_volcano_plot', writes_one)
    ml.perform_regression(parametric_settings(
        screen, toxo=True, legacy_volcano=True,
        metadata_files=[screen['meta']]))
    assert 'Saved volcano plot to' in capsys.readouterr().out


def test_a_finished_run_whose_summary_raises_keeps_its_coefficients(
        screen, monkeypatch, capsys):
    """A summary that fails is reported; the fit it describes is not lost.

    This is the parametric path's own guard: the results are already on disk
    by the time the summary is written, and an hour's fit must not be thrown
    away over prose.
    """
    import spacr.regression_summary as regression_summary

    def boom(*args, **kwargs):
        raise RuntimeError('the summary template is missing')

    monkeypatch.setattr(regression_summary, 'write_run_summary', boom)

    output = ml.perform_regression(parametric_settings(screen))

    printed = capsys.readouterr().out
    assert 'Could not write the run summary' in printed
    assert 'the summary template is missing' in printed
    assert len(output['results']) > 0


def test_a_threshold_no_guide_reaches_is_an_answer_not_a_failure(screen,
                                                                 capsys):
    """A guide_min_wells the screen cannot satisfy skips one panel only.

    ``guide_min_wells`` is a sweep; on a one-plate screen no guide appears in
    ninety-nine wells, and raising there used to throw away the results for
    every other threshold at the drawing stage.
    """
    output = ml.perform_regression(base_settings(
        screen, guide_min_wells=[1, 99], guide_primary_min_wells=1))

    printed = capsys.readouterr().out
    assert 'No guide reached 99 well(s)' in printed
    assert 'The thresholds that did have guides are unaffected' in printed
    assert len(output['primary']) > 0


def test_permutation_diagnostics_that_fail_are_skipped_by_name(screen,
                                                               monkeypatch,
                                                               capsys):
    """The design panels are advisory; losing them costs a line, not the run.

    They are written for every run because the failure this mode exists to
    prevent is invisible on the volcano -- but a diagnostic that cannot be
    drawn must not take the analysis with it.
    """
    import spacr.regression_diagnostics as regression_diagnostics

    def boom(*args, **kwargs):
        raise RuntimeError('the design panel could not be rendered')

    monkeypatch.setattr(regression_diagnostics, 'write_diagnostic_suite', boom)

    output = ml.perform_regression(base_settings(screen))

    printed = capsys.readouterr().out
    assert 'Regression diagnostics were skipped' in printed
    assert 'the design panel could not be rendered' in printed
    assert len(output['primary']) > 0


def test_a_bootstrap_resample_that_builds_no_design_is_counted_and_named(
        screen, monkeypatch, caplog):
    """Selection frequencies say how many resamples they are over.

    ``selection_frequency`` is divided by the number that fitted, so 3 of 4
    resamples dropping gives a frequency computed from a single draw --
    reported in the same column, under the same name, as one over four.
    """
    import logging

    from patsy import dmatrices as real_dmatrices

    dropped = {'n': 0}

    def sometimes(formula, data=None, **kwargs):
        # A bootstrap resample draws WITH REPLACEMENT, so it holds the same
        # (well, gRNA) row more than once; the cleaned frame the fits are
        # built on holds each of them exactly once.
        if data is not None and dropped['n'] < 2 and data.duplicated().any():
            dropped['n'] += 1
            raise ValueError('factor levels are too sparse in this resample')
        return real_dmatrices(formula, data=data, **kwargs)

    monkeypatch.setattr(ml, 'dmatrices', sometimes)

    with caplog.at_level(logging.WARNING, logger='spacr.ml'):
        output = ml.perform_regression(parametric_settings(
            screen, regression_type='lasso', alpha=0.0001,
            lasso_n_boot=4))

    assert dropped['n'] == 2
    messages = [record.getMessage() for record in caplog.records]
    assert any('stability selection: 2 of 4 resamples produced no design'
               in message for message in messages)
    assert 'selection_frequency' in output['results'].columns


def test_a_group_lasso_bootstrap_cross_validates_its_own_penalty(screen):
    """The resamples are fitted at a penalty chosen for the design.

    ``alpha`` is not read by this backend at all, and ``float('auto')`` is not
    a number; the block penalty has to be chosen once, from the reference
    design, or every resample would be a different model.
    """
    output = ml.perform_regression(parametric_settings(
        screen, regression_type='group_lasso', group_lasso_lambda='auto',
        lasso_n_boot=3))

    frequencies = output['results']['selection_frequency'].dropna()
    assert len(frequencies) > 0
    assert frequencies.between(0.0, 1.0).all()


# ===========================================================================
# regression(): the paragraph at the end of a fit
# ===========================================================================

from tests.test_cov_12_ml_regression_pipeline import (  # noqa: E402
    NC, PC, wells_frame)


def test_a_summary_with_nothing_to_say_prints_no_heading(tmp_path, monkeypatch,
                                                         capsys):
    """An empty summary prints nothing at all, heading included.

    The paragraph is the last thing on screen when a run finishes; a SUMMARY
    heading with nothing under it reads as a summary that was lost.
    """
    import spacr.figures.summary as summary_module

    monkeypatch.setattr(summary_module, 'summarise', lambda coef_df: '')
    ml.regression(wells_frame(), str(tmp_path / 'scores.csv'),
                  dependent_variable='predictions', regression_type='ols',
                  nc=NC, pc=PC, dst=None, qc=False)
    assert 'SUMMARY' not in capsys.readouterr().out

    monkeypatch.setattr(summary_module, 'summarise',
                        lambda coef_df: 'two genes carry the screen.')
    ml.regression(wells_frame(), str(tmp_path / 'scores.csv'),
                  dependent_variable='predictions', regression_type='ols',
                  nc=NC, pc=PC, dst=None, qc=False)
    printed = capsys.readouterr().out
    assert 'SUMMARY' in printed
    assert 'two genes carry the screen.' in printed


# ===========================================================================
# ml_analysis: which rows the classifier is trained on
# ===========================================================================

from tests.test_cov_12_ml_analysis import COMMON, feature_frame  # noqa: E402


def _control_wells(rows=('r1', 'r2', 'r3'), per_well=10, plates=('plate1',),
                   seed=3):
    """Control-well features with more than one well per class.

    ``feature_frame`` puts each class in a single well, which a well-grouped
    split refuses before any of the branches below are reached; these rows
    give every class independent wells so the split itself succeeds.
    """
    from tests.test_cov_12_ml_analysis import FEATURES

    rng = np.random.default_rng(seed)
    records, index = [], []
    for plate in plates:
        for row in rows:
            for location, centre in (('c1', 0.3), ('c2', 0.9)):
                for _ in range(per_well):
                    record = {'columnID': location, 'rowID': row,
                              'plateID': plate}
                    for name in FEATURES:
                        record[name] = float(rng.normal(centre, 0.1))
                    records.append(record)
                    index.append(f'{plate}_{row}_{location}_f1_o{len(index)}')
    return pd.DataFrame(records, index=index)


def test_the_refusal_names_only_the_control_that_matched_nothing():
    """A half-matched pair names the half that failed.

    Naming both would send the user to look at the setting that is correct,
    and naming neither is the pandas KeyError this replaced.
    """
    frame = feature_frame(per_class=8)

    with pytest.raises(ValueError) as caught:
        ml.ml_analysis(frame, positive_control='c2', negative_control='nope',
                       **COMMON)
    message = str(caught.value)
    assert "negative_control='nope'" in message
    assert 'positive_control=' not in message

    with pytest.raises(ValueError) as caught:
        ml.ml_analysis(frame, positive_control='nope', negative_control='c1',
                       **COMMON)
    other = str(caught.value)
    assert "positive_control='nope'" in other
    assert 'negative_control=' not in other


def test_a_fold_that_cannot_hold_both_classes_is_refused_by_name():
    """Grouped folds that split the classes apart are stopped.

    Each control well holds one class, so a fold that is one whole well
    trains on one class and is scored on the other -- and reports that as an
    accuracy.
    """
    frame = _control_wells(rows=('r1', 'r2'))

    with pytest.raises(ValueError) as caught:
        ml.ml_analysis(frame, positive_control='c2', negative_control='c1',
                       cross_validation=True, split_by='well', **COMMON)

    message = str(caught.value)
    assert 'cannot put every class in both train and test' in message
    assert 'well-grouped CV fold 1' in message


def test_a_correction_with_one_batch_says_it_corrected_nothing(capsys):
    """The batch correction's own warnings reach the console.

    A correction that could not run and one that ran and found nothing print
    the same centroid-spread line; the warning is the only thing that tells
    them apart.
    """
    frame = _control_wells()

    ml.ml_analysis(frame, positive_control='c2', negative_control='c1',
                   batch_correction='center', batch_column='plateID',
                   **COMMON)

    printed = capsys.readouterr().out
    assert 'Batch correction center:' in printed
    assert ('Warning: batch correction: Only 1 batch was present; correction '
            'was a no-op.') in printed


def test_a_verbose_run_shows_the_panels_it_built(capsys):
    """``verbose`` names the features and shows the importance panels.

    The permutation and importance figures are the QC a user reads to decide
    whether the classifier learned the phenotype or the plate.
    """
    frame = _control_wells()

    output, figures = ml.ml_analysis(
        frame, positive_control='c2', negative_control='c1',
        verbose=True, **{**COMMON, 'model_type': 'logistic_regression'})

    printed = capsys.readouterr().out
    assert 'Features used in training:' in printed
    # logistic regression exposes no feature_importances_, so the importance
    # panel is built from the permutation importance and shown under the same
    # rule as a tree's own
    assert len(figures) == 2 and all(fig is not None for fig in figures)
    assert len(output[2]) > 0


def test_a_control_that_is_neither_a_name_nor_a_list_leaves_no_similarity():
    """Only a string and a list name a control; anything else is not read.

    The similarity block is wrapped, so an unsupported control type costs the
    twelve similarity columns and says so rather than stopping the run -- and
    a caller that believes a tuple works would read their absence as "no
    similarity was measurable".
    """
    frame = pd.DataFrame({
        'columnID': ['c1', 'c1', 'c2', 'c2'],
        'a': [0.1, 0.2, 0.9, 1.0],
        'b': [1.0, 0.9, 0.2, 0.1],
    })

    listed = ml._calculate_similarity(frame.copy(), ['a', 'b'], 'columnID',
                                      ['c2'], ['c1'])
    assert 'similarity_to_pos_euclidean' in listed.columns

    tupled = ml._calculate_similarity(frame.copy(), ['a', 'b'], 'columnID',
                                      ('c2',), ('c1',))
    assert 'similarity_to_pos_euclidean' not in tupled.columns
