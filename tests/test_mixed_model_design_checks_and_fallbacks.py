"""Issue 145: the nested mixed model checks its design and says what it fitted.

A user pressed Stop during a slow MixedLM fit and was told, in the words of a
data problem, that their screen had too few genes. Two things must hold: a
stop is a stop and never a fitting error, and a design that cannot be fitted
is refused before the optimiser runs with the dataset's actual problem and the
setting that changes it. When the optimiser, not the data, is what fails, the
documented fallbacks run and the result records which model was fitted.
"""
from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from spacr import ml
from spacr.cancellation import PipelineCancelled
from spacr.mixed_gpu import MixedBackendUnavailable


def screen(n_plates=4, genes=('geneA', 'geneB', 'geneC'), guides=2, seed=0,
           effect=True):
    """A balanced plate x row x column x gene x guide design."""
    rng = np.random.default_rng(seed)
    records = []
    for plate in range(n_plates):
        for row in ('r1', 'r2'):
            for column in ('c1', 'c2'):
                for gene in genes:
                    for guide in range(guides):
                        records.append({
                            'plateID': f'plate{plate + 1}',
                            'rowID': row,
                            'columnID': column,
                            'prc': f'plate{plate + 1}_{row}_{column}',
                            'gene': gene,
                            'grna': f'{gene}_g{guide}',
                            'fraction': float(rng.uniform(0.05, 0.6)),
                        })
    frame = pd.DataFrame(records)
    frame['gene_fraction'] = frame.groupby(['prc', 'gene'])['fraction'] \
        .transform('sum')
    signal = 0.4 * frame['fraction'] if effect else 0.0
    frame['score'] = signal + rng.normal(0, 0.05, len(frame))
    return frame


FORMULA = 'score ~ gene_fraction:gene'


def check(frame, formula=FORMULA, response='score'):
    return ml._check_mixed_design(frame, formula, response, 'gene', 'grna')


@pytest.fixture
def cuda_present(monkeypatch):
    import spacr.regression_backends as backends

    monkeypatch.setattr(backends, 'cuda_present_without_importing_torch',
                        lambda: True)


class FakeModel:
    """A stand-in for a statsmodels MixedLM whose fits are scripted."""

    def __init__(self, script, vc_formula):
        self.script = script
        self.vc_formula = vc_formula

    def loglike(self, *args, **kwargs):
        return 0.0

    def fit(self, **options):
        step = self.script.pop(0)
        self.script_seen.append((dict(options), sorted(self.vc_formula or {})))
        if isinstance(step, Exception):
            raise step
        return step


def script_fits(monkeypatch, steps):
    """Make ``smf.mixedlm`` hand out models that play ``steps`` in order."""
    seen = []
    script = list(steps)

    def factory(formula, data=None, groups=None, re_formula=None,
                vc_formula=None):
        model = FakeModel(script, vc_formula)
        model.script_seen = seen
        return model

    monkeypatch.setattr(ml.smf, 'mixedlm', factory)
    return seen


def result(converged=True):
    return types.SimpleNamespace(converged=converged)


def fit(frame=None, random_row_column_effects=False):
    frame = screen() if frame is None else frame
    vc = {'grna': '0 + C(grna)'}
    if random_row_column_effects:
        vc.update(rowID='0 + C(rowID)', columnID='0 + C(columnID)')
    return ml._fit_nested_mixedlm(FORMULA, frame, frame['gene'], vc, 'grna',
                                  {'n_rows': len(frame)})


def test_a_stop_during_the_fit_is_a_stop_not_a_data_problem(monkeypatch):
    """The reporter's traceback: Stop raised inside loglike became a ValueError."""
    import spacr.cancellation as cancellation

    def stop():
        raise PipelineCancelled('stopped by the user')

    monkeypatch.setattr(cancellation, 'checkpoint', stop)
    frame = screen()

    with pytest.raises(PipelineCancelled, match='stopped by the user'):
        ml.fit_mixed_model(frame, FORMULA, dst=None)


def test_a_stop_inside_the_gpu_fit_is_also_a_stop(cuda_present, monkeypatch):
    import spacr.mixed_gpu as mixed_gpu

    def stopped(formula, frame, groups, vc_formula=None):
        raise PipelineCancelled('stopped by the user')

    monkeypatch.setattr(mixed_gpu, 'mixedlm_torch', stopped)

    with pytest.raises(PipelineCancelled):
        ml.fit_mixed_model(screen(), FORMULA, dst=None,
                           regression_backend='torch')


def test_a_stop_between_optimisers_is_not_swallowed_by_the_fallback(
        monkeypatch):
    script_fits(monkeypatch, [ValueError('first'),
                              PipelineCancelled('stopped')])

    with pytest.raises(PipelineCancelled):
        fit()


def test_a_good_design_is_described(capsys):
    facts = check(screen(genes=('a', 'b', 'c', 'd')))

    assert facts['n_genes'] == 4
    assert facts['n_guides'] == 8
    assert facts['genes_with_several_guides'] == 4
    assert facts['genes_with_one_guide'] == 0
    assert facts['fixed_rank'] == facts['fixed_columns']
    assert facts['min_rows_per_guide'] == 16


def test_a_response_the_frame_does_not_have_is_left_to_the_formula():
    frame = screen().drop(columns='score')

    assert check(frame, 'gene_fraction ~ gene_fraction:gene',
                 response='absent')['n_genes'] == 3


def test_a_response_with_no_finite_value_names_the_setting():
    frame = screen()
    frame['score'] = np.nan

    with pytest.raises(ValueError) as caught:
        check(frame)

    text = str(caught.value)
    assert 'no finite value' in text and 'dependent_variable' in text
    assert "inference='nonparametric'" in text


def test_rows_without_a_finite_response_are_reported_and_left_out(capsys):
    frame = screen()
    frame.loc[frame.index[:5], 'score'] = np.nan

    facts = check(frame)

    assert facts['n_rows'] == len(frame) - 5
    assert '5 of' in capsys.readouterr().out


def test_a_constant_response_says_its_variance_is_zero():
    frame = screen()
    frame['score'] = 0.25

    with pytest.raises(ValueError, match='variance is zero') as caught:
        check(frame)

    assert 'min_cells_per_well' in str(caught.value)


def test_rows_without_a_gene_label_are_refused_with_the_count():
    frame = screen()
    frame['gene'] = frame['gene'].astype(object)
    frame.loc[frame.index[:3], 'gene'] = None

    with pytest.raises(ValueError, match='3 rows have no') as caught:
        check(frame)

    assert 'exclude_grnas' in str(caught.value)


def test_too_few_genes_among_the_usable_rows_is_refused():
    frame = screen(genes=('geneA', 'geneB'))
    frame.loc[frame['gene'] == 'geneB', 'score'] = np.nan

    with pytest.raises(ValueError, match='at least two clusters') as caught:
        check(frame)

    assert 'regression_type' in str(caught.value)


def test_one_guide_per_gene_is_refused_with_the_fixed_effects_way_out():
    frame = screen(guides=1)

    with pytest.raises(ValueError,
                       match='no gene in this frame has more than one guide'):
        check(frame)


def test_guides_that_appear_once_cannot_carry_a_variance():
    frame = screen(n_plates=1, genes=('geneA', 'geneB'))
    frame = frame.drop_duplicates('grna').reset_index(drop=True)

    with pytest.raises(ValueError, match='exactly one row') as caught:
        check(frame)

    assert 'min_cells_per_well' in str(caught.value)


def test_a_formula_the_frame_cannot_build_is_named():
    with pytest.raises(ValueError, match='could not be built') as caught:
        check(screen(), 'score ~ not_a_column:gene')

    assert 'not_a_column' in str(caught.value)


def test_more_fixed_effect_columns_than_rows_is_refused():
    frame = screen(n_plates=1, genes=('a', 'b'), guides=2)

    with pytest.raises(ValueError,
                       match='no residual degrees of freedom') as caught:
        check(frame, 'score ~ gene_fraction:gene + C(prc):C(grna)')

    assert 'model_plate_position=False' in str(caught.value)


def test_a_gene_that_is_never_present_makes_its_slope_unidentified():
    frame = screen()
    frame.loc[frame['gene'] == 'geneC', 'gene_fraction'] = 0.0

    with pytest.raises(ValueError, match='rank') as caught:
        check(frame)

    text = str(caught.value)
    assert 'geneC' in text and 'not identified' in text
    assert 'and more' not in text
    assert 'model_plate_position=False' in text


def test_many_unidentified_columns_are_listed_briefly():
    genes = tuple(f'g{i}' for i in range(9))
    frame = screen(genes=genes, n_plates=6)
    frame.loc[frame['gene'] != 'g0', 'gene_fraction'] = 0.0

    with pytest.raises(ValueError, match='and more'):
        check(frame)


def test_a_real_fit_on_a_refused_design_never_reaches_the_optimiser(
        monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('the optimiser must not run')

    monkeypatch.setattr(ml.smf, 'mixedlm', forbidden)
    frame = screen(guides=1)

    with pytest.raises(ValueError, match='fixed-effects regression_type'):
        ml.fit_mixed_model(frame, FORMULA, dst=None)


def test_the_default_optimiser_converging_changes_nothing(monkeypatch):
    seen = script_fits(monkeypatch, [result()])

    fitted, record = fit()

    assert fitted.converged
    assert record['model'] == ml._MIXED_MODEL_FULL
    assert record['fallbacks'] == []
    assert record['optimiser'] == 'statsmodels default (bfgs, then lbfgs)'
    assert seen == [({}, ['grna'])]
    assert record['variance_components'] == ['grna']


def test_a_failed_default_falls_back_to_lbfgs_with_reml_and_says_so(
        monkeypatch):
    seen = script_fits(monkeypatch, [np.linalg.LinAlgError('Singular matrix'),
                                     result()])

    _fitted, record = fit()

    assert record['optimiser'] == 'lbfgs'
    assert seen[1][0] == {'method': 'lbfgs', 'reml': True}
    assert record['model'] == ml._MIXED_MODEL_FULL
    assert len(record['fallbacks']) == 1
    assert 'lbfgs with REML was used instead' in record['fallbacks'][0]
    assert 'Singular matrix' in record['attempts'][0]


def test_unconverged_fits_are_retried_until_one_converges(monkeypatch):
    script_fits(monkeypatch, [result(False), result(False), result(True)])

    fitted, record = fit()

    assert fitted.converged and record['optimiser'] == 'powell'
    assert any('did not converge' in line for line in record['attempts'])


def test_when_nothing_converges_the_first_result_is_returned_as_it_is(
        monkeypatch):
    first = result(False)
    script_fits(monkeypatch, [first, result(False), result(False),
                              result(False)])

    fitted, record = fit()

    assert fitted is first
    assert record['model'] == ml._MIXED_MODEL_FULL
    assert record['optimiser'] == 'statsmodels default (bfgs, then lbfgs)'
    assert record['fallbacks'] == []


def test_the_guide_component_is_dropped_only_when_every_optimiser_raised(
        monkeypatch):
    boom = np.linalg.LinAlgError('Singular matrix')
    seen = script_fits(monkeypatch, [boom] * 4 + [result()])

    _fitted, record = fit(random_row_column_effects=True)

    assert record['model'] == ml._MIXED_MODEL_GENE_ONLY
    assert record['variance_components'] == ['rowID', 'columnID']
    assert sorted(seen[4][1]) == ['columnID', 'rowID']
    assert 'Guides that disagree no longer widen' in ' '.join(
        record['fallbacks'])
    assert "inference='nonparametric'" in ' '.join(record['fallbacks'])


def test_a_gene_only_fit_that_does_not_converge_is_still_labelled(monkeypatch):
    boom = ValueError('bad')
    unconverged = result(False)
    script_fits(monkeypatch, [boom] * 4 + [unconverged, boom, boom, boom])

    fitted, record = fit()

    assert fitted is unconverged
    assert record['model'] == ml._MIXED_MODEL_GENE_ONLY
    assert record['variance_components'] == []


def test_a_model_no_optimiser_can_fit_lists_every_attempt(monkeypatch):
    script_fits(monkeypatch, [ValueError(f'bad {i}') for i in range(8)])

    with pytest.raises(ValueError, match='no optimiser could fit') as caught:
        fit()

    text = str(caught.value)
    assert 'bad 0' in text and 'bad 7' in text
    assert 'powell failed' in text


def test_the_real_gene_only_fallback_gives_a_labelled_fit(monkeypatch,
                                                          capsys):
    """End to end: the full model fails, the gene-only model is really fitted."""
    real = ml.smf.mixedlm

    def failing_with_guides(formula, data=None, groups=None, re_formula=None,
                            vc_formula=None):
        if vc_formula and 'grna' in vc_formula:
            raise np.linalg.LinAlgError('Singular matrix')
        return real(formula, data=data, groups=groups,
                    re_formula=re_formula, vc_formula=vc_formula)

    monkeypatch.setattr(ml.smf, 'mixedlm', failing_with_guides)
    frame = screen()

    _fitted, coef_df = ml.fit_mixed_model(frame, FORMULA, dst=None)

    record = coef_df.attrs['mixed_fit']
    printed = capsys.readouterr().out
    assert record['model'] == ml._MIXED_MODEL_GENE_ONLY
    assert 'Mixed model fitted: gene fixed; random gene intercept only' \
        in printed
    assert 'Fallback: every optimiser failed on the full model' in printed
    assert not (coef_df['term_type'] == ml.TERM_BLUP).any()
    assert (coef_df['term_type'] == ml.TERM_FIXED).any()


def test_a_guide_free_response_fits_and_the_zero_variance_is_reported(capsys):
    """A singular guide covariance is a boundary answer, and is said plainly."""
    frame = screen(effect=False, seed=3)

    _fitted, coef_df = ml.fit_mixed_model(frame, FORMULA, dst=None)

    record = coef_df.attrs['mixed_fit']
    printed = capsys.readouterr().out
    assert record['model'] == ml._MIXED_MODEL_FULL
    assert record['design']['n_genes'] == 3
    assert 'Mixed model fitted: gene fixed; guides nested in genes' in printed
    if record['variances_at_zero']:
        assert 'Variance estimated at zero for' in printed


def test_variances_on_the_boundary_are_named_and_the_text_says_what_follows():
    fitted = types.SimpleNamespace(vcomp=np.array([0.0, 0.3]))

    assert ml._mixed_variances_at_zero(fitted, ['grna', 'rowID']) == ['grna']
    assert ml._mixed_variances_at_zero(types.SimpleNamespace(), ['a']) == []

    text = ml._describe_mixed_fit({
        'model': ml._MIXED_MODEL_FULL, 'optimiser': 'lbfgs',
        'fallbacks': ['an optimiser note'],
        'variances_at_zero': ['grna']})
    assert 'Fallback: an optimiser note.' in text
    assert 'Variance estimated at zero for grna' in text
    quiet = ml._describe_mixed_fit({
        'model': ml._MIXED_MODEL_FULL, 'fallbacks': [],
        'variances_at_zero': []})
    assert 'optimiser unknown' in quiet and 'Fallback' not in quiet


def test_a_run_that_did_not_converge_says_so_with_the_fitted_components(
        cuda_present, monkeypatch, capsys):
    """The warning banner reads the components of the model actually fitted."""
    import spacr.mixed_gpu as mixed_gpu

    frame = screen()
    index = ['Intercept', 'gene_fraction:gene[T.geneB]']
    unconverged = types.SimpleNamespace(
        summary_line=lambda: 'torch nested REML: not converged',
        resid=np.zeros(len(frame)), converged=False,
        vcomp=np.array([0.0]), cov_re=np.array([[0.01]]),
        fe_params=pd.Series([0.1, 0.4], index=index),
        params=pd.Series([0.1, 0.4, 0.02], index=index + ['gene Var']),
        pvalues=pd.Series([0.3, 1e-4, 0.7], index=index + ['gene Var']),
        random_effects={})
    monkeypatch.setattr(mixed_gpu, 'mixedlm_torch',
                        lambda *args, **kwargs: unconverged)

    _model, coef_df = ml.fit_mixed_model(frame, FORMULA, dst=None,
                                         regression_backend='torch')

    printed = capsys.readouterr().out
    assert 'WARNING: the mixed model did not converge' in printed
    assert 'grna=0' in printed
    record = coef_df.attrs['mixed_fit']
    assert record['optimiser'] == 'torch nested REML'
    assert record['converged'] is False
    assert record['variances_at_zero'] == ['grna']


def run_regression_with_record(monkeypatch, record):
    coefficients = pd.DataFrame(
        {'feature': ['gene_fraction:gene[gene1]'], 'coefficient': [0.5],
         'p_value': [0.01], 'term_type': [ml.TERM_FIXED]})
    if record is not None:
        coefficients.attrs['mixed_fit'] = record
    monkeypatch.setattr(ml, 'fit_mixed_model',
                        lambda *args, **kwargs: (object(),
                                                 coefficients.copy()))
    rows = []
    for well in range(1, 9):
        for guide, gene, fraction in (("g1", "gene1", well / 20),
                                      ("g2", "gene2", (9 - well) / 20)):
            rows.append({
                "prc": f"p1_r{well}_c1", "plateID": "p1",
                "rowID": f"r{well}", "columnID": "c1", "grna": guide,
                "gene": gene, "fraction": fraction,
                "gene_fraction": fraction,
                "predictions": 0.1 + 0.8 * well / 8,
                "cell_count": 100 + well})

    _model, observed, kind = ml.regression(
        pd.DataFrame(rows), 'counts.csv', regression_type='mixed',
        dependent_variable='predictions', controls=[], dst=None, qc=False,
        plot=False)

    assert kind == 'mixed'
    return observed.attrs['fit_design']


def test_a_mixed_fit_without_a_record_adds_nothing_to_the_design_file(
        monkeypatch):
    design = run_regression_with_record(monkeypatch, None)

    assert 'mixed_model_fitted' not in design


def test_the_fit_record_reaches_the_per_level_design_file(monkeypatch):
    record = {'model': ml._MIXED_MODEL_GENE_ONLY, 'optimiser': 'lbfgs',
              'converged': True, 'fallbacks': ['dropped the guide component'],
              'variances_at_zero': ['rowID']}
    design = run_regression_with_record(monkeypatch, record)

    assert design['mixed_model_fitted'] == ml._MIXED_MODEL_GENE_ONLY
    assert design['mixed_model_optimiser'] == 'lbfgs'
    assert design['mixed_model_converged'] is True
    assert design['mixed_model_fallbacks'] == ['dropped the guide component']
    assert design['mixed_model_variances_at_zero'] == ['rowID']


def test_a_backend_refusal_still_passes_through(cuda_present, monkeypatch):
    import spacr.mixed_gpu as mixed_gpu

    def refuse(formula, frame, groups, vc_formula=None):
        raise MixedBackendUnavailable('no CUDA device on this machine')

    monkeypatch.setattr(mixed_gpu, 'mixedlm_torch', refuse)

    with pytest.raises(MixedBackendUnavailable):
        ml.fit_mixed_model(screen(), FORMULA, dst=None,
                           regression_backend='torch')
