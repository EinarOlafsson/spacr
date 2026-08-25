"""The mixed fits: choosing a control offset, and what the GPU backend costs.

Two things must hold whatever happens. A control-centred response must only be
shifted when there is a real control median to shift it by, because a wrong
offset moves every reported effect. And a fit dispatched to the GPU must come
back with the same numbers or say plainly that it did not run there -- a fit
that silently fell back is the slow run the user was avoiding, reported as the
fast one.
"""
from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from spacr import ml
from spacr.mixed_gpu import MixedBackendUnavailable


@pytest.fixture
def cuda_present(monkeypatch):
    """Let the backend check pass, so the torch branch can be reached."""
    import spacr.regression_backends as backends

    monkeypatch.setattr(backends, 'cuda_present_without_importing_torch',
                        lambda: True)


def mixed_model_frame(n_plates=4, seed=0):
    """A balanced plate x row x column x gene x guide design."""
    rng = np.random.default_rng(seed)
    records = []
    for plate in range(n_plates):
        for row in ('r1', 'r2'):
            for column in ('c1', 'c2'):
                for gene in ('geneA', 'geneB'):
                    for guide in (1, 2):
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
    frame['score'] = (0.4 * frame['fraction']
                      + 0.1 * (frame['gene'] == 'geneA').astype(float)
                      + rng.normal(0, 0.05, len(frame)))
    return frame


# ---------------------------------------------------------------------------
# centring on the negative control
# ---------------------------------------------------------------------------

def test_no_control_named_leaves_the_response_exactly_as_it_was():
    """With no control, or a response column that is not there, nothing shifts.

    Centring on a control the frame does not contain would either shift by a
    number computed from nothing or fail; returning the frame untouched with a
    zero offset lets the caller say so.
    """
    frame = pd.DataFrame({'grna': ['g1', 'g2'], 'score': [0.4, 0.6]})

    assert ml.centre_on_controls(frame, 'score', '') == (frame, 0.0)
    assert ml.centre_on_controls(frame, 'missing_column', 'nc') == (frame, 0.0)
    assert ml.centre_on_controls(frame, 'score', '   ') == (frame, 0.0)


def test_control_rows_with_no_finite_value_shift_nothing():
    """A control whose response is all NaN gives no median, so no offset.

    A median over an empty selection is NaN, and subtracting NaN would erase
    the whole response column.
    """
    frame = pd.DataFrame({'grna': ['nc', 'nc', 'g1'],
                          'score': [np.nan, np.inf, 0.6]})
    shifted, offset = ml.centre_on_controls(frame, 'score', 'nc')

    assert offset == 0.0
    assert shifted is frame


def test_a_control_median_of_exactly_zero_is_not_a_shift():
    """When the control already sits at zero the frame is returned unchanged.

    Copying the frame to subtract zero costs memory on a screen-sized table and
    reports an offset that did nothing.
    """
    frame = pd.DataFrame({'grna': ['nc', 'nc', 'g1'],
                          'score': [-0.2, 0.2, 0.6]})
    shifted, offset = ml.centre_on_controls(frame, 'score', 'nc')

    assert offset == 0.0
    assert shifted is frame


# ---------------------------------------------------------------------------
# the GPU mixed fit
# ---------------------------------------------------------------------------

def test_a_gpu_fit_that_runs_out_of_memory_falls_back_to_the_cpu(
        cuda_present, monkeypatch, capsys):
    """An OOM on the shared card is announced and the same model is fitted on CPU.

    The card is shared, so what was free when the design was checked can be
    gone by the time the fit asks for it. Failing there would lose a run that
    the CPU can complete, twenty minutes in and with nobody to ask.
    """
    import spacr.mixed_gpu as mixed_gpu

    def out_of_memory(y, X, groups):
        raise RuntimeError('CUDA error: out of memory')

    monkeypatch.setattr(mixed_gpu, 'fit_mixed_reml_torch', out_of_memory)

    rng = np.random.default_rng(1)
    design = pd.DataFrame({'a': rng.normal(0, 1, 60),
                           'b': rng.normal(0, 1, 60)})
    groups = np.repeat(np.arange(6), 10)
    y = 1.5 * design['a'] + rng.normal(0, 0.3, 60)

    result = ml.perform_mixed_model(y, design, groups,
                                    regression_backend='torch')

    printed = capsys.readouterr().out
    assert 'The GPU ran out of memory during the mixed fit' in printed
    assert 'same model, same numbers' in printed
    assert list(result.params.index) == ['a', 'b', 'Group Var']
    assert abs(float(result.fe_params['a']) - 1.5) < 0.25


def test_a_gpu_failure_that_is_not_memory_is_not_quietly_retried(
        cuda_present, monkeypatch):
    """Any other GPU error propagates, so a broken backend is not hidden.

    Falling back for every failure would turn a genuinely wrong torch fit into
    a silent CPU run, and nothing would say the GPU path is broken.
    """
    import spacr.mixed_gpu as mixed_gpu

    def broken(y, X, groups):
        raise RuntimeError('the kernel image is invalid for the device')

    monkeypatch.setattr(mixed_gpu, 'fit_mixed_reml_torch', broken)

    rng = np.random.default_rng(2)
    design = pd.DataFrame({'a': rng.normal(0, 1, 40),
                           'b': rng.normal(0, 1, 40)})
    groups = np.repeat(np.arange(4), 10)
    y = design['a'] + rng.normal(0, 0.3, 40)

    with pytest.raises(RuntimeError, match='kernel image'):
        ml.perform_mixed_model(y, design, groups,
                               regression_backend='torch')


def test_a_gpu_fit_that_succeeds_is_returned_with_its_summary_line(
        cuda_present, monkeypatch, capsys):
    """The torch result is handed back as-is, after printing its summary line.

    Nothing downstream may be able to tell which backend ran, so the result
    object is passed through untouched and only the console says.
    """
    import spacr.mixed_gpu as mixed_gpu

    fitted = types.SimpleNamespace(
        summary_line=lambda: 'torch REML: converged in 12 iterations')
    monkeypatch.setattr(mixed_gpu, 'fit_mixed_reml_torch',
                        lambda y, X, groups: fitted)

    rng = np.random.default_rng(3)
    design = pd.DataFrame({'a': rng.normal(0, 1, 40),
                           'b': rng.normal(0, 1, 40)})
    groups = np.repeat(np.arange(4), 10)
    y = design['a'] + rng.normal(0, 0.3, 40)

    assert ml.perform_mixed_model(y, design, groups,
                                  regression_backend='torch') is fitted
    assert 'torch REML: converged' in capsys.readouterr().out


# ---------------------------------------------------------------------------
# the nested mixed fit
# ---------------------------------------------------------------------------

def test_the_nested_fit_on_the_gpu_uses_the_same_call_shape(
        cuda_present, monkeypatch, capsys):
    """``mixedlm_torch`` is handed the formula, frame, groups and vc_formula.

    It takes statsmodels' argument shape on purpose, so who fits the model
    cannot become a second code path with its own bugs.
    """
    import spacr.mixed_gpu as mixed_gpu

    seen = {}

    class Fitted:
        params = pd.Series({'Intercept': 0.1})
        fe_params = pd.Series({'Intercept': 0.1})

        def summary_line(self):
            return 'torch nested REML'

    def record(formula, frame, groups, vc_formula=None):
        seen['formula'] = formula
        seen['groups'] = groups
        seen['vc_formula'] = dict(vc_formula or {})
        raise ValueError('stop here; the call shape is what is under test')

    monkeypatch.setattr(mixed_gpu, 'mixedlm_torch', record)
    frame = mixed_model_frame()
    formula = ml.prepare_formula('score', random_row_column_effects=True,
                                 level='gene')

    with pytest.raises(ValueError, match='MixedLM could not fit'):
        ml.fit_mixed_model(frame, formula, dst=None,
                           random_row_column_effects=True,
                           regression_backend='torch')

    assert seen['formula'] == formula
    assert seen['vc_formula']['grna'] == '0 + C(grna)'
    assert seen['vc_formula']['rowID'] == '0 + C(rowID)'
    assert seen['vc_formula']['columnID'] == '0 + C(columnID)'


def test_a_backend_refusal_survives_the_nested_fit_untouched(cuda_present,
                                                             monkeypatch):
    """``MixedBackendUnavailable`` is re-raised, not rewrapped as a data problem.

    Wrapped in "MixedLM could not fit this frame" it would send the user
    looking at their screen for a missing CUDA device.
    """
    import spacr.mixed_gpu as mixed_gpu

    def refuse(formula, frame, groups, vc_formula=None):
        raise MixedBackendUnavailable('no CUDA device on this machine')

    monkeypatch.setattr(mixed_gpu, 'mixedlm_torch', refuse)
    frame = mixed_model_frame()
    formula = ml.prepare_formula('score', level='gene')

    with pytest.raises(MixedBackendUnavailable, match='no CUDA device'):
        ml.fit_mixed_model(frame, formula, dst=None,
                           regression_backend='torch')


def test_a_cache_that_will_not_empty_does_not_stop_the_cpu_fallback(
        cuda_present, monkeypatch, capsys):
    """Freeing the GPU cache is best-effort; the CPU fit runs either way.

    The fallback exists because the run must finish. A torch that will not
    release its cache is one more reason to be on the CPU, not a reason to fail.
    """
    import torch
    import spacr.mixed_gpu as mixed_gpu

    def out_of_memory(y, X, groups):
        raise MemoryError('the design will not fit on this device')

    def will_not_empty():
        raise RuntimeError('CUDA context is already torn down')

    monkeypatch.setattr(mixed_gpu, 'fit_mixed_reml_torch', out_of_memory)
    monkeypatch.setattr(torch.cuda, 'empty_cache', will_not_empty)

    rng = np.random.default_rng(9)
    design = pd.DataFrame({'a': rng.normal(0, 1, 40),
                           'b': rng.normal(0, 1, 40)})
    groups = np.repeat(np.arange(4), 10)
    y = design['a'] + rng.normal(0, 0.3, 40)

    result = ml.perform_mixed_model(y, design, groups,
                                    regression_backend='torch')

    assert 'The GPU ran out of memory' in capsys.readouterr().out
    assert list(result.params.index) == ['a', 'b', 'Group Var']


def test_a_nested_gpu_fit_that_succeeds_reports_its_backend(cuda_present,
                                                            monkeypatch,
                                                            capsys):
    """A torch nested fit is read exactly as the statsmodels one is.

    Everything after the call reads the result the same way, so the choice of
    who fits it cannot become a second code path with its own bugs -- and the
    run says which backend produced the numbers.
    """
    import spacr.mixed_gpu as mixed_gpu

    frame = mixed_model_frame()
    formula = ml.prepare_formula('score', level='gene')

    index = ['Intercept', 'gene_fraction:gene[T.geneB]']
    fitted = types.SimpleNamespace(
        summary_line=lambda: 'torch nested REML: converged',
        resid=np.zeros(len(frame)),
        fe_params=pd.Series([0.1, 0.4], index=index),
        params=pd.Series([0.1, 0.4, 0.02], index=index + ['gene Var']),
        pvalues=pd.Series([0.3, 1e-4, 0.7], index=index + ['gene Var']),
        random_effects={},
    )
    monkeypatch.setattr(mixed_gpu, 'mixedlm_torch',
                        lambda *a, **k: fitted)

    model, coef_df = ml.fit_mixed_model(frame, formula, dst=None,
                                        regression_backend='torch')

    printed = capsys.readouterr().out
    assert 'torch nested REML: converged' in printed
    assert 'regression_backend=' in printed
    assert model is fitted
    # The variance component keeps its value and loses its p-value: the null
    # it would test sits on the boundary of the parameter space.
    variance_row = coef_df.loc[coef_df['feature'] == 'gene Var']
    assert float(variance_row['coefficient'].iloc[0]) == pytest.approx(0.02)
    assert np.isnan(float(variance_row['p_value'].iloc[0]))
