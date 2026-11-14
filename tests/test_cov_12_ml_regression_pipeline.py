"""What the regression pipeline says the intercept means, and what it draws.

The intercept is a choice with four answers and the coefficients read
differently under each, so the run has to state which one it used -- including
when the answer it was asked for could not be applied. The drawing and
summarising steps around it are courtesies and are checked here for the same
property every courtesy in this module has: failing costs the courtesy and
nothing else.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from spacr import ml


NC = '233460'
PC = '220950'
GENES = [NC, PC, 'gene3', 'gene4']


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


def wells_frame(seed=0, n_rows=4, n_columns=6, screens=None):
    """Long-format score/count table: one row per (well, gRNA)."""
    rng = np.random.default_rng(seed)
    guides = {gene: [f'{gene}_a', f'{gene}_b'] for gene in GENES}
    records = []
    for row in range(n_rows):
        for column in range(n_columns):
            row_id = f'r{row + 1:02d}'
            column_id = f'c{column + 1:02d}'
            well = f'plate1_{row_id}_{column_id}'
            raw = rng.random(len(GENES) * 2) + 0.2
            fractions = raw / raw.sum()
            score = float(rng.normal(0.0, 1.0))
            index = 0
            for gene in GENES:
                for guide in guides[gene]:
                    record = {
                        'plateID': 'plate1', 'rowID': row_id,
                        'columnID': column_id, 'prc': well,
                        'gene': gene, 'grna': guide,
                        'fraction': float(fractions[index]),
                        'cell_count': int(rng.integers(30, 200)),
                        'predictions': score,
                    }
                    if screens is not None:
                        record['screenID'] = screens[
                            (row * n_columns + column) % len(screens)]
                    records.append(record)
                    index += 1
    return pd.DataFrame(records)


def fit(frame, tmp_path, **kwargs):
    kwargs.setdefault('regression_type', 'ols')
    kwargs.setdefault('nc', NC)
    kwargs.setdefault('pc', PC)
    return ml.regression(frame, str(tmp_path / 'scores.csv'),
                         dependent_variable='predictions',
                         dst=None, qc=False, **kwargs)


def case_an_intercept_centred_on_the_controls_says_what_it_shifted_by(
        tmp_path, capsys):
    """The run names the offset, so a coefficient reads as a distance from nc.

    Without the sentence, two runs of the same screen under different intercept
    modes produce different coefficients with nothing on the record saying why.
    """
    _seen = []
    frame = wells_frame(seed=3)
    _model, coef_df, _kind = fit(frame, tmp_path, intercept='control')

    printed = capsys.readouterr().out
    _seen.append(('Intercept set to the negative controls' if 'Intercept set to the negative controls' in (printed) else ("missing", 'Intercept set to the negative controls')))
    _seen.append(bool(
        f"distance from {NC!r}" in printed))
    _seen.append(bool(
        len(coef_df) > 0))
    return _seen


def case_an_intercept_centred_on_a_control_that_is_absent_says_it_was_not(
        tmp_path, capsys):
    """No matching control leaves the intercept fitted, and reports that.

    Silently fitting the intercept anyway would mean the coefficients are not
    what the setting asked for and nothing anywhere would say so.
    """
    _seen = []
    frame = wells_frame(seed=4)
    fit(frame, tmp_path, intercept='control', nc='not-a-guide-in-this-screen')

    printed = capsys.readouterr().out
    _seen.append(('Intercept left as fitted' if 'Intercept left as fitted' in (printed) else ("missing", 'Intercept left as fitted')))
    _seen.append(('no rows match' if 'no rows match' in (printed) else ("missing", 'no rows match')))
    return _seen


def case_an_intercept_pinned_at_a_value_shifts_the_response_by_it(tmp_path,
                                                                  capsys):
    """The response is moved by the number, so the intercept is exactly it.

    An estimated intercept would land near the number and read as though the
    value had been a suggestion.
    """
    _seen = []
    frame = wells_frame(seed=5)
    fit(frame, tmp_path, intercept='value', intercept_value=0.5)

    printed = capsys.readouterr().out
    _seen.append(('Intercept pinned at 0.5' if 'Intercept pinned at 0.5' in (printed) else ("missing", 'Intercept pinned at 0.5')))
    _seen.append(('distance from that value' if 'distance from that value' in (printed) else ("missing", 'distance from that value')))
    return _seen


def case_an_intercept_pinned_at_zero_is_still_announced(tmp_path, capsys):
    """Pinning at zero changes no value but is still stated.

    Zero is a real answer for a centred response; leaving it unsaid would make
    the pinned run indistinguishable from the default one in the log.
    """
    _seen = []
    frame = wells_frame(seed=6)
    before = frame['predictions'].to_numpy(dtype=float).copy()

    fit(frame, tmp_path, intercept='value', intercept_value=0.0)

    _seen.append(('Intercept pinned at 0' if 'Intercept pinned at 0' in (capsys.readouterr().out) else ("missing", 'Intercept pinned at 0')))
    _seen.append(bool(
        np.array_equal(frame['predictions'].to_numpy(dtype=float), before)))
    return _seen


def case_a_multi_screen_frame_is_blocked_on_and_the_screens_are_named(
        tmp_path, capsys):
    """More than one screen becomes a design term, and the run lists them.

    A single-screen frame must not be blocked on -- the term would be a
    constant column and the design rank deficient -- so which happened has to
    be on the record. The decision is taken from the CLEANED frame and patsy
    builds the term from that same frame, so ``screenID`` has to survive
    cleaning; a frame that loses it pools two screens with nothing said and
    charges the difference between the experiments to the guides.
    """
    _seen = []
    frame = wells_frame(seed=7, screens=['screenA', 'screenB'])
    _seen.append(bool(
        ml.screen_is_blockable(frame) is True))

    fit(frame, tmp_path)

    printed = capsys.readouterr().out
    _seen.append(('Blocking on 2 screens' if 'Blocking on 2 screens' in (printed) else ("missing", 'Blocking on 2 screens')))
    _seen.append(("'screenA', 'screenB'" if "'screenA', 'screenB'" in (printed) else ("missing", "'screenA', 'screenB'")))
    return _seen


def case_the_old_histograms_are_drawn_when_the_new_panels_cannot_be(
        tmp_path, monkeypatch, capsys):
    """A distribution panel that declines sends the run back to plot_histogram.

    The point of the fallback is that a figure is never worth losing a fit
    over, and the run must still show the response it fitted.
    """
    _seen = []
    drawn = []

    def declined(frame, response_name, dst, plot=True):
        return False

    def record(data, variable, dst=None, **kwargs):
        drawn.append(variable)

    import spacr.plot as plot_module

    monkeypatch.setattr(ml, '_show_well_distributions', declined)
    monkeypatch.setattr(plot_module, 'plot_histogram', record)

    fit(wells_frame(seed=8), tmp_path, plot=True)

    _seen.append(keep(drawn))
    return _seen


def case_a_summary_that_cannot_be_written_does_not_lose_the_fit(
        tmp_path, monkeypatch, capsys):
    """A failure in the closing paragraph is printed and the fit is returned.

    The summary is prose about numbers that are already computed; nothing about
    it is worth a run.
    """
    _seen = []
    import spacr.figures.summary as summary_module

    def boom(coef_df):
        raise RuntimeError('the summariser could not read this table')

    monkeypatch.setattr(summary_module, 'summarise', boom)

    _model, coef_df, kind = fit(wells_frame(seed=9), tmp_path)

    _seen.append(keep(kind))
    _seen.append(bool(
        len(coef_df) > 0))
    _seen.append(('Could not summarise the run' if 'Could not summarise the run' in (capsys.readouterr().out) else ("missing", 'Could not summarise the run')))
    return _seen


def case_a_mixed_run_says_that_the_level_setting_is_not_read(tmp_path, capsys,
                                                             monkeypatch):
    """``regression_type='mixed'`` is already both levels, and says so.

    The GUI greys the level out, but a script can still set it; quietly
    ignoring it would hand back a gene table to a caller who asked for guides.
    """
    _seen = []
    def one_fit(*args, **kwargs):
        return None, pd.DataFrame({'feature': ['Intercept'],
                                   'coefficient': [0.1],
                                   'p_value': [0.5]}), 'mixed'

    monkeypatch.setattr(ml, 'regression', one_fit)

    ml.regression_levels(wells_frame(seed=10), str(tmp_path / 'scores.csv'),
                         regression_type='mixed', level='grna')

    printed = capsys.readouterr().out
    _seen.append(("regression_type='mixed' fits the gene fixed" if "regression_type='mixed' fits the gene fixed" in (printed) else ("missing", "regression_type='mixed' fits the gene fixed")))
    _seen.append(('is not read' if 'is not read' in (printed) else ("missing", 'is not read')))
    return _seen


from tests.observed_case import keep, observe  # noqa: E402

@pytest.mark.parametrize(("case", "expected"), [
    pytest.param(case_an_intercept_centred_on_the_controls_says_what_it_shifted_by, ['Intercept set to the negative controls', True, True], id="an_intercept_centred_on_the_controls_says_what_it_shifted_by"),
    pytest.param(case_an_intercept_centred_on_a_control_that_is_absent_says_it_was_not, ['Intercept left as fitted', 'no rows match'], id="an_intercept_centred_on_a_control_that_is_absent_says_it_was_not"),
    pytest.param(case_an_intercept_pinned_at_a_value_shifts_the_response_by_it, ['Intercept pinned at 0.5', 'distance from that value'], id="an_intercept_pinned_at_a_value_shifts_the_response_by_it"),
    pytest.param(case_an_intercept_pinned_at_zero_is_still_announced, ['Intercept pinned at 0', True], id="an_intercept_pinned_at_zero_is_still_announced"),
    pytest.param(case_the_old_histograms_are_drawn_when_the_new_panels_cannot_be, [['predictions', 'fraction']], id="the_old_histograms_are_drawn_when_the_new_panels_cannot_be"),
    pytest.param(case_a_summary_that_cannot_be_written_does_not_lose_the_fit, ['ols', True, 'Could not summarise the run'], id="a_summary_that_cannot_be_written_does_not_lose_the_fit"),
    pytest.param(case_a_multi_screen_frame_is_blocked_on_and_the_screens_are_named, [True, 'Blocking on 2 screens', "'screenA', 'screenB'"], id="a_multi_screen_frame_is_blocked_on_and_the_screens_are_named"),
    pytest.param(case_a_mixed_run_says_that_the_level_setting_is_not_read, ["regression_type='mixed' fits the gene fixed", 'is not read'], id="a_mixed_run_says_that_the_level_setting_is_not_read"),
])
def test_12_ml_regression_pipeline_contracts(request, case, expected):
    """One contract per row: the case records what it checked, the row states it."""
    observed = observe(case, request)
    assert observed == expected
