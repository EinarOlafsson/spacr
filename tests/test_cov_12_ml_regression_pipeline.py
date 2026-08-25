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


def test_an_intercept_centred_on_the_controls_says_what_it_shifted_by(
        tmp_path, capsys):
    """The run names the offset, so a coefficient reads as a distance from nc.

    Without the sentence, two runs of the same screen under different intercept
    modes produce different coefficients with nothing on the record saying why.
    """
    frame = wells_frame(seed=3)
    _model, coef_df, _kind = fit(frame, tmp_path, intercept='control')

    printed = capsys.readouterr().out
    assert 'Intercept set to the negative controls' in printed
    assert f"distance from {NC!r}" in printed
    assert len(coef_df) > 0


def test_an_intercept_centred_on_a_control_that_is_absent_says_it_was_not(
        tmp_path, capsys):
    """No matching control leaves the intercept fitted, and reports that.

    Silently fitting the intercept anyway would mean the coefficients are not
    what the setting asked for and nothing anywhere would say so.
    """
    frame = wells_frame(seed=4)
    fit(frame, tmp_path, intercept='control', nc='not-a-guide-in-this-screen')

    printed = capsys.readouterr().out
    assert 'Intercept left as fitted' in printed
    assert 'no rows match' in printed


def test_an_intercept_pinned_at_a_value_shifts_the_response_by_it(tmp_path,
                                                                  capsys):
    """The response is moved by the number, so the intercept is exactly it.

    An estimated intercept would land near the number and read as though the
    value had been a suggestion.
    """
    frame = wells_frame(seed=5)
    fit(frame, tmp_path, intercept='value', intercept_value=0.5)

    printed = capsys.readouterr().out
    assert 'Intercept pinned at 0.5' in printed
    assert 'distance from that value' in printed


def test_an_intercept_pinned_at_zero_is_still_announced(tmp_path, capsys):
    """Pinning at zero changes no value but is still stated.

    Zero is a real answer for a centred response; leaving it unsaid would make
    the pinned run indistinguishable from the default one in the log.
    """
    frame = wells_frame(seed=6)
    before = frame['predictions'].to_numpy(dtype=float).copy()

    fit(frame, tmp_path, intercept='value', intercept_value=0.0)

    assert 'Intercept pinned at 0' in capsys.readouterr().out
    assert np.array_equal(frame['predictions'].to_numpy(dtype=float), before)


@pytest.mark.xfail(strict=True, reason="check_and_clean_data drops screenID "
                                       "before screen_is_blockable is "
                                       "consulted, so regression() never "
                                       "blocks on the screen")
def test_a_multi_screen_frame_is_blocked_on_and_the_screens_are_named(
        tmp_path, capsys):
    """More than one screen becomes a design term, and the run lists them.

    A single-screen frame must not be blocked on -- the term would be a
    constant column and the design rank deficient -- so which happened has to
    be on the record. A two-screen frame reaches ``regression`` with a
    ``screenID`` column and ``screen_is_blockable`` answers True for it, but by
    the time the decision is taken the cleaning step has dropped the column, so
    the two screens are pooled with nothing said.
    """
    frame = wells_frame(seed=7, screens=['screenA', 'screenB'])
    assert ml.screen_is_blockable(frame) is True

    fit(frame, tmp_path)

    printed = capsys.readouterr().out
    assert 'Blocking on 2 screens' in printed
    assert "'screenA', 'screenB'" in printed


def test_the_old_histograms_are_drawn_when_the_new_panels_cannot_be(
        tmp_path, monkeypatch, capsys):
    """A distribution panel that declines sends the run back to plot_histogram.

    The point of the fallback is that a figure is never worth losing a fit
    over, and the run must still show the response it fitted.
    """
    drawn = []

    def declined(frame, response_name, dst, plot=True):
        return False

    def record(data, variable, dst=None, **kwargs):
        drawn.append(variable)

    import spacr.plot as plot_module

    monkeypatch.setattr(ml, '_show_well_distributions', declined)
    monkeypatch.setattr(plot_module, 'plot_histogram', record)

    fit(wells_frame(seed=8), tmp_path, plot=True)

    assert drawn == ['predictions', 'fraction']


def test_a_summary_that_cannot_be_written_does_not_lose_the_fit(
        tmp_path, monkeypatch, capsys):
    """A failure in the closing paragraph is printed and the fit is returned.

    The summary is prose about numbers that are already computed; nothing about
    it is worth a run.
    """
    import spacr.figures.summary as summary_module

    def boom(coef_df):
        raise RuntimeError('the summariser could not read this table')

    monkeypatch.setattr(summary_module, 'summarise', boom)

    _model, coef_df, kind = fit(wells_frame(seed=9), tmp_path)

    assert kind == 'ols'
    assert len(coef_df) > 0
    assert 'Could not summarise the run' in capsys.readouterr().out


def test_a_mixed_run_says_that_the_level_setting_is_not_read(tmp_path, capsys,
                                                             monkeypatch):
    """``regression_type='mixed'`` is already both levels, and says so.

    The GUI greys the level out, but a script can still set it; quietly
    ignoring it would hand back a gene table to a caller who asked for guides.
    """
    def one_fit(*args, **kwargs):
        return None, pd.DataFrame({'feature': ['Intercept'],
                                   'coefficient': [0.1],
                                   'p_value': [0.5]}), 'mixed'

    monkeypatch.setattr(ml, 'regression', one_fit)

    ml.regression_levels(wells_frame(seed=10), str(tmp_path / 'scores.csv'),
                         regression_type='mixed', level='grna')

    printed = capsys.readouterr().out
    assert "regression_type='mixed' fits the gene fixed" in printed
    assert 'is not read' in printed
