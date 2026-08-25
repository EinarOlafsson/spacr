"""A finished run keeps its results when a closing step fails.

``perform_regression`` writes the results and then decorates them: a run
summary, plate panels, metadata merges, a volcano, a guide-support paragraph.
Every one of those runs after the fit is complete, so every one of them has to
be survivable. These drive each into failure on a real end-to-end run and
assert the results are still there and the loss is named.

The synthetic screen and its settings builder are the ones
``tests/test_cov_ml_perform_regression.py`` already defines, so both files
describe the same run rather than two screens that can drift apart.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pytest

from spacr import ml

from tests.test_cov_ml_perform_regression import (base_settings,
                                                  parametric_settings,
                                                  write_counts, write_metadata,
                                                  write_scores)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


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


def test_a_permutation_run_whose_summary_cannot_be_written_still_returns(
        screen, monkeypatch, capsys):
    """A failing run summary is printed and the permutation output is intact.

    The permutation branch has no statsmodels summary to fall back on, so this
    is the only summary that run produces -- and losing the run over prose
    would be the worst possible trade.
    """
    import spacr.regression_summary as regression_summary

    def boom(*args, **kwargs):
        raise RuntimeError('the summary template is missing')

    monkeypatch.setattr(regression_summary, 'write_run_summary', boom)

    output = ml.perform_regression(base_settings(screen))

    assert 'Could not write the run summary' in capsys.readouterr().out
    assert output['res_folder']
    assert os.path.isdir(output['res_folder'])
    assert len(output['primary']) > 0


def test_an_auto_inference_run_says_which_mode_it_chose(screen, capsys):
    """``inference='auto'`` prints the mode and the reason it was picked.

    Auto is a decision about what the numbers mean; a run that made it
    silently could not be reproduced from its own log.
    """
    settings = base_settings(screen, inference='auto')
    output = ml.perform_regression(settings)

    printed = capsys.readouterr().out
    assert "inference='auto':" in printed
    assert settings['analysis_mode'] in {'regression', 'guide_permutation'}
    assert output['res_folder']


def test_a_plate_panel_that_declines_falls_back_to_the_old_plate_figure(
        screen, monkeypatch):
    """When the house-style plate panel cannot draw, plot_plates is used.

    The plate map is how a user sees a positional artefact; losing it silently
    because the new panel declined would hide the thing it exists to show.
    """
    import spacr.plot as plot_module

    drawn = []
    monkeypatch.setattr(ml, '_show_plates', lambda *a, **k: False)
    monkeypatch.setattr(plot_module, 'plot_plates',
                        lambda frame, **kwargs: drawn.append(kwargs))

    ml.perform_regression(parametric_settings(screen))

    assert drawn, 'the fallback plate figure was not drawn'
    assert drawn[0]['grouping'] == 'mean'


def test_a_gene_only_run_writes_an_empty_guide_table_and_says_so(screen,
                                                                 capsys):
    """``level='gene'`` still writes results_grna.csv, empty, and names why.

    A file that is absent is indistinguishable from a run that crashed, so the
    guide table exists with the run's own columns and no rows.
    """
    output = ml.perform_regression(parametric_settings(screen, level='gene'))

    printed = capsys.readouterr().out
    assert "level='gene': no guide fit was run" in printed
    guide_table = os.path.join(output['res_folder'], 'results_grna.csv')
    assert os.path.isfile(guide_table)
    assert len(pd.read_csv(guide_table)) == 0


def test_a_missing_metadata_file_is_skipped_not_fatal(screen, capsys, tmp_path):
    """An absent or empty annotation file costs the decoration, not the fit.

    One bad metadata path used to fail every trial of a sweep that touched it,
    with the coefficients already computed and written.
    """
    empty = tmp_path / 'empty_metadata.csv'
    empty.write_text('')

    output = ml.perform_regression(parametric_settings(
        screen, metadata_files=[str(tmp_path / 'not_here.csv'), str(empty)]))

    printed = capsys.readouterr().out
    assert printed.count('Skipping empty or missing metadata file') == 2
    assert os.path.isfile(os.path.join(output['res_folder'], 'results.csv'))


def test_a_metadata_path_that_cannot_be_stat_ed_is_skipped(screen, monkeypatch,
                                                           tmp_path):
    """A path whose size cannot be read is stepped over rather than raised.

    A metadata file on a mount that went away is not a reason to fail a run
    whose results are already on disk.
    """
    unreadable = tmp_path / 'unreadable.csv'
    unreadable.write_text('Gene ID\nTGGT1_000000\n')
    real_getsize = os.path.getsize

    def hostile(path):
        if str(path) == str(unreadable):
            raise OSError('the mount went away')
        return real_getsize(path)

    monkeypatch.setattr(ml.os.path, 'getsize', hostile)

    output = ml.perform_regression(parametric_settings(
        screen, metadata_files=[str(unreadable)]))

    assert os.path.isfile(os.path.join(output['res_folder'], 'results.csv'))


def test_a_legacy_volcano_that_cannot_be_drawn_is_reported_by_name(
        screen, monkeypatch, capsys):
    """The plain volcano failing is printed; the results are already written.

    The volcano is the figure the module exists to produce, so its absence has
    to be stated rather than left for the user to notice.
    """
    import spacr.plot as plot_module

    def boom(*args, **kwargs):
        raise RuntimeError('the volcano had no finite p-values to plot')

    monkeypatch.setattr(plot_module, 'volcano_plot', boom)

    output = ml.perform_regression(parametric_settings(
        screen, legacy_volcano=True, toxo=False))

    printed = capsys.readouterr().out
    assert 'Could not draw the volcano plot' in printed
    # Nothing is claimed about a file that was not written.
    assert 'Saved volcano plot' not in printed
    assert os.path.isfile(os.path.join(output['res_folder'], 'results.csv'))


def test_a_legacy_volcano_that_is_drawn_says_where_it_went(screen, monkeypatch,
                                                           capsys):
    """A written volcano announces its path, like every other artefact.

    Written silently, a run that had drawn one perfectly well was
    indistinguishable from a run that had drawn none.
    """
    import spacr.plot as plot_module

    def draw(*args, save_path=None, **kwargs):
        with open(save_path, 'w') as handle:
            handle.write('a volcano')
        return None

    monkeypatch.setattr(plot_module, 'volcano_plot', draw)

    output = ml.perform_regression(parametric_settings(
        screen, legacy_volcano=True, toxo=False))

    printed = capsys.readouterr().out
    assert 'Saved volcano plot to ' in printed
    assert os.path.isfile(os.path.join(output['res_folder'], 'results.csv'))


def test_a_guide_support_paragraph_that_fails_costs_only_the_paragraph(
        screen, monkeypatch, capsys):
    """A concordance report that raises is named and the run still returns.

    A gene backed by one guide and a gene whose guides agree are the same dot;
    the paragraph is what tells them apart, and it is worth saying when it is
    missing.
    """
    import spacr.guide_concordance as guide_concordance

    def boom(*args, **kwargs):
        raise RuntimeError('the coefficient table has no guide column')

    monkeypatch.setattr(guide_concordance, 'concordance_report', boom)

    output = ml.perform_regression(parametric_settings(screen))

    assert 'Could not summarise guide support' in capsys.readouterr().out
    assert output['res_folder']
    assert 'results' in output


def test_settings_that_cannot_be_re_saved_do_not_stop_the_run(screen,
                                                              monkeypatch,
                                                              capsys):
    """A failed settings re-save is printed and the fit carries on.

    A value the run chose for itself is written back over the settings CSV so
    the record can be reproduced from. That is provenance; a read-only settings
    folder must not cost the run that was about to be recorded in it.
    """
    import spacr.utils as utils_module

    real_save = utils_module.save_settings
    calls = []

    def sometimes(settings, name='settings', show=False):
        calls.append(name)
        if len(calls) > 1:
            raise OSError('the settings folder is read-only')
        return real_save(settings, name=name, show=show)

    monkeypatch.setattr(utils_module, 'save_settings', sometimes)

    # fraction_threshold unset, so the run derives one and re-saves the
    # settings CSV with the resolved value in it.
    output = ml.perform_regression(
        parametric_settings(screen, fraction_threshold=None))

    assert 'Could not re-save the resolved settings' in capsys.readouterr().out
    assert os.path.isfile(os.path.join(output['res_folder'], 'results.csv'))
