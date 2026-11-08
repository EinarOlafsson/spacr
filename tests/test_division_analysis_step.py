"""Division analysis from spaCR's own tracking output (item 771, alpha).

A tracks CSV and a tracked label stack, as the Make Masks timelapse run
writes them, go in; the tidy tables, the summary and the three figures come
out, and read back to the same numbers. The synthetic movie plants a
pre-division rounding (the mother gets taller in her last three frames) that
the matched controls and the shuffled alignment must not show.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from spacr import division_analysis as da
from spacr.tabular import read_table, write_table
from tests.test_division_analysis import _synthetic_movie


def _tracked_movie(n_dividing=12, n_steady=12, frames=20, birth=10):
    """Rectangles in their own 24x24 tiles: mothers that divide at ``birth``
    into two daughters, and steady cells of the same size that never do."""
    tiles = n_dividing + n_steady
    rng = np.random.default_rng(0)
    stack = np.zeros((frames, 24 * 2, 24 * tiles), np.int32)
    rows, nxt = [], 1
    for tile in range(tiles):
        x0 = 24 * tile + 2
        mother = nxt
        nxt += 1
        dividing = tile < n_dividing
        end = birth - 1 if dividing else frames - 1
        start_width, start_height = int(rng.integers(8, 11)), int(rng.integers(5, 8))
        for t in range(end + 1):
            width = start_width + t // 2
            height = start_height + (3 if dividing and t > end - 3 else 0)
            stack[t, 2:2 + height, x0:x0 + width] = mother
            rows.append({'track_id': mother, 'frame': t, 'x': x0 + width / 2,
                         'y': 2 + height / 2, 'parent_track_id': 0})
        if not dividing:
            continue
        for half in range(2):
            daughter = nxt
            nxt += 1
            for t in range(birth, frames):
                y0 = 2 + 24 * half
                stack[t, y0:y0 + 6, x0:x0 + 6 + (t - birth) // 3] = daughter
                rows.append({'track_id': daughter, 'frame': t, 'x': x0 + 3.0,
                             'y': y0 + 3.0, 'parent_track_id': mother})
    return stack, pd.DataFrame(rows)


def test_lineage_from_tracks_keeps_the_tracker_s_division_links():
    _, tracks = _tracked_movie(n_dividing=2, n_steady=1)
    lineage = da.lineage_from_tracks(tracks)
    assert list(lineage.columns) == list(da.LINEAGE_COLUMNS)
    by_label = lineage.set_index('label')
    assert by_label.loc[1].tolist() == [0, 9, 0]
    assert by_label.loc[2].tolist() == [10, 19, 1] and by_label.loc[3, 'parent'] == 1
    assert by_label.loc[7].tolist() == [0, 19, 0]
    events = da.division_events(lineage, mode='fission')
    assert events['daughters'].tolist() == [(2, 3), (5, 6)]


def test_lineage_from_tracks_reads_an_empty_table():
    empty = pd.DataFrame(columns=['track_id', 'frame', 'x', 'y'])
    assert da.lineage_from_tracks(empty).empty


def test_analyse_divisions_finds_the_planted_change_and_its_controls_do_not():
    meas, lineage = _synthetic_movie()
    result = da.analyse_divisions(meas, lineage, columns=('area', 'minor_axis'),
                                  window=6, n_boot=300, rng=0)
    summary = result['summary']
    assert summary['mode'] == 'fission' and summary['window'] == 6
    assert summary['late_frames'] == 2 and summary['n_events_kept'] > 20
    assert summary['n_controls_matched'] >= 0.8 * summary['n_events_kept']
    effects = result['effects'].set_index(['contrast', 'measure'])
    planted = effects.loc[('dividing_vs_control', 'minor_axis')]
    assert planted['g'] > 3 and planted['ci_low'] > 1
    null = effects.loc[('shuffled_vs_its_control', 'minor_axis')]
    assert abs(null['g']) < 0.8
    assert set(result['trajectories']['series']) == {
        'dividing', 'control', 'shuffled', 'shuffled_control'}
    dividing = result['trajectories'].query("series == 'dividing'")
    assert set(dividing['role']) == {'self', 'daughter'}
    assert len(result['cycles']) and set(result['size_control']['size_measure']) == {'area'}


def test_analyse_divisions_refuses_bad_input():
    meas, lineage = _synthetic_movie(n_lineages=3)
    with pytest.raises(ValueError):
        da.analyse_divisions(meas, lineage, mode='mitosis')
    with pytest.raises(ValueError):
        da.analyse_divisions(meas.drop(columns='area'), lineage)
    with pytest.raises(ValueError):
        da.analyse_divisions(meas, lineage, window=1)


def test_analyse_divisions_records_why_a_division_is_dropped():
    meas, lineage = _synthetic_movie(n_lineages=6)
    first = lineage[lineage['parent'] == 0]['label'].iloc[0]
    meas = meas[~((meas['track_id'] == first) & (meas['t'] % 2 == 0))]
    result = da.analyse_divisions(meas, lineage, columns=('area',), window=6, n_boot=0)
    events = result['events'].set_index('parent')
    assert events.loc[first, 'status'] in ('short_prehistory', 'parent_unseen_at_division')
    assert sum(result['summary']['drop_reasons'].values()) >= 1


def test_results_are_written_read_back_and_drawn(tmp_path):
    meas, lineage = _synthetic_movie()
    result = da.analyse_divisions(meas, lineage, columns=('area', 'minor_axis'), n_boot=100)
    paths = da.write_division_results(result, tmp_path / 'out')
    for key in ('events', 'effects', 'trajectories', 'cycles', 'size_control', 'summary',
                'figure_trajectories', 'figure_forest', 'figure_size_control'):
        assert os.path.isfile(paths[key]), key
    back = da.read_division_results(tmp_path / 'out')
    assert back['summary'] == json.loads(json.dumps(result['summary']))
    assert back['events']['daughters'].tolist() == result['events']['daughters'].tolist()
    pd.testing.assert_frame_equal(back['effects'].reset_index(drop=True),
                                  result['effects'].reset_index(drop=True),
                                  check_dtype=False)
    figures = da.division_figures(back)
    assert set(figures) == {'trajectories', 'forest', 'size_control'}
    assert len(figures['forest'].axes[0].get_yticklabels()) == 2
    with pytest.raises(FileNotFoundError):
        da.read_division_results(tmp_path)


def test_figures_say_so_when_there_is_nothing_to_draw():
    empty = {'effects': pd.DataFrame(), 'trajectories': pd.DataFrame(),
             'cycles': pd.DataFrame(), 'size_control': pd.DataFrame()}
    figures = da.division_figures(empty)
    texts = [t.get_text() for f in figures.values() for ax in f.axes for t in ax.texts]
    assert 'no complete cell cycles' in texts and 'no effect sizes' in texts


def test_the_step_reads_spacr_tracks_and_writes_beside_them(tmp_path):
    stack, tracks = _tracked_movie()
    src = tmp_path / 'plate' / 'masks'
    tracks_dir = tmp_path / 'plate' / 'tracks'
    os.makedirs(src)
    write_table(tracks, tracks_dir / 'trackastra_tracks_cell_A01.csv')
    images = np.ones(stack.shape + (2,), np.float32)
    settings = {'timelapse_division_window': 6, 'save': True,
                'timelapse_division_size_tolerance': 0.4}
    result = da._run_division_step(str(src), 'A01', 'cell', 'trackastra', settings,
                                   label_stack=list(stack), images=images)
    assert result is not None
    out = tracks_dir / 'division' / 'trackastra_tracks_cell_A01'
    assert result['summary']['n_events'] == 12
    assert result['summary']['n_events_kept'] == 12
    assert result['summary']['n_controls_matched'] > 0
    lineage = read_table(out / 'lineage.csv', report=None)
    assert int((lineage['parent'] > 0).sum()) == 24
    effects = read_table(out / 'effects.csv', report=None).set_index(['contrast', 'measure'])
    assert effects.loc[('dividing_vs_control', 'minor_axis'), 'g'] > 1
    assert os.path.isfile(out / 'summary.json')


def test_the_step_reports_instead_of_stopping_the_run(tmp_path, capsys):
    src = tmp_path / 'plate' / 'masks'
    os.makedirs(src)
    assert da._run_division_step(str(src), 'A01', 'cell', 'iou', {}, label_stack=[]) is None
    assert 'no tracks table' in capsys.readouterr().out
    stack, tracks = _tracked_movie(n_dividing=1, n_steady=1)
    write_table(tracks, tmp_path / 'plate' / 'tracks' / 'trackpy_tracks_cell_A01.csv')
    bad = {'timelapse_division_size_tolerance': -1}
    assert da._run_division_step(str(src), 'A01', 'cell', 'iou', bad,
                                 label_stack=list(stack)) is None
    assert 'could not run' in capsys.readouterr().out
    assert da._run_division_step(str(src), 'A01', 'cell', 'iou', {}) is None


def test_division_settings_defaults_and_registration():
    from spacr.settings import (ALPHA_FEATURES, expected_types,
                                set_default_settings_preprocess_generate_masks, tooltips)

    defaults = set_default_settings_preprocess_generate_masks({})
    keys = ('timelapse_division_analysis', 'timelapse_division_mode',
            'timelapse_division_window', 'timelapse_division_size_tolerance')
    assert [defaults[k] for k in keys] == [False, 'auto', None, 0.25]
    assert set(ALPHA_FEATURES[771]['settings']) == set(keys)
    assert ALPHA_FEATURES[771]['widgets'] == ('DivisionAnalysisPanel',)
    for key in keys:
        assert key in expected_types and tooltips[key].endswith(
            f'Default {defaults[key]}.')
    assert da._division_settings(defaults) == {'mode': 'auto', 'window': None,
                                               'size_tol': 0.25}
    with pytest.raises(ValueError):
        da._division_settings({'timelapse_division_size_tolerance': True})
