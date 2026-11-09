"""Division results in minutes and um (item 864) and a movie bootstrap (item 867).

A synthetic movie with a known frame interval and pixel size must give the
cycle length in minutes and the size in um^2 that a hand calculation gives,
and a movie without them must stay in frames and pixels and say so. Cells of
one movie are not independent: with a planted between-movie spread of the
effect, the interval that resamples movies covers the true Hedges' g at about
the nominal rate while the one that resamples cells does not.
"""
import json
import math
import os

import numpy as np
import pandas as pd
import pytest

from spacr import division_analysis as da
from spacr.tabular import read_table, write_table
from tests.test_division_analysis import _synthetic_movie
from tests.test_division_analysis_step import _tracked_movie

COLUMNS = ('area', 'minor_axis', 'speed')


def _movie(seed=0, **calibration):
    meas, lineage = _synthetic_movie(seed=seed)
    meas = da.add_motion_features(meas)
    meas['speed'] = meas['speed'] + np.random.default_rng(seed).normal(0, 0.2, len(meas))
    result = da.analyse_divisions(meas, lineage, columns=COLUMNS, window=6, n_boot=100,
                                  rng=seed)
    return da.calibrate_divisions(result, **calibration) if calibration else result


def _labels(figures):
    return {name: [(ax.get_xlabel(), ax.get_ylabel()) for ax in fig.axes if ax.get_visible()]
            for name, fig in figures.items()}


def test_a_known_interval_and_pixel_size_give_minutes_and_square_micrometres(tmp_path):
    result = _movie(frame_interval_min=5.0, pixel_size_um=0.5)
    summary = result['summary']
    assert summary['frame_interval_min'] == 5.0 and summary['pixel_size_um'] == 0.5
    assert (summary['time_unit'], summary['area_unit'], summary['speed_unit']) == (
        'min', 'um^2', 'um/min')
    assert summary['units_note'] == 'times in minutes, lengths in um'
    # hand calculation: a 12-frame cycle at 5 min per frame is 60 min, the
    # 6-frame window 30 min; area in px^2 times 0.5^2 um^2 per px^2.
    assert summary['median_cycle_min'] == 60.0 and summary['window_min'] == 30.0
    cycles = result['cycles']
    assert (cycles['duration'] == 12).all() and (cycles['duration_min'] == 60.0).all()
    assert np.allclose(cycles['t_division_min'], cycles['t_division'] * 5.0)
    meas, _ = _synthetic_movie()
    first = cycles.iloc[0]
    birth_px = meas.set_index(['track_id', 't']).at[(first['track_id'], first['t_birth']), 'area']
    assert first['area_birth_um2'] == pytest.approx(birth_px * 0.25)
    events = result['events']
    assert np.allclose(events['t_div_min'], events['t_div'] * 5.0)
    assert np.allclose(events['base_size_um2'], events['base_size'] * 0.25, equal_nan=True)
    traj = result['trajectories']
    assert np.allclose(traj['rel_t_min'], traj['rel_t'] * 5.0)
    speed = traj[traj['measure'] == 'speed']
    assert np.allclose(speed['mean_um_per_min'], speed['mean'] * 0.1, equal_nan=True)
    assert traj.loc[traj['measure'] != 'speed', 'mean_um_per_min'].isna().all()
    assert set(traj['unit']) == {'log ratio', 'um/min'}
    effects = result['effects'].set_index(['contrast', 'measure'])
    assert effects.loc[('dividing_vs_control', 'speed'), 'change_unit'] == 'um/min'
    assert effects.loc[('dividing_vs_control', 'area'), 'change_unit'] == 'log ratio'
    row = effects.loc[('dividing_vs_control', 'speed')]
    assert row['mean_change_treated_um_per_min'] == pytest.approx(row['mean_change_treated'] * 0.1)
    labels = _labels(da.division_figures(result))
    assert {x for x, _ in labels['trajectories']} == {'minutes from division'}
    assert ('minutes from division', 'change from baseline (µm/min)') in labels['trajectories']
    assert labels['size_control'] == [('area at birth (µm²)', 'area at division (µm²)')]
    paths = da.write_division_results(result, tmp_path / 'out', plot=False)
    assert os.path.isfile(paths['changes'])
    back = da.read_division_results(tmp_path / 'out')
    assert back['summary'] == json.loads(json.dumps(result['summary']))
    assert set(back['changes']['series']) == set(result['changes']['series'])


def test_a_movie_without_calibration_stays_in_frames_and_pixels_and_says_so():
    result = _movie()
    summary = result['summary']
    assert summary['frame_interval_min'] is None and summary['pixel_size_um'] is None
    assert summary['time_unit'] == 'frames' and summary['length_unit'] == 'px'
    assert summary['units_note'] == ('frame interval unknown: times are in frames; '
                                     'pixel size unknown: lengths are in pixels')
    assert summary['median_cycle_min'] is None and summary['window_min'] is None
    assert not [c for c in result['cycles'] if c.endswith(('_min', '_um2', '_um'))]
    assert 'rel_t_min' not in result['trajectories']
    assert 't_div_min' not in result['events']
    effects = result['effects'].set_index(['contrast', 'measure'])
    assert effects.loc[('dividing_vs_control', 'speed'), 'change_unit'] == 'px/frame'
    finite = result['effects'][np.isfinite(result['effects']['g'])]
    assert len(finite) and (finite['ci_method'] == 'cell').all()
    assert set(result['effects']['ci_method']) <= {'cell', 'none'}
    assert (result['effects']['n_movies'] == 1).all()
    assert (result['effects']['movie_ci_note'] == 'not available: one movie').all()
    assert result['effects']['movie_ci_low'].isna().all()
    assert summary['movie_ci_note'] == 'not available: one movie'
    labels = _labels(da.division_figures(result))
    assert {x for x, _ in labels['trajectories']} == {'frames from division (frame interval unknown)'}
    assert ('frames from division (frame interval unknown)',
            'change from baseline (px/frame)') in labels['trajectories']
    assert labels['size_control'] == [('area at birth (px², pixel size unknown)',
                                       'area at division (px², pixel size unknown)')]
    forest = da.division_figures(result)['forest'].axes[0].get_xlabel()
    assert forest.endswith('(95% CI, cells resampled)')


def test_only_the_interval_known_gives_minutes_but_keeps_pixels():
    result = _movie(frame_interval_min=5.0, pixel_size_um=0.5)
    da.calibrate_divisions(result, frame_interval_min=2)
    assert 'area_birth_um2' not in result['cycles'] and 'base_size_um2' not in result['events']
    assert np.allclose(result['cycles']['duration_min'], result['cycles']['duration'] * 2)
    summary = result['summary']
    assert summary['time_unit'] == 'min' and summary['speed_unit'] == 'px/frame'
    assert summary['units_note'] == 'pixel size unknown: lengths are in pixels'
    assert 'duration_min' in result['cycles'] and 'area_birth_um2' not in result['cycles']
    assert 'mean_um_per_min' not in result['trajectories']


@pytest.mark.parametrize('bad', [-1.0, 0, True, 'fast', float('nan')])
def test_a_calibration_that_is_not_a_positive_number_is_refused(bad):
    meas, lineage = _synthetic_movie(n_lineages=4)
    result = da.analyse_divisions(meas, lineage, columns=('area',), n_boot=0)
    with pytest.raises(ValueError, match='positive number'):
        da.calibrate_divisions(result, pixel_size_um=bad)
    assert da._calibration_value('  ', 'x') is None and da._calibration_value('0.5', 'x') == 0.5


def _old_effect_size(x, y, n_boot, seed):
    """The cell bootstrap as item 771 drew it, for the unchanged-result check."""
    rng = np.random.default_rng(seed)
    n1, n2 = len(x), len(y)
    xs, ys = x[rng.integers(0, n1, size=(n_boot, n1))], y[rng.integers(0, n2, size=(n_boot, n2))]
    sp = np.sqrt(((n1 - 1) * xs.var(axis=1, ddof=1) + (n2 - 1) * ys.var(axis=1, ddof=1))
                 / (n1 + n2 - 2))
    gs = (1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)) * (xs.mean(axis=1) - ys.mean(axis=1)) / sp
    return np.percentile(gs[np.isfinite(gs)], [2.5, 97.5])


def test_one_movie_keeps_the_cell_bootstrap_unchanged():
    rng = np.random.default_rng(1)
    x, y = rng.normal(1, 1, 40), rng.normal(0, 1, 30)
    plain = da.effect_size(x, y, n_boot=500, rng=7)
    unlabelled = da.movie_effect_size(x, y, n_boot=500, rng=7)
    labelled = da.movie_effect_size(x, y, n_boot=500, rng=7, treated_movies=['m'] * 40,
                              control_movies=['m'] * 30)
    expected = _old_effect_size(x, y, 500, 7)
    assert [plain['ci_low'], plain['ci_high']] == pytest.approx(list(expected))
    for res in (unlabelled, labelled):
        assert {k: res[k] for k in plain} == plain
        assert res['ci_method'] == 'cell' and res['n_movies'] == 1
        assert res['movie_ci_note'] == 'not available: one movie'
        assert math.isnan(res['movie_ci_low']) and res['var_g'] == res['var_g_large_sample']
    assert labelled['g'] == plain['g']


def _clustered(rng, n_movies=10, cells=50, delta=0.8, tau=0.6):
    """Cells whose dividing-vs-control effect differs from movie to movie."""
    x, y, mx, my = [], [], [], []
    for movie in range(n_movies):
        effect = delta + tau * rng.normal()
        x.append(rng.normal(effect, 1.0, cells))
        y.append(rng.normal(0.0, 1.0, cells))
        mx += [movie] * cells
        my += [movie] * cells
    return np.concatenate(x), np.concatenate(y), mx, my


def test_the_movie_interval_covers_the_true_g_and_the_cell_interval_does_not():
    delta, tau = 0.8, 0.6
    # population g: mean difference over the pooled SD, (1 + tau^2 + 1) / 2
    true_g = delta / math.sqrt(1.0 + tau ** 2 / 2.0)
    rng = np.random.default_rng(2026)
    movie_hits = cell_hits = 0
    runs = 80
    for _ in range(runs):
        x, y, mx, my = _clustered(rng, delta=delta, tau=tau)
        res = da.movie_effect_size(x, y, n_boot=400, rng=rng, treated_movies=mx, control_movies=my)
        assert res['ci_method'] == 'movie' and res['n_movies'] == 10
        assert res['movie_ci_note'] == 'reported: movies resampled'
        assert (res['ci_low'], res['ci_high']) == (res['movie_ci_low'], res['movie_ci_high'])
        assert res['var_g'] != res['var_g_large_sample']
        movie_hits += res['movie_ci_low'] <= true_g <= res['movie_ci_high']
        cell_hits += res['cell_ci_low'] <= true_g <= res['cell_ci_high']
    assert movie_hits / runs >= 0.8
    assert cell_hits / runs <= 0.65
    assert movie_hits - cell_hits >= 0.25 * runs


def test_few_movies_show_the_movie_interval_but_report_the_cell_one():
    x, y, mx, my = _clustered(np.random.default_rng(0), n_movies=3)
    res = da.movie_effect_size(x, y, n_boot=300, rng=0, treated_movies=mx, control_movies=my)
    assert res['ci_method'] == 'cell' and res['n_movies'] == 3
    assert np.isfinite(res['movie_ci_low']) and res['ci_low'] == res['cell_ci_low']
    assert res['movie_ci_note'] == ('shown only: fewer than 5 movies, '
                                    'the cell interval is reported')
    both = da.movie_effect_size(x, y, n_boot=300, rng=0, treated_movies=mx, control_movies=my,
                          min_movies=3)
    assert both['ci_method'] == 'movie'
    none = da.movie_effect_size(x, y, n_boot=0, treated_movies=mx, control_movies=my)
    assert none['ci_method'] == 'none'
    assert none['movie_ci_note'] == 'not available: no bootstrap resamples'
    flat = da.movie_effect_size([1, 1], [1, 1], treated_movies=[0, 1], control_movies=[0, 1])
    assert flat['movie_ci_note'] == 'not available: no finite effect size'


def test_movie_lists_must_match_their_values():
    with pytest.raises(ValueError, match='both groups'):
        da.movie_effect_size([1, 2, 3], [0, 1, 2], treated_movies=[0, 0, 1])
    with pytest.raises(ValueError, match='its movie'):
        da.movie_effect_size([1, 2, 3], [0, 1, 2], treated_movies=[0, 1], control_movies=[0, 1, 1])


def test_a_movie_resample_without_both_groups_has_no_interval():
    # movie a holds only treated cells, movie b only controls: a resample
    # that draws one movie twice has no comparison.
    notes = set()
    for seed in range(20):
        res = da.movie_effect_size([1.0, 2.0, 3.0], [0.0, 0.5, 1.0], n_boot=1, rng=seed,
                             treated_movies=['a'] * 3, control_movies=['b'] * 3)
        notes.add(res['movie_ci_note'])
    assert 'not available: every movie resample was degenerate' in notes


def test_pooling_movies_resamples_movies_and_keeps_the_planted_change(tmp_path):
    movies = {f'F{seed}': _movie(seed, frame_interval_min=5.0, pixel_size_um=0.5)
              for seed in range(6)}
    pooled = da.pool_division_movies(movies, n_boot=200)
    summary = pooled['summary']
    assert summary['pooled'] and summary['n_movies'] == 6 and summary['ci_method'] == 'movie'
    assert summary['movies'] == list(movies) and summary['min_movies'] == 5
    assert summary['speed_unit'] == 'um/min' and summary['size_unit'] == 'um^2'
    assert summary['units_note'] == 'times in minutes, lengths in um'
    assert summary['n_events_kept'] == sum(m['summary']['n_events_kept'] for m in movies.values())
    assert summary['per_movie']['F0']['frame_interval_min'] == 5.0
    effects = pooled['effects'].set_index(['contrast', 'measure'])
    planted = effects.loc[('dividing_vs_control', 'minor_axis')]
    assert planted['g'] > 2 and planted['ci_low'] > 1 and planted['n_movies'] == 6
    assert planted['ci_low'] == planted['movie_ci_low']
    assert effects.loc[('dividing_vs_control', 'speed'), 'change_unit'] == 'um/min'
    assert set(pooled['events']['movie']) == set(movies)
    assert set(pooled['cycles']['movie']) == set(movies)
    assert set(pooled['size_control']['size_unit']) == {'um^2'}
    traj = pooled['trajectories']
    assert np.allclose(traj['rel_t_min'], traj['rel_t'] * 5.0)
    one = movies['F0']['trajectories']
    key = ['series', 'role', 'rel_t', 'measure']
    merged = traj.merge(one, on=key, suffixes=('', '_one'))
    assert (merged['n'] >= merged['n_one']).all()
    assert 'mean_um_per_min' in traj
    paths = da.write_division_results(pooled, tmp_path / 'pooled')
    assert os.path.isfile(paths['figure_forest'])
    back = da.read_division_results(tmp_path / 'pooled')
    forest = da.division_figures(back)['forest'].axes[0].get_xlabel()
    assert forest.endswith('(95% CI, movies resampled)')
    again = da.pool_division_movies(movies, n_boot=200)
    pd.testing.assert_frame_equal(again['effects'], pooled['effects'])


def test_pooling_says_when_units_or_intervals_differ():
    movies = {'a': _movie(0, frame_interval_min=5.0, pixel_size_um=0.5),
              'b': _movie(1, frame_interval_min=3.0)}
    pooled = da.pool_division_movies(movies, n_boot=100)
    summary = pooled['summary']
    assert summary['ci_method'] == 'cell' and summary['speed_unit'] == 'px/frame'
    assert summary['time_unit'] == 'frames'
    note = summary['units_note']
    assert 'frame intervals differ between movies: times are in frames' in note
    assert 'pixel sizes differ between movies: lengths are in pixels' in note
    assert 'speed pooled in px/frame' in note and 'trajectories not pooled' in note
    assert 'size control pooled in px^2' in note
    assert not len(pooled['trajectories'])
    assert set(pooled['size_control']['size_unit']) == {'px^2'}
    forest = da.division_figures(pooled)['forest'].axes[0].get_xlabel()
    assert forest.endswith('(95% CI, cells resampled)')
    mixed = pooled['effects'].assign(ci_method=['movie', 'cell'] * (len(pooled['effects']) // 2))
    label = da.division_figures({'effects': mixed})['forest'].axes[0].get_xlabel()
    assert label.endswith('(95% CI, cells or movies resampled)')
    plain = da.pool_division_movies({'a': _movie(0), 'b': _movie(1)}, n_boot=50)
    assert plain['summary']['units_note'].startswith('frame interval unknown')
    assert len(plain['trajectories']) and 'rel_t_min' not in plain['trajectories']


def test_pooling_needs_unit_changes_and_skips_movies_without_them():
    empty = {'changes': pd.DataFrame(), 'summary': {}}
    bare = da.calibrate_divisions({}, frame_interval_min=1)
    assert bare['summary']['time_unit'] == 'min' and bare['events'].empty
    with pytest.raises(ValueError, match='no movie'):
        da.pool_division_movies({'a': empty, 'b': empty})
    pooled = da.pool_division_movies({'a': _movie(0), 'b': empty}, n_boot=20)
    assert pooled['summary']['n_movies'] == 2
    assert set(pooled['changes']['movie']) == {'a'}
    assert da._pooled_trajectories([]).empty


def test_pooled_trajectories_weight_movies_by_their_units():
    base = {'series': 'dividing', 'role': 'self', 'rel_t': 0, 'measure': 'area',
            'scale': 'log_ratio_to_baseline'}
    a = pd.DataFrame([{**base, 'mean': 1.0, 'sd': 1.0, 'n': 3}])
    b = pd.DataFrame([{**base, 'mean': 4.0, 'sd': 0.0, 'n': 1}])
    gap = pd.DataFrame([{**base, 'mean': np.nan, 'sd': np.nan, 'n': 0}])
    row = da._pooled_trajectories([a, b]).iloc[0]
    assert row['n'] == 4 and row['mean'] == pytest.approx(1.75)
    # within: 2 * 1^2; between: 3 * 0.75^2 + 1 * 2.25^2
    assert row['sd'] == pytest.approx(math.sqrt((2.0 + 3 * 0.5625 + 5.0625) / 3))
    assert row['sem'] == pytest.approx(row['sd'] / 2)
    assert np.isnan(da._pooled_trajectories([gap]).iloc[0]['mean'])


def _write_field(tmp_path, name, tracks):
    write_table(tracks, tmp_path / 'plate' / 'tracks' / f'trackastra_tracks_cell_{name}.csv')


def test_the_step_reads_the_settings_converts_and_pools_the_fields(tmp_path, capsys):
    stack, tracks = _tracked_movie(n_dividing=8, n_steady=8)
    src = tmp_path / 'plate' / 'masks'
    os.makedirs(src)
    settings = {'timelapse_division_window': 6, 'save': False,
                'timelapse_division_size_tolerance': 0.4,
                'frame_interval_s': 120, 'pixel_size_um': 0.65}
    for name in ('A01', 'A02'):
        _write_field(tmp_path, name, tracks)
        result = da._run_division_step(str(src), name, 'cell', 'trackastra', settings,
                                       label_stack=list(stack))
        assert result is not None
    summary = result['summary']
    assert summary['frame_interval_min'] == 2.0 and summary['pixel_size_um'] == 0.65
    assert summary['frame_interval_source'] == 'frame_interval_s setting'
    assert summary['pixel_size_source'] == 'pixel_size_um setting'
    division = tmp_path / 'plate' / 'tracks' / 'division'
    pooled = division / 'pooled_trackastra_tracks_cell'
    stored = json.loads((pooled / 'summary.json').read_text())
    assert stored['n_movies'] == 2 and stored['movies'] == ['A01', 'A02']
    assert stored['ci_method'] == 'cell'
    assert result['pooled']['summary']['n_movies'] == 2
    out = capsys.readouterr().out
    assert 'pooled 2 fields (cell intervals)' in out
    assert 'times in minutes, lengths in um' in out
    effects = read_table(division / 'trackastra_tracks_cell_A01' / 'effects.csv', report=None)
    assert set(effects['movie_ci_note']) == {'not available: one movie'}


def test_the_step_prefers_the_tracks_own_times_and_notes_a_conflict(tmp_path):
    stack, tracks = _tracked_movie(n_dividing=6, n_steady=6)
    tracks['time_s'] = tracks['frame'] * 30.0
    src = tmp_path / 'plate' / 'masks'
    os.makedirs(src)
    _write_field(tmp_path, 'B01', tracks)
    settings = {'timelapse_division_window': 6, 'save': False, 'frame_interval_s': 60,
                'timelapse_division_size_tolerance': 0.4}
    result = da._run_division_step(str(src), 'B01', 'cell', 'trackastra', settings,
                                   label_stack=list(stack))
    summary = result['summary']
    assert summary['frame_interval_min'] == 0.5
    assert summary['frame_interval_source'] == 'tracks time_s'
    assert summary['frame_interval_note'] == ('tracks time_s gives 30 s per frame, '
                                              'frame_interval_s 60 s; time_s used')
    assert summary['pixel_size_source'] == 'missing'
    assert summary['units_note'] == 'pixel size unknown: lengths are in pixels'
    assert 'pooled' not in result


def test_uneven_track_times_fall_back_to_the_setting_or_to_frames():
    tracks = pd.DataFrame({'frame': [0, 1, 2, 2], 'time_s': [0.0, 10.0, 35.0, 35.0]})
    assert da._movie_calibration(tracks, {'frame_interval_s': 6})[0] == 0.1
    interval, size, sources = da._movie_calibration(tracks, {})
    assert interval is None and size is None
    assert sources == {'frame_interval_source': 'missing', 'pixel_size_source': 'missing'}
    disagree = pd.DataFrame({'frame': [0, 0, 1], 'time_s': [0.0, 1.0, 2.0]})
    assert da._movie_calibration(disagree, {})[0] is None
    with pytest.raises(ValueError, match='frame_interval_s'):
        da._movie_calibration(tracks, {'frame_interval_s': -5})


def test_the_step_reports_a_bad_calibration_and_a_failed_pool(tmp_path, capsys, monkeypatch):
    stack, tracks = _tracked_movie(n_dividing=4, n_steady=4)
    src = tmp_path / 'plate' / 'masks'
    os.makedirs(src)
    _write_field(tmp_path, 'C01', tracks)
    bad = {'timelapse_division_window': 6, 'pixel_size_um': 'big', 'save': False}
    assert da._run_division_step(str(src), 'C01', 'cell', 'trackastra', bad,
                                 label_stack=list(stack)) is None
    assert 'pixel_size_um must be a positive number' in capsys.readouterr().out

    def broken(*args, **kwargs):
        raise RuntimeError('disk full')

    monkeypatch.setattr(da, '_pool_division_folders', broken)
    good = {'timelapse_division_window': 6, 'save': False,
            'timelapse_division_size_tolerance': 0.4}
    assert da._run_division_step(str(src), 'C01', 'cell', 'trackastra', good,
                                 label_stack=list(stack)) is not None
    assert 'could not pool the fields' in capsys.readouterr().out
