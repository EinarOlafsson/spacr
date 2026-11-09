"""The budding-yeast cycle model of the division analysis (item 865)."""
import numpy as np
import pandas as pd
import pytest

from spacr import division_analysis as da

G1, BUD, TAIL = 4, 8, 3


def _budding_movie(rule, founders=40, seed=0, noise=10.0, contacts=True, flat_after=False):
    """Mothers and buds with a planted size rule on the size at cytokinesis.

    Every cycle: an unbudded phase of G1 frames in which the cell gains 30%
    of the cycle's added size, then a bud that grows to the other 70% over
    BUD frames while the mother stays constant. Cytokinesis is the bud's last
    frame touching its mother, so mother + bud at cytokinesis is the planted
    division size.
    """
    rng = np.random.default_rng(seed)
    measures, lineage, touching = [], [], []
    state = {'next': 1}

    def division_size(birth):
        if rule == 'sizer':
            return 900.0 + rng.normal(0, noise)
        if rule == 'adder':
            return birth + 300.0 + rng.normal(0, noise)
        return 1.6 * birth + rng.normal(0, noise)

    def cell(cid, birth, t, cycles, parent, t_first, pre=()):
        """Simulate ``cid`` from its birth frame ``t``; ``pre`` are its frames as a bud."""
        rows = list(pre)
        size = birth
        if not pre:
            rows.append((t, size))
        for _ in range(cycles):
            target = division_size(size)
            delta = target - size
            for i in range(G1):
                rows.append((t + 1 + i, size + 0.3 * delta * (i + 1) / G1))
            mother_size = size + 0.3 * delta
            t_e = t + 1 + G1
            t_c = t_e + BUD - 1
            for f in range(t_e, t_c + 1):
                rows.append((f, mother_size))
            bud = state['next']
            state['next'] += 1
            bud_rows = [(f, 0.05 * delta + (0.65 * delta) * (f - t_e) / (BUD - 1))
                        for f in range(t_e, t_c + 1)]
            for f in range(t_e, t_c + 1):
                touching.append((bud, cid, f, True))
            children.append((bud, cid, 0.7 * delta, t_c, t_e, bud_rows))
            size, t = mother_size, t_c
        for f in range(t + 1, t + 1 + TAIL):
            rows.append((f, size if not flat_after else size))
        measures.extend((cid, f, a) for f, a in rows)
        lineage.append((cid, min(f for f, _ in rows), max(f for f, _ in rows), parent))

    children = []
    for _ in range(founders):
        cid = state['next']
        state['next'] += 1
        cell(cid, rng.uniform(250, 600) if rule != 'timer' else rng.uniform(250, 400),
             0, 3, 0, 0)
    first = list(children)
    children.clear()
    for bud, mother, birth, t_c, t_e, pre in first:
        cell(bud, birth, t_c, 1, mother, t_e, pre=pre)
    for bud, mother, birth, t_c, t_e, pre in list(children):
        cell(bud, birth, t_c, 0, mother, t_e, pre=pre)
    m = pd.DataFrame(measures, columns=['track_id', 't', 'area'])
    lin = pd.DataFrame(lineage, columns=list(da.LINEAGE_COLUMNS))
    alive = m.groupby('track_id')['t'].agg(['min', 'max'])
    rows = []
    for bud, mother, f, on in touching:
        rows.append((bud, mother, f, on))
    seen = pd.DataFrame(rows, columns=['bud', 'mother', 't', 'touching'])
    after = []
    for (bud, mother), g in seen.groupby(['bud', 'mother']):
        end = min(alive.at[bud, 'max'], alive.at[mother, 'max'])
        after += [(bud, mother, f, False) for f in range(int(g['t'].max()) + 1, int(end) + 1)]
    contact = pd.concat([seen, pd.DataFrame(after, columns=seen.columns)], ignore_index=True)
    return m, lin, (contact.sort_values(['bud', 't']).reset_index(drop=True) if contacts else None)


@pytest.mark.parametrize('rule, slope', [('sizer', 0.0), ('adder', 1.0), ('timer', 1.6)])
def test_planted_size_control_comes_back_within_its_interval(rule, slope):
    measures, lineage, contacts = _budding_movie(rule, seed=3)
    events = da.division_events(lineage, mode='budding')
    buds = da.budding_cytokinesis(measures, events, contacts=contacts)
    assert (buds['status'] == 'ok').all() and set(buds['method']) == {'contact'}
    cycles = da.budding_cycles(measures, buds)
    assert set(cycles['branch']) == {'mother', 'daughter'}
    assert (cycles['budded_frames'] == BUD).all() and (cycles['unbudded_frames'] == G1 + 1).all()
    for branch in ('mother', 'daughter'):
        part = cycles[cycles['branch'] == branch]
        fit = da.size_control_fits(part['area_birth'], part['area_division'], n_boot=500, rng=0)
        assert fit['slope_division_ci_low'] <= slope <= fit['slope_division_ci_high'], (branch, fit)
        assert abs(fit['slope_division'] - slope) < 0.15
    np.testing.assert_allclose(cycles['area_added'],
                               cycles['area_division'] - cycles['area_birth'])
    assert (cycles['volume_division'] > cycles['volume_birth']).all()


def test_add_budding_cycles_fills_the_results_and_says_what_it_assumed():
    measures, lineage, contacts = _budding_movie('adder', founders=25, seed=1)
    results = da.analyse_divisions(measures, lineage, mode='auto', n_boot=50)
    assert results['summary']['mode'] == 'budding' and not len(results['cycles'])
    da.add_budding_cycles(results, measures, lineage, contacts=contacts, n_boot=200)
    sc = results['size_control']
    assert set(sc['model']) == {'budding'} and set(sc['branch']) == {'mother', 'daughter'}
    assert set(sc['size_measure']) == {'area', 'volume'}
    row = sc[(sc['branch'] == 'mother') & (sc['size_measure'] == 'area')].iloc[0]
    assert row['slope_division_ci_low'] <= 1.0 <= row['slope_division_ci_high']
    summary = results['summary']
    assert summary['size_control_model'] == 'budding'
    assert summary['n_buds'] == summary['n_buds_fitted'] == len(results['buds'])
    assert summary['cytokinesis_methods'] == {'contact': summary['n_buds']}
    assert 'cytokinesis' in summary['budding_model'] and 'Hartwell' in summary['budding_model']
    assert summary['n_cycles'] == summary['n_cycles_mother'] + summary['n_cycles_daughter']
    da.calibrate_divisions(results, frame_interval_min=5.0, pixel_size_um=0.5)
    cycles = results['cycles']
    np.testing.assert_allclose(cycles['volume_birth_um3'], cycles['volume_birth'] * 0.125)
    np.testing.assert_allclose(cycles['budded_frames_min'], cycles['budded_frames'] * 5.0)
    da.calibrate_divisions(results, frame_interval_min=5.0, pixel_size_um=0.5)
    assert list(results['cycles'].columns).count('volume_birth_um3') == 1


def test_a_fission_movie_keeps_its_fission_fits():
    lineage = pd.DataFrame({'label': [1, 2, 3], 't_start': [0, 5, 5], 't_end': [4, 9, 9],
                            'parent': [0, 1, 1]})
    before = pd.DataFrame({'size_measure': ['area'], 'n': [3], 'slope_division': [1.0]})
    results = {'summary': {'mode': 'fission'}, 'size_control': before.copy()}
    da.add_budding_cycles(results, pd.DataFrame(), lineage)
    assert results['summary']['size_control_model'] == 'fission'
    pd.testing.assert_frame_equal(results['size_control'], before)
    assert not len(results['buds'])


def test_buds_whose_separation_is_not_seen_are_not_fitted_and_say_why():
    measures, lineage, _ = _budding_movie('adder', founders=4, seed=2)
    events = da.division_events(lineage, mode='budding')
    buds = da.budding_cytokinesis(measures, events, contacts=None)
    first = buds['mother'].isin(lineage.loc[lineage['parent'] == 0, 'label'])
    assert (buds.loc[first, 'status'] != 'ok').all()
    assert (buds.loc[first, 't_cytokinesis'] == -1).all()
    assert buds.loc[first, 'status'].str.contains('no contact table').all()
    assert buds.loc[first, 'status'].str.contains('buds').all()
    late = buds[~first & buds['mother'].isin(buds.loc[first, 'bud'])]
    assert len(late) and (late['method'] == 'growth_plateau').all()
    assert (late['t_cytokinesis'] == late['t_emergence'] + BUD - 1).all()
    assert not len(da.budding_cycles(measures, buds))
    touching_forever = pd.DataFrame(
        [(int(b), int(m), int(t), True) for b, m in zip(buds['bud'], buds['mother'])
         for t in measures.loc[measures['track_id'] == b, 't']],
        columns=['bud', 'mother', 't', 'touching'])
    again = da.budding_cytokinesis(measures, events, contacts=touching_forever)
    assert again.loc[first, 'status'].str.contains('separation not seen').all()
    assert (again.loc[~first, 'method'] == 'growth_plateau').all()
    apart = touching_forever.assign(touching=False)
    third = da.budding_cytokinesis(measures, events, contacts=apart)
    assert third.loc[first, 'status'].str.contains('not touching its mother').all()


def test_the_growth_curve_gives_cytokinesis_when_the_masks_cannot():
    lineage = pd.DataFrame({'label': [1, 2], 't_start': [0, 5], 't_end': [30, 30],
                            'parent': [0, 1]})
    bud_area = [10, 30, 50, 70, 90, 110, 120, 121, 122, 122, 123, 123, 124, 124]
    measures = pd.DataFrame(
        {'track_id': [1] * 31 + [2] * len(bud_area),
         't': list(range(31)) + list(range(5, 5 + len(bud_area))),
         'area': [400.0] * 31 + [float(a) for a in bud_area]})
    events = da.division_events(lineage, mode='budding')
    buds = da.budding_cytokinesis(measures, events)
    row = buds.iloc[0]
    assert row['method'] == 'growth_plateau' and row['status'] == 'ok'
    assert row['t_emergence'] == 5 and 9 <= row['t_cytokinesis'] <= 12
    short = measures[(measures['track_id'] == 1) | (measures['t'] < 9)]
    assert 'too short' in da.budding_cytokinesis(short, events)['status'].iloc[0]
    flat = measures.assign(area=np.where(measures['track_id'] == 2, 50.0, measures['area']))
    assert 'does not grow' in da.budding_cytokinesis(flat, events)['status'].iloc[0]
    steady = measures.assign(area=np.where(measures['track_id'] == 2,
                                           measures['t'] * 10.0, measures['area']))
    assert 'does not slow' in da.budding_cytokinesis(steady, events)['status'].iloc[0]
    twice = pd.concat([lineage, pd.DataFrame({'label': [3], 't_start': [10], 't_end': [18],
                                              'parent': [2]})], ignore_index=True)
    early = da.budding_cytokinesis(measures, da.division_events(twice, mode='budding'))
    assert 'too short' in early.loc[early['bud'] == 2, 'status'].iloc[0]
    gone = measures[measures['track_id'] == 1]
    assert da.budding_cytokinesis(gone, events)['status'].iloc[0] == 'bud not measured'


def test_bud_contacts_reads_touching_and_separated_cells_from_masks():
    events = pd.DataFrame({'parent': [1, 4], 't_div': [0, 0], 'kind': ['budding', 'fission'],
                           'n_daughters': [1, 2], 'daughters': [(2,), (5, 6)]})
    planes = []
    for gap in (0, 1, 3):
        m = np.zeros((20, 30), np.int32)
        m[5:15, 2:10] = 1
        m[7:12, 10 + gap:15 + gap] = 2
        planes.append(m)
    lone = np.zeros((20, 30), np.int32)
    lone[5:15, 2:10] = 1
    planes.append(lone)
    contacts = da.bud_contacts(planes, events, times=[3, 4, 5, 6], distance=1)
    assert list(contacts['t']) == [3, 4, 5]
    assert list(contacts['touching']) == [True, False, False]
    seam = da.bud_contacts(planes, events, times=[3, 4, 5, 6])
    assert list(seam['touching']) == [True, True, False]
    assert not len(da.bud_contacts(planes, events.iloc[1:]))
    with pytest.raises(ValueError):
        da.bud_contacts(planes, events, times=[1])
    with pytest.raises(ValueError):
        da.bud_contacts(planes, events, distance=0)
    with pytest.raises(ValueError):
        da.bud_contacts([np.zeros((2, 2, 2), np.int32)], events)


def test_bad_input_is_refused():
    events = pd.DataFrame(columns=['parent', 't_div', 'kind', 'n_daughters', 'daughters'])
    with pytest.raises(ValueError):
        da.budding_cytokinesis(pd.DataFrame({'t': [0]}), events)
    with pytest.raises(ValueError):
        da.budding_cytokinesis(pd.DataFrame({'t': [0], 'track_id': [1], 'area': [1.0]}),
                               events, plateau_fraction=1.5)
    with pytest.raises(ValueError):
        da.budding_cycles(pd.DataFrame({'t': [0]}), pd.DataFrame())
    empty = da.budding_cytokinesis(pd.DataFrame({'t': [0], 'track_id': [1], 'area': [1.0]}),
                                   events)
    assert not len(empty) and 'status' in empty
    assert not len(da.budding_cycles(pd.DataFrame({'t': [0], 'track_id': [1], 'area': [1.0]}),
                                     empty))


def test_budding_results_are_written_pooled_and_drawn(tmp_path):
    movies = {}
    for seed in range(3):
        measures, lineage, contacts = _budding_movie('sizer', founders=12, seed=seed)
        results = da.analyse_divisions(measures, lineage, mode='budding', n_boot=20)
        da.add_budding_cycles(results, measures, lineage, contacts=contacts, n_boot=50)
        da.calibrate_divisions(results, pixel_size_um=0.5 if seed else None)
        da.write_division_results(results, tmp_path / f'm{seed}', plot=False)
        movies[f'm{seed}'] = da.read_division_results(tmp_path / f'm{seed}')
    assert len(movies['m0']['buds']) and 'method' in movies['m0']['buds']
    pooled = da.pool_division_movies(movies, n_boot=50)
    sc = pooled['size_control']
    assert set(sc['branch']) == {'mother', 'daughter'} and set(sc['size_unit']) == {'px^2', 'px^3'}
    assert set(pooled['buds']['movie']) == {'m0', 'm1', 'm2'}
    figures = da.division_figures(pooled)
    ax = figures['size_control'].axes[0]
    assert 'Budding model' in ax.get_title()
    assert any('mothers' in t.get_text() for t in ax.get_legend().get_texts())
    calibrated = {k: v for k, v in movies.items() if k != 'm0'}
    assert set(da.pool_division_movies(calibrated, n_boot=20)['size_control']['size_unit']) == {
        'um^2', 'um^3'}


def _budding_stack(tiles=8, frames=24, seed=0):
    """Mothers that bud twice; each bud grows against her and then moves 3 px away."""
    rng = np.random.default_rng(seed)
    stack = np.zeros((frames, 30, 30 * tiles), np.int32)
    rows, nxt = [], 1
    for tile in range(tiles):
        x0 = 30 * tile + 2
        mother = nxt
        nxt += 1
        side = int(rng.integers(7, 10))
        for t in range(frames):
            stack[t, 5:5 + side, x0:x0 + side] = mother
            rows.append({'track_id': mother, 'frame': t, 'x': x0 + side / 2,
                         'y': 5 + side / 2, 'parent_track_id': 0})
        for start, y in ((3, 5), (13, 18)):
            bud = nxt
            nxt += 1
            for t in range(start, frames):
                grow = min(t - start, 5)
                gap = 0 if t < start + 6 else 3
                bx = x0 + side + gap
                stack[t, y:y + 2 + grow, bx:bx + 2 + grow] = bud
                rows.append({'track_id': bud, 'frame': t, 'x': bx + 1.0, 'y': y + 1.0,
                             'parent_track_id': mother})
    return stack, pd.DataFrame(rows)


def test_the_step_fits_budding_movies_and_predicts_over_three_fields(tmp_path, capsys):
    import os

    from spacr.tabular import write_table

    src = tmp_path / 'plate' / 'masks'
    os.makedirs(src)
    settings = {'timelapse_division_mode': 'budding', 'timelapse_division_window': 4,
                'timelapse_division_prediction': True, 'timelapse_division_prediction_frames': 3,
                'timelapse_division_prediction_folds': 3,
                'timelapse_division_prediction_model': 'logistic', 'save': False}
    for field in range(3):
        stack, tracks = _budding_stack(seed=field)
        write_table(tracks, tmp_path / 'plate' / 'tracks' / f'trackpy_tracks_cell_F{field}.csv')
        result = da._run_division_step(str(src), f'F{field}', 'cell', 'iou', settings,
                                       label_stack=list(stack))
        assert result is not None
        assert result['summary']['size_control_model'] == 'budding'
        assert result['summary']['cytokinesis_methods'].get('contact', 0) > 0
    out = tmp_path / 'plate' / 'tracks' / 'division'
    assert (out / 'trackpy_tracks_cell_F0' / 'buds.csv').is_file()
    assert (out / 'trackpy_tracks_cell_F0' / 'prediction_features.csv').is_file()
    assert 'prediction' in result
    assert (out / 'prediction_trackpy_tracks_cell' / 'summary.json').is_file()
    assert 'Division prediction over 3 fields' in capsys.readouterr().out
