"""Event detection on timelapse tracks: mitosis and host death from track windows.

Synthetic tracked movies with known mitoses (the mother rounds up and
brightens, two daughters start beside her), host deaths (the cell shrinks
and brightens, then its track ends), broken tracks, decoy roundings and
cells that vanish. The detector is trained on annotated fields and scored on
held-out fields; detected mitoses re-link the daughters to their mothers.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
from skimage.draw import ellipse                                   # noqa: E402

from spacr import timelapse as tl                                  # noqa: E402


def _field(seed, n_frames=30, size=160):
    """Labels, images, tracks, annotated events and true division links."""
    rng = np.random.default_rng(seed)
    labels = np.zeros((n_frames, size, size), np.int32)
    images = rng.normal(0.1, 0.02, (n_frames, size, size)).astype(np.float32)
    rows, ann, links = [], [], []
    ids = iter(range(1, 1000))

    def paint(t, tid, x, y, a, b, angle, level):
        rr, cc = ellipse(y, x, a, b, shape=(size, size), rotation=angle)
        free = labels[t, rr, cc] == 0
        labels[t, rr[free], cc[free]] = tid
        images[t, rr, cc] = level + rng.normal(0, 0.03, rr.size)
        rows.append({'frame': t, 'track_id': tid, 'x': x, 'y': y})

    grid = [(x, y) for x in range(25, size - 20, 36) for y in range(25, size - 20, 36)]
    rng.shuffle(grid)
    kinds = ['mitosis', 'mitosis', 'death', 'broken', 'decoy', 'vanish', 'plain']
    for (x, y), kind in zip(grid, kinds):
        tid = next(ids)
        angle = rng.uniform(0, np.pi)
        x, y = float(x), float(y)
        event = int(rng.integers(8, n_frames - 8))
        for t in range(n_frames):
            x += rng.normal(0, 0.7)
            y += rng.normal(0, 0.7)
            if kind in ('mitosis', 'death', 'vanish') and t > event:
                break
            if kind == 'broken' and t == event + 1:
                tid = next(ids)
            a, b, level = 8.0, 5.0, 0.5
            if kind == 'mitosis':
                a, b = 7 + 2 * t / event, 4.5 + 2 * t / event
                if t >= event - 1:
                    a, b, level = 7.5, 7.5, 0.95
            if kind == 'decoy' and abs(t - event) <= 1:
                a, b, level = 7.0, 7.0, 0.85
            level += rng.normal(0, 0.08)
            if kind == 'death' and t >= event - 2:
                shrink = (event - t + 1) / 3
                a, b, level = 3 + 5 * shrink, 2.5 + 2.5 * shrink, 0.9
            paint(t, tid, x, y, a, b, angle, level)
        if kind == 'mitosis':
            ann.append({'track_id': tid, 'frame': event, 'event': 'mitosis'})
            for sign in (-1, 1):
                did = next(ids)
                links.append((did, tid))
                xd = x + sign * 7 * np.cos(angle)
                yd = y + sign * 7 * np.sin(angle)
                for t in range(event + 1, n_frames):
                    paint(t, did, xd, yd, 5.5, 4.0, angle, 0.5)
        if kind == 'death':
            ann.append({'track_id': tid, 'frame': event, 'event': 'host_death'})
    tracks = pd.DataFrame(rows).drop_duplicates(['frame', 'track_id'])
    return labels, images, tracks.reset_index(drop=True), pd.DataFrame(ann), links


@pytest.fixture(scope='module')
def run(tmp_path_factory):
    """A tracked run of four fields in two conditions, with frame features."""
    root = tmp_path_factory.mktemp('events')
    os.makedirs(root / 'tracks')
    anns, truth = [], {}
    for k in range(4):
        name = f'plate1_r1_c{1 + k % 2}_f{k + 1}'
        labels, images, tracks, ann, links = _field(200 + k)
        tracks.to_csv(root / 'tracks' / f'trackpy_tracks_cell_{name}.csv', index=False)
        tl._run_event_features_step(str(root / 'masks'), name, 'cell', labels,
                                    images[..., None], 'trackpy', {})
        anns.append(ann.assign(field=name))
        truth[name] = links
    pd.concat(anns).to_csv(root / 'annotations.csv', index=False)
    result = tl._event_detection(str(root / 'tracks'), 'cell', 'trackpy',
                                 annotations=str(root / 'annotations.csv'),
                                 conditions=['mock=c1', 'drug=c2'], epochs=15)
    return root, truth, result


def test_frame_features_follow_each_track_with_crops():
    labels, images, tracks, _, _ = _field(1, n_frames=20)
    features, crops = tl._event_frame_features(labels, images)
    assert set(zip(features['frame'], features['track_id'])) == set(
        zip(tracks['frame'], tracks['track_id']))
    assert crops.shape == (len(features), 1, tl._EVENT_CROP, tl._EVENT_CROP)
    assert {'area', 'eccentricity', 'intensity_mean_c0'} <= set(features.columns)


def test_windows_are_centred_on_each_frame_and_mark_absence():
    tracks = pd.DataFrame({'frame': [3, 4, 5], 'track_id': [1, 1, 1],
                           'x': [0.0, 3.0, 3.0], 'y': [0.0, 4.0, 4.0]})
    table = tl._event_track_table(tracks)
    x, crops, index = tl._event_windows(table, ['speed'], 5, [0.0], [1.0])
    assert x.shape == (3, 2, 5) and crops is None
    assert list(index['frame']) == [3, 4, 5]
    assert x[1, 0].tolist() == [0.0, 0.0, 5.0, 0.0, 0.0]
    assert x[0, 1].tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]


@pytest.mark.heavy
def test_held_out_scores_state_precision_recall_and_timing(run):
    _, _, result = run
    scores = result['scores'].set_index('event')
    assert list(scores.index) == ['host_death', 'mitosis', 'all']
    assert scores.loc['mitosis', 'annotated'] == 8
    assert scores.loc['all', 'precision'] >= 0.8
    assert scores.loc['all', 'recall'] >= 0.8
    assert scores.loc['all', 'mean_abs_timing_error'] <= 1.0
    assert os.path.isfile(result['paths']['scores'])


@pytest.mark.heavy
def test_detected_mitoses_relink_the_daughters(run):
    root, truth, _ = run
    found = expected = 0
    for name, links in truth.items():
        fixed = pd.read_csv(root / 'tracks' / 'events' / f'trackpy_tracks_cell_{name}_corrected.csv')
        parents = fixed.groupby('track_id')['parent_track_id'].first()
        found += len({(int(t), int(p)) for t, p in parents.items() if p > 0} & set(links))
        expected += len(links)
    assert found >= 0.8 * expected
    assert os.path.isdir(root / 'tracks' / 'events' / 'lineage')


@pytest.mark.heavy
def test_event_timing_is_compared_across_conditions(run):
    _, _, result = run
    summary = pd.read_csv(result['paths']['mitosis_summary'])
    conditions = summary[summary['level'] == 'condition']
    assert set(conditions['group']) == {'mock', 'drug'}
    assert conditions['n'].sum() > conditions['events'].sum() > 0
    assert os.path.isfile(result['paths']['mitosis_figure'])


@pytest.mark.heavy
def test_a_saved_model_is_applied_by_the_run_step(run, tmp_path):
    root, _, result = run
    settings = {'timelapse_mode': 'trackpy', 'timelapse_objects': ['cell'],
                'timelapse_events_model': result['paths']['model'],
                'save': False}
    out = tl._run_event_detection_step(str(root), settings)
    again = out['cell']['events']
    assert out['cell']['scores'] is None
    assert len(again) == len(result['events'])


def test_division_correction_ignores_tracks_far_from_a_mitosis():
    tracks = pd.DataFrame({'frame': [0, 1, 2, 3, 3], 'track_id': [1, 1, 1, 2, 3],
                           'x': [10.0, 10, 10, 14, 90], 'y': [10.0, 10, 10, 10, 90]})
    events = pd.DataFrame({'track_id': [1], 'frame': [2], 'event': ['mitosis']})
    fixed, links = tl._event_correct_divisions(tracks, events, max_distance=20)
    assert links['track_id'].tolist() == [2]
    assert fixed.groupby('track_id')['parent_track_id'].first().to_dict() == {1: 0, 2: 1, 3: 0}


def test_scores_count_misses_and_false_events():
    ann = pd.DataFrame({'field': ['a', 'a'], 'track_id': [1, 2], 'frame': [5, 9],
                        'event': ['mitosis', 'mitosis']})
    det = pd.DataFrame({'field': ['a', 'a'], 'track_id': [1, 3], 'frame': [6, 2],
                        'event': ['mitosis', 'mitosis'], 'probability': [0.9, 0.8]})
    row = tl._event_scores(det, ann, ['mitosis']).iloc[0]
    assert (row['true_positives'], row['precision'], row['recall']) == (1, 0.5, 0.5)
    assert row['mean_abs_timing_error'] == 1.0


def test_detection_without_annotations_or_model_is_refused(tmp_path):
    (tmp_path / 'tracks').mkdir()
    pd.DataFrame({'frame': [0], 'track_id': [1], 'x': [1.0], 'y': [1.0]}).to_csv(
        tmp_path / 'tracks' / 'trackpy_tracks_cell_p_r1_c1_f1.csv', index=False)
    with pytest.raises(ValueError, match='timelapse_events_annotations'):
        tl._event_detection(str(tmp_path / 'tracks'), 'cell', 'trackpy')


def test_a_track_carried_through_its_divisions_is_split_into_cell_cycles():
    spans = pd.DataFrame({'field': ['f', 'f'], 'track_id': [1, 2],
                          'start': [0, 5], 'end': [29, 12]})
    cycles = tl._event_cycles(spans, {('f', 1): [9, 19], ('f', 2): [12]})
    one = cycles[cycles['track_id'] == 1]
    assert one['duration'].tolist() == [10.0, 10.0, 9.0]
    assert one['event'].tolist() == [1, 1, 0]
    assert one['start'].tolist() == [0, 10, 20]
    two = cycles[cycles['track_id'] == 2]
    assert two['duration'].tolist() == [8.0] and two['event'].tolist() == [1]


def test_mitosis_timing_counts_cell_cycles_and_death_the_first_event():
    table = pd.DataFrame({'frame': list(range(30)), 'track_id': 1,
                          'x': 0.0, 'y': 0.0})
    events = pd.DataFrame({'field': ['plate1_r1_c1_f1'] * 3,
                           'track_id': [1, 1, 1], 'frame': [9, 19, 25],
                           'event': ['mitosis', 'mitosis', 'death']})
    timing = tl._event_timing({'plate1_r1_c1_f1': table}, events)
    mitosis = timing['mitosis'][0]
    assert sorted(mitosis['duration'].tolist()) == [9.0, 10.0, 10.0]
    assert int(mitosis['event'].sum()) == 2
    death = timing['death'][0]
    assert death['duration'].tolist() == [25.0] and death['event'].tolist() == [1]


def test_a_death_is_called_once_per_track_and_mitoses_may_repeat():
    index = pd.DataFrame({'track_id': [1] * 12, 'frame': list(range(12))})
    p = np.zeros((12, 3))
    p[:, 0] = 1.0
    for frame, value in ((2, 0.7), (9, 0.9)):
        p[frame] = [0.1, 0.0, value]
    for frame in (3, 10):
        p[frame] = [0.1, 0.9, 0.0]
    found = tl._event_peaks(index, p, ['none', 'mitosis', 'host_death'])
    assert found[found['event'] == 'mitosis']['frame'].tolist() == [3, 10]
    assert found[found['event'] == 'host_death']['frame'].tolist() == [9]
    assert tl._event_is_terminal('Lysis') and not tl._event_is_terminal('egress')


def _two_object_field(seed, n_frames=30):
    """Host and parasite tracks with annotated egress and invasion on hosts.

    Egress: a host holding four parasites ends and its parasites scatter.
    Invasion: an outside parasite glides into a host and stays. Decoys: a
    parasite gliding past a host, and a host ending with no parasites.
    """
    rng = np.random.default_rng(seed)
    hosts, parasites, ann = [], [], []
    pid = iter(range(1, 1000))
    for h, kind in enumerate(['egress', 'invasion', 'passer', 'empty', 'plain', 'egress']):
        hx, hy = 60.0 + 120 * h, 60.0
        event = int(rng.integers(10, n_frames - 8))
        end = event if kind in ('egress', 'empty') else n_frames - 1
        for t in range(end + 1):
            hosts.append({'frame': t, 'track_id': h + 1, 'x': hx + rng.normal(0, 0.5),
                          'y': hy + rng.normal(0, 0.5)})
        if kind == 'egress':
            ann.append({'track_id': h + 1, 'frame': event, 'event': 'egress'})
            for k in range(4):
                tid, ang = next(pid), k * np.pi / 2 + rng.uniform(0, 0.5)
                for t in range(n_frames):
                    r = 4.0 if t <= event else min(4.0 + 12 * (t - event), 50.0)
                    parasites.append({'frame': t, 'track_id': tid,
                                      'x': hx + r * np.cos(ang), 'y': hy + r * np.sin(ang)})
        if kind in ('invasion', 'passer'):
            tid = next(pid)
            for t in range(n_frames):
                d = float(np.clip(12.0 * (event - t), -50, 50))
                if kind == 'invasion':
                    d = max(d, 3.0)
                parasites.append({'frame': t, 'track_id': tid, 'x': hx + d, 'y': hy + 2.0})
            if kind == 'invasion':
                ann.append({'track_id': h + 1, 'frame': event, 'event': 'invasion'})
    return pd.DataFrame(hosts), pd.DataFrame(parasites), pd.DataFrame(ann)


def test_partner_columns_count_parasites_near_each_host():
    hosts, parasites, _ = _two_object_field(3)
    table = tl._event_track_table(hosts, partners={'parasite': parasites})
    cols = tl._event_columns(table)
    assert {'parasite_near', 'parasite_near_change', 'parasite_starts_near',
            'parasite_ends_near'} <= set(cols)
    assert table.loc[table['track_id'] == 1, 'parasite_near'].iloc[0] == 4
    invaded = table.loc[table['track_id'] == 2, 'parasite_near']
    assert invaded.iloc[0] == 0 and invaded.iloc[-1] == 1


@pytest.mark.heavy
def test_egress_and_invasion_are_read_from_host_and_parasite_tracks(tmp_path):
    tracks = tmp_path / 'tracks'
    os.makedirs(tracks)
    anns = []
    for k in range(4):
        name = f'plate1_r1_c1_f{k + 1}'
        hosts, parasites, ann = _two_object_field(300 + k)
        hosts.to_csv(tracks / f'trackpy_tracks_host_{name}.csv', index=False)
        parasites.to_csv(tracks / f'trackpy_tracks_parasite_{name}.csv', index=False)
        anns.append(ann.assign(field=name))
    pd.concat(anns).to_csv(tmp_path / 'annotations.csv', index=False)
    result = tl._event_detection(str(tracks), 'host', 'trackpy',
                                 annotations=str(tmp_path / 'annotations.csv'),
                                 epochs=20, plot=False, partners=['parasite', 'host'])
    scores = result['scores'].set_index('event')
    assert {'egress', 'invasion'} <= set(scores.index)
    assert scores.loc['all', 'precision'] >= 0.8
    assert scores.loc['all', 'recall'] >= 0.8
    assert scores.loc['all', 'mean_abs_timing_error'] <= 1.0
