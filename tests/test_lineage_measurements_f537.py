"""Measured lineage colours require exact saved frame and final-label provenance."""
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr._lineage_measurements import (
    _load_lineage_sources,
    _measured_lineage_inputs,
    _run_measured_lineage_step,
    _write_lineage_sources,
)
from spacr._lineage_trees import _run_lineage_step
from tests.test_lineage_trees_from_tracks import _field


@pytest.fixture
def project(tmp_path):
    tracks = _field()
    directory = tmp_path / 'tracks'
    directory.mkdir()
    path = directory / 'trackpy_tracks_cell_plate1_A01_1.csv'
    tracks.to_csv(path, index=False)
    # Frame positions deliberately do NOT equal acquisition time identifiers.
    names = [f'plate1_A01_1_{100 + frame * 3}.npy' for frame in range(21)]
    labels = []
    measured = []
    for frame, _name in enumerate(names):
        ids = tracks.loc[tracks.frame == frame, 'track_id'].astype(int).to_numpy()
        labels.append(np.pad(ids.reshape(1, -1), ((0, 1), (0, 0))))
        for value in ids:
            measured.append({'prcf': f'plate1_r1_c1_f1_t{100 + frame * 3}',
                             'object_label': int(value), 'cell_intensity': float(frame + value * 100)})
    # Same labels in another field must never contaminate colours.
    measured.append({'prcf': 'plate2_r1_c1_f1_t115', 'object_label': 2, 'cell_intensity': 99999.0})
    dbdir = tmp_path / 'measurements'
    dbdir.mkdir()
    db = dbdir / 'measurements.db'
    with sqlite3.connect(db) as con:
        pd.DataFrame(measured).sample(frac=1, random_state=5).to_sql('cell', con, index=False)
    _write_lineage_sources(path, 'cell', names, labels)
    return path, db, tracks, names, labels


def test_saved_mapping_reads_correct_feature_rows_and_preserves_sources(project):
    path, db, original_tracks, _, _ = project
    csv_before, db_before = path.read_bytes(), db.read_bytes()
    tracks, measured, report = _measured_lineage_inputs(db, path, 'cell_intensity')
    pd.testing.assert_frame_equal(tracks, original_tracks)
    assert report['matched_rows'] == len(tracks) and report['unmatched_rows'] == 0
    np.testing.assert_array_equal(measured['cell_intensity'], measured['frame'] + measured['track_id'] * 100)
    assert path.read_bytes() == csv_before and db.read_bytes() == db_before
    assert report['object_type'] == 'cell'


def test_post_measure_step_writes_separate_outputs_and_correct_segment_means(project):
    path, db, _, _, _ = project
    pre = path.parent / 'lineage'
    pre.mkdir()
    sentinel = pre / 'original.csv'
    sentinel.write_bytes(b'original pre-measure lineage')
    sources = {p: p.read_bytes() for p in (path, db, sentinel)}
    result, = _run_measured_lineage_step(db, {'timelapse_objects': ['cell'], 'save': False,
                                            'timelapse_lineage_color_by': 'cell_intensity'})
    two = result['segments'].query('track_id == 2 and start_frame == 5').iloc[0]
    assert two['color_cell_intensity'] == np.mean(np.arange(5, 13) + 200)
    assert all('lineage_measured' in str(p) for p in result['paths'].values())
    report = json.loads((path.parent / 'lineage_measured' /
                         (path.stem + '_measurement_source.json')).read_text())
    assert report['matched_rows'] == len(pd.read_csv(path))
    assert len(report['measurement_values_sha256']) == 64
    for file, content in sources.items():
        assert file.read_bytes() == content


def test_tracking_hook_records_actual_nonzero_frame_names(project):
    path, _, _, names, labels = project
    source = path.parent.parent / 'merged'
    _run_lineage_step(str(source), 'plate1_A01_1', 'cell', 'iou', {'save': False},
                      frame_sources=names, label_stack=labels)
    _, mapping, provenance = _load_lineage_sources(path)
    assert provenance['frames'][0]['filename'] == 'plate1_A01_1_100.npy'
    assert mapping.loc[mapping.frame == 5, 'prcf'].unique().tolist() == ['plate1_r1_c1_f1_t115']


def test_filtered_measurement_rows_stay_missing_without_reindexing(project):
    path, db, _, _, _ = project
    with sqlite3.connect(db) as con:
        con.execute('DELETE FROM cell WHERE prcf=?', ('plate1_r1_c1_f1_t115',))
    _, measured, report = _measured_lineage_inputs(db, path, 'cell_intensity')
    assert measured.loc[measured.frame == 5, 'cell_intensity'].isna().all()
    assert measured.loc[measured.frame == 6, 'cell_intensity'].notna().all()
    assert report['unmatched_rows'] > 0


@pytest.mark.parametrize('bad', ['missing', 'changed_csv', 'duplicate_frame', 'duplicate_label',
                                'wrong_label', 'omitted_label', 'bad_prcf', 'boolean_frame'])
def test_missing_or_ambiguous_provenance_is_rejected(project, bad):
    path, db, _, _, _ = project
    manifest = path.with_name(path.name + '.lineage_sources.json')
    data = json.loads(manifest.read_text())
    if bad == 'missing':
        manifest.unlink()
    elif bad == 'changed_csv':
        with path.open('a') as stream:
            stream.write('\n')
    else:
        frame = data['frames'][0]
        if bad == 'duplicate_frame':
            data['frames'].append(frame)
        elif bad == 'duplicate_label':
            frame['labels'].append(frame['labels'][0])
        elif bad == 'wrong_label':
            frame['labels'][0]['object_label'] = 999
        elif bad == 'omitted_label':
            frame['labels'].pop()
        elif bad == 'bad_prcf':
            frame['prcf'] = 'plate2_r1_c1_f1_t100'
        else:
            frame['frame'] = False
        manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='lineage'):
        _measured_lineage_inputs(db, path, 'cell_intensity')
    assert not (path.parent / 'lineage_measured').exists()


@pytest.mark.parametrize('bad', ['duplicate', 'text', 'infinite', 'missing_column', 'no_matches'])
def test_measurement_schema_errors_preserve_existing_outputs(project, bad, capsys):
    path, db, _, _, _ = project
    output = path.parent / 'lineage_measured'
    output.mkdir()
    sentinel = output / 'old.csv'
    sentinel.write_bytes(b'prior valid measured colours')
    with sqlite3.connect(db) as con:
        if bad == 'duplicate':
            con.execute('INSERT INTO cell SELECT * FROM cell WHERE prcf=? LIMIT 1', ('plate1_r1_c1_f1_t115',))
        elif bad == 'text':
            con.execute("UPDATE cell SET cell_intensity='wrong' WHERE prcf=?", ('plate1_r1_c1_f1_t115',))
        elif bad == 'infinite':
            con.execute('UPDATE cell SET cell_intensity=? WHERE prcf=?', (float('inf'), 'plate1_r1_c1_f1_t115'))
        elif bad == 'missing_column':
            con.execute('ALTER TABLE cell RENAME COLUMN cell_intensity TO something_else')
        else:
            con.execute('DELETE FROM cell')
    assert _run_measured_lineage_step(db, {'timelapse_lineage_color_by': 'cell_intensity', 'save': False}) == []
    assert 'could not be built' in capsys.readouterr().out
    assert list(output.iterdir()) == [sentinel] and sentinel.read_bytes() == b'prior valid measured colours'


@pytest.mark.parametrize('bad', ['count', 'labels', 'cross_field', 'duplicate_time'])
def test_tracker_provenance_preflight_preserves_old_manifest(project, bad):
    path, _, _, names, labels = project
    manifest = path.with_name(path.name + '.lineage_sources.json')
    before = manifest.read_bytes()
    if bad == 'count':
        labels = labels[:-1]
    elif bad == 'labels':
        labels[0] = np.zeros_like(labels[0])
    elif bad == 'cross_field':
        names[0] = 'plate2_A01_1_100.npy'
    else:
        names[0] = names[1]
    with pytest.raises(ValueError, match='lineage'):
        _write_lineage_sources(path, 'cell', names, labels)
    assert manifest.read_bytes() == before


def test_measured_colour_retains_explicit_tracking_calibration(project, capsys):
    path, db, _, names, labels = project
    _write_lineage_sources(path, 'cell', names, labels, frame_interval_s=900)
    result, = _run_measured_lineage_step(db, {'save': False, 'timelapse_lineage_color_by': 'cell_intensity'})
    complete = result['segments'].dropna(subset=['generation_time'])
    assert complete['generation_time_hours'].tolist() == [2, 2.5]
    paths = list((path.parent / 'lineage_measured').iterdir())
    before = {p: p.read_bytes() for p in paths}
    assert _run_measured_lineage_step(db, {'save': False, 'frame_interval_s': 60,
                                         'timelapse_lineage_color_by': 'cell_intensity'}) == []
    assert 'conflicts with saved tracking calibration' in capsys.readouterr().out
    assert {p: p.read_bytes() for p in paths} == before


def test_tracking_hook_invalid_calibration_publishes_nothing_new(project):
    path, _, _, names, labels = project
    before = {p: p.read_bytes() for p in path.parent.iterdir()}
    result = _run_lineage_step(str(path.parent.parent / 'merged'), 'plate1_A01_1', 'cell', 'iou',
                               {'save': False, 'frame_interval_s': -1},
                               frame_sources=names, label_stack=labels)
    assert result is None
    assert {p: p.read_bytes() for p in path.parent.iterdir()} == before


@pytest.mark.parametrize('malformed', ['duplicate_json', 'nan_json', 'invalid_interval'])
def test_strict_manifest_json_rejects_ambiguous_or_invalid_values(project, malformed):
    path, db, _, _, _ = project
    manifest = path.with_name(path.name + '.lineage_sources.json')
    data = manifest.read_text()
    if malformed == 'duplicate_json':
        data = data.replace('"version": 1,', '"version": 1, "version": 1,')
    elif malformed == 'nan_json':
        data = data.replace('"frame_interval_s": null', '"frame_interval_s": NaN')
    else:
        data = data.replace('"frame_interval_s": null', '"frame_interval_s": true')
    manifest.write_text(data)
    with pytest.raises(ValueError, match='lineage'):
        _measured_lineage_inputs(db, path, 'cell_intensity')


@pytest.mark.parametrize('key,value', [
    ('frame_interval_s', []), ('frame_interval_s', 'bad'), ('frame_interval_s', 10**400),
    ('timelapse_lineage_max_distance', {}), ('timelapse_lineage_max_distance', float('nan')),
    ('timelapse_lineage_max_distance', True), ('timelapse_lineage_max_distance', 10**400),
])
def test_malformed_run_settings_report_failure_without_publishing(project, key, value, capsys):
    path, db, _, _, _ = project
    assert _run_measured_lineage_step(db, {'save': False, 'timelapse_lineage_color_by': 'cell_intensity',
                                         key: value}) == []
    assert 'could not be built' in capsys.readouterr().out
    assert not (path.parent / 'lineage_measured').exists()


def test_unset_distance_uses_legacy_default_and_records_exact_mapping_digest(project):
    import hashlib

    path, db, _, _, _ = project
    manifest = path.with_name(path.name + '.lineage_sources.json')
    result, = _run_measured_lineage_step(db, {'save': False, 'timelapse_lineage_color_by': 'cell_intensity',
                                            'timelapse_lineage_max_distance': None})
    assert result['measurement_report']['mapping_sha256'] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert result['segments']['generation_time'].dropna().tolist() == [8, 10]
