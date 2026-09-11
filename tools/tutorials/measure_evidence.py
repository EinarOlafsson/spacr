"""Read-only checks of the actual Measure tutorial output against raw pixels.

The five recorded merge pairs are explicit evidence, not inferred by fitting
the output values. Each pair must share a nonzero child label in the source.
This verifies area, integrated intensity and mean, not every measurement or
the biological validity of the original segmentation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np


RECORDED_MERGES = {
    'plate1_E01_10_1': {4: 12},
    'plate1_E01_1_1': {7: 8},
    'plate1_E02_11_1': {17: 18},
    'plate1_L02_10_1': {16: 30},
    'plate1_L02_9_1': {9: 12},
}


def verify_field(raw, rows, merges=None):
    """Verify recorded cell areas and four-channel sums/means independently."""
    raw = np.asarray(raw)
    if raw.ndim != 3 or raw.shape[-1] != 7 or raw.dtype != np.uint16:
        raise ValueError('Expected seven uint16 planes from the recorded example')
    if not rows:
        raise ValueError('An empty field proves no measurements')
    merges = dict(merges or {})
    labels = raw[..., 4]
    keys = [int(row['object_label']) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate measured object identity')
    if any(key <= 0 or not np.any(labels == key) for key in keys):
        raise ValueError('Missing source label or background measured as object')
    if not set(merges).issubset(keys):
        raise ValueError('Expected merge survivor is absent from measured rows')
    if len(set(merges.values())) != len(merges) or set(merges.values()) & set(keys):
        raise ValueError('A merged source label would be counted twice')
    bridges = []
    for survivor, other in merges.items():
        if other <= 0 or not np.any(labels == other):
            raise ValueError('Merged source label is absent')
        connections = []
        for plane in (5, 6):
            shared = (set(np.unique(raw[..., plane][labels == survivor]))
                      & set(np.unique(raw[..., plane][labels == other]))) - {0}
            if shared:
                connections.append({'child_plane': plane,
                                    'child_labels': sorted(map(int, shared))})
        if not connections:
            raise ValueError('Merge is not supported by a shared source child label')
        bridges.append({'survivor': survivor, 'merged_raw_label': other,
                        'source_bridges': connections})
    areas = np.bincount(labels.ravel())
    sums = [np.bincount(labels.ravel(), weights=raw[..., ch].ravel())
            for ch in range(4)]
    maximum_mean_error = 0.0
    for row in rows:
        key = int(row['object_label'])
        group = (key, merges[key]) if key in merges else (key,)
        area = sum(int(areas[k]) for k in group)
        if float(row['cell_area']) != area:
            raise ValueError('Saved cell area differs from source pixel count')
        for channel in range(4):
            integrated = sum(int(sums[channel][k]) for k in group)
            if float(row[f'cell_channel_{channel}_integrated_intensity']) != integrated:
                raise ValueError('Saved integrated intensity differs from source pixels')
            error = abs(float(row[f'cell_channel_{channel}_mean_intensity'])
                        - integrated / area)
            if not np.isfinite(error) or error > 1e-9:
                raise ValueError('Saved mean intensity differs from source pixels')
            maximum_mean_error = max(maximum_mean_error, error)
    return {'objects_checked': len(rows), 'values_checked': len(rows) * 9,
            'area_maximum_error': 0, 'integrated_intensity_maximum_error': 0,
            'mean_intensity_maximum_error': maximum_mean_error,
            'source_supported_merges': bridges}


def inspect_project(project: Path, capture: Path):
    """Audit a finished private native run without updating its database."""
    project, capture = Path(project).resolve(), Path(capture).resolve()
    db = project / 'measurements/measurements.db'
    before_hash = hashlib.sha256(db.read_bytes()).hexdigest()
    with sqlite3.connect(db.as_uri() + '?mode=ro', uri=True) as connection:
        connection.row_factory = sqlite3.Row
        statuses = [dict(row) for row in connection.execute('SELECT * FROM run_status')]
        if len(statuses) != 1 or statuses[0]['status'] != 'complete':
            raise ValueError('Expected one complete native run')
        status = statuses[0]
        if (status['n_attempted'], status['n_succeeded'], status['n_failed']) != (16, 16, 0):
            raise ValueError('The complete sixteen-field example has not succeeded')
        fields = [row[0] for row in connection.execute('SELECT DISTINCT file_name FROM cell')]
        if len(fields) != 16:
            raise ValueError('Expected actual cell results for all sixteen fields')
        counts = {table: connection.execute(f'SELECT count(*) FROM {table}').fetchone()[0]
                  for table in ('cell', 'nucleus', 'pathogen', 'cytoplasm', 'png_list', 'intensity_rescale')}
        checks, source_hashes = {}, {}
        columns = ['object_label', 'cell_area'] + [
            f'cell_channel_{channel}_{kind}' for channel in range(4)
            for kind in ('integrated_intensity', 'mean_intensity')]
        for field in sorted(fields):
            if Path(field).name != field:
                raise ValueError('A field name is not a file stem')
            source = project / 'merged' / (field + '.npy')
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            rows = [dict(row) for row in connection.execute(
                'SELECT ' + ','.join(columns) + ' FROM cell WHERE file_name=?', (field,))]
            checks[field] = verify_field(np.load(source, mmap_mode='r'), rows,
                                         RECORDED_MERGES.get(field))
            if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                raise ValueError('Source changed during evidence inspection')
            source_hashes[field] = digest
        paths = [row[0] for row in connection.execute('SELECT png_path FROM png_list')]
        if len(paths) != counts['cell'] or len(set(paths)) != len(paths):
            raise ValueError('Expected one distinct crop path per measured cell')
        expected_prefix = Path('/home/olafsson/.cache/spacr/example_data/plate1')
        from PIL import Image
        for recorded in paths:
            relative = Path(recorded).relative_to(expected_prefix)
            target = (project / relative).resolve()
            if not target.is_relative_to(project):
                raise ValueError('Crop path leaves the private project')
            with Image.open(target) as image:
                if image.size != (224, 224) or image.mode != 'RGB':
                    raise ValueError('Crop does not match the recorded RGB dimensions')
                image.verify()
        rescale = [tuple(row) for row in connection.execute(
            'SELECT original_dtype,rescale_factor,rescale_scope FROM intensity_rescale')]
        if any(row != ('uint16', 1.0, 'identity') for row in rescale):
            raise ValueError('The raw-intensity comparison requires identity rescaling')
    if hashlib.sha256(db.read_bytes()).hexdigest() != before_hash:
        raise ValueError('Database changed during read-only inspection')
    live = json.loads((capture / 'live_variants.json').read_text())
    source = project / 'merged' / Path(live['source']).name
    if hashlib.sha256(np.load(source).tobytes()).hexdigest() != live['source_sha256']:
        raise ValueError('Live-preview source changed after capture')
    if live['before'] != live['restored'] or not 0 < len(live['after']) < len(live['before']):
        raise ValueError('Live filter did not selectively change and restore the grid')
    for key in ('source_unchanged', 'propagation_off_preserved_batch', 'propagation_on_updated_batch'):
        if live[key] is not True:
            raise ValueError('Live filter or propagation proof failed')
    return {'accepted': True, 'run_status': status, 'database_sha256': before_hash,
            'database_bytes': db.stat().st_size, 'table_counts': counts,
            'field_checks': checks, 'source_file_sha256': source_hashes,
            'values_checked': sum(x['values_checked'] for x in checks.values()),
            'decoded_rgb_crops': len(paths), 'crop_size': [224, 224],
            'live_grid_counts': [len(live[key]) for key in ('before', 'after', 'restored')],
            'live_source_array_preserved_after_batch': True,
            'all_measurement_columns_validated': False,
            'biological_segmentation_validated': False,
            'registry_and_gui_staleness_accepted': False}
