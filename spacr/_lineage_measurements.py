import hashlib
import io
import json
import os
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from . import schema

_ROLES = ('cell', 'nucleus', 'pathogen')
_SUFFIX = '.lineage_sources.json'


def _atomic_json(path, data):
    """Replace one complete JSON artifact without exposing a partial file."""
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(data, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _tracks_snapshot(path):
    """Parse and validate the exact CSV bytes bound by the returned digest."""
    with open(path, 'rb') as stream:
        data = stream.read(128 * 1024 * 1024 + 1)
    if len(data) > 128 * 1024 * 1024:
        raise ValueError('lineage tracks CSV exceeds 128 MiB')
    tracks = pd.read_csv(io.BytesIO(data))
    for key in ('frame', 'track_id'):
        if key not in tracks:
            raise ValueError(f'lineage tracks have no {key}')
        numeric = pd.to_numeric(tracks[key], errors='coerce')
        if (tracks[key].map(lambda value: isinstance(value, (bool, np.bool_))).any()
                or not np.isfinite(numeric).all() or (numeric % 1 != 0).any()
                or (numeric < (0 if key == 'frame' else 1)).any()):
            raise ValueError(f'lineage {key} must contain valid integer identities')
        tracks[key] = numeric.astype(np.int64)
    if tracks.duplicated(['frame', 'track_id']).any():
        raise ValueError('lineage tracks contain duplicate frame/track_id identities')
    return tracks, hashlib.sha256(data).hexdigest()


def _source_fields(filenames):
    """Map actual ordered batch basenames to unique canonical field/time keys."""
    records = []
    for ordinal, filename in enumerate(filenames):
        name = str(filename)
        if Path(name).name != name or '\\' in name:
            raise ValueError('lineage source filenames must be basenames')
        field = schema.parse_field_stem(name, timelapse=True, strict=True)
        records.append({'frame': ordinal, 'filename': name, 'prcf': field.prcf})
    if not records or len({r['prcf'] for r in records}) != len(records):
        raise ValueError('lineage source timepoints must be nonempty and unique')
    fields = {r['prcf'].rsplit('_', 1)[0] for r in records}
    if len(fields) != 1:
        raise ValueError('lineage frame sources must belong to one imaging field')
    return records


def _prepare_lineage_sources(tracks_path, object_type, filenames, label_stack,
                             frame_interval_s=None):
    """Bind actual tracker frame order and final mask labels to a CSV snapshot."""
    if object_type not in _ROLES:
        raise ValueError('lineage object must be cell, nucleus or pathogen')
    tracks, digest = _tracks_snapshot(tracks_path)
    frames = _source_fields(filenames)
    if len(label_stack) != len(frames):
        raise ValueError('lineage source count differs from final mask frame count')
    if len(tracks) and tracks['frame'].max() >= len(frames):
        raise ValueError('lineage tracks reference an unmapped frame')
    for record in frames:
        ids = tracks.loc[tracks.frame == record['frame'], 'track_id'].tolist()
        labels = np.unique(label_stack[record['frame']])
        if not set(ids) <= set(labels):
            raise ValueError('lineage track IDs differ from the saved mask labels')
        record['labels'] = [{'track_id': int(value), 'object_label': int(value)}
                            for value in sorted(ids)]
    from ._lineage_trees import _lineage_calibrate_time, _lineage_segments

    _, calibration = _lineage_calibrate_time(
        _lineage_segments(tracks), tracks, frame_interval_s)
    interval = calibration['calibration_frame_interval_s']
    return {'version': 1, 'tracks_sha256': digest, 'object_type': object_type,
            'frame_interval_s': float(interval) if np.isfinite(interval) else None,
            'frames': frames}


def _write_lineage_sources(tracks_path, object_type, filenames, label_stack,
                           frame_interval_s=None):
    """Validate and atomically save one track-to-image provenance manifest."""
    result = _prepare_lineage_sources(
        tracks_path, object_type, filenames, label_stack, frame_interval_s)
    _atomic_json(str(tracks_path) + _SUFFIX, result)
    return result


def _unique_json_object(pairs):
    """Reject duplicate keys instead of silently choosing one JSON value."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('lineage source mapping contains duplicate JSON keys')
        result[key] = value
    return result


def _invalid_json_constant(value):
    """Reject nonstandard NaN and infinity tokens in saved provenance."""
    raise ValueError(f'lineage source mapping contains invalid JSON value {value}')


def _load_lineage_sources(tracks_path):
    """Read a verified mapping, never inferring time from sorted database rows."""
    tracks, digest = _tracks_snapshot(tracks_path)
    manifest = Path(str(tracks_path) + _SUFFIX)
    if not manifest.is_file():
        raise ValueError('lineage frame/source mapping is missing; rerun tracking with lineage enabled')
    with manifest.open('rb') as stream:
        data = stream.read(32 * 1024 * 1024 + 1)
    if len(data) > 32 * 1024 * 1024:
        raise ValueError('lineage source mapping exceeds 32 MiB')
    try:
        provenance = json.loads(data, object_pairs_hook=_unique_json_object,
                                parse_constant=_invalid_json_constant)
        if (not isinstance(provenance, dict) or type(provenance.get('version')) is not int
                or provenance['version'] != 1 or provenance.get('object_type') not in _ROLES
                or provenance.get('tracks_sha256') != digest):
            raise ValueError('lineage source mapping does not match its tracks CSV')
        interval = provenance.get('frame_interval_s')
        if interval is not None and (type(interval) not in (int, float)
                or not np.isfinite(interval) or interval <= 0):
            raise ValueError('lineage source mapping has invalid frame_interval_s')
        records = provenance['frames']
        if not isinstance(records, list):
            raise ValueError('lineage frames must be a list')
        expected = _source_fields([r['filename'] for r in records])
        rows = []
        for actual, base in zip(records, expected):
            if any(actual.get(key) != value for key, value in base.items()):
                raise ValueError('lineage source mapping contains inconsistent frame identities')
            if type(actual.get('frame')) is not int or not isinstance(actual.get('labels'), list):
                raise ValueError('lineage frame and labels have invalid types')
            for label in actual['labels']:
                if (not isinstance(label, dict) or any(type(label.get(k)) is not int
                        or label[k] <= 0 for k in ('track_id', 'object_label'))
                        or label['track_id'] != label['object_label']):
                    raise ValueError('lineage mapping has invalid final object labels')
                rows.append(dict(frame=actual['frame'], prcf=actual['prcf'], **label))
        mapping = pd.DataFrame(rows, columns=['frame', 'prcf', 'track_id', 'object_label'])
        if mapping.duplicated(['frame', 'track_id']).any():
            raise ValueError('lineage mapping has duplicate object identities')
        expected_ids = set(map(tuple, tracks[['frame', 'track_id']].to_numpy()))
        if set(map(tuple, mapping[['frame', 'track_id']].to_numpy())) != expected_ids:
            raise ValueError('lineage mapping does not cover the exact tracked objects')
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('malformed lineage source mapping') from exc
    provenance['mapping_sha256'] = hashlib.sha256(data).hexdigest()
    return tracks, mapping, provenance


def _measured_lineage_inputs(db_path, tracks_path, column):
    """Read a single measured feature with exact field, frame and label identities."""
    from .database_concurrency import connect
    from .tabular import _quote_identifier, _read_query

    tracks, mapping, provenance = _load_lineage_sources(tracks_path)
    table = provenance['object_type']
    if not column or column in ('prcf', 'object_label', 'frame', 'track_id'):
        raise ValueError('lineage colour must name a numeric measurement feature')
    quote = _quote_identifier
    with closing(connect(db_path, readonly=True)) as connection:
        columns = {row[1] for row in connection.execute(f'PRAGMA table_info({quote(table)})')}
        if not {'prcf', 'object_label', column} <= columns:
            raise ValueError(f'lineage measurement table {table} lacks identity or feature column {column!r}')
        pieces = []
        fields = sorted(mapping['prcf'].unique())
        for offset in range(0, len(fields), 500):
            wanted = fields[offset:offset + 500]
            query = (f'SELECT prcf, object_label, {quote(column)} FROM {quote(table)} '
                     f'WHERE prcf IN ({",".join("?" for _ in wanted)})')
            pieces.append(_read_query(connection, query, params=wanted,
                                      canonicalise=False, report=None))
    measured = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(
        columns=['prcf', 'object_label', column])
    labels = pd.to_numeric(measured['object_label'], errors='coerce')
    if (not np.isfinite(labels).all() or (labels <= 0).any() or (labels % 1 != 0).any()):
        raise ValueError('lineage measurement labels must be positive integers')
    measured['object_label'] = labels.astype(np.int64)
    if measured.duplicated(['prcf', 'object_label']).any():
        raise ValueError('lineage measurements have ambiguous duplicate object identities')
    values = pd.to_numeric(measured[column], errors='coerce')
    if ((measured[column].notna() & values.isna()).any()
            or np.isinf(values.to_numpy(dtype=float)).any()):
        raise ValueError(f'lineage measurement {column!r} must be numeric and finite or missing')
    measured[column] = values
    joined = mapping.merge(measured, on=['prcf', 'object_label'], how='left',
                           validate='one_to_one', indicator=True)
    matched = int((joined['_merge'] == 'both').sum())
    if not matched and len(mapping):
        raise ValueError('no measured objects match the saved lineage identities')
    result = joined[['frame', 'track_id', column]].copy()
    canonical = [[int(frame), int(track), None if pd.isna(value) else float(value).hex()]
                 for frame, track, value in result.sort_values(
                     ['frame', 'track_id']).itertuples(index=False, name=None)]
    content = json.dumps(canonical, separators=(',', ':'), allow_nan=False)
    report = {'tracks_sha256': provenance['tracks_sha256'],
              'mapping_sha256': provenance['mapping_sha256'],
              'measurements_db': str(Path(db_path).resolve()), 'object_type': table,
              'column': column, 'frame_interval_s': provenance.get('frame_interval_s'),
              'matched_rows': matched, 'unmatched_rows': len(joined) - matched,
              'missing_feature_rows': int(joined[column].isna().sum()),
              'measurement_values_sha256': hashlib.sha256(content.encode()).hexdigest()}
    return tracks, result, report


def _measured_lineage_options(settings, stored_interval):
    """Validate numeric run options and retain explicit saved timing calibration."""
    def number(key, value, positive=False):
        """Reject boolean, nonnumeric and unrepresentable numeric settings."""
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f'{key} must be a finite number, not a boolean')
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f'{key} must be a finite number') from exc
        if not np.isfinite(result) or (positive and result <= 0):
            raise ValueError(f'{key} must be finite' + (' and positive' if positive else ''))
        return result

    raw_distance = settings.get('timelapse_lineage_max_distance')
    if raw_distance is None or raw_distance == '':
        distance = 30.0
    else:
        distance = number('timelapse_lineage_max_distance', raw_distance) or 30.0
    interval = settings.get('frame_interval_s')
    if interval is None:
        interval = stored_interval
    else:
        interval = number('frame_interval_s', interval, positive=True)
        if stored_interval is not None and not np.isclose(
                interval, stored_interval, rtol=1e-9, atol=0):
            raise ValueError('Measure frame_interval_s conflicts with saved tracking calibration')
    return interval, distance


def _run_measured_lineage_step(db_path, settings):
    """Build measured-colour trees separately, reporting failures per tracks file."""
    from ._lineage_trees import _LINEAGE_SEGMENT_STATS, _lineage_trees_from_tracks

    directory = Path(db_path).resolve().parent.parent / 'tracks'
    roles = settings.get('timelapse_objects') or ['cell']
    if isinstance(roles, str):
        roles = [roles]
    if any(role not in _ROLES for role in roles):
        raise ValueError('lineage objects must be cell, nucleus or pathogen')
    column = str(settings.get('timelapse_lineage_color_by') or 'generation_time')
    results = []
    paths = sorted({(path, role) for role in roles
                    for path in directory.glob(f'*_tracks_{role}_*.csv')})
    if not paths:
        print('Measured lineage trees skipped: no tracks tables found')
    for path, expected_role in paths:
        try:
            if column in _LINEAGE_SEGMENT_STATS:
                tracks, _, provenance = _load_lineage_sources(path)
                measurements = None
                report = {'tracks_sha256': provenance['tracks_sha256'],
                          'mapping_sha256': provenance['mapping_sha256'], 'column': column,
                          'measurements_db': str(Path(db_path).resolve()),
                          'object_type': provenance['object_type'],
                          'frame_interval_s': provenance.get('frame_interval_s')}
            else:
                tracks, measurements, report = _measured_lineage_inputs(db_path, path, column)
            if report['object_type'] != expected_role:
                raise ValueError('lineage manifest object type differs from selected object')
            interval, distance = _measured_lineage_options(settings, report.get('frame_interval_s'))
            result = _lineage_trees_from_tracks(
                path, directory / 'lineage_measured', color_by=column,
                measurements=measurements, tracks_snapshot=tracks,
                max_distance=distance,
                frame_interval_s=interval,
                plot=bool(settings.get('save', True) or settings.get('plot', False)))
            report_path = directory / 'lineage_measured' / (path.stem + '_measurement_source.json')
            _atomic_json(report_path, report)
            result['measurement_report'] = report
            results.append(result)
            coverage = (f"; {report['matched_rows']} matched, {report['unmatched_rows']} unmatched, "
                        f"{report['missing_feature_rows']} missing feature values"
                        if 'matched_rows' in report else '')
            print(f'Measured lineage trees: {path.name}; colour {column}{coverage}; outputs {report_path.parent}')
        except (ValueError, TypeError, OverflowError, OSError, KeyError, sqlite3.Error) as exc:
            print(f'Measured lineage trees could not be built for {path}: {exc}')
    return results
