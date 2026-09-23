"""Read-only, bounded field previews using the Host–Pathogen analysis engine."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path
import sqlite3


def _quote(name):
    """Quote a nonempty SQLite identifier without interpreting SQL."""
    if not isinstance(name, str) or not name:
        raise ValueError('Choose a nonempty measurement table name')
    return '"' + name.replace('"', '""') + '"'


def _columns(connection, table):
    """Map canonical identity names to the database's stored column names."""
    import pandas as pd
    from .utils import correct_metadata_column_names

    names = [row[1] for row in connection.execute(f'PRAGMA table_info({_quote(table)})')]
    if not names:
        raise ValueError(f'Measurement table not found: {table}')
    canonical = correct_metadata_column_names(pd.DataFrame(columns=names)).columns
    return dict(zip(canonical, names))


def preview_fields(settings, *, limit=50):
    """List at most ``limit`` measured fields, in stable source/identity order.

    :param settings: Host–Pathogen settings; src is a project, database or list.
    :param limit: positive maximum number of fields offered by the preview.
    :returns: field records and a boolean saying whether additional fields exist.
        Database connections are read-only; no migration or analysis is written.
    """
    from .utils import _time_column
    from .host_pathogen import default_settings

    if not isinstance(limit, int) or limit < 1:
        raise ValueError('The preview field limit must be positive')
    config = default_settings(settings)
    sources = config.get('src')
    sources = sources if isinstance(sources, list) else [sources]
    fields, seen = [], set()
    for source in sources:
        path = Path(str(source or '')).expanduser().resolve()
        database = path if path.is_file() else path / 'measurements' / 'measurements.db'
        if not database.is_file():
            raise FileNotFoundError(f'Measurements database not found: {database}')
        if database in seen:
            raise ValueError('The same project database was provided more than once')
        seen.add(database)
        with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as connection:
            columns = _columns(connection, 'cell')
            keys = ['plateID', 'rowID', 'columnID', 'fieldID']
            time = _time_column(columns)
            if time:
                keys.append(time)
            if not set(keys) <= columns.keys():
                raise ValueError('Host measurements lack plate/well/field identities')
            projections = []
            for table in dict.fromkeys(['cell', config['hp_vacuole_table']]):
                columns = _columns(connection, table)
                if not set(keys) <= columns.keys():
                    raise ValueError(f'{table} lacks plate/well/field/time identities')
                projection = ', '.join(f'{_quote(columns[key])} AS {_quote(key)}' for key in keys)
                projections.append(f'SELECT {projection} FROM {_quote(table)}')
            query = ' UNION '.join(projections)
            order = ', '.join(_quote(key) for key in keys)
            rows = connection.execute(
                f'{query} ORDER BY {order} LIMIT ?',
                (limit + 1 - len(fields),)).fetchall()
            for row in rows:
                fields.append(dict(database=str(database), identity=dict(zip(keys, row))))
            if len(fields) > limit:
                return fields[:limit], True
    return fields, False


def preview_field(settings, field, *, planes=None, image_channel=0, row_limit=100000):
    """Analyze one measured field without modifying source or result files.

    :param settings: current Host–Pathogen settings, shared with the full run.
    :param field: one record returned by :func:`preview_fields`.
    :param planes: optional explicit host/vacuole/parasite mask plane indices.
        Missing entries use a valid merged-plane manifest; absent metadata never
        silently chooses mask planes. A negative index disables that overlay.
    :param image_channel: intensity plane shown as grayscale, display-only.
    :param row_limit: refuse fields exceeding this many rows in any input table.
    :returns: real analysis tables, display image/masks and explicit image notes.
        Infection denominators cover this field only, including uninfected hosts.
    """
    from .host_pathogen import default_settings, summarize_tables
    from .tabular import _read_query
    from .utils import correct_metadata

    config = default_settings(settings)
    if not isinstance(row_limit, int) or row_limit < 1:
        raise ValueError('The preview object limit must be positive')
    database = Path(field['database'])
    tables = ['cell', config['hp_vacuole_table'], config['hp_reference_table']]
    if config['hp_parasite_table']:
        tables.append(config['hp_parasite_table'])
    frames = []
    with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as connection:
        connection.execute('BEGIN')
        for table in tables:
            columns = _columns(connection, table)
            identity = field['identity']
            if not set(identity) <= columns.keys():
                raise ValueError(f'{table} lacks field/time identities needed for this preview')
            where = ' AND '.join(f'{_quote(columns[key])} IS ?' for key in identity)
            frame = _read_query(connection,
                f'SELECT * FROM {_quote(table)} WHERE {where} LIMIT ?',
                params=[*identity.values(), row_limit + 1])
            if len(frame) > row_limit:
                raise ValueError(f'{table} exceeds the preview limit of {row_limit} objects in one field')
            frames.append(correct_metadata(frame))
    result = summarize_tables(*frames, settings=config, source=str(database))
    try:
        display = _field_image(database, field['identity'], config, planes or {}, image_channel)
    except (OSError, ValueError, TypeError) as exc:
        display = dict(image=None, masks={}, planes={}, path='', image_note=f'Image unavailable: {exc}')
    return dict(field=field, results=result, **display)


def _field_image(database, identity, settings, planes, image_channel):
    """Resolve a recorded field to a merged stack and read only display planes."""
    import numpy as np
    from . import schema
    from .crops import read_merged_plane_layout
    from .utils import _time_column

    root = database.parent.parent if database.parent.name == 'measurements' else database.parent
    time_key = _time_column(identity)
    field = schema.FieldID.build(identity['plateID'], row=identity['rowID'],
        column=identity['columnID'], field=identity['fieldID'],
        time=identity[time_key] if time_key else None)
    plate = schema.escape_filename_component(field.plateID)
    base = f'{plate}_{field.well}_{field.fieldID.removeprefix("f")}'
    time = field.timeID.removeprefix('t') if field.timeID else '1'
    names = [f'{base}_{time}.npy']
    if not time_key:
        names.append(f'{base}.npy')
    if field.plateID == 'synthetic_hp' and (root / 'example_manifest.json').is_file():
        names.append(f'synthetic_hp_{field.well}_{field.fieldID.removeprefix("f")}_{time}.npy')
    paths = [root / 'merged' / name for name in names if (root / 'merged' / name).is_file()]
    empty = dict(image=None, masks={}, planes={}, path='', image_note='No matching merged image; measurements remain available.')
    if len(paths) != 1:
        if paths:
            empty['image_note'] = 'Several merged images match this field; no image was selected automatically.'
        return empty
    path = paths[0]
    try:
        data = np.load(path, mmap_mode='r', allow_pickle=False)
        if data.ndim != 3 or not np.issubdtype(data.dtype, np.number):
            raise ValueError('Merged image must have numeric H × W × planes data')
        if not 0 <= image_channel < data.shape[-1]:
            raise ValueError('The selected image channel is outside this merged stack')
        layout = read_merged_plane_layout(str(path))
        roles = {'host': 'cell', 'vacuole': settings['hp_vacuole_table'],
                 'parasite': settings['hp_parasite_table']}
        chosen = {name: (layout or {}).get('mask_dims', {}).get(role, -1) for name, role in roles.items()}
        chosen.update(planes)
        if layout and image_channel >= len(layout['intensity_channels']):
            raise ValueError('Choose an intensity channel, not a recorded mask plane')
        stride = max(1, int(np.ceil(max(data.shape[:2]) / 1024)))
        values = np.asarray(data[::stride, ::stride, image_channel], dtype=float)
        finite = values[np.isfinite(values)]
        lo, hi = np.percentile(finite, [1, 99.8]) if finite.size else (0, 0)
        image = np.uint8(np.clip(np.nan_to_num((values - lo) / max(hi - lo, 1e-12)), 0, 1) * 255)
        masks = {}
        for role, plane in chosen.items():
            if plane < 0:
                continue
            if plane >= data.shape[-1] or (layout and plane < len(layout['intensity_channels'])):
                raise ValueError(f'{role} mask plane is not a label plane in this stack')
            mask = np.asarray(data[::stride, ::stride, plane])
            if not np.isfinite(mask).all() or (mask < 0).any() or (mask != np.floor(mask)).any():
                raise ValueError(f'{role} mask plane must contain nonnegative integer labels')
            masks[role] = np.array(mask, copy=True)
        note = 'Display contrast only; ratios use stored measurements. Preview covers one field.'
        if not layout:
            note += ' No plane manifest: choose mask planes explicitly to show outlines.'
        return dict(image=image, masks=masks, planes=chosen, path=str(path), image_note=note)
    except (OSError, ValueError) as exc:
        return {**empty, 'path': str(path), 'image_note': f'Image unavailable: {exc}'}
