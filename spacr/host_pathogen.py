"""Vacuole recruitment and replication with explicit host-cell denominators."""
from __future__ import annotations

import json
import math
from pathlib import Path

DEFAULTS = {
    'src': 'path',
    'hp_vacuole_table': 'pathogen',
    'hp_vacuole_prefix': 'pathogen',
    'hp_reference_table': 'cytoplasm',
    'hp_reference_prefix': 'cytoplasm',
    'hp_marker_channels': [0],
    'hp_marker_thresholds': {},
    'hp_parasite_table': '',
    'hp_parasite_parent': 'pathogen_id',
    'hp_count_column': '',
    'save': True,
}


def default_settings(settings=None):
    """Apply Host–Pathogen defaults; replication requires explicit count inputs.

    :param settings: optional overrides, copied without modifying the caller.
    :returns: settings with independent mutable defaults.
    """
    return {**{key: value.copy() if isinstance(value, (dict, list)) else value
               for key, value in DEFAULTS.items()}, **(settings or {})}


def vacuole_links(child_mask, vacuole_mask):
    """Link each child label only when one vacuole covers most of its pixels.

    :param child_mask: integer parasite/object mask, zero background.
    :param vacuole_mask: co-registered integer whole-vacuole mask.
    :returns: label, pathogen_id and pathogen_overlap_fraction columns.
        A parent must cover strictly more than half the child; ties, outside
        objects and ambiguous overlaps retain missing parent identities.
        This geometric link does not assign biological meaning to an object.
    """
    import numpy as np
    import pandas as pd
    if child_mask.shape != vacuole_mask.shape:
        raise ValueError('Child and vacuole masks must have matching dimensions')
    rows = []
    for label in np.unique(child_mask):
        if label <= 0:
            continue
        covered = vacuole_mask[child_mask == label]
        parents, counts = np.unique(covered[covered > 0], return_counts=True)
        fraction = float(counts.max() / len(covered)) if counts.size else 0.
        parent = int(parents[counts.argmax()]) if fraction > .5 else np.nan
        rows.append((label, parent, fraction))
    return pd.DataFrame(rows, columns=['label', 'pathogen_id', 'pathogen_overlap_fraction'])


def _prepare(frame, source, name, time_column):
    """Normalize field identities without collapsing separate acquisitions."""
    from .utils import _time_column
    frame = frame.copy()
    required = ['plateID', 'rowID', 'columnID', 'fieldID', 'object_label']
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f'{name} is missing identity columns: {sorted(missing)}')
    for key in required:
        if frame[key].isna().any():
            raise ValueError(f'{name} has missing {key} identities')
    actual_time = _time_column(frame.columns)
    if time_column and actual_time is None and len(frame):
        raise ValueError(f'{name} lacks time identities present in other tables')
    frame['timepoint'] = frame[actual_time].astype(str) if actual_time else ''
    if actual_time and frame[actual_time].isna().any():
        raise ValueError(f'{name} has missing time identities')
    frame['source_id'] = str(source)
    for key in ('plateID', 'rowID', 'columnID', 'fieldID'):
        frame[key] = frame[key].astype(str)
    from pandas import to_numeric
    labels = to_numeric(frame['object_label'], errors='raise')
    if ((labels <= 0) | (labels % 1 != 0)).any():
        raise ValueError(f'{name} object labels must be positive integers')
    frame['object_label'] = labels.astype('int64')
    keys = ['source_id', 'plateID', 'rowID', 'columnID', 'fieldID', 'timepoint']
    if frame.duplicated(keys + ['object_label']).any():
        raise ValueError(f'{name} has duplicate objects within a field/timepoint')
    return frame, keys


def summarize_tables(cells, vacuoles, reference, parasites=None, *, settings=None, source='experiment'):
    """Join per-vacuole signals to host references and retain every host cell.

    :param cells: one row per host, including uninfected cells.
    :param vacuoles: one row per segmented vacuole, with object_label/cell_id.
    :param reference: one row per host reference compartment, object_label
        identifying its host, with channel mean-intensity columns.
    :param parasites: optional independently segmented parasites with an
        explicit parent-vacuole column; never infer parentage from cell ID.
    :param settings: hp_* options. Marker thresholds apply to vacuole/reference
        ratios. A count column or parasite table is required for replication;
        otherwise counts remain missing. These two inputs are mutually exclusive.
    :param source: experiment/acquisition identity, preserved in every output.
    :returns: dict of vacuoles, cells, wells, marker_states and orphan_parasites
        DataFrames. Missing references and invalid denominators stay unknown;
        extracellular or unlinked vacuoles do not inflate host infection rates.
    :raises ValueError: ambiguous identities, missing columns or invalid policy.
    """
    import numpy as np
    import pandas as pd
    from .utils import _time_column

    config = default_settings(settings)
    channels = config['hp_marker_channels']
    if not isinstance(channels, (list, tuple)) or not channels or any(
            isinstance(c, bool) or not isinstance(c, int) or c < 0 for c in channels):
        raise ValueError('hp_marker_channels must list nonnegative channel indices')
    channels = list(dict.fromkeys(channels))
    if not isinstance(config['hp_marker_thresholds'], dict):
        raise ValueError('Marker thresholds must be a dictionary mapping channel indices to ratios')
    thresholds = {int(key): float(value) for key, value in config['hp_marker_thresholds'].items()}
    if any(key not in channels or not math.isfinite(value) or value < 0
           for key, value in thresholds.items()):
        raise ValueError('Marker thresholds must be finite nonnegative ratios for selected channels')
    frames = [cells, vacuoles, reference] + ([] if parasites is None else [parasites])
    timed = any(_time_column(frame.columns) for frame in frames)
    cells, keys = _prepare(cells, source, 'cell table', timed)
    vacuoles, _ = _prepare(vacuoles, source, 'vacuole table', timed)
    reference, _ = _prepare(reference, source, 'reference table', timed)
    if 'cell_id' not in vacuoles:
        raise ValueError('Vacuoles need explicit cell_id host links')
    vacuoles['cell_id'] = pd.to_numeric(vacuoles['cell_id'], errors='raise').astype('Int64')
    vacuoles = vacuoles.rename(columns={'object_label': 'vacuole_id'})
    cells = cells.rename(columns={'object_label': 'host_id'})
    host_keys = keys + ['host_id']
    linked = vacuoles.merge(cells[host_keys].assign(host_present=True),
                            left_on=keys + ['cell_id'], right_on=host_keys,
                            how='left', validate='many_to_one')
    linked['host_present'] = linked['host_present'].eq(True)
    ref_columns = [f"{config['hp_reference_prefix']}_channel_{channel}_mean_intensity" for channel in channels]
    missing = set(ref_columns) - set(reference.columns)
    if missing:
        raise ValueError(f'Reference table is missing marker intensities: {sorted(missing)}')
    refs = reference[keys + ['object_label'] + ref_columns].rename(
        columns={'object_label': 'reference_host_id', **{name: 'reference_' + name for name in ref_columns}})
    linked = linked.merge(refs, left_on=keys + ['cell_id'], right_on=keys + ['reference_host_id'],
                          how='left', validate='many_to_one')
    state_columns = []
    for channel, reference_column in zip(channels, ref_columns):
        numerator = f"{config['hp_vacuole_prefix']}_channel_{channel}_mean_intensity"
        if numerator not in linked:
            raise ValueError(f'Vacuole table is missing {numerator}')
        values = pd.to_numeric(linked[numerator], errors='raise')
        denominator = pd.to_numeric(linked['reference_' + reference_column], errors='raise')
        valid = np.isfinite(values) & np.isfinite(denominator) & (denominator > 0) & linked['host_present']
        ratio = f'channel_{channel}_recruitment_ratio'
        state = f'channel_{channel}_state'
        linked[ratio] = (values / denominator.where(valid)).where(valid)
        linked[state] = 'unknown'
        if channel in thresholds:
            known = valid & linked[ratio].notna()
            linked.loc[known, state] = np.where(linked.loc[known, ratio] >= thresholds[channel], 'positive', 'negative')
        state_columns.append(state)
    linked['joint_marker_state'] = linked[state_columns].apply(
        lambda row: '|'.join(f'{channel}:{state}' for channel, state in zip(channels, row)), axis=1)
    vacuole_keys = keys + ['vacuole_id']
    count_column = config['hp_count_column']
    if parasites is not None and count_column:
        raise ValueError('Choose a parasite table OR hp_count_column, not both')
    orphans = pd.DataFrame(columns=keys + ['object_label', 'vacuole_id', 'orphan_reason'])
    if parasites is not None:
        parasites, _ = _prepare(parasites, source, 'parasite table', timed)
        parent = config['hp_parasite_parent']
        if parent not in parasites:
            raise ValueError(f'Parasite table needs explicit parent-vacuole column {parent!r}')
        parasites['vacuole_id'] = pd.to_numeric(parasites[parent], errors='raise').astype('Int64')
        joined = parasites.merge(linked[vacuole_keys + ['cell_id']].rename(columns={'cell_id': 'vacuole_host_id'}),
                                  on=vacuole_keys, how='left', validate='many_to_one', indicator=True)
        valid = joined['_merge'].eq('both')
        joined['orphan_reason'] = np.where(valid, '', 'unmatched_vacuole')
        if 'cell_id' in joined:
            host = pd.to_numeric(joined['cell_id'], errors='raise')
            mismatch = host.notna() & joined['vacuole_host_id'].notna() & host.ne(joined['vacuole_host_id'])
            joined.loc[mismatch & valid, 'orphan_reason'] = 'host_identity_mismatch'
            valid &= ~mismatch
        orphans = joined.loc[~valid].copy()
        counts = joined.loc[valid].groupby(vacuole_keys).size().rename('parasite_count').reset_index()
        linked = linked.merge(counts, on=vacuole_keys, how='left', validate='one_to_one')
        linked['parasite_count'] = linked['parasite_count'].fillna(0).astype('Int64')
        linked['replication_method'] = 'linked_parasite_count'
    elif count_column:
        if count_column not in linked:
            raise ValueError(f'Missing parasite count column {count_column!r}')
        counts = pd.to_numeric(linked[count_column], errors='raise')
        if ((counts.dropna() < 0) | (counts.dropna() % 1 != 0)).any():
            raise ValueError('Parasite counts must be nonnegative integers or missing')
        linked['parasite_count'] = counts.astype('Int64')
        linked['replication_method'] = 'provided_count'
    else:
        linked['parasite_count'] = pd.Series(pd.NA, index=linked.index, dtype='Int64')
        linked['replication_method'] = 'not_measured'
    infected = linked.loc[linked['host_present']].groupby(keys + ['cell_id']).size().rename('vacuole_count').reset_index()
    cells = cells.merge(infected, left_on=host_keys, right_on=keys + ['cell_id'], how='left', validate='one_to_one')
    cells['vacuole_count'] = cells['vacuole_count'].fillna(0).astype(int)
    cells['infected'] = cells['vacuole_count'].gt(0)
    cells['multiply_infected'] = cells['vacuole_count'].gt(1)
    well_keys = keys[:4] + ['timepoint']
    wells = cells.groupby(well_keys, dropna=False).agg(
        host_cells=('host_id', 'size'), infected_cells=('infected', 'sum'),
        multiply_infected_cells=('multiply_infected', 'sum')).reset_index()
    wells['infection_fraction'] = wells['infected_cells'] / wells['host_cells']
    vac_summary = linked.groupby(well_keys, dropna=False).agg(
        vacuoles=('vacuole_id', 'size'), linked_vacuoles=('host_present', 'sum'),
        vacuoles_with_counts=('parasite_count', 'count')).reset_index()
    wells = wells.merge(vac_summary, on=well_keys, how='outer', validate='one_to_one')
    for name in ('host_cells', 'infected_cells', 'multiply_infected_cells', 'vacuoles', 'linked_vacuoles', 'vacuoles_with_counts'):
        wells[name] = wells[name].fillna(0).astype(int)
    states = linked.groupby(well_keys + ['joint_marker_state'], dropna=False).size().rename('vacuoles').reset_index()
    states['all_vacuoles'] = states.groupby(well_keys)['vacuoles'].transform('sum')
    states['fraction_of_all_vacuoles'] = states['vacuoles'] / states['all_vacuoles']
    distribution = linked.dropna(subset=['parasite_count']).groupby(
        well_keys + ['parasite_count'], dropna=False).size().rename('vacuoles').reset_index()
    distribution['vacuoles_with_counts'] = distribution.groupby(well_keys)['vacuoles'].transform('sum')
    distribution['fraction_of_counted_vacuoles'] = distribution['vacuoles'] / distribution['vacuoles_with_counts']
    wells['host_denominator'] = 'retained measured host cells; include uninfected cells in Measure'
    return dict(vacuoles=linked, cells=cells, wells=wells, marker_states=states,
                replication_distribution=distribution, orphan_parasites=orphans)


def analyze_host_pathogen(settings=None):
    """Analyze measured projects and save a unified, source-separated report.

    :param settings: hp_* settings from :func:`default_settings`; src accepts
        a project, measurements.db, or a list of either. Each canonical DB
        path is a distinct source identity, even when plate/field IDs repeat.
    :returns: combined DataFrames from :func:`summarize_tables` and settings.
        CSV files and the exact settings JSON are written to the first
        project's results/host_pathogen directory when save is true.
        Host infection fractions describe retained measured cells. Run Measure
        with uninfected=True to include uninfected hosts in that denominator;
        previously discarded hosts cannot be reconstructed from these tables.
    """
    import pandas as pd
    from .io import _read_db
    from .cancellation import checkpoint

    config = default_settings(settings)
    sources = config['src'] if isinstance(config['src'], list) else [config['src']]
    databases = []
    for source in sources:
        path = Path(source).expanduser().resolve()
        database = path if path.is_file() else path / 'measurements' / 'measurements.db'
        if not database.is_file():
            raise FileNotFoundError(f'Measurements database not found: {database}')
        if database in databases:
            raise ValueError('The same project database was provided more than once')
        databases.append(database)
    if not databases:
        raise ValueError('Choose at least one measured project')
    tables = ['cell', config['hp_vacuole_table'], config['hp_reference_table']]
    if config['hp_parasite_table']:
        tables.append(config['hp_parasite_table'])
    outputs = []
    for database in databases:
        checkpoint()
        frames = _read_db(database, tables)
        outputs.append(summarize_tables(*frames, settings=config, source=str(database)))
    result = {key: pd.concat([out[key] for out in outputs], ignore_index=True) for key in outputs[0]}
    if config['save']:
        from .tabular import write_table

        root = databases[0].parent.parent if databases[0].parent.name == 'measurements' else databases[0].parent
        directory = root / 'results' / 'host_pathogen'
        directory.mkdir(parents=True, exist_ok=True)
        for name, frame in result.items():
            write_table(frame, directory / f'{name}.csv')
        (directory / 'settings.json').write_text(json.dumps(config, indent=2, default=str) + '\n', encoding='utf-8')
        print(f'Host–Pathogen Analysis saved to {directory}')
    return dict(result, settings=config)
