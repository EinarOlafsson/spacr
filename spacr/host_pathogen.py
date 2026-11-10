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
        ratios. Threshold channel keys must be integer indices or their JSON
        string form; duplicate aliases are rejected. A count column or parasite
        table is required for replication;
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
    thresholds = {}
    for key, value in config['hp_marker_thresholds'].items():
        if isinstance(key, bool) or not str(key).isdigit():
            raise ValueError('Marker threshold channel indices must be nonnegative integers')
        channel = int(key)
        if channel in thresholds:
            raise ValueError(f'Marker thresholds contain duplicate channel {channel}')
        try:
            thresholds[channel] = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError('Marker thresholds must be finite nonnegative ratios') from error
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


TITRATION_DEFAULTS = {
    'vacuole_from_parasites': False,
    'vacuole_gap_px': 4,
    'vacuole_min_size': 15,
    'host_min_size': 300,
    'host_assignment': 'majority_overlap',
    'titration_merged_folder': '',
    'titration_object_tables': '',
    'titration_plate_map': '',
    'titration_reference_values': '',
    'titration_min_well_cells': 0.3,
    'replication_classifier': 'none',
    'replication_training_table': '',
    'replication_training_columns': {'Num Pix': 'area_px', 'Circularity': 'circularity',
                                     'Convexity': 'solidity', 'Extend': 'extent',
                                     'Diameter': 'feret_max'},
    'replication_class_column': 'Parasites/vacuole',
    'replication_classes': '1,2,4,8',
    'replication_seed': 0,
    'replication_cv_folds': 5,
    'size_calibration': 'none',
    'size_calibration_reference_px': 57.0,
    'infection_signal_floor': 'none',
    'infection_floor_percentile': 95,
    'infection_uninfected_wells': '',
    'infection_signal_column': 'tox_mean',
}

#: Training-table columns that are lengths and so scale by sqrt(k) under a size calibration.
_LENGTH_COLUMNS = ('feret_max',)
#: Training-table columns that are areas and so scale by k.
_AREA_COLUMNS = ('area_px',)


def vacuoles_from_parasites(parasite_mask, gap_px=4):
    """Group touching or nearly touching parasites into vacuoles.

    The parasite foreground is closed with a 4-connected structuring element
    ``gap_px`` times (parasites up to about ``gap_px`` pixels apart join),
    every original parasite pixel is kept, and the result is labelled.
    Parasite-to-vacuole parentage is :func:`vacuole_links` (majority overlap).

    :param parasite_mask: integer parasite mask, zero background.
    :param gap_px: closing iterations.
    :returns: ``(vacuole_mask, links)``; ``links`` is None for an empty mask.
    """
    import numpy as np
    from scipy import ndimage as ndi

    parasite_mask = np.asarray(parasite_mask)
    foreground = parasite_mask > 0
    if not foreground.any():
        return np.zeros_like(parasite_mask, dtype=np.int32), None
    structure = ndi.generate_binary_structure(2, 1)
    closed = ndi.binary_closing(foreground, structure=structure, iterations=int(gap_px)) | foreground
    vacuoles, _ = ndi.label(closed)
    vacuoles = vacuoles.astype(np.int32)
    return vacuoles, vacuole_links(parasite_mask, vacuoles)


def vacuole_objects(cell_mask, parasite_mask, intensity=None, settings=None, well='', field=0):
    """Per-vacuole and per-cell tables of one field.

    Vacuoles come from :func:`vacuoles_from_parasites`; each one under
    ``vacuole_min_size`` pixels is dropped. Its host is the cell label most
    of its pixels lie on (``host_assignment='majority_overlap'``), 0 when it
    lies on no cell or the cell is under ``host_min_size`` pixels. Cells
    under ``host_min_size`` are dropped; a cell is infected when it hosts a
    vacuole.

    :param cell_mask: integer host-cell mask.
    :param parasite_mask: integer parasite mask of the same shape.
    :param intensity: the parasite-channel image for ``tox_mean``, or None.
    :param settings: :data:`TITRATION_DEFAULTS` keys.
    :param well: the well name written into both tables.
    :param field: the field number written into both tables.
    :returns: ``(cells, vacuoles)`` with the columns of the reference
        titration_objects script.
    :raises ValueError: for another ``host_assignment``.
    """
    import numpy as np
    import pandas as pd
    from skimage.measure import regionprops

    config = {**TITRATION_DEFAULTS, **(settings or {})}
    if config['host_assignment'] != 'majority_overlap':
        raise ValueError("host_assignment must be 'majority_overlap'")
    min_cell, min_vac = int(config['host_min_size']), int(config['vacuole_min_size'])
    cell_mask = np.asarray(cell_mask).astype(np.int32)
    parasite_mask = np.asarray(parasite_mask).astype(np.int32)
    image = (np.zeros(cell_mask.shape, np.float32) if intensity is None
             else np.asarray(intensity).astype(np.float32))
    vacuole_mask, links = vacuoles_from_parasites(parasite_mask, gap_px=config['vacuole_gap_px'])
    counts = (links.dropna().groupby('pathogen_id').size() if links is not None
              else pd.Series(dtype=int))
    cell_area = np.bincount(cell_mask.ravel())
    rows = []
    for region in regionprops(vacuole_mask, intensity_image=image):
        if region.area < min_vac:
            continue
        hosts = cell_mask[region.coords[:, 0], region.coords[:, 1]]
        hosts = hosts[hosts > 0]
        host = int(np.bincount(hosts).argmax()) if hosts.size else 0
        if host and cell_area[host] < min_cell:
            host = 0
        perimeter = region.perimeter if region.perimeter > 0 else np.nan
        rows.append(dict(well=well, field=field, vacuole=region.label, host_cell=host,
                         area_px=int(region.area), perimeter=perimeter,
                         circularity=(min(1.0, 4 * np.pi * region.area / perimeter ** 2)
                                      if perimeter == perimeter else np.nan),
                         solidity=region.solidity, extent=region.extent,
                         feret_max=region.feret_diameter_max,
                         tox_mean=region.intensity_mean,
                         n_parasites_cellpose=int(counts.get(region.label, 0))))
    vacuoles = pd.DataFrame(rows, columns=_VACUOLE_COLUMNS)
    ids = np.nonzero(cell_area)[0]
    ids = ids[(ids > 0) & (cell_area[ids] >= min_cell)]
    infected = set(vacuoles.loc[vacuoles['host_cell'] > 0, 'host_cell'])
    cells = pd.DataFrame(dict(well=well, field=field, cell=ids, area_px=cell_area[ids],
                              infected=[int(i) in infected for i in ids],
                              n_vacuoles=[int((vacuoles['host_cell'] == i).sum()) for i in ids]))
    return cells, vacuoles


_VACUOLE_COLUMNS = ['well', 'field', 'vacuole', 'host_cell', 'area_px', 'perimeter', 'circularity',
                    'solidity', 'extent', 'feret_max', 'tox_mean', 'n_parasites_cellpose']


def modal_area(areas, min_size=15, bins=60):
    """The modal object area on a log scale.

    Areas under ``min_size`` are left out; the log areas are binned into
    ``bins`` equal bins and the centre of the fullest bin is returned.

    :param areas: object areas in pixels.
    :param min_size: smallest area counted.
    :param bins: histogram bins.
    :returns: the modal area in pixels.
    """
    import numpy as np

    areas = np.asarray(areas, float)
    logs = np.log(areas[areas >= min_size])
    counts, edges = np.histogram(logs, bins=bins)
    i = int(np.argmax(counts))
    return float(np.exp((edges[i] + edges[i + 1]) / 2))


def size_calibration(vacuole_areas, settings=None):
    """The factor that puts spaCR vacuole sizes on an external reference scale.

    With ``size_calibration='modal_single'``,
    k = ``size_calibration_reference_px`` / :func:`modal_area` of the
    vacuole areas: the reference single-parasite area over spaCR's modal
    vacuole area. Areas are multiplied by k and lengths by sqrt(k);
    ``'none'`` gives k = 1.

    :param vacuole_areas: areas of the vacuoles of infected wells.
    :param settings: ``size_calibration``, ``size_calibration_reference_px``
        and ``vacuole_min_size``.
    :returns: dict with ``k``, ``reference_px`` and ``modal_px``.
    :raises ValueError: for an unknown ``size_calibration``.
    """
    config = {**TITRATION_DEFAULTS, **(settings or {})}
    mode = config['size_calibration']
    if mode == 'none':
        return {'k': 1.0, 'reference_px': None, 'modal_px': None}
    if mode != 'modal_single':
        raise ValueError("size_calibration must be 'none' or 'modal_single'")
    modal = modal_area(vacuole_areas, min_size=int(config['vacuole_min_size']))
    reference = float(config['size_calibration_reference_px'])
    return {'k': reference / modal, 'reference_px': reference, 'modal_px': modal}


def _training_columns(config):
    """The training-table-to-spaCR column mapping as a dict."""
    mapping = config['replication_training_columns']
    if isinstance(mapping, str):
        mapping = json.loads(mapping)
    return dict(mapping)


def _replication_model(seed):
    """The replication classifier: histogram gradient boosting."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_iter=200, random_state=int(seed))


def replication_classifier_cv(training, settings=None):
    """Cross-validate the replication classifier on its own training table.

    Reports the majority-class accuracy, the area quantiles (10%, 50%, 90%)
    per class, and the ``replication_cv_folds``-fold accuracy of the
    classifier on the area column alone and on every training column, each
    for the classes themselves and for 'more than one parasite'.

    :param training: the training table (e.g. HRMAn's Training_dataset.csv).
    :param settings: ``replication_training_columns``,
        ``replication_class_column``, ``replication_classes``,
        ``replication_seed`` and ``replication_cv_folds``.
    :returns: dict with ``majority``, ``quantiles`` (a DataFrame),
        ``scores`` (a DataFrame of feature set, target and accuracy) and
        ``text`` (the report as written).
    """
    import pandas as pd
    from sklearn.model_selection import cross_val_score

    config = {**TITRATION_DEFAULTS, **(settings or {})}
    columns = list(_training_columns(config))
    target_column = config['replication_class_column']
    training = _replication_rows(training, config)
    target = training[target_column]
    majority = float(target.value_counts(normalize=True).max())
    quantiles = training.groupby(target_column)[columns[0]].quantile([.1, .5, .9]).unstack()
    rows = []
    for features in ([columns[0]], columns):
        for name, y in ((f'{target.nunique()}class', target), ('binary>1', target > 1)):
            score = cross_val_score(_replication_model(config['replication_seed']),
                                    training[features], y, cv=int(config['replication_cv_folds']))
            rows.append({'features': features, 'target': name, 'accuracy': float(score.mean())})
    lines = [f'majority {majority:.3f}', quantiles.to_string()]
    lines += [f"{row['features']} {row['target']} {row['accuracy']:.3f}" for row in rows]
    return {'majority': majority, 'quantiles': quantiles, 'scores': pd.DataFrame(rows),
            'text': '\n'.join(lines) + '\n'}


def _replication_rows(training, config):
    """The training rows of the configured classes."""
    classes = [float(c) for c in str(config['replication_classes']).split(',') if c.strip()]
    target = config['replication_class_column']
    return training[training[target].isin(classes)] if classes else training


def classify_replication(vacuoles, training, settings=None, k=1.0):
    """Parasites per vacuole from a classifier trained on an external table.

    The classifier (``replication_classifier='hist_gbm'``) is trained on the
    training table's columns; each vacuole is described by the mapped spaCR
    columns, areas multiplied by ``k`` and lengths by sqrt(``k``), missing
    values filled with the column median.

    :param vacuoles: the vacuole table.
    :param training: the training table.
    :param settings: the replication settings.
    :param k: the size calibration factor (:func:`size_calibration`).
    :returns: the predicted parasites per vacuole, one per vacuole row.
    :raises ValueError: for ``replication_classifier`` other than
        ``'hist_gbm'``.
    """
    import numpy as np
    import pandas as pd

    config = {**TITRATION_DEFAULTS, **(settings or {})}
    if config['replication_classifier'] != 'hist_gbm':
        raise ValueError("replication_classifier must be 'hist_gbm' to classify")
    mapping = _training_columns(config)
    training = _replication_rows(training, config)
    model = _replication_model(config['replication_seed']).fit(
        training[list(mapping)], training[config['replication_class_column']])
    scale = {c: k for c in _AREA_COLUMNS}
    scale.update({c: np.sqrt(k) for c in _LENGTH_COLUMNS})
    features = pd.DataFrame({name: vacuoles[column] * scale.get(column, 1.0)
                             for name, column in mapping.items()})
    return model.predict(features.fillna(features.median()))


def infection_floor(vacuoles, settings=None):
    """The parasite-signal floor below which a vacuole is not counted.

    ``infection_signal_floor`` is ``'none'`` (no floor), a number, or
    ``'uninfected_percentile'``: the ``infection_floor_percentile``
    percentile of ``infection_signal_column`` over the vacuoles found in the
    ``infection_uninfected_wells`` (or the wells with status
    ``uninfected``).

    :param vacuoles: the vacuole table (with ``status`` when no wells are
        named).
    :param settings: the floor settings.
    :returns: the floor, or None for no floor.
    :raises ValueError: when the percentile has no uninfected vacuole to read.
    """
    import numpy as np

    config = {**TITRATION_DEFAULTS, **(settings or {})}
    floor = config['infection_signal_floor']
    if floor in (None, '', 'none'):
        return None
    if floor != 'uninfected_percentile':
        return float(floor)
    wells = [w.strip() for w in str(config['infection_uninfected_wells'] or '').split(',')
             if w.strip()]
    chosen = (vacuoles['well'].isin(wells) if wells
              else vacuoles.get('status', '') == 'uninfected')
    values = vacuoles.loc[chosen, config['infection_signal_column']].dropna()
    if not len(values):
        raise ValueError('infection_signal_floor=uninfected_percentile found no uninfected vacuole')
    return float(np.percentile(values, float(config['infection_floor_percentile'])))


def titration_wells(cells, vacuoles, settings=None):
    """Per-well infection, replication and vacuole size.

    Columns as the reference titration_summary: cells, infected_cells,
    vacuoles, vacuoles_unhosted, mean_vacuole_area_px, parasites per
    vacuole and % replicating (classifier and linked-parasite counts),
    pct_infected, vacuoles_per_100_cells and ``qc_pass`` -- False for a well
    with fewer than ``titration_min_well_cells`` times the median cell count
    of its condition.

    :param cells: per-cell table with ``well``, ``strain``, ``status``,
        ``dose``, ``cell`` and ``infected``.
    :param vacuoles: per-vacuole table with ``n_parasites_clf``,
        ``n_parasites_cellpose``, ``repl_clf`` and ``repl_cellpose``.
    :param settings: ``titration_min_well_cells``.
    :returns: the per-well table, sorted by strain, status, dose and well.
    """
    config = {**TITRATION_DEFAULTS, **(settings or {})}
    counted = cells.groupby(['well', 'strain', 'status', 'dose'], dropna=False).agg(
        cells=('cell', 'size'), infected_cells=('infected', 'sum')).reset_index()
    per_vacuole = vacuoles.groupby('well').agg(
        vacuoles=('vacuole', 'size'),
        vacuoles_unhosted=('host_cell', lambda s: int((s == 0).sum())),
        mean_vacuole_area_px=('area_px', 'mean'),
        parasites_per_vacuole_clf=('n_parasites_clf', 'mean'),
        parasites_per_vacuole_cellpose=('n_parasites_cellpose', 'mean'),
        pct_replicating_clf=('repl_clf', lambda s: 100 * s.mean()),
        pct_replicating_cellpose=('repl_cellpose', lambda s: 100 * s.mean())).reset_index()
    wells = counted.merge(per_vacuole, on='well', how='left')
    wells['vacuoles'] = wells['vacuoles'].fillna(0).astype(int)
    wells['pct_infected'] = 100 * wells['infected_cells'] / wells['cells']
    wells['vacuoles_per_100_cells'] = 100 * wells['vacuoles'] / wells['cells']
    median = wells.groupby(['strain', 'status', 'dose'], dropna=False)['cells'].transform('median')
    wells['qc_pass'] = wells['cells'] >= float(config['titration_min_well_cells']) * median
    return wells.sort_values(['strain', 'status', 'dose', 'well'])


TITRATION_METRICS = ('pct_infected', 'parasites_per_vacuole_clf', 'pct_replicating_clf',
                     'pct_replicating_cellpose', 'mean_vacuole_area_px', 'vacuoles_per_100_cells')


def titration_conditions(wells, reference=None):
    """Per-condition means and SDs of the wells that pass QC, beside reference values.

    :param wells: the table from :func:`titration_wells`.
    :param reference: optional reference values with ``strain``, ``dose``,
        ``metric`` (``repl`` or ``size``), ``value`` and ``err``; they become
        the ``HRMAn pct_replicating`` and ``HRMAn mean_size`` columns and
        their ``err(as stored)`` columns.
    :returns: one row per strain, status and dose, values rounded to 2
        decimals, with ``wells``, ``cells`` and ``vacuoles``.
    """
    import numpy as np

    kept = wells[wells['qc_pass'].astype(bool)]
    grouped = kept.groupby(['strain', 'status', 'dose'], dropna=False)
    table = grouped[list(TITRATION_METRICS)].agg(['mean', 'std']).round(2)
    table.columns = [f'{a} {b}' for a, b in table.columns]
    table['wells'] = grouped.size()
    table['cells'] = grouped['cells'].sum()
    table['vacuoles'] = grouped['vacuoles'].sum()
    table = table.reset_index()
    if reference is not None and len(reference):
        names = {'repl': 'HRMAn pct_replicating', 'size': 'HRMAn mean_size'}
        for metric in ('repl', 'size'):
            if (reference['metric'] == metric).any():
                table[names[metric]] = np.nan
                table[names[metric] + ' err(as stored)'] = np.nan
        for row in reference.itertuples(index=False):
            chosen = (table['strain'] == row.strain) & (table['dose'] == float(row.dose))
            column = names[row.metric]
            table.loc[chosen, column] = row.value
            table.loc[chosen, column + ' err(as stored)'] = row.err
    return table


def _plate_map(path):
    """The plate map (well, strain, status, dose) from a table spaCR reads."""
    from .tabular import read_table

    plate = read_table(path, canonicalise=False, report=None)
    missing = {'well', 'strain', 'status', 'dose'} - set(plate.columns)
    if missing:
        raise ValueError(f'titration_plate_map lacks columns {sorted(missing)}')
    return plate[['well', 'strain', 'status', 'dose']]


def _titration_objects(config):
    """The cell and vacuole tables of a titration run.

    From ``titration_object_tables`` (a folder with cells.csv and
    vacuoles.csv) or, with ``vacuole_from_parasites``, from the merged
    arrays in ``titration_merged_folder`` (channel 1 parasite signal, mask
    channels 3 cells and 5 parasites, as Make Masks writes them with
    merge_pathogens off).
    """
    import glob
    import os

    import numpy as np
    import pandas as pd

    from .tabular import read_table

    if config['titration_object_tables']:
        folder = config['titration_object_tables']
        return (read_table(os.path.join(folder, 'cells.csv'), canonicalise=False, report=None),
                read_table(os.path.join(folder, 'vacuoles.csv'), canonicalise=False, report=None))
    if not config['vacuole_from_parasites'] or not config['titration_merged_folder']:
        raise ValueError('Set titration_object_tables, or vacuole_from_parasites with '
                         'titration_merged_folder')
    cells, vacuoles = [], []
    for path in sorted(glob.glob(os.path.join(config['titration_merged_folder'], '*.npy'))):
        parts = os.path.basename(path)[:-4].split('_')
        stack = np.load(path)
        c, v = vacuole_objects(stack[..., 3], stack[..., 5], stack[..., 1], config,
                               well=parts[1], field=int(parts[2]))
        cells.append(c)
        if len(v):
            vacuoles.append(v)
    return (pd.concat(cells, ignore_index=True),
            pd.concat(vacuoles, ignore_index=True) if vacuoles
            else pd.DataFrame(columns=_VACUOLE_COLUMNS))


def analyze_titration(settings=None):
    """An HRMAn-style infection and replication readout across a titration.

    Vacuoles and cells come from :func:`vacuole_objects` (or saved tables).
    Every well gets its strain, status and dose from
    ``titration_plate_map``. Vacuoles under the parasite-signal floor
    (:func:`infection_floor`) are dropped and cells are called infected from
    the kept vacuoles. With ``replication_classifier='hist_gbm'`` the
    classifier is cross-validated on ``replication_training_table``
    (:func:`replication_classifier_cv`) and predicts parasites per vacuole
    after the size calibration (:func:`size_calibration`, on the kept
    vacuoles of infected wells). Writes titration_vacuoles.csv,
    titration_per_well.csv, titration_per_condition.csv,
    titration_calibration.txt and replication_classifier_cv.txt to
    ``<src>/results/titration/``.

    :param settings: ``src`` and the :data:`TITRATION_DEFAULTS` keys.
    :returns: dict with ``vacuoles``, ``wells``, ``conditions``,
        ``calibration``, ``floor`` and ``cv`` (None without a classifier).
    """
    import os

    import numpy as np

    from .tabular import read_table, write_table

    config = {**TITRATION_DEFAULTS, **(settings or {})}
    cells, vacuoles = _titration_objects(config)
    plate = _plate_map(config['titration_plate_map'])
    cells = cells.drop(columns=['strain', 'status', 'dose'], errors='ignore').merge(
        plate, on='well', how='left')
    vacuoles = vacuoles.drop(columns=['strain', 'status', 'dose'], errors='ignore').merge(
        plate, on='well', how='left')
    floor = infection_floor(vacuoles, config)
    if floor is not None:
        vacuoles = vacuoles[vacuoles[config['infection_signal_column']] >= floor].copy()
    kept = set(zip(vacuoles['well'], vacuoles['field'], vacuoles['host_cell']))
    cells['infected'] = [(w, f, c) in kept
                         for w, f, c in zip(cells['well'], cells['field'], cells['cell'])]
    infected_areas = vacuoles.loc[vacuoles['status'] == 'infected', 'area_px'].to_numpy()
    calibration = size_calibration(infected_areas, config)
    cv = None
    if config['replication_classifier'] != 'none':
        training = read_table(config['replication_training_table'], canonicalise=False,
                              report=None)
        cv = replication_classifier_cv(training, config)
        vacuoles['n_parasites_clf'] = classify_replication(vacuoles, training, config,
                                                           k=calibration['k'])
    else:
        vacuoles['n_parasites_clf'] = np.nan
    vacuoles['repl_clf'] = vacuoles['n_parasites_clf'] > 1
    vacuoles['repl_cellpose'] = vacuoles['n_parasites_cellpose'] > 1
    wells = titration_wells(cells, vacuoles, config)
    reference = (read_table(config['titration_reference_values'], canonicalise=False, report=None)
                 if config['titration_reference_values'] else None)
    conditions = titration_conditions(wells, reference)
    result = {'vacuoles': vacuoles, 'wells': wells, 'conditions': conditions,
              'calibration': calibration, 'floor': floor, 'cv': cv}
    if config.get('save', True):
        out = os.path.join(config['src'], 'results', 'titration')
        os.makedirs(out, exist_ok=True)
        write_table(vacuoles, os.path.join(out, 'titration_vacuoles.csv'), canonicalise=False)
        write_table(wells, os.path.join(out, 'titration_per_well.csv'), canonicalise=False)
        write_table(conditions, os.path.join(out, 'titration_per_condition.csv'),
                    canonicalise=False)
        with open(os.path.join(out, 'titration_calibration.txt'), 'w', encoding='utf-8') as fh:
            fh.write(f"size calibration: {config['size_calibration']}; k = {calibration['k']:.3f} "
                     f"(reference {calibration['reference_px']} px, spaCR modal "
                     f"{calibration['modal_px']} px)\nparasite-signal floor: {floor}\n")
        if cv is not None:
            with open(os.path.join(out, 'replication_classifier_cv.txt'), 'w',
                      encoding='utf-8') as fh:
                fh.write(cv['text'])
        result['folder'] = out
    return result


TRANSLOCATION_DEFAULTS = {
    'translocation': False,
    'translocation_merged_folder': '',
    'cell_from_nucleus_expand_px': 0,
    'cell_from_nucleus_min_nucleus_size': 250,
    'translocation_channel': 2,
    'translocation_control_channel': 0,
    'translocation_exclude_px': 2,
    'translocation_min_pathogen_size': 15,
    'translocation_background': 'outside_cells_median',
    'translocation_background_min_px': 1000,
    'translocation_background_fallback_percentile': 5,
    'translocation_min_nucleus_size': 250,
    'translocation_k_column': 'n_vacuoles',
    'translocation_kmax': 10,
    'translocation_groups': '',
    'translocation_min_cells_per_bin': 3,
    'translocation_null_percentile': 99,
    'translocation_shuffle_n': 200,
    'translocation_seed': 0,
    'translocation_reference': '',
}


def nucleus_territories(nucleus_mask, expand_px=40, min_nucleus_size=250):
    """Cell masks from nuclei when there is no cell stain.

    Nuclei under ``min_nucleus_size`` pixels are removed, then every nucleus
    grows by up to ``expand_px`` pixels without overlapping another
    (:func:`skimage.segmentation.expand_labels`, a distance-capped Voronoi
    territory). The territories are not cell outlines.

    :param nucleus_mask: integer nucleus mask.
    :param expand_px: the largest growth in pixels.
    :param min_nucleus_size: the smallest nucleus kept.
    :returns: ``(territories, nuclei)`` as int32 masks; a territory has its
        nucleus's label.
    """
    import numpy as np
    from skimage.segmentation import expand_labels

    nuclei = np.asarray(nucleus_mask).astype(np.int32)
    area = np.bincount(nuclei.ravel())
    small = np.nonzero((area > 0) & (area < int(min_nucleus_size)))[0]
    small = small[small > 0]
    if small.size:
        nuclei[np.isin(nuclei, small)] = 0
    return expand_labels(nuclei, distance=int(expand_px)).astype(np.int32), nuclei


def _majority_parent(labels_a, labels_b, n_a):
    """For each label 1..n_a of ``labels_a``, the ``labels_b`` label holding most of its pixels."""
    import numpy as np
    import pandas as pd

    both = (labels_a > 0) & (labels_b > 0)
    out = np.zeros(n_a + 1, dtype=np.int64)
    if not both.any():
        return out
    pairs = pd.DataFrame({'a': labels_a[both], 'b': labels_b[both]})
    best = (pairs.groupby(['a', 'b']).size().reset_index(name='n').sort_values('n')
            .drop_duplicates('a', keep='last'))
    out[best['a'].to_numpy()] = best['b'].to_numpy()
    return out


def translocation_cells(stack, settings=None, well='', field=0):
    """Per-host-cell nuclear effector signal and vacuole counts of one field.

    ``stack`` is a merged array ``[..., c]`` with the image channels and, at
    planes 3, 4 and 5, the cell (territory), nucleus and parasite masks.
    Parasites under ``translocation_min_pathogen_size`` pixels are dropped;
    vacuoles are :func:`vacuoles_from_parasites` with ``vacuole_gap_px``;
    parasites and vacuoles go to the cell holding most of their pixels.
    Nucleus pixels within ``translocation_exclude_px`` of a parasite pixel
    are excluded from the corrected readout; the field background is the
    median effector signal outside every cell and parasite (the
    ``translocation_background_fallback_percentile`` of the field when fewer
    than ``translocation_background_min_px`` such pixels).

    :param stack: the merged array.
    :param settings: :data:`TRANSLOCATION_DEFAULTS` and ``vacuole_gap_px``.
    :param well: the well name written into the table.
    :param field: the field number written into the table.
    :returns: one row per cell with a nucleus: nucleus_area, n_parasites,
        n_vacuoles, ha_total_nucleus (raw sum over the nucleus),
        ha_mean_nucleus, ha_total_nucleus_excl, ha_mean_nucleus_excl,
        frac_nucleus_excluded, ha_mean_cyto, cyto_area, dna_mean_nucleus and
        bg_ha (``ha`` is the ``translocation_channel``, ``dna`` the control
        channel).
    """
    import numpy as np
    import pandas as pd
    from scipy import ndimage as ndi

    config = {**TITRATION_DEFAULTS, **TRANSLOCATION_DEFAULTS, **(settings or {})}
    stack = np.asarray(stack)
    dna = stack[..., int(config['translocation_control_channel'])].astype(np.float64)
    ha = stack[..., int(config['translocation_channel'])].astype(np.float64)
    cell, nuc, par = (stack[..., k].astype(np.int64) for k in (3, 4, 5))
    parea = np.bincount(par.ravel())
    tiny = np.nonzero((parea > 0) & (parea < int(config['translocation_min_pathogen_size'])))[0]
    tiny = tiny[tiny > 0]
    if tiny.size:
        par[np.isin(par, tiny)] = 0
    foreground = par > 0
    exclude = int(config['translocation_exclude_px'])
    near = (ndi.binary_dilation(foreground, iterations=exclude)
            if foreground.any() and exclude > 0 else foreground)
    vacuoles, _ = vacuoles_from_parasites(par.astype(np.int32), gap_px=config['vacuole_gap_px'])
    vacuoles = vacuoles.astype(np.int64)
    n_cells = max(int(cell.max()), int(nuc.max()))
    par_host = _majority_parent(par, cell, int(par.max()))
    vac_host = _majority_parent(vacuoles, cell, int(vacuoles.max()))
    n_par = (np.bincount(par_host[1:][par_host[1:] > 0], minlength=n_cells + 1) if par.max()
             else np.zeros(n_cells + 1, int))
    n_vac = (np.bincount(vac_host[1:][vac_host[1:] > 0], minlength=n_cells + 1)
             if vacuoles.max() else np.zeros(n_cells + 1, int))
    idx = np.arange(1, n_cells + 1)
    ones = np.ones_like(ha)
    area = ndi.sum(ones, nuc, idx)
    keep = area > 0
    ha_total = ndi.sum(ha, nuc, idx)
    dna_mean = ndi.mean(dna, nuc, idx)
    nuc_kept = np.where(near, 0, nuc)
    area_kept = ndi.sum(ones, nuc_kept, idx)
    ha_kept = ndi.sum(ha, nuc_kept, idx)
    cyto = np.where((nuc > 0) | near, 0, cell)
    area_cyto = ndi.sum(ones, cyto, idx)
    ha_cyto = ndi.sum(ha, cyto, idx)
    if config['translocation_background'] == 'none':
        background = 0.0
    else:
        outside = (cell == 0) & ~near
        background = (float(np.median(ha[outside]))
                      if outside.sum() > int(config['translocation_background_min_px'])
                      else float(np.percentile(
                          ha, float(config['translocation_background_fallback_percentile']))))
    with np.errstate(invalid='ignore', divide='ignore'):
        table = pd.DataFrame(dict(
            well=well, field=field, cell=idx, nucleus_area=area,
            n_parasites=n_par[idx], n_vacuoles=n_vac[idx],
            ha_total_nucleus=ha_total, ha_mean_nucleus=ha_total / area,
            ha_total_nucleus_excl=ha_kept, ha_mean_nucleus_excl=ha_kept / area_kept,
            frac_nucleus_excluded=1 - area_kept / area,
            ha_mean_cyto=ha_cyto / area_cyto, cyto_area=area_cyto,
            dna_mean_nucleus=dna_mean, bg_ha=background))
    return table[keep]


def _translocation_groups(spec):
    """The well-row -> group mapping: a dict, a 'B:TgIST, C:TgIST' string or a plate-map table."""
    import os

    if isinstance(spec, dict):
        return {str(k): str(v) for k, v in spec.items()}
    spec = str(spec or '')
    if spec and os.path.isfile(spec):
        from .tabular import read_table

        plate = read_table(spec, canonicalise=False, report=None)
        return {str(w): str(g) for w, g in zip(plate['well'], plate['group'])}
    return {k.strip(): v.strip() for k, v in
            (part.split(':') for part in spec.split(',') if ':' in part)}


def _translocation_curve(cells, k_column, value, kmax):
    """Mean and count per group and k, and the ratio to k = 0 of the same group."""
    g = (cells[cells[k_column] <= kmax].groupby(['effector', k_column])[value]
         .agg(['mean', 'count']).reset_index())
    base = g[g[k_column] == 0].set_index('effector')['mean']
    g['ratio'] = g['mean'] / g['effector'].map(base)
    return g.rename(columns={k_column: 'k'})


def _translocation_per_well(cells, k_column, value, kmax, min_cells):
    """Per-well ratios of each k bin to the k = 0 cells of the same well."""
    import pandas as pd

    rows = []
    for (group, well), d in cells[cells[k_column] <= kmax].groupby(['effector', 'well']):
        m0 = d.loc[d[k_column] == 0, value].mean()
        for k, dk in d.groupby(k_column):
            if len(dk) >= min_cells and m0 > 0:
                rows.append(dict(effector=group, well=well, k=int(k),
                                 ratio=dk[value].mean() / m0, n=len(dk)))
    return pd.DataFrame(rows, columns=['effector', 'well', 'k', 'ratio', 'n'])


def _translocation_shuffle(cells, value, n, kmax, rng):
    """Ratio at each k after permuting k among the cells of each field, ``n`` times."""
    import numpy as np
    import pandas as pd

    rows = []
    for group, d in cells.groupby('effector'):
        d = d[['well', 'field', 'n_vacuoles', value]].copy()
        keys = d['well'] + '_' + d['field'].astype(str)
        idx = [np.flatnonzero(keys.to_numpy() == u) for u in keys.unique()]
        v = d[value].to_numpy()
        kk = d['n_vacuoles'].to_numpy()
        acc = []
        for _ in range(int(n)):
            perm = kk.copy()
            for ix in idx:
                perm[ix] = rng.permutation(kk[ix])
            s = pd.Series(v).groupby(perm).mean()
            acc.append((s / s.loc[0]).reindex(range(kmax + 1)).to_numpy())
        acc = np.array(acc)
        for k in range(kmax + 1):
            rows.append(dict(effector=group, k=k, null_ratio_mean=np.nanmean(acc[:, k]),
                             null_ratio_p2_5=np.nanpercentile(acc[:, k], 2.5),
                             null_ratio_p97_5=np.nanpercentile(acc[:, k], 97.5)))
    return pd.DataFrame(rows)


#: The curves of the translocation readout: name, k column, value column.
TRANSLOCATION_READOUTS = (
    ('hrman_equiv', 'n_vacuoles', 'ha_total_nucleus'),
    ('corrected', 'n_vacuoles', 'ha_corr_total'),
    ('corrected_mean', 'n_vacuoles', 'ha_corr_mean'),
    ('hrman_equiv_by_parasites', 'n_parasites', 'ha_total_nucleus'),
    ('dna_control', 'n_vacuoles', 'dna_mean_nucleus'),
    ('nucleus_area', 'n_vacuoles', 'nucleus_area'),
    ('cyto_ha', 'n_vacuoles', 'ha_mean_cyto'),
    ('frac_nucleus_excluded', 'n_vacuoles', 'frac_nucleus_excluded'))


def translocation_summary(cells, settings=None, reference=None):
    """Effector translocation against vacuoles per cell, with its controls.

    Every cell gets its group from ``translocation_groups`` (by well row, or
    by well from a plate map) and two readouts: raw (``ha_total_nucleus``)
    and corrected (``(ha_mean_nucleus_excl - bg_ha) x nucleus_area``). Cells
    with nuclei under ``translocation_min_nucleus_size`` are left out.
    Curves bin cells by k = vacuoles per cell up to ``translocation_kmax``
    and divide by the k = 0 cells of the same group; per-well ratios use
    wells with ``translocation_min_cells_per_bin`` cells in the bin. The
    controls are the k = 0 reference, the ``translocation_null_percentile``
    threshold of the corrected mean in k = 0 cells (% positive nuclei per k
    and per well), the within-field shuffle of k (``translocation_shuffle_n``
    times, seed ``translocation_seed``, group by group, raw then corrected),
    the DNA control, nucleus area and the excluded fraction.

    :param cells: :func:`translocation_cells` rows of every field.
    :param settings: the translocation settings.
    :param reference: an optional reference curve (effector, k,
        hrman_total, hrman_ratio) for the agreement table.
    :returns: dict of DataFrames ``cells``, ``counts``, ``curves``,
        ``per_well``, ``per_well_summary``, ``null_thresholds``,
        ``pct_positive``, ``pct_positive_per_well``, ``shuffle_null`` and
        ``agreement`` (empty without a reference).
    """
    import numpy as np
    import pandas as pd

    config = {**TRANSLOCATION_DEFAULTS, **(settings or {})}
    kmax = int(config['translocation_kmax'])
    groups = _translocation_groups(config['translocation_groups'])
    c = cells.copy()
    c['effector'] = c['well'].map(groups) if set(c['well']) & set(groups) else \
        c['well'].str[0].map(groups)
    c['ha_corr_total'] = (c['ha_mean_nucleus_excl'] - c['bg_ha']) * c['nucleus_area']
    c['ha_corr_mean'] = c['ha_mean_nucleus_excl'] - c['bg_ha']
    c = c[c['effector'].notna() & (c['nucleus_area'] >= float(config['translocation_min_nucleus_size']))]
    counts = c.groupby('effector').agg(
        cells=('cell', 'size'), wells=('well', 'nunique'),
        infected=('n_vacuoles', lambda s: (s > 0).mean() * 100),
        over_kmax=('n_vacuoles', lambda s: int((s > kmax).sum())))
    parts = []
    for name, k_column, value in TRANSLOCATION_READOUTS:
        k_column = config['translocation_k_column'] if k_column == 'n_vacuoles' else k_column
        g = _translocation_curve(c, k_column, value, kmax)
        g['readout'] = name
        parts.append(g)
    curves = pd.concat(parts, ignore_index=True)
    if reference is not None:
        curves = curves.merge(reference, on=['effector', 'k'], how='left')
    per_well = []
    for name, value in (('hrman_equiv', 'ha_total_nucleus'), ('corrected', 'ha_corr_total'),
                        ('dna_control', 'dna_mean_nucleus')):
        p = _translocation_per_well(c, config['translocation_k_column'], value, kmax,
                                    int(config['translocation_min_cells_per_bin']))
        p['readout'] = name
        per_well.append(p)
    per_well = pd.concat(per_well, ignore_index=True)
    per_well_summary = per_well.groupby(['readout', 'effector', 'k'])['ratio'].agg(
        well_mean='mean', well_sem=lambda s: s.std(ddof=1) / np.sqrt(len(s)),
        n_wells='count').reset_index()
    k0 = c[c['n_vacuoles'] == 0]
    threshold = k0.groupby('effector')['ha_corr_mean'].quantile(
        float(config['translocation_null_percentile']) / 100)
    c['ha_positive'] = c['ha_corr_mean'] > c['effector'].map(threshold)
    positive = (c[c['n_vacuoles'] <= kmax].groupby(['effector', 'n_vacuoles'])['ha_positive']
                .agg(pct_positive=lambda s: s.mean() * 100, n='count').reset_index()
                .rename(columns={'n_vacuoles': 'k'}))
    thresholds = pd.DataFrame({'effector': threshold.index,
                               'null_threshold_bgsub_mean_HA': threshold.values})
    per_well_positive = c.groupby(['effector', 'well']).apply(
        lambda d: pd.Series(dict(
            pct_pos_infected=d.loc[d['n_vacuoles'] > 0, 'ha_positive'].mean() * 100,
            pct_pos_uninfected=d.loc[d['n_vacuoles'] == 0, 'ha_positive'].mean() * 100,
            n_inf=int((d['n_vacuoles'] > 0).sum()), n_uninf=int((d['n_vacuoles'] == 0).sum())))
    ).reset_index()
    rng = np.random.default_rng(int(config['translocation_seed']))
    null = []
    for name, value in (('hrman_equiv', 'ha_total_nucleus'), ('corrected', 'ha_corr_total')):
        s = _translocation_shuffle(c, value, int(config['translocation_shuffle_n']), kmax, rng)
        s['readout'] = name
        null.append(s)
    agreement = []
    if reference is not None:
        for (readout, group), g in curves[curves['k'] >= 1].groupby(['readout', 'effector']):
            g = g.dropna(subset=['ratio', 'hrman_ratio'])
            if len(g) < 3:
                continue
            at_kmax = g['k'] == kmax
            agreement.append(dict(
                readout=readout, effector=group,
                pearson_r=np.corrcoef(g['ratio'], g['hrman_ratio'])[0, 1],
                spearman_rho=g['ratio'].rank().corr(g['hrman_ratio'].rank()),
                mean_abs_diff=(g['ratio'] - g['hrman_ratio']).abs().mean(),
                ratio_k10=g.loc[at_kmax, 'ratio'].squeeze() if at_kmax.any() else np.nan,
                hrman_k10=g.loc[at_kmax, 'hrman_ratio'].squeeze() if at_kmax.any() else np.nan))
    return {'cells': c, 'counts': counts, 'curves': curves, 'per_well': per_well,
            'per_well_summary': per_well_summary, 'null_thresholds': thresholds,
            'pct_positive': positive, 'pct_positive_per_well': per_well_positive,
            'shuffle_null': pd.concat(null), 'agreement': pd.DataFrame(agreement)}


_TRANSLOCATION_FILES = {
    'counts': 'counts_per_effector.csv', 'curves': 'curves_all_readouts.csv',
    'per_well': 'per_well_ratios.csv', 'per_well_summary': 'per_well_summary.csv',
    'null_thresholds': 'null_thresholds.csv', 'pct_positive': 'pct_ha_positive_nuclei.csv',
    'pct_positive_per_well': 'pct_ha_positive_per_well.csv', 'shuffle_null': 'shuffle_null.csv',
    'agreement': 'agreement_with_hrman.csv'}


def nuclear_translocation(settings=None):
    """The host-nucleus effector translocation readout of a plate, from settings.

    Reads the merged arrays of ``translocation_merged_folder`` (or
    ``<src>/merged``). With ``cell_from_nucleus_expand_px`` above 0 the cell
    plane is made from the nuclei first (:func:`nucleus_territories`,
    ``cell_source`` nucleus_territory; the territories are not cell
    outlines): a five-plane array ``[c0, c1, c2, nucleus, parasite]``
    becomes the six-plane layout. Then :func:`translocation_cells` per field
    and :func:`translocation_summary`, written to
    ``<src>/translocation/`` (cells_translocation.csv and the summary
    tables, plus reference_curve.csv when ``translocation_reference`` names
    a reference curve table).

    :param settings: ``src`` and the translocation settings.
    :returns: the :func:`translocation_summary` dict with ``folder``.
    """
    import glob
    import os

    import numpy as np
    import pandas as pd

    from .tabular import read_table, write_table

    config = {**TITRATION_DEFAULTS, **TRANSLOCATION_DEFAULTS, **(settings or {})}
    folder = config['translocation_merged_folder'] or os.path.join(config['src'], 'merged')
    tables = []
    for path in sorted(glob.glob(os.path.join(folder, '*.npy'))):
        stem = os.path.basename(path)[:-4]
        stem = stem[:-len('_None')] if stem.endswith('_None') else stem
        parts = stem.split('_')
        stack = np.load(path)
        if int(config['cell_from_nucleus_expand_px']) > 0:
            territory, nuclei = nucleus_territories(
                stack[..., 3], config['cell_from_nucleus_expand_px'],
                config['cell_from_nucleus_min_nucleus_size'])
            stack = np.stack([stack[..., 0], stack[..., 1], stack[..., 2],
                              territory.astype(stack.dtype), nuclei.astype(stack.dtype),
                              stack[..., 4]], axis=-1)
        tables.append(translocation_cells(stack, config, well=parts[-2], field=int(parts[-1])))
    cells = pd.concat(tables, ignore_index=True)
    if int(config['cell_from_nucleus_expand_px']) > 0:
        cells['cell_source'] = 'nucleus_territory'
    reference = (read_table(config['translocation_reference'], canonicalise=False, report=None)
                 if config['translocation_reference'] else None)
    result = translocation_summary(cells, config, reference)
    out = os.path.join(config['src'], 'translocation')
    os.makedirs(out, exist_ok=True)
    write_table(cells, os.path.join(out, 'cells_translocation.csv'), canonicalise=False)
    for key, name in _TRANSLOCATION_FILES.items():
        write_table(result[key], os.path.join(out, name), canonicalise=False,
                    index=key == 'counts')
    if reference is not None:
        write_table(reference, os.path.join(out, 'reference_curve.csv'), canonicalise=False)
    result['folder'] = out
    return result


DEFAULTS.update(TITRATION_DEFAULTS)
DEFAULTS.update(TRANSLOCATION_DEFAULTS)


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
        With ``titration_plate_map`` set, the run is the titration readout of
        :func:`analyze_titration` instead, and returns its dict; with
        ``translocation`` it is :func:`nuclear_translocation`.
    """
    import pandas as pd
    from .io import _read_db
    from .cancellation import checkpoint

    config = default_settings(settings)
    if config.get('translocation'):
        return nuclear_translocation(config)
    if config.get('titration_plate_map'):
        return analyze_titration(config)
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
