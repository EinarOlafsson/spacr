"""Run a whole screen plate from one settings file: import, masks, measure, export, aggregate.

A plate of an image-based screen (BBBC021, BBBC022, JUMP CPJUMP1) goes
through six phases, and :func:`run_screen` runs them in order from one
settings dict, so the settings file saved with the plate reproduces it:

* ``import``: :func:`import_plate` links (or copies) the raw images into
  the plate folder under spaCR's ``cellvoyager`` names and records every
  link in ``<src>/settings/import_map.csv``;
* ``prep`` and ``masks``: :func:`spacr.core.preprocess_generate_masks`,
  in ONE call with ``preprocess=True, masks=True`` when both phases run,
  so ``gen_mask_settings.csv`` holds the whole mask run;
* ``measure``: :func:`spacr.measure.measure_crop`;
* ``export``: :func:`export_plate_parquet` turns ``measurements.db`` into
  one per-cell Parquet file per plate and writes each field's cell counts
  to the ``image`` table;
* ``aggregate``: :func:`aggregate_wells` writes the well means, medians,
  per-well heterogeneity and the control cells' percentiles.

With ``record_timing`` each phase's wall and CPU seconds go to
``<src>/settings/screen_timing.json``.

The mask phases take the mask settings from the file and the measure phase
the measure settings; a key both pipelines read (``channels``, ``n_jobs``)
goes to both, except ``normalize``, which means something different in
each and so is left to the measure defaults. ``mask_overrides`` and
``measure_overrides`` set a key for one phase alone, such as::

    run_screen({'src': 'work/Week10_40111',
                'import_source': 'images/Week10_40111',
                'channels': [0, 1, 2], 'nucleus_channel': 0, 'cell_channel': 1,
                'magnification': 20,
                'mask_overrides': {'illumination_correction': False},
                'measure_overrides': {'illumination_correction': True,
                                      'cell_min_size': 177,
                                      'nucleus_min_size': 177},
                'aggregate_metadata': 'BBBC021_v1_image.csv',
                'aggregate_metadata_columns': 'Image_Metadata_Plate_DAPI,'
                                              'Image_Metadata_Well_DAPI,'
                                              'Image_Metadata_Compound'})

``spacr-run screen --settings plate.csv`` does the same from a file.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Key the settings of this module are registered under.
APP_KEY = 'screen'

#: The phases :func:`run_screen` knows, in the order it runs them.
PHASES = ('import', 'prep', 'masks', 'measure', 'export', 'aggregate')

#: The import namings :func:`import_plate` writes.
IMPORT_NAMINGS = ('cellvoyager',)

#: How :func:`import_plate` places a file.
IMPORT_MODES = ('symlink', 'copy')

#: The well tables :func:`aggregate_wells` can write.
AGGREGATE_LEVELS = ('well_mean', 'well_median')

#: The object tables :func:`export_plate_parquet` reads, and the prefixes of
#: the feature columns :func:`aggregate_wells` aggregates.
FEATURE_PREFIXES = ('cell_', 'nucleus_', 'cytoplasm_', 'pathogen_')

#: Default pattern of a raw file name: well, site and channel, as BBBC021's
#: ``<plate>_<well>_s<site>_w<channel><GUID>.tif``.
DEFAULT_IMPORT_REGEX = r'(?P<wellID>[A-P]\d{2})_s(?P<fieldID>\d+)_w(?P<chanID>\d)'

_KEYS_MEASURED_DIFFERENTLY = frozenset({'normalize'})

_META_COLUMNS = ('plateID', 'rowID', 'columnID', 'fieldID', 'timeID', 'prc', 'prcf',
                 'prcfo', 'file_name', 'path_name')

_POSITION = re.compile(r'(centroid(_weighted)?-[01]$|bbox-[0-3]$|distance_to_field_edge$)')

_ONE_CHUNK = 2 ** 31 - 1

_BOOKKEEPING = re.compile(r'(measurement_ndim|n_z|voxel_size)')

_DEFAULTS: Dict[str, Any] = {
    'screen_phases': 'import,prep,masks,measure,export,aggregate',
    'import_source': '',
    'import_naming': 'cellvoyager',
    'import_regex': DEFAULT_IMPORT_REGEX,
    'import_mode': 'symlink',
    'import_map': '',
    'import_wells': '',
    'mask_overrides': {},
    'measure_overrides': {},
    'export_parquet': True,
    'export_parquet_dir': '',
    'aggregate_levels': 'well_mean,well_median',
    'aggregate_metadata': '',
    'aggregate_metadata_columns': 'plate,well,compound',
    'aggregate_control': 'DMSO',
    'aggregate_control_percentiles': '1,99',
    'aggregate_heterogeneity': False,
    'aggregate_dir': '',
    'record_timing': True,
}

_EXPECTED_TYPES: Dict[str, Any] = {key: type(value) for key, value in _DEFAULTS.items()}

_TOOLTIPS: Dict[str, str] = {
    'screen_phases': "(str) - Comma-separated phases to run, in this order: import, prep, masks, measure, export, aggregate. prep and masks together are one mask run with preprocess and masks both on. Default all six.",
    'import_source': "(str) - Folder of the plate's raw images. The import phase links or copies them into src under spaCR's cellvoyager names. Blank skips the import. Default blank.",
    'import_naming': "(str) - File naming the import writes. 'cellvoyager': <plate>_<well>_T0001F<site>L01A01Z01C<channel>.tif, the names the mask pipeline reads by default. Default 'cellvoyager'.",
    'import_regex': "(str) - Regular expression found in each raw file name, with named groups wellID, fieldID and chanID, and optionally plateID and timeID. Without plateID the plate is the source folder's name. Default BBBC021's <well>_s<site>_w<channel>.",
    'import_mode': "(str) - 'symlink' links each raw file (nothing is copied); 'copy' copies it. Default 'symlink'.",
    'import_map': "(str) - Optional table with source and target columns that names each imported file itself, instead of the regex. Default blank.",
    'import_wells': "(str) - Comma-separated wells to import, such as B02,B03; blank imports every well. Default blank.",
    'mask_overrides': "(dict) - Settings given to the mask phases alone, such as {'illumination_correction': False}, for a key the mask and measure pipelines both read. Default {}.",
    'measure_overrides': "(dict) - Settings given to the measure phase alone, such as {'illumination_correction': True, 'cell_min_size': 177}. Default {}.",
    'export_parquet': "(bool) - Write one Parquet file per plate with one row per cell (cell, cytoplasm and largest-nucleus features) and each field's cell counts in the image table. Default True.",
    'export_parquet_dir': "(str) - Folder of the per-plate Parquet files. Blank writes to <src>/parquet. Default blank.",
    'aggregate_levels': "(str) - Well tables to write: well_mean (wells_mean.parquet) and well_median (wells_median.parquet). Default 'well_mean,well_median'.",
    'aggregate_metadata': "(str) - Table naming each well's treatment, such as BBBC021_v1_image.csv, used to find the control wells. Blank writes no control percentiles. Default blank.",
    'aggregate_metadata_columns': "(str) - The metadata table's plate, well and treatment columns, comma-separated, such as Image_Metadata_Plate_DAPI,Image_Metadata_Well_DAPI,Image_Metadata_Compound. Default 'plate,well,compound'.",
    'aggregate_control': "(str) - Treatment of the control wells, such as DMSO. Their cells give the per-plate percentiles that Ljosa scaling divides by. Default 'DMSO'.",
    'aggregate_control_percentiles': "(str) - The two percentiles of the control cells to write per plate and feature, comma-separated. Default '1,99'.",
    'aggregate_heterogeneity': "(bool) - Also write wells_heterogeneity.parquet: each feature's per-well standard deviation and 10-90% spread. Default False.",
    'aggregate_dir': "(str) - Folder of the well tables. Blank writes to <src>/tables. Default blank.",
    'record_timing': "(bool) - Write each phase's wall-clock and CPU seconds to <src>/settings/screen_timing.json. Default True.",
}

_CATEGORIES: Dict[str, List[str]] = {
    'Screen staging': ['screen_phases', 'import_source', 'import_naming', 'import_regex',
                       'import_mode', 'import_map', 'import_wells', 'mask_overrides',
                       'measure_overrides', 'record_timing'],
    'Screen export and aggregate': ['export_parquet', 'export_parquet_dir', 'aggregate_levels',
                                    'aggregate_metadata', 'aggregate_metadata_columns',
                                    'aggregate_control', 'aggregate_control_percentiles',
                                    'aggregate_heterogeneity', 'aggregate_dir'],
}

_DESCRIPTION = (
    "Run a screen plate from one settings file: import the raw images, make "
    "the masks in one run, measure, export one per-cell Parquet per plate "
    "and write the well means, medians, heterogeneity and control percentiles."
)


def _mask_defaults(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """The mask pipeline's defaults for these settings."""
    from .settings import set_default_settings_preprocess_generate_masks

    return set_default_settings_preprocess_generate_masks(
        {'src': settings.get('src') or 'src'})


def _measure_defaults(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """The measure pipeline's defaults for these settings."""
    from .settings import get_measure_crop_settings

    return get_measure_crop_settings({'src': settings.get('src') or 'src'})


def screen_defaults(settings: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The settings of :func:`run_screen`, filled in with their defaults.

    Holds the mask pipeline's keys, the measure pipeline's keys and this
    module's own; a key the two pipelines share takes the mask default.

    :param settings: settings given so far; a key already present is kept.
    :returns: a new dict with every key :func:`run_screen` reads.
    """
    given = dict(settings or {})
    merged = dict(_measure_defaults(given))
    merged.update(_mask_defaults(given))
    merged.update({key: (dict(value) if isinstance(value, dict) else value)
                   for key, value in _DEFAULTS.items()})
    merged.update(given)
    return merged


def _register() -> None:
    """Register this module's settings through the defaults seam, once per process."""
    from .settings import has_registered_defaults, register_defaults

    if has_registered_defaults(APP_KEY):
        return
    register_defaults(APP_KEY, screen_defaults, expected_types=_EXPECTED_TYPES,
                      tooltips=_TOOLTIPS, categories=_CATEGORIES,
                      description=_DESCRIPTION)


_register()


def _names(value: Any) -> List[str]:
    """A list of names from a comma-separated string or a sequence."""
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(',') if part.strip()]
    return [str(part).strip() for part in value if str(part).strip()]


def _cellvoyager_name(plate: str, well: str, field: int, channel: int, time_point: int = 1) -> str:
    """The cellvoyager name of one raw image."""
    return f'{plate}_{well}_T{time_point:04d}F{field:03d}L01A01Z01C{channel:02d}.tif'


def _place(source: str, target: str, mode: str) -> bool:
    """Link or copy one file unless the target, or spaCR's moved original, is there."""
    folder, name = os.path.split(target)
    if os.path.lexists(target) or os.path.lexists(os.path.join(folder, 'orig', name)):
        return False
    if mode == 'symlink':
        os.symlink(source, target)
    else:
        shutil.copy2(source, target)
    return True


def import_plate(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Link or copy a plate's raw images into ``src`` under cellvoyager names.

    Each ``.tif``/``.tiff`` file of ``import_source`` whose name matches
    ``import_regex`` becomes ``<plate>_<well>_T<time>F<site>L01A01Z01C<channel>.tif``
    in ``src``; with ``import_map`` the table's ``source`` and ``target``
    columns name the files instead. A target already there, or already moved
    to ``src/orig`` by the mask pipeline, is left as it is, so an import can
    be repeated. The mapping is written to ``<src>/settings/import_map.csv``
    with ``target`` and ``source`` (the raw file's full path).

    :param settings: ``src``, ``import_source``, ``import_naming``,
        ``import_regex``, ``import_mode``, ``import_map`` and
        ``import_wells``; missing keys take their defaults.
    :returns: ``{"map", "path", "files", "fields", "placed"}``: the mapping
        frame, the CSV written, the files mapped, the distinct
        (well, site) fields and the files newly placed.
    :raises ValueError: without ``src`` or ``import_source``, for an unknown
        naming or mode, a regex without the needed groups, or two raw files
        mapped to one name.
    """
    from .tabular import read_table, write_table

    settings = {**_DEFAULTS, **dict(settings)}
    src, source = str(settings.get('src') or ''), str(settings.get('import_source') or '')
    if not src or not source:
        raise ValueError('import needs src (the plate folder) and import_source (the raw images)')
    naming = str(settings['import_naming']).strip().lower()
    if naming not in IMPORT_NAMINGS:
        raise ValueError(f'import_naming must be one of {IMPORT_NAMINGS}, not {naming!r}')
    mode = str(settings['import_mode']).strip().lower()
    if mode not in IMPORT_MODES:
        raise ValueError(f'import_mode must be one of {IMPORT_MODES}, not {mode!r}')
    wells = set(_names(settings.get('import_wells')))
    rows: List[Tuple[str, str]] = []
    fields: Set[Tuple[str, int]] = set()
    if settings.get('import_map'):
        table = read_table(settings['import_map'], canonicalise=False, report=None)
        missing = {'source', 'target'} - set(table.columns)
        if missing:
            raise ValueError(f'import_map lacks the column(s) {sorted(missing)}')
        for raw, target in zip(table['source'].astype(str), table['target'].astype(str)):
            path = raw if os.path.isabs(raw) else os.path.join(source, raw)
            rows.append((target, path))
    else:
        pattern = re.compile(str(settings['import_regex']))
        needed = {'wellID', 'fieldID', 'chanID'} - set(pattern.groupindex)
        if needed:
            raise ValueError(f'import_regex lacks the named group(s) {sorted(needed)}')
        default_plate = os.path.basename(os.path.normpath(source))
        for name in sorted(os.listdir(source)):
            if not name.lower().endswith(('.tif', '.tiff')):
                continue
            match = pattern.search(name)
            if not match:
                continue
            groups = match.groupdict()
            well = groups['wellID']
            if wells and well not in wells:
                continue
            plate = groups.get('plateID') or default_plate
            field = int(groups['fieldID'])
            time_point = int(groups['timeID']) if groups.get('timeID') else 1
            target = _cellvoyager_name(plate, well, field, int(groups['chanID']), time_point)
            rows.append((target, os.path.join(source, name)))
            fields.add((well, field))
    frame = pd.DataFrame(rows, columns=['target', 'source'])
    doubled = frame['target'][frame['target'].duplicated()]
    if len(doubled):
        raise ValueError(f'{len(doubled)} raw files map to one name, such as {doubled.iloc[0]}')
    os.makedirs(src, exist_ok=True)
    placed = sum(_place(path, os.path.join(src, target), mode)
                 for target, path in rows)
    path = write_table(frame, os.path.join(src, 'settings', 'import_map.csv'), canonicalise=False)
    logger.info('import: %d files, %d fields, %d placed', len(frame), len(fields), placed)
    return {'map': frame, 'path': path, 'files': int(len(frame)),
            'fields': int(len(fields)), 'placed': int(placed)}


def _numeric(frame: pd.DataFrame, keep: Sequence[str]) -> pd.DataFrame:
    """The kept key columns beside every numeric feature, as float32."""
    drop = [column for column in (*_META_COLUMNS, 'object_label', 'cell_id')
            if column in frame.columns]
    values = frame.drop(columns=drop).select_dtypes(include=[np.number]).astype('float32')
    return pd.concat([frame[list(keep)].reset_index(drop=True),
                      values.reset_index(drop=True)], axis=1)


def _field_counts(out: pd.DataFrame, cells: pd.DataFrame) -> List[Dict[str, Any]]:
    """Each field's exported cells, cells without a nucleus and cells with several."""
    names = cells[['prcf', 'file_name']].drop_duplicates('prcf').set_index('prcf')['file_name']
    nuclei = out['n_nuclei'] if 'n_nuclei' in out.columns else pd.Series(1, index=out.index)
    counts = pd.DataFrame({'prcf': out['prcf'], 'none': nuclei == 0, 'several': nuclei > 1})
    grouped = counts.groupby('prcf', sort=True)
    rows = []
    for prcf, part in grouped:
        rows.append({'file_name': str(names.get(prcf, prcf)), 'n_cells': int(len(part)),
                     'cells_without_nucleus': int(part['none'].sum()),
                     'cells_multi_nucleus': int(part['several'].sum())})
    return rows


def export_plate_parquet(src: Any, dst: Any = None, *, plate: Optional[str] = None,
                         write_image_counts: bool = True) -> Dict[str, Any]:
    """Turn a plate's ``measurements.db`` into one Parquet file with one row per cell.

    Cell and cytoplasm rows share ``(prcf, object_label)``; nucleus rows
    name their cell in ``cell_id``. A cell with several nuclei gets the
    features of its largest nucleus and ``Metadata_n_nuclei``; nuclei
    without a cell are not exported. The keys are ``plate``, ``well``
    (such as B02), ``site`` and ``Metadata_cell_id``; positions
    (centroids, bounding boxes, distance to the field edge) are kept as
    ``Location_*`` columns so profile scorers skip them, and bookkeeping
    columns are dropped, as are the ``pathogen`` distance features a plate
    without a pathogen table carries. Features are ``float32``.

    With ``write_image_counts`` each field's ``n_cells``,
    ``cells_without_nucleus`` and ``cells_multi_nucleus`` are written to the
    ``image`` table of ``measurements.db`` (item 814).

    :param src: the plate folder holding ``measurements/measurements.db``.
    :param dst: the Parquet folder; ``None`` writes to ``<src>/parquet``.
    :param plate: the plate name for the file and the ``plate`` column;
        ``None`` takes the database's one plate.
    :param write_image_counts: write the per-field counts to ``image``.
    :returns: the export record: ``path``, ``n_cells``, ``n_cells_db``,
        ``n_features``, ``n_fields``, ``n_wells``, ``tables`` and, with a
        nucleus table, ``n_nuclei_db``, ``nuclei_without_cell``,
        ``cells_without_nucleus`` and ``cells_multi_nucleus``.
    :raises ValueError: when the database is missing, holds no cell table,
        repeats a cell, or holds more than one plate.
    """
    from .tabular import _write_parquet, database_tables, read_database

    db = os.path.join(os.fspath(src), 'measurements', 'measurements.db')
    if not os.path.exists(db):
        raise ValueError(f'no measurements database at {db}')
    tables = set(database_tables(db))
    if 'cell' not in tables:
        raise ValueError(f'{db} holds no cell table')
    wanted = [name for name in ('cell', 'cytoplasm', 'nucleus') if name in tables]
    frames = dict(zip(wanted, read_database(db, wanted, canonicalise=False, report=None,
                                            migrate=False, read_only=True,
                                            chunksize=_ONE_CHUNK)))
    cell = frames['cell']
    if cell.duplicated(['prcf', 'object_label']).any():
        raise ValueError(f'{db}: a cell appears twice')
    out = _numeric(cell, ['plateID', 'rowID', 'columnID', 'fieldID', 'prcf', 'object_label'])
    info: Dict[str, Any] = {'n_cells_db': int(len(cell))}
    if 'cytoplasm' in frames:
        out = out.merge(_numeric(frames['cytoplasm'], ['prcf', 'object_label']),
                        on=['prcf', 'object_label'], how='left', validate='one_to_one')
    if 'nucleus' in frames:
        nucleus = frames['nucleus']
        linked = nucleus[nucleus['cell_id'].notna()].copy()
        linked['cell_id'] = linked['cell_id'].astype(int)
        counts = linked.groupby(['prcf', 'cell_id']).size().rename('n_nuclei').reset_index()
        largest = linked.sort_values('nucleus_area', ascending=False).drop_duplicates(['prcf', 'cell_id'])
        largest = _numeric(largest, ['prcf', 'cell_id']).merge(counts, on=['prcf', 'cell_id'])
        largest = largest.rename(columns={'cell_id': 'object_label'})
        out = out.merge(largest, on=['prcf', 'object_label'], how='left', validate='one_to_one')
        out['n_nuclei'] = out['n_nuclei'].fillna(0).astype('int16')
        info.update(n_nuclei_db=int(len(nucleus)),
                    nuclei_without_cell=int(nucleus['cell_id'].isna().sum()),
                    cells_without_nucleus=int((out['n_nuclei'] == 0).sum()),
                    cells_multi_nucleus=int((out['n_nuclei'] > 1).sum()))
    plates = out['plateID'].unique()
    if len(plates) != 1:
        raise ValueError(f'{db}: expected one plate, found {list(plates)}')
    name = str(plate) if plate else str(plates[0])
    row = out['rowID'].astype(str).str.lstrip('r').astype(int)
    column = out['columnID'].astype(str).str.lstrip('c').astype(int)
    keys = pd.DataFrame({'plate': name,
                         'well': [f'{chr(64 + r)}{c:02d}' for r, c in zip(row, column)],
                         'site': out['fieldID'].astype(str).str.lstrip('f').astype('int16'),
                         'Metadata_cell_id': out['object_label'].astype('int32')})
    features = out.drop(columns=['plateID', 'rowID', 'columnID', 'fieldID', 'prcf', 'object_label'])
    no_pathogen = 'pathogen' not in tables
    features = features.drop(columns=[
        column for column in features.columns
        if _BOOKKEEPING.match(column) or column == 'nucleus_cell_id'
        or (no_pathogen and 'pathogen' in column)])
    positions = [column for column in features.columns if _POSITION.search(column)]
    features = features.rename(columns={column: f'Location_{column}' for column in positions})
    if 'n_nuclei' in features.columns:
        features = features.rename(columns={'n_nuclei': 'Metadata_n_nuclei'})
    result = pd.concat([keys, features], axis=1)
    folder = Path(dst) if dst else Path(os.fspath(src)) / 'parquet'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{name}.parquet'
    info.update(path=str(path), n_cells=int(len(result)),
                n_features=sum(1 for column in features.columns
                               if not column.startswith(('Location_', 'Metadata_'))),
                n_fields=int(result[['well', 'site']].drop_duplicates().shape[0]),
                n_wells=int(result['well'].nunique()), tables=sorted(tables))
    _write_parquet(result, path, canonicalise=False,
                   metadata={key: value for key, value in info.items() if key != 'path'})
    if write_image_counts:
        from .measure import _write_image_rows

        _write_image_rows(db, _field_counts(out, cell))
    return info


def control_wells(metadata: Any, *, columns: Any = ('plate', 'well', 'compound'),
                  control: str = 'DMSO') -> Set[Tuple[str, str]]:
    """The ``(plate, well)`` pairs of a metadata table whose treatment is the control.

    :param metadata: a frame or table path with one row per image or well.
    :param columns: the plate, well and treatment columns, as a sequence or
        a comma-separated string.
    :param control: the control's treatment value, such as ``'DMSO'``.
    :returns: the control wells.
    :raises ValueError: for other than three columns, or a column the table
        lacks.
    """
    from .tabular import read_table

    names = _names(columns)
    if len(names) != 3:
        raise ValueError(f'name three metadata columns (plate, well, treatment), not {names}')
    table = metadata if isinstance(metadata, pd.DataFrame) else read_table(
        metadata, canonicalise=False, report=None)
    missing = [name for name in names if name not in table.columns]
    if missing:
        raise ValueError(f'the metadata table lacks {missing}')
    plate, well, treatment = names
    chosen = table.loc[table[treatment].astype(str) == str(control), [plate, well]]
    return set(map(tuple, chosen.astype(str).values))


def _feature_columns(frame: pd.DataFrame) -> List[str]:
    """The per-cell feature columns of an exported plate."""
    return [column for column in frame.columns if column.startswith(FEATURE_PREFIXES)]


def _percentile_pair(value: Any) -> List[float]:
    """The control percentiles as quantiles, from '1,99' or a pair."""
    parts = [float(part) for part in _names(value)] if not isinstance(value, (list, tuple)) \
        else [float(part) for part in value]
    if not parts or any(not 0.0 <= part <= 100.0 for part in parts):
        raise ValueError(f'control percentiles must lie in 0-100, not {value!r}')
    return [part / 100.0 for part in parts]


def aggregate_wells(parquet_files: Iterable[Any], dst: Any, *,
                    levels: Any = AGGREGATE_LEVELS,
                    controls: Optional[Set[Tuple[str, str]]] = None,
                    control_name: str = 'DMSO', percentiles: Any = (1, 99),
                    heterogeneity: bool = False) -> Dict[str, Any]:
    """Aggregate per-cell plate files into well tables.

    Each plate file is aggregated on its own (a well lies on one plate) by
    ``plate`` and ``well`` over its ``cell_``, ``nucleus_``, ``cytoplasm_``
    and ``pathogen_`` columns:

    * ``wells_mean.parquet`` / ``wells_median.parquet``: the mean or median
      of every feature, with ``n_cells``;
    * ``wells_heterogeneity.parquet``: each feature's standard deviation
      (``<feature>__sd``) and 10-90% spread (``<feature>__p90mp10``), with
      ``n_cells``;
    * ``<control>_percentiles.parquet`` (such as
      ``dmso_percentiles.parquet``) when ``controls`` are given: per plate
      and quantile ``q``, the percentile of every feature over the control
      wells' cells, the reference of Ljosa et al. 2013's scaling
      (:func:`spacr.sp_stats._normalize_profiles`, ``ljosa_percentile``).

    :param parquet_files: the per-plate files of :func:`export_plate_parquet`.
    :param dst: the output folder; created if absent.
    :param levels: ``'well_mean'`` and/or ``'well_median'``.
    :param controls: the control ``(plate, well)`` pairs (see
        :func:`control_wells`); ``None`` writes no percentiles.
    :param control_name: names the percentile file.
    :param percentiles: the percentiles to write, such as ``(1, 99)``.
    :param heterogeneity: also write the heterogeneity table.
    :returns: ``{"paths", "plates", "n_cells", "n_features"}``.
    :raises ValueError: for an unknown level, no input files, or files
        without feature columns.
    """
    from .tabular import read_table, write_table

    chosen = [level.strip().lower() for level in _names(levels)]
    unknown = [level for level in chosen if level not in AGGREGATE_LEVELS]
    if unknown:
        raise ValueError(f'aggregate levels must be among {AGGREGATE_LEVELS}, not {unknown}')
    files = sorted(os.fspath(path) for path in parquet_files)
    if not files:
        raise ValueError('no plate files to aggregate')
    quantiles = _percentile_pair(percentiles)
    means, medians, spreads, limits = [], [], [], []
    n_cells, n_features = 0, 0
    for path in files:
        frame = read_table(path, canonicalise=False, report=None)
        features = _feature_columns(frame)
        if not features:
            raise ValueError(f'{path} has no feature columns')
        grouped = frame.groupby(['plate', 'well'])[features]
        size = grouped.size()
        if 'well_mean' in chosen:
            mean = grouped.mean().copy()
            mean['n_cells'] = size
            means.append(mean.reset_index())
        if 'well_median' in chosen:
            median = grouped.median().copy()
            median['n_cells'] = size
            medians.append(median.reset_index())
        if heterogeneity:
            spread = pd.concat([grouped.std().add_suffix('__sd'),
                                (grouped.quantile(0.9) - grouped.quantile(0.1)).add_suffix('__p90mp10')],
                               axis=1)
            spread['n_cells'] = size
            spreads.append(spread.reset_index())
        if controls is not None:
            inside = np.fromiter(((str(p), str(w)) in controls
                                  for p, w in zip(frame['plate'], frame['well'])),
                                 dtype=bool, count=len(frame))
            limit = frame.loc[inside, features].quantile(quantiles)
            limit.insert(0, 'q', limit.index.values)
            limit.insert(0, 'plate', frame['plate'].iat[0])
            limits.append(limit.reset_index(drop=True))
        n_cells += len(frame)
        n_features = max(n_features, len(features))
        logger.info('aggregate: %s, %d cells, %d features', os.path.basename(path),
                    len(frame), len(features))
    folder = Path(dst)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, parts in (('wells_mean', means), ('wells_median', medians),
                        ('wells_heterogeneity', spreads),
                        (f'{str(control_name).strip().lower() or "control"}_percentiles', limits)):
        if parts:
            paths.append(write_table(pd.concat(parts, ignore_index=True),
                                     folder / f'{name}.parquet', canonicalise=False))
    return {'paths': paths, 'plates': len(files), 'n_cells': int(n_cells),
            'n_features': int(n_features)}


def _phases(value: Any) -> List[str]:
    """The phases to run, in :data:`PHASES` order."""
    chosen = [phase.lower() for phase in _names(value)]
    unknown = [phase for phase in chosen if phase not in PHASES]
    if unknown:
        raise ValueError(f'screen_phases must be among {PHASES}, not {unknown}')
    return [phase for phase in PHASES if phase in chosen]


def _stage_settings(settings: Mapping[str, Any], defaults: Mapping[str, Any],
                    overrides: Any, *, own_keys: bool) -> Dict[str, Any]:
    """One pipeline's settings: its keys from the file, then its overrides."""
    staged = dict(defaults)
    for key in defaults:
        if key in settings and (own_keys or key not in _KEYS_MEASURED_DIFFERENTLY):
            staged[key] = settings[key]
    staged.update(dict(overrides or {}))
    return staged


class _Clock:
    """Wall-clock and CPU seconds of each phase, the children's CPU included."""

    def __init__(self) -> None:
        """Start an empty record."""
        self.phases: List[Dict[str, Any]] = []

    @staticmethod
    def _cpu() -> float:
        """CPU seconds of this process and its finished children."""
        import resource

        own = resource.getrusage(resource.RUSAGE_SELF)
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime

    def run(self, phase: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Run ``fn(*args, **kwargs)`` and record its seconds under ``phase``."""
        wall, cpu = time.time(), self._cpu()
        try:
            return fn(*args, **kwargs)
        finally:
            self.phases.append({'phase': phase, 'wall_s': time.time() - wall,
                                'cpu_s': self._cpu() - cpu, 'device': _device()})


def _device() -> str:
    """'gpu' when torch sees a CUDA device, else 'cpu'."""
    try:
        import torch

        return 'gpu' if torch.cuda.is_available() else 'cpu'
    except Exception:
        return 'cpu'


def run_screen(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Run the phases of one screen plate from one settings dict.

    See the module docstring for the phases. The settings are saved to
    ``<src>/settings/screen_settings.csv`` first, so the plate can be rerun
    with ``spacr-run screen --settings`` on that file.

    :param settings: ``src`` and the keys of :func:`screen_defaults`;
        missing keys take their defaults.
    :returns: ``{"phases", "import", "export", "aggregate", "timing"}``:
        the phases run, the import record, one export record per plate, the
        aggregate record and the per-phase seconds.
    :raises ValueError: without ``src``, or for unknown phases.
    """
    from .utils import save_settings

    settings = screen_defaults(settings)
    if not settings.get('src'):
        raise ValueError('src is blank: name the plate folder')
    src = os.fspath(settings['src'])
    phases = _phases(settings['screen_phases'])
    os.makedirs(os.path.join(src, 'settings'), exist_ok=True)
    save_settings(dict(settings), name='screen_settings')
    clock = _Clock()
    result: Dict[str, Any] = {'phases': phases, 'import': None, 'export': [], 'aggregate': None}
    if 'import' in phases and settings.get('import_source'):
        result['import'] = clock.run('import', import_plate, settings)
    mask_phases = [phase for phase in ('prep', 'masks') if phase in phases]
    if mask_phases:
        from .core import preprocess_generate_masks

        staged = _stage_settings(settings, _mask_defaults(settings),
                                 settings.get('mask_overrides'), own_keys=True)
        staged.update(preprocess='prep' in mask_phases, masks='masks' in mask_phases)
        clock.run('+'.join(mask_phases), preprocess_generate_masks, staged)
    if 'measure' in phases:
        from .measure import measure_crop
        from .settings import get_measure_crop_settings

        staged = _stage_settings(settings, _measure_defaults(settings),
                                 settings.get('measure_overrides'), own_keys=False)
        clock.run('measure', measure_crop, get_measure_crop_settings(staged))
    parquet_dir = settings.get('export_parquet_dir') or os.path.join(src, 'parquet')
    if 'export' in phases and settings.get('export_parquet'):
        result['export'].append(clock.run('export', export_plate_parquet, src, parquet_dir))
    if 'aggregate' in phases:
        controls = None
        if settings.get('aggregate_metadata'):
            controls = control_wells(settings['aggregate_metadata'],
                                     columns=settings['aggregate_metadata_columns'],
                                     control=settings['aggregate_control'])
        files = sorted(Path(parquet_dir).glob('*.parquet'))
        result['aggregate'] = clock.run(
            'aggregate', aggregate_wells, files,
            settings.get('aggregate_dir') or os.path.join(src, 'tables'),
            levels=settings['aggregate_levels'], controls=controls,
            control_name=settings['aggregate_control'],
            percentiles=settings['aggregate_control_percentiles'],
            heterogeneity=bool(settings['aggregate_heterogeneity']))
    result['timing'] = clock.phases
    if settings.get('record_timing'):
        path = Path(src) / 'settings' / 'screen_timing.json'
        path.write_text(json.dumps({'phases': clock.phases}, indent=1), encoding='utf-8')
        result['timing_path'] = str(path)
    return result
