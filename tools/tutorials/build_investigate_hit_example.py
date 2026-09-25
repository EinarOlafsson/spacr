"""Build the Investigate Hit tutorial example from real screen data only.

Every input comes from one screen. Nothing is simulated:

* Regression's downloadable test data (``plate1-4_dv.csv`` score tables and
  ``plate_1-4_unique_combinations.csv`` count tables), checked against
  :data:`spacr.example_data_manifest.FILES`. Each score row is one cell's
  real CV prediction (``pred``) and names that cell's crop.
* A completed Regression run on exactly those tables (lesson 13's settings).
  Its ``regression_data.csv`` supplies the per-well guide fractions and its
  ``results_gene.csv`` supplies the gene under investigation.
* The same screen's per-cell measurement frame (the merged measurements of
  the four plate databases). Every scored cell matches exactly one measured
  cell on plate, row, column, field and object label.

The example keeps only the wells the investigation needs: every analysed
well that contains a guide for ``--gene`` and, from the same plates, a seeded
random sample of analysed wells that contain none. Measurements, predictions
and wells are copied, never altered; the only derived values are identity
keys (``prcf``/``prcfo``) and the per-well consistency checks below.

Output (refused if it exists)::

    <output>/measurements/measurements.db   cell table: identity + real features
    <output>/cv_predictions.csv             prcfo, crop name, pred, cv_predictions
    <output>/regression_run/                the complete Regression run folder
    <output>/README.txt, example_manifest.json

``--zip`` also writes a deterministic ZIP of that folder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

IDENTITY = ['object_label', 'plateID', 'rowID', 'columnID', 'fieldID']
WELL = ['plateID', 'rowID', 'columnID']
#: A fixed, interpretable subset of the real cell measurements: shape and
#: size, then each channel's mean intensity. Chosen before any attribution
#: was run; the example does not search for features that separate groups.
FEATURES = [
    'cell_area', 'cell_perimeter', 'cell_major_axis_length',
    'cell_minor_axis_length', 'cell_eccentricity', 'cell_solidity',
    'cell_extent', 'cell_equivalent_diameter_area', 'cell_feret_diameter_max',
    'cell_channel_0_mean_intensity', 'cell_channel_1_mean_intensity',
    'cell_channel_2_mean_intensity', 'cell_channel_3_mean_intensity',
]
RUN_TABLES = ('results_gene.csv', 'results_grna.csv', 'regression_data.csv')


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_regression_example(folder: Path) -> dict:
    """Check the eight downloaded tables against spaCR's own manifest."""
    from spacr.example_data import is_whole
    from spacr.example_data_manifest import FILES
    checked = {}
    for entry in FILES:
        path = folder / entry['name']
        if not is_whole(path, entry):
            raise ValueError(f'{path} is not the published Regression test file')
        checked[entry['name']] = entry['sha256']
    return checked


def read_scores(folder: Path) -> pd.DataFrame:
    """The four real score tables as one frame with measurement identities."""
    frames = []
    for plate in range(1, 5):
        frame = pd.read_csv(folder / f'plate{plate}_dv.csv', index_col=0)
        frame = frame.rename(columns={'column': 'col'})
        frames.append(frame)
    scores = pd.concat(frames, ignore_index=True)
    if not scores['plate'].astype(str).str.fullmatch(r'pplate\d+').all():
        raise ValueError('unexpected plate spelling in the score tables')
    scores['plateID'] = scores['plate'].str[1:]
    scores['rowID'] = scores['row'].astype(str)
    scores['columnID'] = scores['col'].astype(str)
    scores['fieldID'] = scores['field'].astype(str)
    scores['object_label'] = scores['object'].astype(int)
    if scores.duplicated(IDENTITY).any() or scores['path'].duplicated().any():
        raise ValueError('score tables repeat a cell')
    return scores


def choose_wells(run: Path, gene: str, controls_per_target: int,
                 seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Target wells for ``gene`` and a seeded same-plate control sample."""
    fractions = pd.read_csv(run / 'regression_data.csv')
    genes = pd.read_csv(run / 'results_gene.csv', dtype={'gene': str})
    hit = genes[genes['gene'] == gene]
    if len(hit) != 1:
        raise ValueError(f'gene {gene} is not one row of results_gene.csv')
    fractions['gene'] = fractions['gene'].astype(str)
    fractions['grna'] = fractions['grna'].astype(str)
    target_rows = fractions[fractions['gene'] == gene]
    targets = target_rows[WELL].drop_duplicates()
    wells = fractions[WELL + ['pred', 'cell_count']].drop_duplicates(WELL)
    target_keys = set(map(tuple, targets.to_numpy()))
    free = wells[[tuple(k) not in target_keys for k in wells[WELL].to_numpy()]]
    rng = np.random.default_rng(seed)
    controls = []
    for plate, n_target in targets.groupby('plateID').size().items():
        pool = free[free['plateID'] == plate].sort_values(WELL)
        take = min(len(pool), controls_per_target * int(n_target))
        controls.append(pool.iloc[np.sort(rng.choice(len(pool), take, replace=False))])
    controls = pd.concat(controls)[WELL]
    summary = dict(
        gene=gene, target_guides=sorted(target_rows['grna'].unique()),
        target_wells=len(targets), control_wells=len(controls),
        plates=sorted(targets['plateID'].unique()),
        regression_row=json.loads(hit.iloc[0].to_json()))
    return (targets.assign(role='target'), controls.assign(role='control'),
            summary)


def read_measurements(path: Path, wells: pd.DataFrame) -> pd.DataFrame:
    """Real cell rows of the selected wells, identity plus FEATURES."""
    keep = set(map(tuple, wells[WELL].to_numpy()))
    parts = []
    for chunk in pd.read_csv(path, usecols=IDENTITY + FEATURES,
                             chunksize=50_000, low_memory=False):
        mask = [tuple(k) in keep for k in chunk[WELL].astype(str).to_numpy()]
        parts.append(chunk.loc[mask])
    cells = pd.concat(parts, ignore_index=True)
    cells['object_label'] = cells['object_label'].astype(int)
    for column in ('plateID', 'rowID', 'columnID', 'fieldID'):
        cells[column] = cells[column].astype(str)
    if cells.duplicated(IDENTITY).any():
        raise ValueError('measurement frame repeats a cell')
    return cells


def build(args) -> dict:
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new folder')
    example = Path(args.regression_example)
    run_root = Path(args.regression_run)
    run = run_root / 'results' / 'guide_permutation'
    for name in RUN_TABLES:
        if not (run / name).is_file():
            raise FileNotFoundError(run / name)
    downloaded = verify_regression_example(example)
    scores = read_scores(example)
    targets, controls, summary = choose_wells(
        run, str(args.gene), args.controls_per_target, args.seed)
    wells = pd.concat([targets, controls], ignore_index=True)
    cells = read_measurements(Path(args.measurements), wells)

    joined = cells.merge(scores, on=IDENTITY, how='outer', indicator=True,
                         validate='one_to_one')
    unmatched = joined['_merge'].value_counts().to_dict()
    selected = set(map(tuple, wells[WELL].to_numpy()))
    in_wells = [tuple(k) in selected for k in joined[WELL].to_numpy()]
    joined = joined.loc[in_wells]
    if (joined['_merge'] != 'both').any():
        raise ValueError(f'scored and measured cells differ: {unmatched}')

    # Regression aggregated these very predictions: check each kept well's
    # mean score and cell count against regression_data.csv.
    fractions = pd.read_csv(run / 'regression_data.csv')
    reported = fractions.drop_duplicates(WELL).set_index(WELL)
    observed = joined.groupby(WELL).agg(pred=('pred', 'mean'),
                                        cell_count=('pred', 'size'))
    reported = reported.loc[observed.index]
    if not np.allclose(observed['pred'], reported['pred'], rtol=0, atol=1e-9):
        raise ValueError('well mean scores differ from the Regression run')
    if not (observed['cell_count'].to_numpy()
            == reported['cell_count'].to_numpy()).all():
        raise ValueError('well cell counts differ from the Regression run')

    joined = joined.sort_values(IDENTITY[1:] + ['object_label'])
    joined['prcf'] = joined[['plateID', 'rowID', 'columnID', 'fieldID']].agg(
        '_'.join, axis=1)
    joined['prcfo'] = joined['prcf'] + '_o' + joined['object_label'].astype(str)

    (output / 'measurements').mkdir(parents=True)
    database = output / 'measurements' / 'measurements.db'
    cell_columns = IDENTITY + ['prcf'] + (['prcfo'] if args.cell_prcfo else []) + FEATURES
    with sqlite3.connect(database) as connection:
        joined[cell_columns].to_sql('cell', connection, index=False)
    from spacr.database_schema import ensure_database_schema
    ensure_database_schema(str(database))

    predictions = joined[['prcfo', 'path', 'pred', 'cv_predictions']]
    predictions.to_csv(output / 'cv_predictions.csv', index=False)
    shutil.copytree(run_root, output / 'regression_run')

    from spacr.io import _read_and_join_tables
    read_back = _read_and_join_tables(str(database))
    if len(read_back) != len(joined):
        raise ValueError('spaCR does not read back every example cell')

    well_table = wells.merge(
        observed.reset_index(), on=WELL, how='left').sort_values(
            ['role', 'plateID', 'rowID', 'columnID'])
    manifest = dict(
        purpose='Investigate Hit tutorial example; real screen data only',
        synthetic_data=False,
        gene=summary, controls_per_target=args.controls_per_target,
        control_sample_seed=args.seed,
        cells=int(len(joined)), wells=int(len(wells)),
        features=FEATURES, cell_table_has_prcfo=bool(args.cell_prcfo),
        well_table=json.loads(well_table.to_json(orient='records')),
        regression_test_data_sha256=downloaded,
        regression_run_tables_sha256={
            name: sha(run / name) for name in RUN_TABLES},
        measurement_source=dict(
            description='merged per-cell measurements of the same four '
                        'plate databases', rows_matched=int(unmatched.get('both', 0)),
            only_scored=int(unmatched.get('right_only', 0)),
            only_measured=int(unmatched.get('left_only', 0))),
        outputs={str(p.relative_to(output)): sha(p) for p in (
            database, output / 'cv_predictions.csv')})
    (output / 'example_manifest.json').write_text(
        json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    shutil.copy2(Path(__file__).with_name('investigate_hit_README.txt'),
                 output / 'README.txt')
    if args.zip:
        write_zip(output, Path(args.zip))
        manifest['zip_sha256'] = sha(Path(args.zip))
    return manifest


def write_zip(folder: Path, target: Path) -> None:
    """Deterministic archive: sorted names, fixed timestamps."""
    if target.exists():
        raise FileExistsError(f'{target} exists; preserve published downloads')
    files = sorted(p for p in folder.rglob('*') if p.is_file())
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            info = zipfile.ZipInfo(str(path.relative_to(folder.parent)),
                                   date_time=(2026, 9, 25, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise ValueError('ZIP failed its own check')
        for path in files:
            if archive.read(str(path.relative_to(folder.parent))) != path.read_bytes():
                raise ValueError('packaged bytes differ')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--regression-example', required=True,
                        help='folder holding the eight downloaded Regression test tables')
    parser.add_argument('--regression-run', required=True,
                        help='Regression run folder (src) made from those tables')
    parser.add_argument('--measurements', required=True,
                        help="the screen's merged per-cell measurement CSV")
    parser.add_argument('--gene', default='225160')
    parser.add_argument('--controls-per-target', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cell-prcfo', action='store_true',
                        help='also store the prcfo key in the cell table')
    parser.add_argument('--output', required=True)
    parser.add_argument('--zip')
    args = parser.parse_args(argv)
    manifest = build(args)
    print(json.dumps({k: manifest[k] for k in ('cells', 'wells', 'gene')},
                     indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
