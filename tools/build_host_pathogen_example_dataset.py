#!/usr/bin/env python
"""Measure two real THP-1/RNF213 fields for the Host–Pathogen example.

Inputs are the already public spacr-example-recruitment dataset. Its acquired
images are preserved; existing masks receive Measure's parent reconciliation. The current
Measure pipeline creates a fresh database covering exactly the included fields.
No synthetic pixels, segmentation, or replication counts are invented.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import tarfile
from pathlib import Path

import numpy as np

SOURCE_REPO = 'einarolafsson/spacr-example-recruitment'
SOURCE_REVISION = '0861a3ab27e19b2793e656d0f9ac1677d5088a2d'
SOURCE_ARCHIVE_SHA256 = 'a9508b0f6ea208498f1ae909eb679881bcb6c67db1f1ec9de027be4f64611862'
FIELDS = ('PLATE1_E01_1_1', 'PLATE1_E02_1_1')
SOURCE_FIELD_HASHES = {
    'PLATE1_E01_1_1': '13b5dae09813fbda64581080e7c5659b73ae34e4aa33e3a4f10eeac0f45764f8',
    'PLATE1_E02_1_1': '0c9fd28d5198017b59aae627b3ec5499fe665527065ab7cc56702ee6c4240892',
}
REPO = 'einarolafsson/spacr-example-host-pathogen'
ARCHIVE = 'spacr-example-host-pathogen.tar'


def build(source, destination):
    """Build a measured example in a new folder; refuse to replace existing data."""
    from spacr.measure import measure_crop
    from spacr.host_pathogen import analyze_host_pathogen
    from spacr.utils import _merge_overlapping_objects, _exclude_objects

    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    merged = destination / 'merged'
    merged.mkdir()
    provenance = []
    for field in FIELDS:
        path = source / 'merged' / f'{field}.npy'
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if source_hash != SOURCE_FIELD_HASHES[field]:
            raise ValueError(f'{field} does not match the pinned published source')
        data = np.load(path, mmap_mode='r', allow_pickle=False)
        if data.ndim != 3 or data.shape[-1] != 7:
            raise ValueError(f'{field} must have the source dataset\'s seven planes')
        prepared = np.array(data, copy=True)
        nucleus, cell = _merge_overlapping_objects(prepared[:, :, 5], prepared[:, :, 4])
        vacuole, cell = _merge_overlapping_objects(prepared[:, :, 6], cell)
        cytoplasm = np.where((nucleus > 0) | (vacuole > 0), 0, cell)
        cell, nucleus, vacuole, _ = _exclude_objects(cell, nucleus, vacuole, cytoplasm, uninfected=True)
        prepared[:, :, 4], prepared[:, :, 5], prepared[:, :, 6] = cell, nucleus, vacuole
        np.save(merged / path.name, prepared, allow_pickle=False)
        provenance.append({'field': field, 'sha256': source_hash,
                           'prepared_sha256': hashlib.sha256((merged / path.name).read_bytes()).hexdigest(),
                           'changed_mask_pixels': [int(np.count_nonzero(prepared[:, :, i] != data[:, :, i])) for i in (4, 5, 6)],
                           'shape': list(data.shape),
                           'source_cells': len(np.unique(data[:, :, 4])) - 1,
                           'source_vacuoles': len(np.unique(data[:, :, 6])) - 1})
    settings = dict(src=str(merged), channels=[0, 1, 2, 3],
                    cell_mask_dim=4, nucleus_mask_dim=5, pathogen_mask_dim=6,
                    organelle_mask_dim=None, cell_min_size=0, nucleus_min_size=0,
                    pathogen_min_size=0, cytoplasm_min_size=0, cytoplasm=True,
                    uninfected=True, merge_edge_pathogen_cells=False,
                    save_measurements=True, save_png=False, save_arrays=False,
                    radial_dist=False, spatial_measurements=False, object_distances=False,
                    object_distance_maxima=False, object_distance_intensity=False,
                    calculate_correlation=False, homogeneity=False, zernike=False,
                    plot=False, n_jobs=2, timelapse=False, test_mode=False,
                    experiment='host_pathogen_example')
    layout = {'version': 1, 'intensity_channels': [0, 1, 2, 3],
              'mask_plane_order': ['cell', 'nucleus', 'pathogen'],
              'mask_dims': {'cell': 4, 'nucleus': 5, 'pathogen': 6}}
    (merged / '.spacr_plane_layout.json').write_text(json.dumps(layout, indent=2) + '\n')
    measure_crop(settings)
    database = destination / 'measurements/measurements.db'
    with sqlite3.connect(database) as connection:
        for table in ('cell', 'nucleus', 'pathogen', 'cytoplasm'):
            count = connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            if count < 1:
                raise ValueError(f'Measure produced no {table} objects')
            columns = {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}
            column = 'columnID' if 'columnID' in columns else 'column_name'
            if connection.execute(f'SELECT COUNT(DISTINCT "{column}") FROM "{table}"').fetchone()[0] != 2:
                raise ValueError(f'Measure did not complete both fields for {table}')
        tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        for table in tables:
            quoted = table.replace('"', '""')
            columns = connection.execute(f'PRAGMA table_info("{quoted}")').fetchall()
            for column in columns:
                name = column[1].replace('"', '""')
                if str(column[2]).upper() in ('TEXT', ''):
                    connection.execute(f'UPDATE "{quoted}" SET "{name}" = replace("{name}", ?, ?) '
                                       f'WHERE typeof("{name}") = \'text\'',
                                       (str(destination), '<dataset>'))
        connection.commit()
    config = {'src': '<dataset>', 'hp_vacuole_table': 'pathogen',
              'hp_vacuole_prefix': 'pathogen', 'hp_reference_table': 'cytoplasm',
              'hp_reference_prefix': 'cytoplasm', 'hp_marker_channels': [1],
              'hp_marker_thresholds': {}, 'hp_parasite_table': '',
              'hp_count_column': '', 'save': True}
    (destination / 'settings').mkdir(exist_ok=True)
    with (destination / 'settings/host_pathogen_settings.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['Key', 'Value'])
        for key, value in config.items():
            writer.writerow([key, value if isinstance(value, str) else repr(value)])
    result = analyze_host_pathogen(dict(config, src=str(destination), save=False))
    counts = {key: len(value) for key, value in result.items() if hasattr(value, 'columns')}
    (destination / 'example_manifest.json').write_text(json.dumps({
        'version': 1, 'synthetic': False, 'source_repo': SOURCE_REPO,
        'source_revision': SOURCE_REVISION, 'source_archive_sha256': SOURCE_ARCHIVE_SHA256,
        'fields': provenance, 'analysis_rows': counts,
        'replication_counts': 'unavailable: masks represent whole vacuoles, not individual parasites',
        'channels': ['Hoechst', 'RNF213 target', 'Toxoplasma', 'CellMask'],
        'masks': 'Existing automatic masks, reconciled with Measure parent/child and nucleus filtering; not manual truth',
    }, indent=2) + '\n')
    for key in ('wells', 'cells', 'vacuoles'):
        frame = result[key].copy()
        for column in frame.select_dtypes(include=['object', 'string']):
            frame[column] = frame[column].map(lambda value: value.replace(str(destination), '<dataset>')
                                              if isinstance(value, str) else value)
        frame.to_csv(destination / f'example_{key}.csv', index=False)
    for path in (destination / 'settings').glob('measure_crop_settings.*'):
        path.write_text(path.read_text().replace(str(destination), '<dataset>'))
    verify(source, destination)
    return destination


def verify(source, destination):
    """Check all saved areas/means against pixels and every preview against its field."""
    from spacr.host_pathogen_preview import preview_fields, preview_field

    source, destination = Path(source), Path(destination)
    checked = 0
    with sqlite3.connect(destination / 'measurements/measurements.db') as connection:
        for field in FIELDS:
            original = np.load(source / 'merged' / f'{field}.npy', mmap_mode='r')
            data = np.load(destination / 'merged' / f'{field}.npy', mmap_mode='r')
            np.testing.assert_array_equal(data[:, :, :4], original[:, :, :4])
            masks = dict(cell=data[:, :, 4], nucleus=data[:, :, 5], pathogen=data[:, :, 6])
            masks['cytoplasm'] = np.where((masks['nucleus'] > 0) | (masks['pathogen'] > 0), 0, masks['cell'])
            for table, mask in masks.items():
                rows = connection.execute(f'SELECT object_label, "{table}_area", '
                    + ', '.join(f'"{table}_channel_{i}_mean_intensity"' for i in range(4))
                    + f' FROM "{table}" WHERE file_name=?', (field,)).fetchall()
                labels = set(np.unique(mask)) - {0}
                assert {row[0] for row in rows} == labels, (field, table, 'labels')
                areas = np.bincount(mask.ravel().astype(int))
                sums = [np.bincount(mask.ravel().astype(int), weights=data[:, :, i].ravel()) for i in range(4)]
                for label, area, *means in rows:
                    assert area == areas[label], (field, table, label, 'area')
                    np.testing.assert_allclose(means, [s[label] / areas[label] for s in sums], rtol=1e-10)
                    checked += 1
    config = dict(src=str(destination), hp_marker_channels=[1], hp_marker_thresholds={})
    fields, truncated = preview_fields(config)
    assert len(fields) == 2 and not truncated
    for field in fields:
        result = preview_field(config, field, image_channel=2)
        assert result['image'] is not None and set(result['masks']) == {'host', 'vacuole'}
        assert result['results']['vacuoles']['replication_method'].eq('not_measured').all()
    print(f'Verified {checked} objects: saved mask labels, areas and all four intensity means; both previews resolve.', flush=True)


def package(destination):
    """Archive only portable inputs, measurements, settings and provenance."""
    destination = Path(destination)
    archive = destination.parent / ARCHIVE
    members = ['merged', 'measurements/measurements.db', 'settings',
               'example_manifest.json', 'example_cells.csv', 'example_vacuoles.csv', 'example_wells.csv']
    with tarfile.open(archive, 'w') as handle:
        for name in members:
            handle.add(destination / name, arcname=name)
    print(f'{archive}: {archive.stat().st_size} bytes; SHA256 {hashlib.sha256(archive.read_bytes()).hexdigest()}', flush=True)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--archive', action='store_true', help='Also package the measured project as a tar')
    args = parser.parse_args()
    folder = build(args.source, args.out)
    if args.archive:
        package(folder)
    print(folder, flush=True)


if __name__ == '__main__':
    main()
