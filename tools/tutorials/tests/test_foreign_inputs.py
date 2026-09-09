"""The neutral import example measures actual pixels, not invented table rows."""
from pathlib import Path
import csv
import sys

import numpy as np
import pytest
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capture_foreign import FIELDS, compare_object_measurements, digest, measure_labels, prepare_inputs


def test_sparse_ids_have_exact_areas_and_means_without_background():
    image = np.array([[1, 3, 100], [7, 9, 100]], dtype=np.uint16)
    labels = np.array([[17, 17, 0], [42, 42, 0]], dtype=np.uint16)
    before = image.copy(), labels.copy()
    assert measure_labels(image, labels) == [(17, 2, 2.0), (42, 2, 8.0)]
    assert np.array_equal(image, before[0]) and np.array_equal(labels, before[1])


def test_real_existing_blank_labels_are_rejected():
    with pytest.raises(ValueError, match='no labelled objects'):
        measure_labels(np.ones((3, 3)), np.zeros((3, 3), dtype=np.uint16))


@pytest.mark.parametrize('shape', [(2, 3), (3, 3, 1)])
def test_nonmatching_or_nonplanar_inputs_are_rejected(shape):
    with pytest.raises(ValueError, match='identical two-dimensional'):
        measure_labels(np.ones((3, 3)), np.ones(shape, dtype=np.uint16))


def write_sources(root, planes=7):
    folder = root / 'example_data/plate1/merged'
    folder.mkdir(parents=True)
    for index, stem in enumerate(FIELDS):
        data = np.zeros((8, 8, planes), dtype=np.uint16)
        data[..., 1] = 20 + index
        data[1:4, 2:6, 4] = 17
        np.save(folder / (stem + '.npy'), data)
    return folder


def test_prepared_csv_and_tiffs_match_exact_source_objects(tmp_path):
    folder = write_sources(tmp_path)
    before = {p.name: digest(p) for p in folder.iterdir()}
    run, images, masks, table, records, rows = prepare_inputs(tmp_path)
    assert run.is_relative_to(tmp_path / 'foreign_runs')
    assert len(records) == len(rows) == 2
    with table.open(newline='') as stream:
        saved = list(csv.DictReader(stream))
    assert [int(r['Area_px2']) for r in saved] == [12, 12]
    assert [float(r['MeanIntensity_ER']) for r in saved] == [20.0, 21.0]
    assert [int(r['ObjectNumber']) for r in saved] == [17, 17]
    for record in records:
        source = np.load(record['source'])
        assert np.array_equal(tifffile.imread(record['image']), source[..., 1])
        assert np.array_equal(tifffile.imread(record['mask']), source[..., 4])
    assert {p.name: digest(p) for p in folder.iterdir()} == before


def test_wrong_example_layout_is_not_reinterpreted(tmp_path):
    write_sources(tmp_path, planes=6)
    with pytest.raises(ValueError, match='seven-plane uint16'):
        prepare_inputs(tmp_path)


def test_per_object_comparison_accepts_the_same_values_and_identities():
    rows = {('first.tif', 17): (12, 20.0), ('second.tif', 17): (15, 30.0)}
    compare_object_measurements(rows, dict(reversed(list(rows.items()))))


def test_equal_totals_do_not_hide_measurements_swapped_between_fields():
    expected = {('first.tif', 17): (12, 20.0), ('second.tif', 17): (15, 30.0)}
    swapped = {('first.tif', 17): (15, 30.0), ('second.tif', 17): (12, 20.0)}
    with pytest.raises(ValueError, match='object assignment'):
        compare_object_measurements(expected, swapped)


def test_identical_values_cannot_hide_a_changed_object_identity():
    with pytest.raises(ValueError, match='object identities'):
        compare_object_measurements({('first.tif', 17): (12, 20.0)},
                                    {('first.tif', 42): (12, 20.0)})
