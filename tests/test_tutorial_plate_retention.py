"""The retention proof must detect wrong data, not merely missing files."""
import csv
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/tutorials'))
from capture_plate_retention import COLUMNS, verify_export, verify_report, verify_wells


@pytest.fixture
def wells():
    records = [dict(zip(COLUMNS, ('plate1', 'B01', 'r2', 'c1', 2, 1, 356, 356.0, 0, True))),
               dict(zip(COLUMNS, ('plate1', 'B02', 'r2', 'c2', 2, 2, 358, 358.0, 1, False)))]
    independent = {('plate1', 'r2', 'c1'): {'n': 356, 'mean': 19.2},
                   ('plate1', 'r2', 'c2'): {'n': 358, 'mean': 11.3},
                   ('plate1', 'r3', 'c14'): {'n': 300, 'mean': 8.1}}
    return records, independent


def test_matching_count_and_mean_with_positive_restoration(wells):
    records, independent = wells
    assert verify_wells(records, independent, 'count', 350)['dropped'] == 1
    means = [dict(r, value=independent[(r['plateID'], r['rowID'], r['columnID'])]['mean']) for r in records]
    assert verify_wells(means, independent, 'mean', 350)['wells'] == 2
    restored = records + [dict(zip(COLUMNS, ('plate1', 'C14', 'r3', 'c14', 3, 14, 300, 300., 2, False)))]
    assert verify_wells(restored, independent, 'count', 100)['wells'] == 3


@pytest.mark.parametrize('field,value', [('value', 357), ('value', float('nan')),
                                        ('n', 355), ('plateID', 'other')])
def test_changed_value_count_or_identity_refused(wells, field, value):
    records, independent = wells
    records[0][field] = value
    with pytest.raises(ValueError):
        verify_wells(records, independent, 'count', 350)


def test_missing_and_duplicate_wells_refused(wells):
    records, independent = wells
    for wrong in (records[:-1], [records[0], records[0]]):
        with pytest.raises(ValueError):
            verify_wells(wrong, independent, 'count', 350)


def test_blank_must_not_be_padded_as_zero(wells):
    records, independent = wells
    records.append(dict(zip(COLUMNS, ('plate1', 'C14', 'r3', 'c14', 3, 14, 300, 0., 2, False))))
    with pytest.raises(ValueError):
        verify_wells(records, independent, 'count', 350)


def test_report_requires_exact_recorded_statistics():
    original = {'n_wells': 196, 'pct_difference': -1.07, 'edge_detected': False}
    verify_report(dict(original), original)
    with pytest.raises(ValueError):
        verify_report(dict(original, edge_detected=True), original)


def test_export_full_values_identity_and_order(tmp_path, wells):
    records, _ = wells
    path = tmp_path / 'real.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    assert verify_export(path, records)['all_values_and_order_exact']
    with pytest.raises(ValueError):
        verify_export(path, records[::-1])
    records[0]['ring'] = 8
    with pytest.raises(ValueError):
        verify_export(path, records)
