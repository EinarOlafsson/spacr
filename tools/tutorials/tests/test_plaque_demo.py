"""Small positive and corrupt-output tests for the tutorial's plaque evidence."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plaque_demo import label_areas, area_summary, verify_tables, require_preserved, verify_filtered
from replication_demo import digest


def test_label_counts_use_area_and_separate_objects():
    mask = np.array([[1, 1, 0, 2], [1, 0, 0, 2]], dtype=np.uint16)
    assert label_areas(mask) == [3, 2]
    mask[1, 3] = 1
    with pytest.raises(ValueError, match='disconnected'):
        label_areas(mask)


def test_live_minimum_keeps_the_equality_boundary_and_exact_object_pixels():
    raw = np.array([[1, 1, 0, 2], [1, 0, 0, 2]], dtype=np.uint16)
    actual = np.array([[9, 9, 0, 0], [9, 0, 0, 0]], dtype=np.uint16)
    assert verify_filtered(raw, raw, 0)['objects'] == 2
    assert verify_filtered(raw, actual, 3)['objects'] == 1
    assert verify_filtered(raw, np.zeros_like(raw), 4)['objects'] == 0
    bad = actual.copy(); bad[1, 0] = 0; bad[1, 3] = 9
    with pytest.raises(ValueError, match='wrong pixels'):
        verify_filtered(raw, bad, 3)
    with pytest.raises(ValueError, match='wrong pixels'):
        verify_filtered(raw, raw[:, :3], 3)


def test_filter_rejects_merging_touching_labels_even_when_foreground_matches():
    raw = np.array([[1, 1, 2, 2]], dtype=np.uint16)
    assert verify_filtered(raw, raw, 0)['objects'] == 2
    with pytest.raises(ValueError, match='membership'):
        verify_filtered(raw, np.ones_like(raw), 0)


@pytest.mark.parametrize('mask', [np.ones((2, 2, 2), dtype=int), np.ones((2, 2)), np.array([[-1]])])
def test_invalid_label_arrays(mask):
    assert label_areas(np.array([[0, 1]], dtype=int)) == [1]
    with pytest.raises(ValueError, match='label mask'):
        label_areas(mask)


def test_calibration_absence_has_a_positive_scaled_counterpart():
    plain = area_summary([4, 8])
    scaled = area_summary([4, 8], 2)
    assert plain['average_size'] == 6 and plain['std_dev_size'] == 2
    assert plain['average_size_mm2'] is plain['std_dev_size_mm2'] is None
    assert scaled['average_size_mm2'] == 1.5 and scaled['std_dev_size_mm2'] == .5


@pytest.mark.parametrize('scale', [0, -2, float('nan')])
def test_invalid_scale(scale):
    assert area_summary([4, 8], 2)['average_size_mm2'] == 1.5
    with pytest.raises(ValueError, match='scale'):
        area_summary([4, 8], scale)


def sample():
    reference = {'a.tif': dict(areas=[4, 8], summary=area_summary([4, 8]))}
    stats = dict(reference['a.tif']['summary'], file='a.tif', well_diameter_px=None)
    summary = {k: v for k, v in stats.items() if k not in ('plaque_count', 'std_dev_size', 'std_dev_size_mm2')}
    summary['object_count'] = 2
    tables = dict(summary=[summary], stats=[stats], details=[
        dict(file='a.tif', plaque_size=a, plaque_size_mm2=None) for a in [4, 8]])
    return tables, reference


@pytest.mark.parametrize('defect', ['table', 'duplicate', 'missing', 'count', 'area', 'units', 'detail', 'detail units'])
def test_corrupted_database_result(defect):
    tables, reference = sample()
    assert verify_tables(tables, reference)['plaques'] == 2
    bad = copy.deepcopy(tables)
    if defect == 'table':
        bad['extra'] = []
    elif defect == 'duplicate':
        bad['summary'].append(copy.deepcopy(bad['summary'][0]))
    elif defect == 'missing':
        bad['summary'][0].pop('average_size_mm2')
    elif defect == 'count':
        bad['summary'][0]['object_count'] = 3
    elif defect == 'area':
        bad['stats'][0]['average_size'] = 7
    elif defect == 'units':
        bad['stats'][0]['average_size_mm2'] = 6
    elif defect == 'detail':
        bad['details'][0]['plaque_size'] = 5
    else:
        bad['details'][0]['plaque_size_mm2'] = 4
    with pytest.raises(ValueError):
        verify_tables(bad, reference)


@pytest.mark.parametrize('changed', ['source', 'copy'])
def test_preserved_files(tmp_path, changed):
    originals = {name: tmp_path/name for name in ('source', 'copy')}
    for p in originals.values():
        p.write_bytes(b'unchanged')
    records = [{**{k: str(v) for k, v in originals.items()}, 'sha256': digest(originals['source'])}]
    require_preserved(records)
    originals[changed].write_bytes(b'changed')
    with pytest.raises(ValueError, match='input changed'):
        require_preserved(records)
