"""Plaque rows use the warning-free area diameter API without changing values."""
import math
import sqlite3
import warnings

import numpy as np
import pytest
import tifffile
from skimage.measure import regionprops


def known_labels():
    labels = np.zeros((48, 48), np.uint16)
    labels[2, 2] = 2
    labels[5:8, 5:9] = 7
    labels[5:8, 15:19] = 12
    labels[20:27, 20:27] = 20
    labels[22:25, 22:25] = 0
    labels[35:40, 35:37] = 31
    labels[38:40, 35:42] = 31
    return labels


def test_installed_regionprops_diameter_aliases_have_identical_values():
    regions = regionprops(known_labels())
    assert [int(region.area) for region in regions] == [1, 12, 12, 40, 20]
    for region in regions:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always', FutureWarning)
            legacy = float(region.equivalent_diameter)
        assert legacy == float(region.equivalent_diameter_area)
        assert legacy == pytest.approx(2 * math.sqrt(region.area / math.pi))
    assert regions[1].equivalent_diameter_area == regions[2].equivalent_diameter_area


def test_actual_per_plaque_rows_retain_ids_areas_and_diameter_field_without_warning(tmp_path, monkeypatch):
    from spacr import submodules as sm

    masks = tmp_path / 'masks'
    masks.mkdir()
    labels = known_labels()
    tifffile.imwrite(masks / 'known_geometry.tif', labels)
    monkeypatch.setattr(sm, '_resolve_plaque_model', lambda *_a, **_k: 'unused-existing-mask-model')
    monkeypatch.setattr('spacr.utils.save_settings', lambda *_a, **_k: None)
    monkeypatch.setattr(sm, '_segment_plaque_folder', lambda *_a, **_k: pytest.fail('Existing masks must not segment'))
    with warnings.catch_warnings():
        warnings.simplefilter('error', FutureWarning)
        sm.analyze_plaques({'src': str(tmp_path), 'masks': False, 'well_detection': False})
    with sqlite3.connect(masks / 'plaques_analysis.db') as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        columns = [row[1] for row in db.execute('PRAGMA table_info(per_plaque)')]
        rows = db.execute('SELECT file, plaque_id, area_px, equivalent_diameter_px FROM per_plaque ORDER BY plaque_id').fetchall()
    assert {'summary', 'stats', 'details', 'per_image', 'per_plaque'} <= tables
    assert columns.count('equivalent_diameter_px') == 1
    assert 'equivalent_diameter_area' not in columns
    assert len(rows) == 5
    assert [row[1] for row in rows] == [2, 7, 12, 20, 31]
    assert [row[2] for row in rows] == [1, 12, 12, 40, 20]
    assert {row[0] for row in rows} == {'known_geometry.tif'}
    for row in rows:
        assert row[3] == pytest.approx(2 * math.sqrt(row[2] / math.pi))
    assert rows[1][3] == rows[2][3]
    assert np.array_equal(tifffile.imread(masks / 'known_geometry.tif'), labels)
