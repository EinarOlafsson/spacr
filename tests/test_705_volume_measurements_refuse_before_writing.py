"""3D image-set measurement reads saved sessions and refuses what it cannot measure.

Every refusal here is checked for its reason and for leaving the image
folder without a new or changed database, because a half-written
``measurements.db`` is worse than a sentence explaining why nothing ran.
"""

import numpy as np
import pandas as pd
import pytest
import tifffile

from spacr import measure, tabular
from spacr.qt import mask_engine


def _write(path, data, *, axes='ZYX', spacing=(2., .5, .5), unit='um'):
    tifffile.imwrite(path, data, photometric='minisblack',
                     metadata={'axes': axes, 'voxel_spacing': spacing, 'unit': unit})
    return str(path)


def _labels():
    labels = np.zeros((10, 8, 8), dtype=np.uint16)
    labels[1:3, 2:5, 2:5] = 4
    labels[6:8, 2:5, 2:5] = 19
    return labels


def _image(shape=(10, 8, 8)):
    return np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 3. + .25


def _row(tmp_path, name='image.tif', **snapshot):
    image = _image()
    source = _write(tmp_path / name, image)
    state = {'channels': (image,), 'mask': _labels(), 'axes': 'ZYX',
             'spacing': (2., .5, .5), 'unit': 'um'}
    state.update(snapshot)
    return {'source_path': source, 'channel_paths': (), 'snapshot': state}


def _read(path, table='cell'):
    return tabular.read_database(str(path), table, canonicalise=False, migrate=False)[0]


def test_a_saved_image_set_without_an_explicit_mask_reads_its_saved_volume_mask(tmp_path):
    source = _write(tmp_path / 'saved.tif', _image())
    mask_path = mask_engine.mask_save_path(str(tmp_path), 'saved.tif')
    (tmp_path / 'masks').mkdir(exist_ok=True)
    _write(mask_path, _labels())

    result = measure.measure_volume_rows([{'source_path': source}])

    assert result['objects'] == 2
    frame = _read(result['db_paths'][0]).sort_values('object_label')
    assert frame.object_label.tolist() == [4, 19]
    np.testing.assert_allclose(frame.cell_volume_um3, [9., 9.])
    assert frame.file_name.tolist() == ['saved.tif', 'saved.tif']


@pytest.mark.parametrize('snapshot, reason', [
    ({'axes': 'CYX'}, 'exactly three spatial axes'),
    ({'unit': 'furlong'}, 'Unsupported volume measurement unit: furlong'),
    ({'channels': tuple(_image() for _ in range(17))}, 'up to 16 channels'),
])
def test_a_snapshot_that_cannot_be_measured_names_why_and_writes_nothing(
        tmp_path, snapshot, reason):
    with pytest.raises(ValueError, match=reason):
        measure.measure_volume_rows([_row(tmp_path, **snapshot)])
    assert not (tmp_path / 'measurements.db').exists()


def test_a_missing_source_is_refused_by_name(tmp_path):
    row = _row(tmp_path)
    row['source_path'] = str(tmp_path / 'gone.tif')
    with pytest.raises(ValueError, match='does not exist: .*gone.tif'):
        measure.measure_volume_rows([row])
    assert not (tmp_path / 'measurements.db').exists()


def test_a_multichannel_mask_file_is_not_taken_for_one_label_volume(tmp_path):
    row = _row(tmp_path)
    row.pop('snapshot')
    stacked = np.stack([_labels(), _labels()])
    row['mask_path'] = _write(tmp_path / 'two_masks.tif', stacked, axes='CZYX')
    with pytest.raises(ValueError, match='single 3D label volume'):
        measure.measure_volume_rows([row])
    assert not (tmp_path / 'measurements.db').exists()


@pytest.mark.parametrize('object_type, rows, reason', [
    ('banana', None, 'supported object type'),
    ('cell', (), 'Add an image set'),
])
def test_the_entry_point_refuses_an_unknown_role_or_nothing_to_measure(
        tmp_path, object_type, rows, reason):
    rows = [_row(tmp_path)] if rows is None else rows
    with pytest.raises(ValueError, match=reason):
        measure.measure_volume_rows(rows, object_type=object_type)
    assert not (tmp_path / 'measurements.db').exists()


def test_a_progress_display_that_fails_does_not_change_the_saved_result(tmp_path):
    calls = []

    def closed_window(message):
        calls.append(message)
        raise RuntimeError('Internal C++ object already deleted')

    result = measure.measure_volume_rows([_row(tmp_path)], progress=closed_window)

    assert result['objects'] == 2
    assert len(calls) == 2
    assert _read(result['db_paths'][0]).object_label.tolist() == [4, 19]


def test_an_existing_object_table_without_field_identity_is_preserved(tmp_path):
    db = tmp_path / 'measurements.db'
    old = pd.DataFrame({'measurement_ndim': [3], 'measurement_units': ['um'],
                        'cell_area': [5.]})
    tabular.write_database(old, db, 'cell')
    with pytest.raises(ValueError, match='no field identity'):
        measure.measure_volume_rows([_row(tmp_path)])
    pd.testing.assert_frame_equal(_read(db), old)


def test_rows_of_the_same_field_without_an_owner_are_never_replaced(tmp_path):
    row = _row(tmp_path)
    _target, frame = measure._volume_measure_frame(row, 'cell')
    db = tmp_path / 'measurements.db'
    foreign = pd.DataFrame({'prcf': [frame.prcf.iloc[0]], 'measurement_ndim': [3],
                            'measurement_units': ['um'], 'cell_area': [5.]})
    tabular.write_database(foreign, db, 'cell')
    with pytest.raises(ValueError, match='belongs to other measurements'):
        measure.measure_volume_rows([row])
    pd.testing.assert_frame_equal(_read(db), foreign)
