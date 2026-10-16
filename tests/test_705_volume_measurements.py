"""Native volume rows retain their voxels, units and canonical database keys."""

import json
import sqlite3

import numpy as np
import pandas as pd
import pytest
import tifffile

from spacr import measure, schema, tabular


def _write(path, data, *, axes='ZYX', spacing=(2., .5, .5), unit='um'):
    tifffile.imwrite(path, data, photometric='minisblack',
                     metadata={'axes': axes, 'voxel_spacing': spacing, 'unit': unit})
    return str(path)


def _row(tmp_path, name='image.tif', *, axes='ZYX', spacing=(2., .5, .5), unit='um'):
    labels = np.zeros((10, 8, 8), dtype=np.uint16)
    labels[1:3, 2:5, 2:5] = 4
    labels[6:8, 2:5, 2:5] = 19
    image = np.arange(labels.size, dtype=np.float32).reshape(labels.shape) / 3. + .25
    order = tuple('ZYX'.index(axis) for axis in axes)
    image, labels = image.transpose(order), labels.transpose(order)
    source = _write(tmp_path / name, image, axes=axes, spacing=spacing, unit=unit)
    return {'source_path': source, 'channel_paths': (),
            'snapshot': {'channels': (image,), 'mask': labels, 'axes': axes,
                         'spacing': spacing, 'unit': unit}}


def _read(path, table='cell'):
    return tabular.read_database(str(path), table, canonicalise=False, migrate=False)[0]


def test_current_unsaved_labels_measure_true_3d_geometry_float_channels_and_correlations(tmp_path):
    row = _row(tmp_path)
    snapshot = row['snapshot']
    second = snapshot['channels'][0] * 2. + .125
    extra = _write(tmp_path / 'extra.tif', second)
    row['channel_paths'] = (extra,)
    saved_mask = tmp_path / 'unrelated_saved_mask.tif'
    _write(saved_mask, np.ones_like(snapshot['mask']))
    before = saved_mask.read_bytes()
    row['mask_path'] = str(saved_mask)

    result = measure.measure_volume_rows([row])

    assert result == {'db_paths': (str(tmp_path / 'measurements.db'),), 'images': 1, 'objects': 2}
    frame = _read(result['db_paths'][0]).sort_values('object_label')
    schema.validate_object_table_frame(frame, 'cell')
    assert frame.object_label.tolist() == [4, 19]
    assert frame.measurement_ndim.tolist() == [3, 3]
    assert frame.measurement_units.tolist() == ['um', 'um']
    assert frame.n_z.tolist() == [10, 10]
    np.testing.assert_allclose(frame.cell_volume_voxels, [18, 18])
    np.testing.assert_allclose(frame.cell_volume_um3, [9., 9.])
    np.testing.assert_allclose(frame.cell_area, [9., 9.])
    np.testing.assert_allclose(frame.cell_centroid_z, [3., 13.])
    np.testing.assert_allclose(frame.cell_nearest_neighbor_distance, [10., 10.])
    np.testing.assert_allclose(frame.cell_channel_0_channel_1_Pearson_correlation, [1., 1.])
    for label, actual in zip(frame.object_label, frame.cell_channel_0_mean_intensity):
        expected = snapshot['channels'][0][snapshot['mask'] == label].mean()
        assert actual == pytest.approx(expected)
    assert 'cell_perimeter' not in frame.columns
    assert 'cell_eccentricity' not in frame.columns
    assert json.loads(frame.measurement_channel_sources.iloc[0]) == [
        {'path': row['source_path'], 'channel': 0}, {'path': extra, 'channel': 0}]
    assert saved_mask.read_bytes() == before


def test_axis_permutation_and_nanometres_preserve_physical_distances(tmp_path):
    row = _row(tmp_path, axes='XYZ', spacing=(500., 750., 2000.), unit='nm')
    result = measure.measure_volume_rows([row], object_type='nucleus')
    frame = _read(result['db_paths'][0], 'nucleus').sort_values('object_label')
    np.testing.assert_allclose(frame.nucleus_volume_um3, [13.5, 13.5])
    np.testing.assert_allclose(frame.nucleus_nearest_neighbor_distance, [10., 10.])
    assert frame.measurement_spacing_x.tolist() == [.5, .5]
    assert frame.measurement_spacing_y.tolist() == [.75, .75]
    assert frame.voxel_size_xy_um.isna().all()
    assert frame.voxel_size_z_um.tolist() == [2., 2.]


def test_added_image_set_reads_saved_source_and_explicit_mask(tmp_path):
    row = _row(tmp_path)
    mask_path = _write(tmp_path / 'labels.tif', row['snapshot']['mask'])
    row.pop('snapshot')
    row['mask_path'] = mask_path
    result = measure.measure_volume_rows([row])
    assert _read(result['db_paths'][0]).object_label.tolist() == [4, 19]


def test_append_and_repeat_preserve_other_images_and_existing_tables(tmp_path):
    first = _row(tmp_path, 'one.tif')
    second = _row(tmp_path, 'two.tif')
    existing = tmp_path / 'measurements' / 'measurements.db'
    tabular.write_database(pd.DataFrame({'note': ['keep me']}), existing, 'notes')
    result = measure.measure_volume_rows([first, second])
    assert result['db_paths'] == (str(existing),)
    original = _read(existing)
    assert len(original) == 4
    first['snapshot']['mask'][first['snapshot']['mask'] == 19] = 0
    repeated = measure.measure_volume_rows([first])
    actual = _read(existing)
    assert repeated['objects'] == 1
    assert len(actual) == 3
    pd.testing.assert_frame_equal(
        original[original.file_name == 'two.tif'].reset_index(drop=True),
        actual[actual.file_name == 'two.tif'].reset_index(drop=True))
    assert _read(existing, 'notes').note.tolist() == ['keep me']


def test_later_channel_shape_error_is_rejected_before_any_database_write(tmp_path):
    first = _row(tmp_path, 'one.tif')
    second = _row(tmp_path, 'two.tif')
    second['channel_paths'] = (_write(tmp_path / 'wrong.tif', np.ones((3, 4, 5), dtype=np.float32)),)
    with pytest.raises(ValueError, match='same 3D grid'):
        measure.measure_volume_rows([first, second])
    assert not (tmp_path / 'measurements.db').exists()


def test_incompatible_existing_2d_table_is_not_changed(tmp_path):
    db = tmp_path / 'measurements.db'
    old = pd.DataFrame({'prcf': ['old'], 'value': [12.]})
    tabular.write_database(old, db, 'cell')
    with pytest.raises(ValueError, match='refusing to append'):
        measure.measure_volume_rows([_row(tmp_path)])
    pd.testing.assert_frame_equal(_read(db), old)


def test_failed_second_write_rolls_back_replacement_append_and_column_growth(tmp_path, monkeypatch):
    first = _row(tmp_path, 'one.tif')
    second = _row(tmp_path, 'two.tif')
    result = measure.measure_volume_rows([first])
    db = result['db_paths'][0]
    original = _read(db)
    first['snapshot']['mask'][first['snapshot']['mask'] == 19] = 0
    second['channel_paths'] = (_write(tmp_path / 'extra.tif', second['snapshot']['channels'][0]),)
    real_write = tabular.write_database
    calls = []

    def fail_second(*args, **kwargs):
        calls.append(args[0])
        if len(calls) == 2:
            raise OSError('disk failure')
        return real_write(*args, **kwargs)

    monkeypatch.setattr(tabular, 'write_database', fail_second)
    with pytest.raises(OSError, match='disk failure'):
        measure.measure_volume_rows([first, second])
    assert len(calls) == 2
    pd.testing.assert_frame_equal(_read(db), original)


@pytest.mark.parametrize('problem', ['spacing', 'labels', 'empty', 'duplicate'])
def test_invalid_input_does_not_create_database(tmp_path, problem):
    row = _row(tmp_path)
    rows = [row]
    if problem == 'spacing':
        row['snapshot']['spacing'] = (0., 1., 1.)
    elif problem == 'labels':
        row['snapshot']['mask'] = row['snapshot']['mask'].astype(float) + .5
    elif problem == 'empty':
        row['snapshot']['mask'].fill(0)
    else:
        rows.append(row)
    with pytest.raises(ValueError):
        measure.measure_volume_rows(rows)
    assert not (tmp_path / 'measurements.db').exists()


def test_foreground_without_zero_label_still_has_channel_correlation(tmp_path):
    row = _row(tmp_path, unit='pixel', spacing=(1., 1., 1.))
    row['snapshot']['mask'].fill(6)
    image = row['snapshot']['channels'][0]
    row['snapshot']['channels'] = (image, image * 2.)
    result = measure.measure_volume_rows([row])
    frame = _read(result['db_paths'][0])
    assert frame.object_label.tolist() == [6]
    assert frame.measurement_units.tolist() == ['px']
    np.testing.assert_allclose(frame.cell_channel_0_channel_1_Pearson_correlation, [1.])


def test_foreign_identity_is_never_replaced(tmp_path):
    row = _row(tmp_path)
    result = measure.measure_volume_rows([row])
    db = result['db_paths'][0]
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE cell SET measurement_source='other entry point'")
    before = _read(db)
    with pytest.raises(ValueError, match='other measurements'):
        measure.measure_volume_rows([row])
    pd.testing.assert_frame_equal(_read(db), before)


def test_declared_extra_channel_spacing_cannot_silently_change_geometry(tmp_path):
    row = _row(tmp_path)
    row['channel_paths'] = (_write(tmp_path / 'wrong-spacing.tif',
                                  row['snapshot']['channels'][0], spacing=(3., .5, .5)),)
    with pytest.raises(ValueError, match='voxel spacing does not match'):
        measure.measure_volume_rows([row])
    assert not (tmp_path / 'measurements.db').exists()


def test_declared_mask_spacing_cannot_silently_change_geometry(tmp_path):
    row = _row(tmp_path)
    row['mask_path'] = _write(tmp_path / 'wrong-mask.tif', row['snapshot']['mask'], spacing=(3., .5, .5))
    row.pop('snapshot')
    with pytest.raises(ValueError, match='mask voxel spacing does not match'):
        measure.measure_volume_rows([row])
    assert not (tmp_path / 'measurements.db').exists()


def test_saved_destination_is_reported_even_if_a_later_folder_write_fails(tmp_path, monkeypatch):
    first = _row(tmp_path, 'one.tif')
    other = tmp_path / 'another-folder'
    other.mkdir()
    second = _row(other, 'two.tif')
    real_write = measure._write_volume_measurements
    messages = []

    def write(path, *args):
        if path == str(other / 'measurements.db'):
            raise OSError('second destination is full')
        return real_write(path, *args)

    monkeypatch.setattr(measure, '_write_volume_measurements', write)
    with pytest.raises(OSError, match='second destination'):
        measure.measure_volume_rows([first, second], progress=messages.append)
    assert f'Measurements saved: {tmp_path / "measurements.db"}' in messages
    assert len(_read(tmp_path / 'measurements.db')) == 2
    assert not (other / 'measurements.db').exists()
