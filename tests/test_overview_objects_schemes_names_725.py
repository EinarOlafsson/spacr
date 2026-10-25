"""Item 725 round 4: radial texture, imported objects, well annotation schemes and named groups."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tifffile

from spacr import schema
from spacr.measure import (
    _image_level_features,
    _image_object_features,
    _image_object_options,
    _image_object_record,
    _import_well_annotation_scheme,
    _object_well_table,
    _read_comparison_groups,
    _read_image_annotations,
    _read_well_annotation_scheme,
    _save_comparison_groups,
    _validate_object_masks,
)
from spacr.sp_stats import (
    _compare_image_groups,
    _image_comparison_figure,
    _image_comparison_table,
    _image_group_names,
    _rank_image_differences,
)


def test_radial_texture_follows_local_detail_from_the_centre():
    rng = np.random.default_rng(0)
    rows, columns = np.indices((128, 128))
    distance = np.hypot(rows - 64, columns - 64) / 90.0
    image = 1000 + rng.normal(0, 1, (128, 128)) * 200 * (1 - distance).clip(0)
    record = _image_level_features({0: image}, groups=('radial_texture',), confluency=False)
    bins = [record[f'channel_0_radial_texture_bin_{index}'] for index in range(6)]
    assert bins[0] > bins[-1] and record['channel_0_radial_texture_edge_centre_ratio'] < 0.5
    flat = _image_level_features({0: np.full((32, 32), 5.0)}, groups=('radial_texture',), confluency=False)
    assert np.isnan(flat['channel_0_radial_texture_edge_centre_ratio'])
    assert flat['channel_0_radial_texture_downsample'] == 1


def _labels():
    labels = np.zeros((40, 40), np.uint16)
    labels[2:12, 2:12] = 1
    labels[20:30, 20:26] = 2
    labels[35, 35] = 3
    return labels


def _plate(tmp_path, roles=('cell', 'nucleus')):
    for role in roles:
        stack = tmp_path / 'masks' / f'{role}_mask_stack'
        stack.mkdir(parents=True)
        tifffile.imwrite(stack / 'plate1_A01_1.tif', _labels())
        np.save(stack / 'plate1_B01_1.npy', _labels()[:20])
    return tmp_path


def test_only_mask_generation_outputs_are_accepted(tmp_path):
    index = _validate_object_masks(_plate(tmp_path / 'good'))
    assert set(index['roles']) == {'cell', 'nucleus'} and index['count'] == 4
    assert _validate_object_masks(tmp_path / 'good' / 'masks')['count'] == 4
    assert set(_validate_object_masks(tmp_path / 'good' / 'masks' / 'cell_mask_stack')['roles']) == {'cell'}
    with pytest.raises(ValueError, match='not a folder'):
        _validate_object_masks(tmp_path / 'missing')
    with pytest.raises(ValueError, match='not a folder'):
        _validate_object_masks('')
    (tmp_path / 'plain').mkdir()
    with pytest.raises(ValueError, match='no <object>_mask_stack'):
        _validate_object_masks(tmp_path / 'plain')
    bad = tmp_path / 'bad' / 'masks'
    (bad / 'banana_mask_stack').mkdir(parents=True)
    (bad / 'cell_mask_stack').mkdir()
    (bad / 'nucleus_mask_stack').mkdir()
    (bad / 'nucleus_mask_stack' / 'notes.txt').write_text('x')
    tifffile.imwrite(bad / 'nucleus_mask_stack' / 'rgb.tif', np.zeros((8, 8, 3), np.uint8))
    tifffile.imwrite(bad / 'nucleus_mask_stack' / 'float.tif', np.zeros((8, 8), np.float32))
    (bad / 'nucleus_mask_stack' / 'broken.npy').write_text('not numpy')
    with pytest.raises(ValueError) as refused:
        _validate_object_masks(bad.parent)
    message = str(refused.value)
    for part in ("'banana'", 'holds no mask images', 'notes.txt', 'float.tif', 'broken.npy', 'and 1 more'):
        assert part in message
    (bad / 'nucleus_mask_stack' / 'notes.txt').unlink()
    (bad / 'nucleus_mask_stack' / 'float.tif').unlink()
    (bad / 'nucleus_mask_stack' / 'broken.npy').unlink()
    with pytest.raises(ValueError, match=r'rgb.tif: not a 2-D integer label image \(shape \(8, 8, 3\)'):
        _validate_object_masks(bad / 'nucleus_mask_stack')


def test_object_features_compare_counts_shape_intensity_and_ratios():
    labels = _labels()
    nuclei = np.zeros_like(labels)
    nuclei[4:8, 4:8] = 1
    plane = np.full(labels.shape, 10.0)
    plane[labels == 1] = 110.0
    record = _image_object_features({0: plane}, {'cell': labels, 'nucleus': nuclei}, dict(min_area=2))
    assert record['cell_objects_count'] == 2 and record['nucleus_objects_count'] == 1
    assert record['cell_objects_area_mean'] == pytest.approx(80.0)
    assert record['cell_objects_area_fraction'] == pytest.approx(160 / 1600)
    assert record['cell_objects_per_megapixel'] == pytest.approx(2e6 / 1600)
    assert record['cell_objects_channel_0_intensity_mean'] == pytest.approx(60.0)
    assert record['cell_objects_channel_0_integrated_mean'] == pytest.approx((100 * 110 + 60 * 10) / 2)
    inside = 100 * 110 + 60 * 10
    assert record['cell_objects_channel_0_inside_fraction'] == pytest.approx(inside / plane.sum())
    assert record['cell_objects_channel_0_inside_outside_ratio'] == pytest.approx((inside / 160) / 10.0)
    assert record['nucleus_objects_per_cell'] == pytest.approx(0.5)
    assert record['cell_objects_per_nucleus'] == pytest.approx(2.0)
    assert 0 < record['cell_objects_form_factor_mean'] <= 1.2
    empty = _image_object_features({0: plane}, {'cell': np.zeros_like(labels)},
                                   dict(count=True, morphology=True, intensity=True, ratios=False))
    assert empty['cell_objects_count'] == 0 and np.isnan(empty['cell_objects_area_mean'])
    assert np.isnan(empty['cell_objects_channel_0_intensity_mean'])
    assert np.isnan(empty['cell_objects_channel_0_inside_outside_ratio'])
    lonely = _image_object_features({0: plane}, {'cell': labels, 'nucleus': np.zeros_like(labels)})
    assert np.isnan(lonely['cell_objects_per_nucleus'])
    only_counts = _image_object_features({0: np.zeros_like(plane)}, {'cell': labels},
                                         dict(morphology=False, intensity=True, ratios=False))
    assert 'cell_objects_area_mean' not in only_counts
    assert np.isnan(only_counts['cell_objects_channel_0_inside_fraction'])
    ratios_only = _image_object_features({0: plane}, {'cell': labels, 'nucleus': nuclei},
                                         dict(count=False, morphology=False, intensity=False, min_area=2))
    assert ratios_only == {'cell_objects_per_nucleus': 2.0, 'nucleus_objects_per_cell': 0.5}


def test_object_record_reads_imported_masks_by_field_and_size(tmp_path):
    root = _plate(tmp_path)
    plane = {0: np.ones((40, 40))}
    assert _image_object_record(plane, (), 'plate1', 'A01', '1', {}) == {}
    settings = dict(overview_objects_path=str(root))
    record = _image_object_record(plane, (('0', str(tmp_path / 'x.tif')),), 'plate1', 'A01', '1', settings)
    assert record['object_masks'] == 'cell,nucleus' and record['cell_objects_count'] == 3
    other = _image_object_record(plane, (), 'plate1', 'B01', '1', settings)
    assert other['object_masks'] == '' and 'cell: mask is (20, 40)' in other['object_masks_skipped']
    assert _image_object_record(plane, (), 'plate1', 'C01', '1', settings)['object_masks'] == ''
    by_stem = _image_object_record(plane, (('0', str(tmp_path / 'plate1_A01_1.tif')),), 'p', 'w', 'f', settings)
    assert by_stem['nucleus_objects_count'] == 3
    with pytest.raises(ValueError):
        _image_object_options(dict(overview_objects_unit='plate'))
    assert _image_object_options(dict(overview_objects_count=False, overview_objects_min_area=-4)) == dict(
        count=False, morphology=True, intensity=True, ratios=True, path='', min_area=0, unit='image')


def test_well_table_sums_counts_and_recomputes_ratios():
    frame = pd.DataFrame(dict(plateID='p', wellID=['A01', 'A01', 'B01'],
                              cell_objects_count=[2, 4, 1], nucleus_objects_count=[2, 2, 0],
                              cell_objects_area_mean=[10.0, 20.0, 30.0],
                              nucleus_objects_per_cell=[1.0, 0.5, 0.0]))
    table = _object_well_table(frame).set_index('wellID')
    assert table.loc['A01', 'images'] == 2 and table.loc['A01', 'cell_objects_count'] == 6
    assert table.loc['A01', 'cell_objects_area_mean'] == pytest.approx(15.0)
    assert table.loc['A01', 'nucleus_objects_per_cell'] == pytest.approx(4 / 6)
    assert table.loc['B01', 'nucleus_objects_per_cell'] == pytest.approx(0.0)


def _fields(tmp_path):
    return [dict(plate=plate, well=well, row=well[0], column=int(well[1:]), field='1',
                 files=(('0', str(tmp_path / f'{plate}_{well}_1.tif')),))
            for plate in ('plate1', 'plate2') for well in ('A01', 'A02', 'B01')]


def test_scheme_list_annotates_every_image_of_its_wells(tmp_path):
    scheme = tmp_path / 'scheme.csv'
    scheme.write_text('plate,well,cell_type,replicate\nplate1,a1,HeLa,1\nplate1,A2,HFF,\n'
                      'plate1,C07,HeLa,2\nplate9,A01,HeLa,01\n')
    db = tmp_path / 'measurements.db'
    report = _import_well_annotation_scheme(db, scheme, _fields(tmp_path), source=tmp_path)
    assert report['columns'] == ['cell_type', 'replicate'] and report['wells'] == 2
    assert sorted(report['unmatched']) == [('plate1', 'C07'), ('plate9', 'A01')]
    table = _read_image_annotations(db).set_index(['plateID', 'wellID'])
    assert table.loc[('plate1', 'A01'), 'cell_type'] == 'HeLa' and table.loc[('plate1', 'A01'), 'replicate'] == '1'
    assert table.loc[('plate1', 'A02'), 'cell_type'] == 'HFF' and pd.isna(table.loc[('plate1', 'A02'), 'replicate'])
    assert ('plate2', 'A01') not in table.index


def test_scheme_without_plate_or_as_plate_map_applies_to_every_plate(tmp_path):
    workbook = tmp_path / 'rows.xlsx'
    pd.DataFrame(dict(Row=['A', 'B'], Column=[2, 1], drug=['DMSO', 'Taxol'])).to_excel(workbook, index=False)
    db = tmp_path / 'measurements.db'
    assert _import_well_annotation_scheme(db, workbook, _fields(tmp_path))['wells'] == 4
    grid = tmp_path / 'genotype.csv'
    pd.DataFrame({'': ['A', 'B'], '1': ['WT', 'KO'], '2': ['KO', None]}).to_csv(grid, index=False)
    assert list(_read_well_annotation_scheme(grid).columns) == ['plate', 'well', 'genotype']
    report = _import_well_annotation_scheme(db, grid, _fields(tmp_path), name='line')
    assert report['columns'] == ['line'] and report['wells'] == 6 and report['unmatched'] == []
    table = _read_image_annotations(db).set_index(['plateID', 'wellID'])
    assert table.loc[('plate2', 'B01'), 'drug'] == 'Taxol' and table.loc[('plate2', 'B01'), 'line'] == 'KO'


def test_bad_schemes_are_refused_before_writing(tmp_path):
    db = tmp_path / 'measurements.db'
    nothing = tmp_path / 'nothing.csv'
    pd.DataFrame(dict(sample=['x'], drug=['y'])).to_csv(nothing, index=False)
    with pytest.raises(ValueError, match='needs a well column'):
        _import_well_annotation_scheme(db, nothing, _fields(tmp_path))
    wrong = tmp_path / 'wrong.csv'
    pd.DataFrame(dict(well=['Z'], drug=['y'])).to_csv(wrong, index=False)
    with pytest.raises(ValueError, match='not a well name'):
        _import_well_annotation_scheme(db, wrong, _fields(tmp_path))
    reserved = tmp_path / 'reserved.csv'
    pd.DataFrame(dict(well=['A01'], file=['y'])).to_csv(reserved, index=False)
    with pytest.raises(ValueError):
        _import_well_annotation_scheme(db, reserved, _fields(tmp_path))
    assert not db.exists()


def _frame():
    rng = np.random.default_rng(1)
    return pd.DataFrame([dict(plate='p', well=well, field=str(index), cell_type=cell, status='ok',
                              channel_0_intensity_mean=base + rng.normal(0, 1))
                         for well, cell, base in (('A01', 'HeLa', 100.0), ('B01', 'HFF', 120.0))
                         for index in range(5)])


def test_named_groups_label_tables_and_figures_and_are_saved(tmp_path):
    groups = [{'cell_type': 'HeLa'}, {'cell_type': 'HFF'}]
    assert _image_group_names(groups) == ['HeLa', 'HFF']
    assert _image_group_names(groups, ['Control', ' ']) == ['Control', 'HFF']
    with pytest.raises(ValueError, match='2 groups'):
        _image_group_names(groups, ['one'])
    with pytest.raises(ValueError, match='share the label'):
        _image_group_names(groups, ['same', 'same'])
    names = ['Control cells', 'Fibroblasts']
    result = _compare_image_groups(_frame(), ['channel_0_intensity_mean'], [], groups=groups, group_names=names)
    assert list(result['summary']['group']) == names
    table = _image_comparison_table(result)
    assert (table.iloc[0]['group_a'], table.iloc[0]['group_b']) == tuple(names)
    ranked = _rank_image_differences(_frame(), groups=groups, group_names=names)
    assert set(ranked['group_a']) | set(ranked['group_b']) == set(names)
    figure, _ = _image_comparison_figure(_frame(), 'channel_0_intensity_mean', [], groups=groups,
                                         group_names=names)
    assert [label.get_text() for label in figure.axes[0].get_xticklabels()] == names
    db = tmp_path / 'measurements.db'
    assert _read_comparison_groups(db) == ([], [])
    assert _save_comparison_groups(db, groups, ['Control cells', None]) == ['Control cells', 'HFF']
    _save_comparison_groups(db, groups[:1], ['Only'], set_name='other')
    assert _read_comparison_groups(db) == (groups, ['Control cells', 'HFF'])
    assert _read_comparison_groups(db, set_name='other') == (groups[:1], ['Only'])
    _save_comparison_groups(db, groups[::-1], names)
    assert _read_comparison_groups(db) == (groups[::-1], names)
    assert schema.table_key_columns('image_comparison_groups') == ('set_name', 'position')
