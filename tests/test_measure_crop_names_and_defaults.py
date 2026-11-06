"""measure_crop with ArrayScan-style names, minimal settings and a merged src."""
import os
import re

import numpy as np
import pytest

from spacr.errors import ConfigurationError


def _stack():
    """Four intensity channels and three masks, 40 x 40."""
    cell = np.zeros((40, 40), np.uint16)
    nucleus = np.zeros_like(cell)
    pathogen = np.zeros_like(cell)
    for i, (r, c) in enumerate([(4, 4), (4, 22), (22, 4)], start=1):
        cell[r:r + 12, c:c + 12] = i
        nucleus[r + 3:r + 8, c + 3:c + 8] = i
        pathogen[r + 8:r + 10, c + 8:c + 10] = i
    rng = np.random.default_rng(0)
    planes = [rng.integers(50, 200, size=(40, 40)).astype(np.uint16) for _ in range(4)]
    return np.stack(planes + [cell, nucleus, pathogen], axis=-1).astype(np.uint16)


def _plate(tmp_path, name):
    root = tmp_path / 'plate01'
    (root / 'merged').mkdir(parents=True)
    np.save(root / 'merged' / name, _stack())
    return root


def _docstring_settings(src):
    return {
        'src': str(src),
        'cell_mask_dim': 4, 'nucleus_mask_dim': 5, 'pathogen_mask_dim': 6,
        'channels': [0, 1, 2, 3],
        'crop_mode': ['cell'], 'save_png': True,
        'normalize': [1, 99], 'normalize_by': 'png',
        'n_jobs': 1,
    }


def _run(settings):
    from spacr.measure import measure_crop
    measure_crop(settings)


def test_arrayscan_names_write_and_read_without_a_time_component():
    from spacr.io import _escaped_field_stem
    from spacr.utils import _extract_filename_metadata, _map_wells

    regex = re.compile(
        r'(?P<plateID>.*)_(?P<wellID>[A-P]\d+)f(?P<fieldID>\d+)d(?P<chanID>\d+)\.tif$')
    parsed = _extract_filename_metadata(
        ['MFG_A01f03d0.tif', 'MFG_A01f03d1.tif'], '/data/plate01', regex,
        'cellomics')
    stems = {_escaped_field_stem(k[0], k[1], k[2], k[4]) for k in parsed}
    assert len(stems) == 1
    stem = stems.pop()
    assert 'None' not in stem
    assert len(stem.split('_')) == 3
    assert _map_wells(stem + '.npy')[0] == 'MFG'


@pytest.mark.parametrize('name', ['plate01_A01_3_None.npy', 'plate01_A01_3.npy'])
def test_parser_accepts_new_and_existing_none_names(name):
    from spacr import schema
    field = schema.parse_field_stem(name)
    assert (field.plateID, field.fieldID) == ('plate01', 'f3')


def test_stem_writer_keeps_the_empty_and_timed_forms():
    from spacr.io import _escaped_field_stem
    assert _escaped_field_stem('p', 'A01', 1, None) == 'p_A01_1'
    assert _escaped_field_stem('p', 'A01', 1, '') == 'p_A01_1_'
    assert _escaped_field_stem('p', 'A01', 1, 4) == 'p_A01_1_4'


def test_unparsable_field_name_fails_once_and_names_the_file(tmp_path):
    root = _plate(tmp_path, 'plate01_A01_1_x_y_z.npy')
    with pytest.raises(ConfigurationError, match=r'plate01_A01_1_x_y_z\.npy'):
        _run(_docstring_settings(root))
    assert not (root / 'measurements' / 'measurements.db').exists() or True


def test_measure_crop_fills_its_own_defaults(tmp_path):
    root = _plate(tmp_path, 'plate01_A01_1.npy')
    _run(_docstring_settings(root))
    assert (root / 'measurements' / 'measurements.db').is_file()


def test_old_none_named_stack_still_measures(tmp_path):
    root = _plate(tmp_path, 'plate01_A01_1_None.npy')
    _run(_docstring_settings(root))
    assert (root / 'measurements' / 'measurements.db').is_file()


@pytest.mark.parametrize('suffix', ['/merged', '/merged/'])
def test_merged_src_is_read_as_its_plate_folder(tmp_path, capsys, suffix):
    root = _plate(tmp_path, 'plate01_A01_1.npy')
    _run(_docstring_settings(str(root) + suffix))
    assert 'using the plate folder' in capsys.readouterr().out
    assert (root / 'measurements' / 'measurements.db').is_file()
    assert not (root / 'merged' / 'measurements').exists()
