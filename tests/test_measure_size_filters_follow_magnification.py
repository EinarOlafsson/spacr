"""Item 800: Measure's size floors follow the pixel size or magnification.

cell_min_size (8000 px), nucleus_min_size (2000 px) and pathogen_min_size
(500 px) were fixed pixel counts for 40-60x. At 20x they removed nearly every
BBBC021 cell (CellProfiler median cell area ~2,800 px), silently.
"""

import numpy as np
import pytest

from spacr import measure as M
from spacr.utils import _filter_object
from spacr.settings import ALPHA_FEATURES, _alpha_names, get_measure_crop_settings


def _floors(**given):
    s = get_measure_crop_settings(dict(given))
    return s['cell_min_size'], s['nucleus_min_size'], s['pathogen_min_size']


def test_without_a_scale_the_floors_are_the_fixed_40_to_60x_defaults():
    assert _floors() == (8000, 2000, 500)
    assert get_measure_crop_settings({})['pixel_size_um'] is None


def test_magnification_20_derives_floors_that_keep_bbbc021_cells():
    # A disc one fifth of Mask's expected diameter: 120, 60 and 20 px at 20x.
    assert _floors(magnification=20) == (452, 113, 13)
    assert _floors(magnification=40) == (804, 177, 50)


def test_the_pixel_size_wins_over_magnification_as_for_mask():
    # 18, 10 and 6.5 um over 0.325 um/px: 55, 31 and 20 px.
    assert _floors(pixel_size_um=0.325) == (95, 30, 13)
    assert _floors(pixel_size_um=0.325, magnification=60) == (95, 30, 13)


def test_a_typed_floor_is_kept_and_the_form_default_is_replaced():
    assert _floors(magnification=20, cell_min_size=300)[0] == 300
    assert _floors(magnification=20, nucleus_min_size=0)[1] == 0
    assert _floors(magnification=20, pathogen_min_size=None)[2] is None
    # A form holds the fixed default until edited; a CSV holds it as text.
    assert _floors(magnification=20, cell_min_size=8000)[0] == 452
    assert _floors(magnification=20, cell_min_size='8000')[0] == 452


def test_filling_the_defaults_twice_does_not_move_a_derived_floor():
    once = get_measure_crop_settings({'magnification': 20})
    twice = get_measure_crop_settings(dict(once))
    assert twice['cell_min_size'] == once['cell_min_size'] == 452


def test_an_unusable_scale_keeps_the_fixed_defaults():
    assert _floors(pixel_size_um=0) == (8000, 2000, 500)
    assert _floors(magnification='') == (8000, 2000, 500)


def test_the_new_measure_setting_is_alpha_on_measure_only():
    assert ALPHA_FEATURES[800] == {'module_settings': {
        'measure': ('pixel_size_um',), 'external_masks': ('pixel_size_um',)}}
    assert 'pixel_size_um' in _alpha_names('settings', app_key='measure')
    assert 'pixel_size_um' in _alpha_names('settings', app_key='external_masks')
    assert 'pixel_size_um' not in _alpha_names('settings', app_key='mask')
    assert 'pixel_size_um' not in _alpha_names('settings')


def test_a_filter_that_removes_most_objects_of_a_field_warns(capsys):
    s = {'cell_min_size': 8000, 'cell_max_size': None}
    assert M._warn_when_a_size_filter_removes_most('f.npy', 'cell', 10, 1, s)
    out = capsys.readouterr().out
    assert 'removed 9 of 10 cell objects' in out
    assert 'pixel_size_um' in out and 'cell_min_size' in out
    assert not M._warn_when_a_size_filter_removes_most('f.npy', 'cell', 10, 5, s)
    assert not M._warn_when_a_size_filter_removes_most('f.npy', 'cell', 0, 0, s)
    assert capsys.readouterr().out == ''


def test_a_20x_field_keeps_its_cells_with_the_derived_floor():
    """Cells of BBBC021's median area (~2,800 px) survive at magnification 20;
    the fixed 8000 px floor removed all of them."""
    yy, xx = np.mgrid[:400, :400]
    mask = np.zeros((400, 400), dtype=np.int32)
    for label, (cy, cx) in enumerate(
            [(60, 60), (60, 200), (60, 340), (200, 60), (200, 200)], start=1):
        mask[(yy - cy) ** 2 + (xx - cx) ** 2 <= 30 ** 2] = label   # 2,821 px
    floor = get_measure_crop_settings({'magnification': 20})['cell_min_size']
    kept = _filter_object(mask.copy(), floor)
    assert len(np.unique(kept)) - 1 == 5
    fixed = _filter_object(mask.copy(), 8000)
    assert len(np.unique(fixed)) - 1 == 0


@pytest.mark.parametrize('obj', ['cell', 'nucleus', 'pathogen'])
def test_the_tooltips_state_the_real_defaults(obj):
    from spacr.settings import tooltips

    text = tooltips[f'{obj}_min_size']
    legacy = {'cell': 8000, 'nucleus': 2000, 'pathogen': 500}[obj]
    assert f'Default {legacy}' in text
    assert '0 (default)' not in text and 'the default, disables' not in text
    assert 'pixel_size_um' in text
