"""F548 voxel helpers: ND2/LIF stand-ins, unreadable OME-TIFFs, src forms, measure snapshots."""

from types import SimpleNamespace

import numpy as np
import pytest

from spacr import core


def test_lif_coordinates_in_metres_become_micrometres():
    image = SimpleNamespace(coords={'X': np.arange(4) * 2e-7, 'Y': np.arange(3) * 2e-7,
                                    'Z': np.array([0.0, 5e-7, 1e-6])})
    assert core._watch_lif_voxel(image, {'Z': 3}) == pytest.approx({'xy': 0.2, 'z': 0.5})
    assert core._watch_lif_voxel(image, {'Z': 1}) == pytest.approx({'xy': 0.2, 'z': None})
    descending = SimpleNamespace(coords={'X': np.array([4e-7, 2e-7]),
                                         'Y': np.array([0.0, 2e-7]), 'Z': np.array([1e-6])})
    assert core._watch_lif_voxel(descending, {'Z': 2}) == pytest.approx({'xy': 0.2, 'z': None})
    missing_y = SimpleNamespace(coords={'X': np.arange(4) * 2e-7})
    assert core._watch_lif_voxel(missing_y, {'Z': 2}) == {'xy': None, 'z': None}


def test_unreadable_tiff_declares_nothing(tmp_path):
    broken = tmp_path / 'broken.ome.tif'
    broken.write_bytes(b'not a tiff at all')
    assert core._watch_tiff_voxel(str(broken)) == {'xy': None, 'z': None}
    assert core._watch_tiff_voxel(str(tmp_path / 'missing.ome.tif')) == {'xy': None, 'z': None}


def test_vendor_recipe_leaves_several_or_invalid_sources_alone(tmp_path):
    base = {'z_stack': True, 'z_segmentation_mode': None, 'z_axis': None}
    # 2026-10-11: since 9ffa83cba (#138/#139) normalize_src_path drops a
    # repeated path, so [tmp_path, tmp_path] is one source; two sources
    # must be two different folders.
    (tmp_path / 'a').mkdir()
    (tmp_path / 'b').mkdir()
    two = {**base, 'src': [str(tmp_path / 'a'), str(tmp_path / 'b')]}
    assert core._watch_vendor_recipe(two) == (two, {})
    for bad in (123, 4.5, {'folder': str(tmp_path)}):
        settings = {**base, 'src': bad}
        result, filled = core._watch_vendor_recipe(settings)
        assert result is settings and filled == {}
    nested = {**base, 'src': [[str(tmp_path)]]}
    assert core._watch_vendor_recipe(nested) == (nested, {})
