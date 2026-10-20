"""F548 voxel helpers: ND2/LIF stand-ins, unreadable OME-TIFFs, src forms, measure snapshots."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from spacr import core
from tests.test_watch_folder_and_analyse import MASK, MEASURE
from tests.test_watch_nested_f548 import _fast


def _ledger(folder):
    return json.loads((folder / 'spacr_watch' / 'watch_ledger.json').read_text(encoding="utf-8"))['fields']


def _ome_volume(path, **physical):
    volume = np.full((3, 2, 32, 32), 40, np.uint16)
    volume[:, :, 5:17, 5:17] = 500
    tifffile.imwrite(path, volume, ome=True, metadata={'axes': 'ZCYX', **physical})


def _bare_z_watch(folder):
    """A Z watch of vendor stacks that sets only z_stack and leaves voxel sizes unset."""
    return {**MASK, 'z_stack': True, 'save_original_images': False, **_fast(folder)}


class _Nd2:
    """An ``nd2.ND2File`` stand-in exposing only what the voxel reader uses."""

    def __init__(self, calibrated, voxel):
        volume = SimpleNamespace(axesCalibrated=calibrated)
        self.metadata = SimpleNamespace(channels=[SimpleNamespace(volume=volume)])
        self._voxel = voxel

    def voxel_size(self):
        return self._voxel


def test_nd2_calibrated_axes_give_pixel_size_and_z_step():
    handle = _Nd2((True, True, True), (0.13, 0.13, 0.5))
    assert core._watch_nd2_voxel(handle, {'Z': 4}) == pytest.approx({'xy': 0.13, 'z': 0.5})
    assert core._watch_nd2_voxel(handle, {'Z': 1}) == pytest.approx({'xy': 0.13, 'z': None})
    assert core._watch_nd2_voxel(handle, {}) == pytest.approx({'xy': 0.13, 'z': None})


def test_nd2_uncalibrated_axes_count_as_undeclared():
    assert core._watch_nd2_voxel(_Nd2((False, True, True), (0.13, 0.13, 0.5)),
                                 {'Z': 4}) == pytest.approx({'xy': None, 'z': 0.5})
    assert core._watch_nd2_voxel(_Nd2((True, True, False), (0.13, 0.13, 0.5)),
                                 {'Z': 4}) == pytest.approx({'xy': 0.13, 'z': None})
    assert core._watch_nd2_voxel(_Nd2((True, True, True), (0.1, 0.3, 0.5)),
                                 {'Z': 4}) == pytest.approx({'xy': None, 'z': 0.5})


def test_nd2_without_volume_metadata_declares_nothing():
    handle = _Nd2((True, True, True), (0.13, 0.13, 0.5))
    handle.metadata = SimpleNamespace(channels=[])
    assert core._watch_nd2_voxel(handle, {'Z': 4}) == {'z': None, 'xy': None}


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


def test_vendor_recipe_reads_one_folder_given_as_a_list(tmp_path):
    base = {'z_stack': True, 'z_segmentation_mode': None, 'z_axis': None}
    expected = {'z_segmentation_mode': 'volumetric', 'z_axis': 0}
    settings, filled = core._watch_vendor_recipe({**base, 'src': [str(tmp_path)]})
    assert filled == expected and settings['z_axis'] == 0
    assert settings['src'] == [str(tmp_path)]
    assert core._watch_vendor_recipe({**base, 'src': repr([str(tmp_path)])})[1] == expected


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


def test_vendor_measure_without_voxel_size_waits_for_the_file(tmp_path):
    _ome_volume(tmp_path / 'plate-1.ome.tif', PhysicalSizeX=0.1, PhysicalSizeY=0.1,
                PhysicalSizeZ=0.4)
    measure = {key: value for key, value in MEASURE.items() if key != 'anisotropy'}
    recipe = tmp_path / 'measure.json'
    recipe.write_text(json.dumps(measure), encoding="utf-8")
    settings = dict(_bare_z_watch(tmp_path), watch_pipeline='mask_measure',
                    watch_measure_settings=str(recipe))
    received = []

    def analyse(_field_dir, field_settings):
        received.append(field_settings)
        raise RuntimeError('stop after staging')

    result = core._watch_folder_and_analyse(settings, analyse)
    assert result['failed'] == ['plate1'] and len(received) == 1
    snapshot = received[0]['watch_measure_snapshot']
    assert (snapshot['voxel_size_z_um'], snapshot['voxel_size_xy_um']) == pytest.approx(
        (0.4, 0.1))
    assert _ledger(tmp_path)['plate1']['voxel_size']['source'] == {
        'z': 'OME-TIFF metadata', 'xy': 'OME-TIFF metadata'}
