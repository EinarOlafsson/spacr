"""F548 follow-ups from the workstation ND2 run: vendor voxel sizes and recipe defaults.

A Z or T-by-Z watch of vendor stacks reads each field's pixel size and Z step
from its ND2, CZI, LIF or OME-TIFF metadata when the settings leave
``voxel_size_*`` unset (a value in the settings always wins), fills the
volumetric recipe keys the mode needs, and still analyses one field at a time.
"""

import json
import os
import shutil
import threading
import time

import numpy as np
import pytest
import tifffile

from spacr import core
from tests.conftest import MISSING_CHANNEL_AXIS, check_cellpose_eval_call
from tests.test_watch_folder_and_analyse import MASK
from tests.test_watch_native_z_t1_f548 import _mask as _volume_mask
from tests.test_watch_nested_f548 import _fast
from tests.test_watch_vendor_formats_f548 import (DATA, _batch, _modern_czifile, _public,
                                                  _same_merged, _series, _series_settings)

pytest_plugins = ['tests.test_cov_object_masks_sam']

HEADER_XY = 0.12863494437945


def _ledger(folder):
    return json.loads((folder / 'spacr_watch' / 'watch_ledger.json').read_text(encoding="utf-8"))['fields']


def _voxel(path):
    fields = core._watch_container_fields(str(path), path.name)
    assert len({json.dumps(field['voxel_um'], sort_keys=True) for field in fields}) == 1
    return fields[0]['voxel_um']


def test_real_nd2_declares_pixel_size_and_z_step():
    pytest.importorskip('nd2')
    voxel = _voxel(_public('header_test2.nd2'))
    assert voxel['z'] == pytest.approx(0.5) and voxel['xy'] == pytest.approx(HEADER_XY)
    voxel = _voxel(_public('MeOh_high_fluo_003.nd2'))
    assert voxel['z'] is None and voxel['xy'] == pytest.approx(0.37)


def test_lif_coordinates_give_pixel_size_and_z_step():
    pytest.importorskip('liffile')
    voxel = _voxel(DATA / 'leica_lif' / 'PR2729_frameOrderCombinedScanTypes.lif')
    assert voxel['xy'] == pytest.approx(3.968254, rel=1e-5)
    assert voxel['z'] == pytest.approx(6.227615, rel=1e-5)
    voxel = _voxel(_public('FRAP.lif'))
    assert voxel['z'] is None and voxel['xy'] == pytest.approx(0.2064025, rel=1e-5)


def test_czi_scaling_is_read_in_metres_and_zero_means_undeclared():
    _modern_czifile()
    assert _voxel(DATA / 'zeiss_czi' / 'two_scenes_tzc.czi') == {'z': None, 'xy': None}
    text = ('<ImageDocument><Metadata><Scaling><Items>'
            '<Distance Id="X"><Value>1.0E-07</Value><DefaultUnitFormat>µm'
            '</DefaultUnitFormat></Distance>'
            '<Distance Id="Y"><Value>1.0E-07</Value></Distance>'
            '<Distance Id="Z"><Value>3.0E-07</Value></Distance>'
            '</Items></Scaling></Metadata></ImageDocument>')
    voxel = core._watch_czi_voxel(text, 4)
    assert voxel['xy'] == pytest.approx(0.1) and voxel['z'] == pytest.approx(0.3)
    assert core._watch_czi_voxel(text, 1)['z'] is None
    assert core._watch_czi_voxel(text.replace('Id="Y"><Value>1.0E-07',
                                              'Id="Y"><Value>2.0E-07'), 4)['xy'] is None


def test_ome_tiff_physical_sizes_and_units(tmp_path):
    stack = np.ones((3, 2, 8, 8), np.uint16)
    tifffile.imwrite(tmp_path / 'um.ome.tif', stack, ome=True,
                     metadata={'axes': 'ZCYX', 'PhysicalSizeX': 0.2, 'PhysicalSizeY': 0.2,
                               'PhysicalSizeZ': 0.5})
    assert core._watch_tiff_voxel(str(tmp_path / 'um.ome.tif')) == pytest.approx(
        {'xy': 0.2, 'z': 0.5})
    tifffile.imwrite(tmp_path / 'nm.ome.tif', stack, ome=True,
                     metadata={'axes': 'ZCYX', 'PhysicalSizeX': 65, 'PhysicalSizeXUnit': 'nm',
                               'PhysicalSizeY': 65, 'PhysicalSizeYUnit': 'nm',
                               'PhysicalSizeZ': 200, 'PhysicalSizeZUnit': 'nm'})
    assert core._watch_tiff_voxel(str(tmp_path / 'nm.ome.tif')) == pytest.approx(
        {'xy': 0.065, 'z': 0.2})
    tifffile.imwrite(tmp_path / 'plain.tif', stack, imagej=True, metadata={'axes': 'ZCYX'})
    assert core._watch_tiff_voxel(str(tmp_path / 'plain.tif')) == {'xy': None, 'z': None}


def test_a_value_in_the_settings_always_wins():
    settings = {'voxel_size_z_um': 1.0, 'voxel_size_xy_um': None,
                'watch_measure_snapshot': {'voxel_size_z_um': None, 'voxel_size_xy_um': 0.3}}
    resolved, record = core._watch_field_voxel(settings, {'z': 0.5, 'xy': 0.1}, 'ND2 metadata')
    assert (resolved['voxel_size_z_um'], resolved['voxel_size_xy_um']) == (1.0, 0.1)
    assert record == {'z_um': 1.0, 'xy_um': 0.1,
                      'source': {'z': 'settings', 'xy': 'ND2 metadata'}}
    assert resolved['watch_measure_snapshot'] == {'voxel_size_z_um': 1.0,
                                                  'voxel_size_xy_um': 0.3}
    assert settings['voxel_size_xy_um'] is None
    assert settings['watch_measure_snapshot']['voxel_size_z_um'] is None
    resolved, record = core._watch_field_voxel(
        {'anisotropy': 4.0, 'watch_measure_snapshot': {'anisotropy': 2.0}}, None, 'LIF metadata')
    assert record['source'] == {'z': 'not declared', 'xy': 'not declared'}
    assert 'voxel_size_z_um' not in resolved['watch_measure_snapshot']
    core._watch_check_field_voxel({'anisotropy': 4.0})
    with pytest.raises(ValueError, match='declares no voxel size'):
        core._watch_check_field_voxel({'voxel_size_xy_um': 0.1})


def test_vendor_z_and_t_watches_default_their_volumetric_recipe(tmp_path):
    base = {'src': str(tmp_path), 'z_stack': True, 'z_segmentation_mode': 'project',
            'z_axis': None}
    settings, filled = core._watch_vendor_recipe(base)
    assert filled == {'z_segmentation_mode': 'volumetric', 'z_axis': 0}
    assert settings == {**base, **filled} and base['z_segmentation_mode'] == 'project'
    assert core._watch_vendor_recipe({**base, 'z_segmentation_mode': 'stitch',
                                      'z_axis': 2})[1] == {}
    _, filled = core._watch_vendor_recipe({**base, 't_stack': True, 't_axis_order': None})
    assert filled == {'z_segmentation_mode': 'volumetric', 't_axis_order': 'TZYX'}
    assert core._watch_vendor_recipe({**base, 't_stack': True, 't_axis_order': 'ZTYX'})[1] == {
        'z_segmentation_mode': 'volumetric'}
    assert core._watch_vendor_recipe({**base, 'z_stack': False})[1] == {}
    (tmp_path / 'conversion_map.csv').write_text('target,source\n', encoding="utf-8")
    assert core._watch_vendor_recipe(base)[1] == {}


def _anisotropy_model(monkeypatch):
    """A volumetric Cellpose double that records the anisotropy each call gets."""
    import spacr.object as spacr_object

    seen = []

    def volume_eval(self, x, batch_size=8, resample=True, channels=None,
                    channel_axis=MISSING_CHANNEL_AXIS, z_axis=None,
                    normalize=True, invert=False, rescale=None, diameter=None,
                    flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False,
                    anisotropy=None, flow3D_smooth=0, stitch_threshold=0.0,
                    min_size=15, max_size_fraction=0.4, niter=None,
                    augment=False, tile_overlap=0.1, bsize=256,
                    compute_masks=True, progress=None):
        converted = check_cellpose_eval_call(x, channel_axis, z_axis=z_axis, do_3D=do_3D)
        assert do_3D
        seen.append(anisotropy)
        labels = np.zeros(converted[0].shape[:-1], np.uint16)
        labels[:, 5:17, 5:17] = 1
        return labels, None, None

    monkeypatch.setattr(spacr_object.cp_models.CellposeModel, 'eval', volume_eval)
    return seen


def test_real_nd2_t_by_z_runs_from_its_own_voxel_size_and_defaults(tmp_path, fake_model,
                                                                   monkeypatch, capsys):
    """The ND2 alone supplies the voxel size and the recipe defaults; the result
    equals a batch run given the same voxel size by hand."""
    nd2 = pytest.importorskip('nd2')
    source = _public('header_test2.nd2')
    stack = np.asarray(nd2.imread(str(source))).reshape(4, 5, 1, 520, 696)
    seen = _anisotropy_model(monkeypatch)
    batch = _batch(tmp_path, {('headertest2', 'A01', 1): stack}, 'series')
    core.preprocess_generate_masks(_series_settings(
        batch, cell_channel=None, channels=[0], anisotropy=None,
        voxel_size_z_um=0.5, voxel_size_xy_um=HEADER_XY))
    batch_seen = list(seen)
    assert batch_seen and all(value == pytest.approx(0.5 / HEADER_XY) for value in batch_seen)
    watched = tmp_path / 'watched'
    watched.mkdir()
    shutil.copy2(source, watched / source.name)
    capsys.readouterr()
    result = core._watch_folder_and_analyse(_series(
        watched, channels=[0], anisotropy=None, z_segmentation_mode='project',
        t_axis_order=None))
    assert result['done'] == ['headertest2_A01_1'] and not result['failed']
    assert seen[len(batch_seen):] == batch_seen
    _same_merged(batch, watched, 4)
    record = _ledger(watched)['headertest2_A01_1']['voxel_size']
    assert record['z_um'] == pytest.approx(0.5)
    assert record['xy_um'] == pytest.approx(HEADER_XY)
    assert record['source'] == {'z': 'ND2 metadata', 'xy': 'ND2 metadata'}
    log = capsys.readouterr().out
    assert "z_segmentation_mode='volumetric', t_axis_order='TZYX'" in log
    assert 'headertest2_A01_1 voxel size z 0.5 um (ND2 metadata)' in log


def test_real_nd2_user_voxel_size_wins_per_axis(tmp_path):
    pytest.importorskip('nd2')
    source = _public('header_test2.nd2')
    shutil.copy2(source, tmp_path / source.name)
    received = []

    def analyse(_field_dir, settings):
        received.append(settings)
        raise RuntimeError('stop after staging')

    result = core._watch_folder_and_analyse(
        _series(tmp_path, channels=[0], anisotropy=None, voxel_size_z_um=1.0), analyse)
    assert result['failed'] == ['headertest2_A01_1']
    assert received[0]['voxel_size_z_um'] == 1.0
    assert received[0]['voxel_size_xy_um'] == pytest.approx(HEADER_XY)
    assert _ledger(tmp_path)['headertest2_A01_1']['voxel_size']['source'] == {
        'z': 'settings', 'xy': 'ND2 metadata'}


def _ome_volume(path, **physical):
    volume = np.full((3, 2, 32, 32), 40, np.uint16)
    volume[:, :, 5:17, 5:17] = 500
    tifffile.imwrite(path, volume, ome=True, metadata={'axes': 'ZCYX', **physical})


def _bare_z_watch(folder):
    """A Z watch that sets only z_stack, as the workstation run had to avoid."""
    settings = {**_volume_mask(folder), **_fast(folder)}
    for key in ('z_segmentation_mode', 'z_axis', 'anisotropy'):
        settings.pop(key)
    return settings


def test_ome_tiff_z_watch_needs_only_z_stack(tmp_path):
    _ome_volume(tmp_path / 'plate-1.ome.tif', PhysicalSizeX=0.1, PhysicalSizeY=0.1,
                PhysicalSizeZ=0.4)
    received = []

    def analyse(_field_dir, settings):
        received.append(settings)
        raise RuntimeError('stop after staging')

    result = core._watch_folder_and_analyse(_bare_z_watch(tmp_path), analyse)
    assert result['failed'] == ['plate1']
    field = received[0]
    assert (field['z_segmentation_mode'], field['z_axis']) == ('volumetric', 0)
    assert (field['voxel_size_z_um'], field['voxel_size_xy_um']) == pytest.approx((0.4, 0.1))
    assert _ledger(tmp_path)['plate1']['voxel_size']['source'] == {
        'z': 'OME-TIFF metadata', 'xy': 'OME-TIFF metadata'}


def test_a_stack_without_a_declared_voxel_size_fails_before_analysis(tmp_path):
    _ome_volume(tmp_path / 'plate-1.ome.tif')
    received = []
    result = core._watch_folder_and_analyse(_bare_z_watch(tmp_path),
                                            lambda *args: received.append(args))
    assert result['failed'] == ['plate1'] and not received
    assert 'declares no voxel size' in _ledger(tmp_path)['plate1']['error']


def test_ready_fields_are_analysed_one_at_a_time(tmp_path):
    for name in ('a', 'b', 'c'):
        _ome_volume(tmp_path / f'plate{name}.ome.tif', PhysicalSizeX=0.1,
                    PhysicalSizeY=0.1, PhysicalSizeZ=0.4)
    active, spans, lock = [], [], threading.Lock()

    def analyse(field_dir, _settings):
        with lock:
            active.append(field_dir)
            assert len(active) == 1
        start = time.monotonic()
        time.sleep(0.05)
        with lock:
            active.remove(field_dir)
        spans.append((start, time.monotonic()))
        raise RuntimeError('stop after timing')

    result = core._watch_folder_and_analyse(_bare_z_watch(tmp_path), analyse)
    assert result['failed'] == ['platea', 'plateb', 'platec']
    spans.sort()
    assert all(end <= later for (_start, end), (later, _end) in zip(spans, spans[1:]))
    assert os.path.isdir(tmp_path / 'spacr_watch' / 'fields' / 'platea')
