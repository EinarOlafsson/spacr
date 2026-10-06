"""Fixed Convert planes become native TZYXC Mask fields without projection."""

import hashlib
import json
import os
import shutil

import numpy as np
import pytest
import tifffile

from spacr import core, io
from spacr.zstack import UnknownAnisotropyError
from tests.conftest import MISSING_CHANNEL_AXIS, check_cellpose_eval_call
from tests.test_cov_object_masks_sam import fake_model
from tests.test_watch_folder_and_analyse import MASK
from tests.test_watch_timelapse_series_f548 import _converted_series
from tests.test_watch_native_z_t1_f548 import _inputs as _converted_t1


def _settings(folder, **changes):
    settings = dict(MASK, src=str(folder), z_stack=True, t_stack=True,
                    t_axis_order='TZYX', z_axis=None,
                    z_segmentation_mode='volumetric', anisotropy=2,
                    frame_interval_s=60.0,
                    save_original_images=False, batch_size=2)
    settings.update(changes)
    return settings


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fixed_map_tzyx_preserves_every_original_plane_and_frame(tmp_path):
    source, rows = _converted_series(tmp_path)
    originals = {row['target']: _digest(source / row['target']) for row in rows}
    spatial = tifffile.imread(source / rows[0]['target']).shape
    settings, returned = io.preprocess_img_data(_settings(source))

    assert returned == str(source)
    assert settings['cellpose_nucleus_channel'] == 0
    assert settings['cellpose_cell_channel'] == 1
    assert {_row: _digest(source / _row) for _row in originals} == originals
    for time in (1, 2):
        filename = f'plate1_A01_1_{time}.npy'
        stack = np.load(source / 'stack' / filename)
        assert stack.shape == (2, *spatial, 2)
        for row in rows:
            if int(row['t']) == time:
                plane = tifffile.imread(source / row['target'])
                np.testing.assert_array_equal(
                    stack[int(row['z']) - 1, :, :, int(row['channel']) - 1], plane)
    archive = source / 'masks/plate1_A01_1_norm_timelapse.npz'
    with np.load(archive, allow_pickle=False) as content:
        assert content['data'].shape == (2, 2, *spatial, 2)
        assert content['filenames'].tolist() == [
            'plate1_A01_1_1.npy', 'plate1_A01_1_2.npy']
        assert np.isfinite(content['data']).all()
    receipt = json.loads((source / 'stack/.spacr_volume_series_ingest.json').read_text())
    assert receipt['axes'] == 'TZYXC'
    assert set(receipt['inputs']) == set(originals)
    assert set(receipt['stacks']) == {
        'plate1_A01_1_1.npy', 'plate1_A01_1_2.npy'}
    assert set(receipt['archives']) == {archive.name}
    before = _digest(archive)
    io.preprocess_img_data(_settings(source))
    assert _digest(archive) == before


def test_two_mapped_native_series_keep_their_timepoints_separate(tmp_path):
    source, rows = _converted_series(tmp_path, wells=('A01', 'A02'))
    io.preprocess_img_data(_settings(source))
    assert {path.name for path in (source / 'masks').glob('*.npz')} == {
        'plate1_A01_1_norm_timelapse.npz',
        'plate1_A02_1_norm_timelapse.npz'}
    for well in ('A01', 'A02'):
        with np.load(source / f'masks/plate1_{well}_1_norm_timelapse.npz',
                     allow_pickle=False) as archive:
            assert archive['filenames'].tolist() == [
                f'plate1_{well}_1_1.npy', f'plate1_{well}_1_2.npy']
        originals = [row for row in rows if row['well'] == well]
        first = next(row for row in originals if int(row['t']) == 1
                     and int(row['z']) == 1 and int(row['channel']) == 1)
        saved = np.load(source / f'stack/plate1_{well}_1_1.npy')
        np.testing.assert_array_equal(saved[0, :, :, 0],
                                      tifffile.imread(source / first['target']))


def test_one_selected_channel_matches_its_position_in_reordered_output(tmp_path):
    source, _rows = _converted_series(tmp_path)
    one_channel = tmp_path / 'one_channel'
    shutil.copytree(source, one_channel)
    selected, _ = io.preprocess_img_data(_settings(
        one_channel, nucleus_channel=1, cell_channel=None))
    reordered, _ = io.preprocess_img_data(_settings(
        source, nucleus_channel=1, cell_channel=0))
    assert selected['cellpose_nucleus_channel'] == 0
    assert reordered['cellpose_nucleus_channel'] == 0
    assert reordered['cellpose_cell_channel'] == 1
    name = 'plate1_A01_1_norm_timelapse.npz'
    with (np.load(one_channel / 'masks' / name, allow_pickle=False) as only,
          np.load(source / 'masks' / name, allow_pickle=False) as both):
        assert only['data'].shape[-1] == 1
        assert both['data'].shape[-1] == 2
        np.testing.assert_array_equal(only['data'][..., 0], both['data'][..., 0])


@pytest.mark.parametrize('damage', [
    'missing', 'extra', 'axes', 'linked', 'source_changed'])
def test_native_tzyx_rejects_nonmatching_map_or_planes_before_outputs(tmp_path, damage):
    source, rows = _converted_series(tmp_path)
    target = source / rows[-1]['target']
    if damage == 'missing':
        target.unlink()
    elif damage == 'extra':
        tifffile.imwrite(source / 'unexpected.tif', np.ones((32, 32), np.uint16))
    elif damage == 'axes':
        tifffile.imwrite(target, np.ones((2, 32, 32), np.uint16),
                         metadata={'axes': 'ZYX'})
    elif damage == 'linked':
        original = tmp_path / 'original.tif'
        target.rename(original)
        target.symlink_to(original)
    else:
        io.preprocess_img_data(_settings(source))
        tifffile.imwrite(target, np.ones((32, 32), np.uint16))
    with pytest.raises(ValueError):
        io.preprocess_img_data(_settings(source))
    if damage != 'source_changed':
        assert not (source / 'stack').exists()
        assert not (source / 'masks').exists()


@pytest.mark.parametrize('damage', ['map', 'stack', 'archive', 'extra_stack'])
def test_native_tzyx_resume_refuses_changed_source_or_outputs(tmp_path, damage):
    source, _rows = _converted_series(tmp_path)
    io.preprocess_img_data(_settings(source))
    if damage == 'map':
        with (source / 'conversion_map.csv').open('ab') as handle:
            handle.write(b'\n')
    elif damage == 'stack':
        output = source / 'stack/plate1_A01_1_1.npy'
        values = np.load(output)
        values[0, 0, 0, 0] += 1
        np.save(output, values)
    elif damage == 'archive':
        with (source / 'masks/plate1_A01_1_norm_timelapse.npz').open('ab') as handle:
            handle.write(b'tampered')
    else:
        np.save(source / 'stack/plate1_A02_1_1.npy', np.zeros((2, 4, 4, 2)))
    before = {str(path.relative_to(source)): _digest(path)
              for path in source.rglob('*') if path.is_file()}
    with pytest.raises(ValueError):
        io.preprocess_img_data(_settings(source))
    assert {str(path.relative_to(source)): _digest(path)
            for path in source.rglob('*') if path.is_file()} == before


def test_native_tzyx_source_change_during_normalization_never_publishes(
        tmp_path, monkeypatch):
    source, rows = _converted_series(tmp_path)
    target = source / rows[0]['target']
    original = io._normalize_img_batch

    def change_source(*args, **kwargs):
        normalized = original(*args, **kwargs)
        tifffile.imwrite(target, np.full((64, 64), 40, np.uint16))
        return normalized

    monkeypatch.setattr(io, '_normalize_img_batch', change_source)
    with pytest.raises(ValueError, match='changed before publication'):
        io.preprocess_img_data(_settings(source))
    assert not (source / 'stack').exists()
    assert not (source / 'masks').exists()


@pytest.mark.parametrize('case', [
    'illumination', 'metadata', 'missing_map', 'only_t1', 'no_object',
    'old_stack', 'linked_receipt', 'oversized_receipt',
])
def test_native_tzyx_preflight_refuses_unbound_or_unsupported_inputs(
        tmp_path, case):
    if case == 'only_t1':
        source, _batch, _rows = _converted_t1(tmp_path)
    else:
        source, _rows = _converted_series(tmp_path)
    settings = _settings(source)
    if case == 'illumination':
        settings['illumination_correction'] = True
    elif case == 'metadata':
        settings['metadata_type'] = 'arrayscan'
    elif case == 'missing_map':
        (source / 'conversion_map.csv').unlink()
    elif case == 'no_object':
        settings.update(nucleus_channel=None, cell_channel=None)
    elif case == 'old_stack':
        (source / 'stack').mkdir()
    elif case in ('linked_receipt', 'oversized_receipt'):
        io.preprocess_img_data(settings)
        receipt = source / 'stack/.spacr_volume_series_ingest.json'
        if case == 'linked_receipt':
            copy = tmp_path / 'receipt.json'
            receipt.rename(copy)
            receipt.symlink_to(copy)
        else:
            with receipt.open('ab') as handle:
                handle.write(b' ' * (16 * 1024 * 1024))
    with pytest.raises(ValueError):
        io.preprocess_img_data(settings)
    if case not in ('linked_receipt', 'oversized_receipt', 'old_stack'):
        assert not (source / 'stack').exists()
        assert not (source / 'masks').exists()


@pytest.mark.parametrize('case', ['nonfinite', 'shape', 'dtype'])
def test_native_tzyx_rejects_incompatible_planes_without_publication(
        tmp_path, case):
    source, rows = _converted_series(tmp_path)
    target = source / rows[-1]['target']
    if case == 'nonfinite':
        image = np.ones((64, 64), np.float32)
        image[0, 0] = np.nan
    else:
        image = np.ones((32, 64) if case == 'shape' else (64, 64),
                        np.uint8 if case == 'dtype' else np.uint16)
    tifffile.imwrite(target, image)
    with pytest.raises(ValueError):
        io.preprocess_img_data(_settings(source))
    assert not (source / 'stack').exists()
    assert not (source / 'masks').exists()


@pytest.mark.parametrize('stage', ['map', 'source_preflight', 'source_read',
                                   'source_decode', 'source_digest'])
def test_native_tzyx_refuses_a_racing_map_or_plane(tmp_path, monkeypatch, stage):
    source, rows = _converted_series(tmp_path)
    target = source / sorted(row['target'] for row in rows)[0]
    if stage == 'map':
        original = core._watch_map_bytes
        calls = 0

        def changing_map(folder):
            nonlocal calls
            calls += 1
            data = original(folder)
            return data + b'\n' if calls == 2 else data

        monkeypatch.setattr(core, '_watch_map_bytes', changing_map)
    elif stage in ('source_preflight', 'source_digest'):
        original = core._watch_artifact_sha256
        calls = 0

        def changing_digest(path):
            nonlocal calls
            digest = original(path)
            if os.fspath(path) == str(target):
                calls += 1
                if calls == (1 if stage == 'source_preflight' else 2):
                    tifffile.imwrite(target, np.full((64, 64), 40, np.uint16))
                    if stage == 'source_digest':
                        return original(path)
            return digest

        monkeypatch.setattr(core, '_watch_artifact_sha256', changing_digest)
    elif stage == 'source_read':
        original = os.open
        calls = 0

        def changing_open(path, flags, *args, **kwargs):
            nonlocal calls
            if os.fspath(path) == str(target):
                calls += 1
                if calls == 2:
                    tifffile.imwrite(target, np.full((64, 64), 40, np.uint16))
            return original(path, flags, *args, **kwargs)

        monkeypatch.setattr(io.os, 'open', changing_open)
    else:
        original = tifffile.TiffFile

        class ChangingTiff:
            def __init__(self, handle, **kwargs):
                self.actual = original(handle, **kwargs)
                self.name = kwargs['name']

            def __enter__(self):
                self.actual.__enter__()
                owner = self

                class Plane:
                    axes = owner.actual.series[0].axes

                    def asarray(self):
                        pixels = owner.actual.series[0].asarray()
                        if owner.name == str(target):
                            with target.open('ab') as handle:
                                handle.write(b'changed')
                        return pixels

                self.series = [Plane()]
                return self

            def __exit__(self, *args):
                return self.actual.__exit__(*args)

        monkeypatch.setattr(tifffile, 'TiffFile', ChangingTiff)
    with pytest.raises(ValueError):
        io.preprocess_img_data(_settings(source))
    assert not (source / 'stack').exists()
    assert not (source / 'masks').exists()


def test_native_tzyx_publication_failure_removes_only_its_new_outputs(
        tmp_path, monkeypatch):
    source, _rows = _converted_series(tmp_path)
    original = os.link
    calls = 0

    def failing_link(src, dst, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError('injected output failure')
        return original(src, dst, *args, **kwargs)

    monkeypatch.setattr(io.os, 'link', failing_link)
    with pytest.raises(OSError, match='injected output failure'):
        io.preprocess_img_data(_settings(source))
    assert not (source / 'stack').exists()
    assert not (source / 'masks').exists()


@pytest.mark.parametrize('change', [
    {'timelapse': True}, {'watch_pipeline': 'mask_measure'},
    {'watch_pipeline': 'mask_measure_classify'}, {'t_axis_order': 'ZTYX'},
    {'anisotropy': None}, {'frame_interval_s': None},
    {'adjust_cells': True}, {'plot': True},
    {'segmentation_backend': 'cellpose3'},
])
def test_native_tzyx_refuses_unproved_modes_before_outputs(tmp_path, change):
    source, _rows = _converted_series(tmp_path)
    with pytest.raises((ValueError, UnknownAnisotropyError)):
        io.preprocess_img_data(_settings(source, **change))
    assert not (source / 'stack').exists()
    assert not (source / 'masks').exists()


def test_real_mask_pipeline_receives_each_native_volume_with_z_axis(
        tmp_path, fake_model, monkeypatch):
    import spacr.object as spacr_object

    source, rows = _converted_series(tmp_path)
    seen = []
    model_class = spacr_object.cp_models.CellposeModel

    def volume_eval(self, x, batch_size=8, resample=True, channels=None,
                    channel_axis=MISSING_CHANNEL_AXIS, z_axis=None,
                    normalize=True, invert=False, rescale=None, diameter=None,
                    flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False,
                    anisotropy=None, flow3D_smooth=0, stitch_threshold=0.0,
                    min_size=15, max_size_fraction=0.4, niter=None,
                    augment=False, tile_overlap=0.1, bsize=256,
                    compute_masks=True, progress=None):
        converted = check_cellpose_eval_call(
            x, channel_axis, z_axis=z_axis, do_3D=do_3D)
        assert do_3D and z_axis == 0 and channel_axis == -1
        assert anisotropy == 2 and len(converted) == 1
        assert converted[0].shape == (2, 64, 64, 1)
        seen.append(converted[0].copy())
        labels = np.zeros((2, 64, 64), np.uint16)
        labels[:, 5:17, 5:17] = 1
        return labels, None, None

    monkeypatch.setattr(model_class, 'eval', volume_eval)
    core.preprocess_generate_masks(_settings(source, cell_channel=None))
    assert len(seen) == 2
    for time, frame in enumerate(seen, start=1):
        assert frame.shape == (2, 64, 64, 1)
        raw_first = tifffile.imread(source / next(
            row['target'] for row in rows if int(row['t']) == time
            and int(row['z']) == 1 and int(row['channel']) == 1))
        assert np.array_equal(frame[0, :, :, 0] > frame[0, :, :, 0].min(),
                              raw_first > raw_first.min())
        merged = np.load(source / f'merged/plate1_A01_1_{time}.npy')
        assert merged.shape == (2, 64, 64, 3)
        np.testing.assert_array_equal(merged[0, :, :, 0], raw_first)
        assert np.count_nonzero(merged[..., -1]) > 0
