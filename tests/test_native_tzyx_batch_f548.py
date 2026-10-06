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


@pytest.mark.parametrize('channel_count, selected', [
    (3, [2, 0, 1]), (4, [3, 1])])
def test_native_three_time_three_z_archive_matches_scalar_stack(
        tmp_path, channel_count, selected):
    """The app's float32 archive keeps full-field quantiles and channel order."""
    from spacr import convert

    source = tmp_path / 'raw' / 'A01'
    source.mkdir(parents=True)
    ramp = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    for channel in range(channel_count):
        frames = np.stack([
            np.stack([
                ((ramp + 37 * time + 53 * z + 71 * channel) % 3000).astype(np.uint16)
                for z in range(3)])
            for time in range(3)])
        tifffile.imwrite(source / f'field01_C{channel + 1}.tif', frames,
                         metadata={'axes': 'TZYX'}, photometric='minisblack')
    converted = tmp_path / 'converted'
    assert convert.convert_folder(dict(src=str(tmp_path / 'raw'),
                                       dst=str(converted), z_handling='keep',
                                       preview_rows=0)).is_complete
    settings, returned = io.preprocess_img_data(_settings(
        converted, nucleus_channel=selected[0], cell_channel=selected[1],
        pathogen_channel=selected[2] if len(selected) > 2 else None,
        pathogen_background=300, pathogen_signal_to_noise=2,
        remove_background_pathogen=True, lower_percentile=17))
    assert returned == str(converted)
    staged = sorted((converted / 'stack').glob('*.npy'))
    assert len(staged) == 3
    raw = np.stack([np.load(path, allow_pickle=False) for path in staged])
    expected = io._normalize_img_batch(
        raw, selected, np.float32, settings)[..., selected]
    archive = converted / 'masks/plate1_A01_1_norm_timelapse.npz'
    with np.load(archive, allow_pickle=False) as content:
        actual = content['data']
        assert content['filenames'].tolist() == [path.name for path in staged]
    assert actual.shape == (3, 3, 64, 64, len(selected))
    assert actual.dtype == np.float32
    assert actual.tobytes() == expected.tobytes()
    reference = tmp_path / 'reference.npz'
    io._save_npz_atomic(reference, data=expected,
                        filenames=[path.name for path in staged])
    assert _digest(archive) == _digest(reference)


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
    original = io._normalize_img_channels

    def change_source(*args, **kwargs):
        normalized = original(*args, **kwargs)
        tifffile.imwrite(target, np.full((64, 64), 40, np.uint16))
        return normalized

    monkeypatch.setattr(io, '_normalize_img_channels', change_source)
    with pytest.raises(ValueError, match='changed before publication'):
        io.preprocess_img_data(_settings(source))
    assert not (source / 'stack').exists()
    assert not (source / 'masks').exists()


def test_native_tzyx_staged_stack_change_during_normalization_never_publishes(
        tmp_path, monkeypatch):
    """A re-read staged stack must still match its pre-normalization hash."""
    source, _rows = _converted_series(tmp_path)
    original = io._normalize_img_channels

    def change_staged_stack(*args, **kwargs):
        normalized = original(*args, **kwargs)
        staged = sorted(source.glob('.spacr-volume-series-*/stack/*.npy'))
        assert len(staged) == 2
        values = np.load(staged[0], allow_pickle=False)
        values[0, 0, 0, 0] += 1
        np.save(staged[0], values)
        return normalized

    monkeypatch.setattr(io, '_normalize_img_channels', change_staged_stack)
    with pytest.raises(ValueError, match='staged stack changed'):
        io.preprocess_img_data(_settings(source))
    assert not (source / 'stack').exists()
    assert not (source / 'masks').exists()
    assert not list(source.glob('.spacr-volume-series-*'))


def test_native_tzyx_staged_stack_change_before_normalization_never_publishes(
        tmp_path, monkeypatch):
    """A changed first timepoint cannot seed an archive after staging ends."""
    source, _rows = _converted_series(tmp_path)
    original = io._save_array_atomic
    staged = []

    def change_first_stack(path, values):
        result = original(path, values)
        if '.spacr-volume-series-' in str(path) and str(path).endswith('.npy'):
            staged.append(path)
            if len(staged) == 2:
                first = np.load(staged[0], allow_pickle=False)
                first[0, 0, 0, 0] += 1
                np.save(staged[0], first)
        return result

    monkeypatch.setattr(io, '_save_array_atomic', change_first_stack)
    with pytest.raises(ValueError, match='staged stack changed'):
        io.preprocess_img_data(_settings(source))
    assert not (source / 'stack').exists()
    assert not (source / 'masks').exists()
    assert not list(source.glob('.spacr-volume-series-*'))


def test_native_tzyx_staged_stack_shape_change_during_read_never_publishes(
        tmp_path, monkeypatch):
    """A damaged private stack cannot enter the channel-wise normalization."""
    source, _rows = _converted_series(tmp_path)
    original = io.np.load
    damaged = False

    def change_mapped_stack(path, *args, **kwargs):
        nonlocal damaged
        if (not damaged and kwargs.get('mmap_mode') == 'r'
                and '.spacr-volume-series-' in str(path)):
            damaged = True
            values = original(path, allow_pickle=False)
            np.save(path, values[..., :1])
        return original(path, *args, **kwargs)

    monkeypatch.setattr(io.np, 'load', change_mapped_stack)
    with pytest.raises(ValueError, match='staged stack shape or dtype changed'):
        io.preprocess_img_data(_settings(source))
    assert damaged
    assert not (source / 'stack').exists()
    assert not (source / 'masks').exists()
    assert not list(source.glob('.spacr-volume-series-*'))


def test_native_tzyx_cancel_during_staged_channel_read_discards_private_stage(
        tmp_path, monkeypatch):
    """A cancellation between channel reads leaves no publishable output."""
    from spacr.cancellation import (CancellationToken, PipelineCancelled,
                                    installed_token)

    source, _rows = _converted_series(tmp_path)
    token = CancellationToken()
    original = io._normalize_img_channels

    def cancel_second_channel(output, channels, dtype, settings, load_channel,
                              **kwargs):
        reads = 0

        def cancellable_load(channel):
            nonlocal reads
            reads += 1
            if reads == 2:
                token.cancel()
            return load_channel(channel)

        return original(output, channels, dtype, settings, cancellable_load,
                        **kwargs)

    monkeypatch.setattr(io, '_normalize_img_channels', cancel_second_channel)
    with installed_token(token), pytest.raises(PipelineCancelled):
        io.preprocess_img_data(_settings(source))
    assert not (source / 'stack').exists()
    assert not (source / 'masks').exists()
    assert not list(source.glob('.spacr-volume-series-*'))


def test_native_tzyx_cancel_after_quantile_closes_maps_before_discard(
        tmp_path, monkeypatch):
    """A late stop releases mapped quantiles, channels and selected output."""
    from spacr.cancellation import CancellationToken, PipelineCancelled, installed_token

    source, _rows = _converted_series(tmp_path)
    token = CancellationToken()
    original_percentile = io.np.percentile
    original_close = io._close_private_memmap
    closed = []
    calls = 0

    def cancel_after_first_quantile(*args, **kwargs):
        nonlocal calls
        result = original_percentile(*args, **kwargs)
        calls += 1
        if calls == 1:
            token.cancel()
        return result

    def record_close(mapped):
        closed.append(os.path.basename(mapped.filename))
        original_close(mapped)

    monkeypatch.setattr(io.np, 'percentile', cancel_after_first_quantile)
    monkeypatch.setattr(io, '_close_private_memmap', record_close)
    with installed_token(token), pytest.raises(PipelineCancelled):
        io.preprocess_img_data(_settings(source))
    assert calls >= 1
    assert any(name.endswith('-quantiles.bin') for name in closed)
    assert any(name.startswith('channel-') and name.endswith('.npy')
               for name in closed)
    assert 'selected.npy' in closed
    assert not (source / 'stack').exists()
    assert not (source / 'masks').exists()
    assert not list(source.glob('.spacr-volume-series-*'))


def test_native_tzyx_rescale_failure_closes_private_stage(tmp_path, monkeypatch):
    """A failed plane operation leaves no mapped workspace or output."""
    source, _rows = _converted_series(tmp_path)
    original_close = io._close_private_memmap
    closed = []

    def record_close(mapped):
        closed.append(os.path.basename(mapped.filename))
        original_close(mapped)

    def fail_rescale(*_args, **_kwargs):
        raise RuntimeError('rescale failed')

    monkeypatch.setattr(io, '_close_private_memmap', record_close)
    monkeypatch.setattr(io.exposure, 'rescale_intensity', fail_rescale)
    with pytest.raises(RuntimeError, match='rescale failed'):
        io.preprocess_img_data(_settings(source))
    assert any(name.endswith('-quantiles.bin') for name in closed)
    assert any(name.startswith('channel-') and name.endswith('.npy')
               for name in closed)
    assert 'selected.npy' in closed
    assert not (source / 'stack').exists()
    assert not (source / 'masks').exists()
    assert not list(source.glob('.spacr-volume-series-*'))


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


@pytest.mark.parametrize('conversion', ['installed', 'three_channels'])
def test_real_mask_pipeline_receives_each_native_volume_with_z_axis(
        tmp_path, fake_model, monkeypatch, conversion):
    """Cellpose 4.0.7 pads one native channel to three without changing it."""
    from cellpose import transforms
    import spacr.object as spacr_object

    if conversion == 'three_channels':
        installed_convert = transforms.convert_image

        def padded_conversion(*args, **kwargs):
            """Exercise Cellpose's supported three-channel converted layout."""
            converted = installed_convert(*args, **kwargs)
            first = converted[..., :1]
            return np.concatenate((first, np.zeros_like(first),
                                   np.zeros_like(first)), axis=-1)

        monkeypatch.setattr(transforms, 'convert_image', padded_conversion)

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
        native = np.asarray(x)
        converted = check_cellpose_eval_call(
            x, channel_axis, z_axis=z_axis, do_3D=do_3D)
        assert do_3D and z_axis == 0 and channel_axis == -1
        assert anisotropy == 2 and len(converted) == 1
        assert native.shape == (2, 64, 64, 1)
        assert converted[0].shape in ((2, 64, 64, 1), (2, 64, 64, 3))
        if conversion == 'three_channels':
            assert converted[0].shape[-1] == 3
        np.testing.assert_array_equal(converted[0][..., 0], native[..., 0])
        if converted[0].shape[-1] == 3:
            assert not np.any(converted[0][..., 1:])
        seen.append(converted[0].copy())
        labels = np.zeros((2, 64, 64), np.uint16)
        labels[:, 5:17, 5:17] = 1
        return labels, None, None

    monkeypatch.setattr(model_class, 'eval', volume_eval)
    core.preprocess_generate_masks(_settings(source, cell_channel=None))
    assert len(seen) == 2
    for time, converted in enumerate(seen, start=1):
        raw_first = tifffile.imread(source / next(
            row['target'] for row in rows if int(row['t']) == time
            and int(row['z']) == 1 and int(row['channel']) == 1))
        assert np.array_equal(converted[0, :, :, 0] > converted[0, :, :, 0].min(),
                              raw_first > raw_first.min())
        merged = np.load(source / f'merged/plate1_A01_1_{time}.npy')
        assert merged.shape == (2, 64, 64, 3)
        np.testing.assert_array_equal(merged[0, :, :, 0], raw_first)
        assert np.count_nonzero(merged[..., -1]) > 0
