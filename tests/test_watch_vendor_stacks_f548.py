"""Self-describing vendor stacks feed the mapped T/Z watch routes without a map."""

import hashlib
import json
import shutil

import numpy as np
import pytest
import tifffile

from spacr import convert, core
from tests.test_watch_folder_and_analyse import (MASK, MEASURE, Recorder, _rows,
                                                 real_pipeline)
from tests.test_native_tzyx_batch_f548 import _settings as _series_settings
from tests.test_watch_nested_f548 import _fast
from tests.test_watch_native_t_series_f548 import _model, _watch as _series_watch
from tests.test_watch_timelapse_series_f548 import _converted_series

pytest_plugins = ['tests.test_cov_object_masks_sam']


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stack_from_planes(folder, rows):
    """Assemble converted planes back into one TZCYX acquisition array."""
    times = max(int(row['t']) for row in rows)
    planes = max(int(row['z']) for row in rows)
    channels = max(int(row['channel']) for row in rows)
    first = tifffile.imread(folder / rows[0]['target'])
    stack = np.zeros((times, planes, channels, *first.shape), first.dtype)
    for row in rows:
        stack[int(row['t']) - 1, int(row['z']) - 1, int(row['channel']) - 1] = (
            tifffile.imread(folder / row['target']))
    return stack


def _partial_ome(path, stack, axes, pages):
    """Write OME metadata declaring the whole stack but only its first pages."""
    whole = path.with_name('whole_' + path.name)
    tifffile.imwrite(whole, stack, ome=True, metadata={'axes': axes})
    with tifffile.TiffFile(whole) as handle:
        description = handle.pages[0].description
    whole.unlink()
    planes = stack.reshape(-1, *stack.shape[-2:])
    with tifffile.TiffWriter(path) as writer:
        for index in range(pages):
            writer.write(planes[index], description=description if index == 0 else None,
                         metadata=None, contiguous=False)


def test_declared_planes_are_the_completion_signal(tmp_path):
    stack = np.arange(5 * 2 * 8 * 8, dtype=np.uint16).reshape(5, 2, 8, 8) + 1
    whole = tmp_path / 'whole.ome.tif'
    tifffile.imwrite(whole, stack, ome=True, metadata={'axes': 'ZCYX'})
    assert core._watch_vendor_unreadable(str(whole), 'volume') is None
    growing = tmp_path / 'growing.ome.tif'
    _partial_ome(growing, stack, 'ZCYX', 7)
    assert '7 of 10 declared planes' in core._watch_vendor_unreadable(str(growing), 'volume')
    imagej = tmp_path / 'imagej.tif'
    tifffile.imwrite(imagej, stack, imagej=True, metadata={'axes': 'ZCYX'})
    assert core._watch_vendor_unreadable(str(imagej), 'volume') is None
    cut = tmp_path / 'cut.tif'
    cut.write_bytes(imagej.read_bytes()[:-300])
    assert core._watch_vendor_unreadable(str(cut), 'volume') is not None
    undeclared = tmp_path / 'undeclared.tif'
    tifffile.imwrite(undeclared, stack, metadata=None)
    assert 'does not declare' in core._watch_vendor_unreadable(str(undeclared), 'volume')
    assert 'do not fit the series' in core._watch_vendor_unreadable(str(whole), 'series')
    assert 'do not fit the timelapse' in core._watch_vendor_unreadable(str(whole), 'timelapse')
    other = tmp_path / 'stack.czi'
    other.write_bytes(b'not a tiff')
    assert 'convert other formats' in core._watch_vendor_unreadable(str(other), 'volume')


@pytest.mark.parametrize('writer', ['ome', 'imagej'])
def test_split_follows_the_declared_axes(tmp_path, writer):
    stack = np.random.default_rng(3).integers(1, 4000, (2, 3, 2, 6, 7), dtype=np.uint16)
    source = tmp_path / 'acq.tif'
    if writer == 'ome':
        tifffile.imwrite(source, stack, ome=True, metadata={'axes': 'TZCYX'})
    else:
        tifffile.imwrite(source, stack, imagej=True, metadata={'axes': 'TZCYX'})
    field = tmp_path / 'field'
    field.mkdir()
    digests, plan = core._watch_vendor_stage(
        str(field), str(source), 'acq', 'series',
        {'channels': [0, 1], 'nucleus_channel': 0})
    assert len(digests) == 12 and plan['names'] == sorted(digests)
    for t in range(2):
        for z in range(3):
            for c in range(2):
                name = convert.target_name('acq', 'A01', 1, c + 1, z + 1, t + 1)
                np.testing.assert_array_equal(tifffile.imread(field / name), stack[t, z, c])
                assert digests[name] == _digest(field / name)
    manifest, map_sha256 = core._watch_map_manifest(
        str(field), {'metadata_type': 'cellvoyager', 'timelapse': True, 'channels': [0, 1]})
    assert map_sha256 == plan['map_sha256']
    assert manifest == {'acq_A01_001': set(digests)}
    assert plan['stacks'] == ['acq_A01_1_1.npy', 'acq_A01_1_2.npy']
    with pytest.raises(ValueError, match='declared channel'):
        core._watch_vendor_stage(str(tmp_path / 'other'), str(source), 'acq', 'series',
                                 {'channels': [2], 'nucleus_channel': 0})


def test_vendor_series_waits_for_declared_planes_then_matches_batch(
        tmp_path, fake_model, monkeypatch):
    source, rows = _converted_series(tmp_path)
    stack = _stack_from_planes(source, rows)
    batch, watched = tmp_path / 'batch', tmp_path / 'watched'
    shutil.copytree(source, batch)
    watched.mkdir()
    seen = _model(monkeypatch)
    core.preprocess_generate_masks(_series_settings(batch, cell_channel=None))
    batch_calls = len(seen)
    vendor = watched / 'plate1.ome.tif'
    _partial_ome(vendor, stack, 'TZCYX', 6)
    first = core._watch_folder_and_analyse(_series_watch(watched))
    assert not first['done'] and first['incomplete'] == ['plate1']
    assert len(seen) == batch_calls
    assert not (watched / 'spacr_watch/fields').exists()
    vendor.unlink()
    tifffile.imwrite(vendor, stack, ome=True, metadata={'axes': 'TZCYX'})
    acquired = _digest(vendor)
    result = core._watch_folder_and_analyse(_series_watch(watched))
    assert result['done'] == ['plate1'] and not result['failed']
    assert len(seen) == batch_calls + 2
    field = watched / 'spacr_watch/fields/plate1'
    assert _digest(field / '.watch_vendor/plate1.ome.tif') == acquired == _digest(vendor)
    for row in rows:
        np.testing.assert_array_equal(tifffile.imread(field / row['target']),
                                      tifffile.imread(source / row['target']))
    names = sorted(path.name for path in (batch / 'merged').glob('*.npy'))
    assert names == sorted(path.name for path in
                           (watched / 'spacr_watch/merged').glob('*.npy'))
    for name in names:
        np.testing.assert_array_equal(np.load(batch / 'merged' / name),
                                      np.load(watched / 'spacr_watch/merged' / name))
    ledger = json.loads((watched / 'spacr_watch/watch_ledger.json').read_text())
    entry = ledger['fields']['plate1']
    assert ledger['vendor_stacks'] == 'series'
    assert entry['vendor_sha256'] == {'plate1.ome.tif': acquired}
    assert entry['collection_checkpoint']['artifacts'][
        '.watch_vendor/plate1.ome.tif'] == acquired
    resumed = core._watch_folder_and_analyse(_series_watch(watched))
    assert resumed['done'] == ['plate1'] and len(seen) == batch_calls + 2


def test_vendor_volume_matches_raw_volume_batch_and_measure(tmp_path, fake_model,
                                                            monkeypatch):
    from spacr.measure import measure_crop
    from tests.test_watch_native_z_t1_f548 import _mask

    seen = _model(monkeypatch)
    volume = np.full((3, 2, 32, 32), 40, np.uint16)
    for channel in (0, 1):
        volume[:, channel, 5:17, 5:17] = 500 + channel
        volume[:, channel, 18:29, 18:29] = 800 + channel
    batch, watched = tmp_path / 'batch', tmp_path / 'watched'
    batch.mkdir()
    watched.mkdir()
    for channel in (0, 1):
        tifffile.imwrite(batch / convert.target_name('plate1', 'A01', 1, channel + 1),
                         volume[:, channel], metadata={'axes': 'ZYX'},
                         photometric='minisblack')
    core.preprocess_generate_masks(_mask(batch))
    measure = dict(MEASURE, anisotropy=2)
    measure_crop(dict(measure, src=str(batch / 'merged')))
    recipe = tmp_path / 'measure.json'
    recipe.write_text(json.dumps(measure))
    tifffile.imwrite(watched / 'plate-1.ome.tif', volume, ome=True,
                     metadata={'axes': 'ZCYX'})
    settings = dict(_mask(watched), **_fast(watched), watch_pipeline='mask_measure',
                    watch_measure_settings=str(recipe))
    result = core._watch_folder_and_analyse(settings)
    assert result['done'] == ['plate1'] and not result['failed'] and not result['incomplete']
    field = watched / 'spacr_watch/fields/plate1'
    assert len(list((field / '.watch_planar').glob('*.tif'))) == 6
    for channel in (0, 1):
        name = convert.target_name('plate1', 'A01', 1, channel + 1)
        np.testing.assert_array_equal(tifffile.imread(field / name), volume[:, channel])
    names = sorted(path.name for path in (batch / 'merged').glob('*.npy'))
    assert len(names) == 1
    for name in names:
        np.testing.assert_array_equal(np.load(batch / 'merged' / name),
                                      np.load(watched / 'spacr_watch/merged' / name))
    for table in ('cell', 'nucleus'):
        assert _rows(batch / 'measurements/measurements.db', table) == _rows(
            watched / 'spacr_watch/measurements/measurements.db', table)
    assert len(seen) == 4


def test_vendor_timelapse_matches_projected_batch(tmp_path, real_pipeline):
    source, rows = _converted_series(tmp_path)
    stack = _stack_from_planes(source, rows)
    batch, watched = tmp_path / 'batch', tmp_path / 'watched'
    batch.mkdir()
    watched.mkdir()
    for row in rows:
        shutil.copy2(source / row['target'], batch / row['target'])
    mask = dict(MASK, timelapse=True)
    core.preprocess_generate_masks(dict(mask, src=str(batch)))
    tifffile.imwrite(watched / 'plate1.tif', stack, imagej=True,
                     metadata={'axes': 'TZCYX'})
    result = core._watch_folder_and_analyse(dict(mask, **_fast(watched)))
    assert result['done'] == ['plate1'] and not result['failed']
    names = sorted(path.name for path in (batch / 'merged').glob('*.npy'))
    assert len(names) == 2
    assert sorted(path.name for path in
                  (watched / 'spacr_watch/merged').glob('*.npy')) == names
    for name in names:
        np.testing.assert_array_equal(np.load(batch / 'merged' / name),
                                      np.load(watched / 'spacr_watch/merged' / name))


def test_per_plane_files_without_a_map_stop_a_series_watch(tmp_path):
    source, _rows_ = _converted_series(tmp_path)
    (source / convert.MAP_FILENAME).unlink()
    with pytest.raises(ValueError, match='per-plane series.*conversion_map.csv'):
        core._watch_folder_and_analyse(dict(MASK, **_fast(source), timelapse=True),
                                       Recorder())
    assert not (source / 'spacr_watch').exists()


def test_vendor_stacks_need_the_cellvoyager_convention(tmp_path):
    with pytest.raises(ValueError, match='self-describing'):
        core._watch_folder_and_analyse(
            {**MASK, **_fast(tmp_path), 'timelapse': True, 'metadata_type': 'custom',
             'custom_regex': r'(?P<plateID>.*)'}, Recorder())
    assert not (tmp_path / 'spacr_watch').exists()


def test_two_stacks_with_one_token_wait_without_combining(tmp_path):
    stack = np.ones((2, 1, 8, 8), np.uint16)
    for name in ('a_1.tif', 'a-1.tif'):
        tifffile.imwrite(tmp_path / name, stack, imagej=True, metadata={'axes': 'TCYX'})
    recorder = Recorder()
    result = core._watch_folder_and_analyse(
        dict(MASK, **_fast(tmp_path), timelapse=True), recorder)
    assert not recorder.calls and result['incomplete'] == ['a1']


@pytest.mark.parametrize('damage', ['none', 'vendor'])
def test_vendor_collection_resume_binds_the_stack_copy(tmp_path, monkeypatch, damage):
    stack = np.arange(2 * 2 * 8 * 8, dtype=np.uint16).reshape(2, 2, 8, 8) + 1
    tifffile.imwrite(tmp_path / 'acq.ome.tif', stack, ome=True, metadata={'axes': 'TCYX'})
    settings = dict(MASK, **_fast(tmp_path), timelapse=True)

    def analyse(field_dir, _settings):
        (tmp_path / 'spacr_watch/fields/acq/merged').mkdir()
        np.save(tmp_path / 'spacr_watch/fields/acq/merged/acq_A01_1_1.npy', np.zeros(3))

    def fail_collection(*_args, **_kwargs):
        raise OSError('injected collection failure')

    with monkeypatch.context() as patch:
        patch.setattr(core, '_watch_collect', fail_collection)
        first = core._watch_folder_and_analyse(settings, analyse)
    assert first['failed'] == ['acq']
    if damage == 'vendor':
        with (tmp_path / 'spacr_watch/fields/acq/.watch_vendor/acq.ome.tif').open('ab') as handle:
            handle.write(b'tampered')
    calls = []
    resumed = core._watch_folder_and_analyse(settings, lambda *args: calls.append(args))
    assert not calls
    if damage == 'none':
        assert resumed['done'] == ['acq']
        assert (tmp_path / 'spacr_watch/merged/acq_A01_1_1.npy').exists()
    else:
        assert resumed['failed'] == ['acq']
        ledger = json.loads((tmp_path / 'spacr_watch/watch_ledger.json').read_text())
        assert 'vendor stack copy changed' in ledger['fields']['acq']['error']


def test_a_per_plane_file_arriving_later_stops_the_watch(tmp_path):
    source, rows = _converted_series(tmp_path)
    watched = tmp_path / 'watched'
    watched.mkdir()
    tifffile.imwrite(watched / 'acq.tif', np.ones((2, 1, 8, 8), np.uint16),
                     imagej=True, metadata={'axes': 'TCYX'})

    def analyse(field_dir, _settings):
        shutil.copy2(source / rows[0]['target'], watched / rows[0]['target'])
        (tmp_path / 'watched/spacr_watch/fields/acq/merged').mkdir()
        np.save(tmp_path / 'watched/spacr_watch/fields/acq/merged/acq_A01_1_1.npy',
                np.zeros(3))

    with pytest.raises(ValueError, match='per-plane series'):
        core._watch_folder_and_analyse(
            {**MASK, **_fast(watched), 'timelapse': True, 'channels': [0],
             'cell_channel': None}, analyse)
    ledger = json.loads((watched / 'spacr_watch/watch_ledger.json').read_text())
    assert ledger['fields']['acq']['status'] == 'done'


def test_volume_normalization_rescales_one_plane_at_a_time(monkeypatch):
    from skimage import exposure

    from spacr import io

    stack = np.random.default_rng(5).integers(1, 5000, (2, 4, 6, 7, 2), dtype=np.uint16)
    calls = []
    rescale = exposure.rescale_intensity

    def recorded(image, in_range, out_range):
        calls.append((image.ndim, in_range))
        return rescale(image, in_range=in_range, out_range=out_range)

    monkeypatch.setattr(io.exposure, 'rescale_intensity', recorded)
    settings = {'lower_percentile': 2, 'nucleus_channel': 0, 'cell_channel': 1,
                'pathogen_channel': None, 'background': 100, 'Signal_to_noise': 10,
                'remove_background': False, 'nucleus_background': 100,
                'cell_background': 100, 'nucleus_signal_to_noise': 10,
                'cell_signal_to_noise': 10, 'remove_background_nucleus': False,
                'remove_background_cell': False}
    result = io._normalize_img_batch(stack.copy(), [0, 1], np.float32, settings)
    assert {ndim for ndim, _range in calls} == {2} and len(calls) == 2 * 2 * 4
    ranges = {}
    for index, (_ndim, in_range) in enumerate(calls):
        ranges.setdefault(index // 8, set()).add(in_range)
    for channel in (0, 1):
        (in_range,) = ranges[channel]
        for batch in range(2):
            expected = rescale(stack[batch, ..., channel], in_range=in_range,
                               out_range=(0, 1)).astype(np.float32)
            np.testing.assert_array_equal(result[batch, ..., channel], expected)
