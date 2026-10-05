"""A fixed T1 Convert map feeds the raw ZYX batch route without losing planes."""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import tifffile

from spacr import convert, core, io
from spacr.cancellation import CancellationToken, PipelineCancelled, installed_token
from spacr.zstack import ZStackError
from tests.cellpose_api_contract import MISSING_CHANNEL_AXIS
from tests.conftest import check_cellpose_eval_call
from tests.test_cov_object_masks_sam import fake_model
from tests.test_watch_folder_and_analyse import MASK, MEASURE, Recorder, _rows
from tests.test_watch_nested_f548 import _fast
from tests.test_raw_volumetric_tiffs_f551 import _settings as _raw_settings


def _inputs(tmp_path):
    """Convert genuine raw ZYX acquisitions into a fixed six-plane T1 map."""
    raw, watched, batch = (tmp_path / name for name in ('raw', 'watched', 'batch'))
    (raw / 'A01').mkdir(parents=True)
    batch.mkdir()
    for channel in (1, 2):
        volume = np.full((3, 32, 32), 40, np.uint16)
        volume[:, 5:17, 5:17] = 500 + channel
        volume[:, 18:29, 18:29] = 800 + channel
        name = convert.target_name('plate1', 'A01', 1, channel)
        tifffile.imwrite(raw / 'A01' / f'field01_C{channel}.tif', volume,
                         metadata={'axes': 'ZYX'}, photometric='minisblack')
        tifffile.imwrite(batch / name, volume, metadata={'axes': 'ZYX'},
                         photometric='minisblack')
    result = convert.convert_folder({'src': str(raw), 'dst': str(watched),
                                     'z_handling': 'keep', 'preview_rows': 0})
    assert result.is_complete and len(result.rows()) == 6
    return watched, batch, result.rows()


def _mask(folder):
    """Use the same one-field volumetric recipe for batch and watch."""
    return dict(MASK, src=str(folder), z_stack=True,
                z_segmentation_mode='volumetric', z_axis=0, anisotropy=2,
                save_original_images=False)


def test_raw_volume_channel_ten_keeps_numeric_position(tmp_path):
    """The existing raw-volume route must map C10 to zero-based index nine."""
    for channel in range(1, 11):
        image = np.full((2, 8, 9), channel, np.uint16)
        tifffile.imwrite(tmp_path / convert.target_name('plate1', 'A01', 1, channel),
                         image, metadata={'axes': 'ZYX'}, photometric='minisblack')
    settings = _raw_settings(tmp_path, channels=list(range(10)),
                             nucleus_channel=9)
    io.preprocess_img_data(settings)
    stack = np.load(tmp_path / 'stack/plate1_A01_1_1.npy')
    assert stack.shape == (2, 8, 9, 10)
    for index in range(10):
        np.testing.assert_array_equal(stack[..., index], index + 1)
    receipt = json.loads((tmp_path / 'stack/.spacr_volume_ingest.json').read_text())
    assert receipt['channels'] == [str(index) for index in range(1, 11)]


def test_raw_volume_named_channels_keep_their_existing_order(tmp_path):
    """Numeric sorting leaves custom alphabetic channel tokens unchanged."""
    for channel, value in (('beta', 20), ('alpha', 10)):
        image = np.full((2, 8, 9), value, np.uint16)
        tifffile.imwrite(tmp_path / f'plate1_A01_F001C{channel}.tif', image,
                         metadata={'axes': 'ZYX'}, photometric='minisblack')
    pattern = (r'(?P<plateID>[^_]+)_(?P<wellID>A01)_F(?P<fieldID>\d+)'
               r'C(?P<chanID>[a-z]+)')
    settings = _raw_settings(tmp_path, metadata_type='custom',
                             custom_regex=pattern, channels=[0, 1])
    io.preprocess_img_data(settings)
    receipt = json.loads((tmp_path / 'stack/.spacr_volume_ingest.json').read_text())
    assert receipt['channels'] == ['alpha', 'beta']
    stack = np.load(next((tmp_path / 'stack').glob('*.npy')))
    np.testing.assert_array_equal(stack[..., 0], 10)
    np.testing.assert_array_equal(stack[..., 1], 20)


def test_native_z_t1_matches_raw_volume_batch_and_measure(tmp_path, fake_model,
                                                          monkeypatch):
    """Only the model forward pass is doubled; ingest and Measure run real."""
    import spacr.object as spacr_object
    from spacr.measure import measure_crop

    model_class = spacr_object.cp_models.CellposeModel

    def volume_eval(self, x, batch_size=8, resample=True, channels=None,
                    channel_axis=MISSING_CHANNEL_AXIS, z_axis=None,
                    normalize=True, invert=False, rescale=None, diameter=None,
                    flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False,
                    anisotropy=None, flow3D_smooth=0, stitch_threshold=0.0,
                    min_size=15, max_size_fraction=0.4, niter=None,
                    augment=False, tile_overlap=0.1, bsize=256,
                    compute_masks=True, progress=None):
        """Paint stable 3-D labels after checking Cellpose's real axis contract."""
        converted = check_cellpose_eval_call(
            x, channel_axis, z_axis=z_axis, do_3D=do_3D)
        assert do_3D and z_axis == 0 and channel_axis == -1
        assert len(converted) == 1 and converted[0].shape[:3] == (3, 32, 32)
        self.eval_kwargs.append({'channel_axis': channel_axis,
                                 'z_axis': z_axis, 'do_3D': do_3D})
        mask = np.zeros((3, 32, 32), np.uint16)
        mask[:, 5:17, 5:17] = 1
        mask[:, 18:29, 18:29] = 2
        return mask, None, None

    monkeypatch.setattr(model_class, 'eval', volume_eval)
    watched, batch, rows = _inputs(tmp_path)
    source_hashes = {row['target']: hashlib.sha256(
        (watched / row['target']).read_bytes()).hexdigest() for row in rows}
    mask = _mask(batch)
    core.preprocess_generate_masks(mask)
    measure = dict(MEASURE, anisotropy=2)
    measure_crop(dict(measure, src=str(batch / 'merged')))
    recipe = tmp_path / 'measure.json'
    recipe.write_text(json.dumps(measure))
    settings = dict(_mask(watched), **_fast(watched),
                    watch_pipeline='mask_measure',
                    watch_measure_settings=str(recipe))
    result = core._watch_folder_and_analyse(settings)
    assert len(result['done']) == 1 and not result['failed'] and not result['incomplete']
    field = watched / 'spacr_watch' / 'fields' / result['done'][0]
    assert {p.name for p in (field / '.watch_planar').glob('*.tif')} == {
        row['target'] for row in rows}
    assert {row['target']: hashlib.sha256(
        (field / '.watch_planar' / row['target']).read_bytes()).hexdigest()
        for row in rows} == source_hashes
    assert {row['target']: hashlib.sha256(
        (watched / row['target']).read_bytes()).hexdigest()
        for row in rows} == source_hashes
    for channel in (1, 2):
        name = convert.target_name('plate1', 'A01', 1, channel)
        with tifffile.TiffFile(field / name) as image:
            assert image.series[0].axes == 'ZYX'
            np.testing.assert_array_equal(image.series[0].asarray(),
                                          tifffile.imread(batch / name))
    batch_names = sorted(path.name for path in (batch / 'merged').glob('*.npy'))
    assert len(batch_names) == 1
    for name in batch_names:
        np.testing.assert_array_equal(np.load(batch / 'merged' / name),
                                      np.load(field / 'merged' / name))
    for table in ('cell', 'nucleus'):
        assert _rows(batch / 'measurements/measurements.db', table) == _rows(
            watched / 'spacr_watch/measurements/measurements.db', table)
    artifacts = json.loads((watched / 'spacr_watch/watch_ledger.json').read_text())[
        'fields'][result['done'][0]]['collection_checkpoint']['artifacts']
    assert len([name for name in artifacts if name.startswith('.watch_planar/')]) == 6
    for channel in (1, 2):
        name = convert.target_name('plate1', 'A01', 1, channel)
        assert artifacts[name] == hashlib.sha256((field / name).read_bytes()).hexdigest()


@pytest.mark.parametrize('damage', ['none', 'planar', 'derived', 'linked_planar'])
def test_native_z_collection_resume_binds_both_inputs(tmp_path, monkeypatch,
                                                       damage):
    """An interrupted collection reuses only unchanged planar and ZYX bytes."""
    watched, _batch, rows = _inputs(tmp_path)
    settings = dict(_mask(watched), **_fast(watched))
    recorder = Recorder()

    def fail_collection(*_args, **_kwargs):
        """Interrupt after analysis and fingerprinting, before collection."""
        raise OSError('injected collection failure')

    with monkeypatch.context() as patch:
        patch.setattr(core, '_watch_collect', fail_collection)
        first = core._watch_folder_and_analyse(settings, recorder)
    assert len(first['failed']) == len(recorder.calls) == 1
    field = watched / 'spacr_watch' / 'fields' / first['failed'][0]
    ledger = json.loads((watched / 'spacr_watch/watch_ledger.json').read_text())
    assert ledger['fields'][first['failed'][0]]['collection_checkpoint']['artifacts']
    if damage == 'linked_planar':
        planar = field / '.watch_planar'
        planar.rename(field / '.watch_planar_original')
        planar.symlink_to(field / '.watch_planar_original', target_is_directory=True)
    elif damage != 'none':
        path = (field / '.watch_planar' / rows[0]['target'] if damage == 'planar'
                else field / convert.target_name('plate1', 'A01', 1, 1))
        with path.open('ab') as handle:
            handle.write(b'tampered')
    resumed = core._watch_folder_and_analyse(settings, recorder)
    assert len(recorder.calls) == 1
    if damage == 'none':
        assert len(resumed['done']) == 1 and not resumed['failed']
    else:
        assert resumed['failed'] == first['failed'] and not resumed['done']
        saved = json.loads((watched / 'spacr_watch/watch_ledger.json').read_text())
        expected = ('Collection planar snapshots' if damage == 'linked_planar'
                    else 'Collection native Z inputs changed')
        assert expected in saved['fields'][first['failed'][0]]['error']


@pytest.mark.parametrize('damage', ['axes', 'nonfinite', 'shape', 'dtype'])
def test_native_z_stage_refuses_incompatible_real_planes(tmp_path, damage):
    """A stable but scientifically invalid plane never reaches batch ingest."""
    watched, _batch, rows = _inputs(tmp_path)
    path = watched / rows[1]['target']
    if damage == 'axes':
        tifffile.imwrite(path, np.ones((2, 32, 32), np.uint16),
                         metadata={'axes': 'ZYX'}, photometric='minisblack')
    elif damage == 'nonfinite':
        bad = np.ones((32, 32), np.float32)
        bad[0, 0] = np.nan
        tifffile.imwrite(path, bad, photometric='minisblack')
    else:
        shape = (16, 32) if damage == 'shape' else (32, 32)
        dtype = np.uint8 if damage == 'dtype' else np.uint16
        tifffile.imwrite(path, np.ones(shape, dtype), photometric='minisblack')
    settings = dict(_mask(watched), **_fast(watched))
    result = core._watch_folder_and_analyse(settings, Recorder())
    assert len(result['failed']) == 1 and not result['done']
    field = watched / 'spacr_watch' / 'fields' / result['failed'][0]
    assert not list(field.glob('*.tif'))
    assert not (watched / 'spacr_watch/merged').exists()


def test_native_z_plan_rejects_map_drift_and_other_time_or_field(tmp_path):
    """The private plan binds the same map bytes and exact T1 field set."""
    import csv

    watched, _batch, _rows = _inputs(tmp_path)
    settings = _mask(watched)
    manifest, digest = core._watch_map_manifest(str(watched), settings)
    with pytest.raises(ValueError, match='map fields changed'):
        core._watch_volume_plan(str(watched), settings, {}, digest)
    path = watched / convert.MAP_FILENAME
    with path.open(newline='') as handle:
        mapped = list(csv.DictReader(handle))
    mapped[0]['t'] = '2'
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=convert.MAP_COLUMNS)
        writer.writeheader()
        writer.writerows(mapped)
    with pytest.raises(ValueError, match='changed during volume preflight'):
        core._watch_volume_plan(str(watched), settings, manifest, digest)
    new_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='exactly T1'):
        core._watch_volume_plan(str(watched), settings, manifest, new_digest)


@pytest.mark.parametrize('damage', ['remove_planar', 'remove_derived',
                                    'change_planar', 'change_derived'])
def test_native_z_checkpoint_refuses_missing_or_changed_analysis_inputs(
        tmp_path, damage):
    """A good merged output cannot bless an incomplete or altered input set."""
    watched, _batch, rows = _inputs(tmp_path)
    settings = dict(_mask(watched), **_fast(watched))
    recorder = Recorder()

    def analyse(folder, recipe):
        """Run the field recorder, then change one private analysis input."""
        recorder(folder, recipe)
        folder = Path(folder)
        source = folder / '.watch_planar' / rows[0]['target']
        derived = folder / convert.target_name('plate1', 'A01', 1, 1)
        path = source if damage.endswith('planar') else derived
        if damage.startswith('remove'):
            path.unlink()
        else:
            with path.open('ab') as handle:
                handle.write(b'changed after analysis')

    result = core._watch_folder_and_analyse(settings, analyse)
    assert len(result['failed']) == len(recorder.calls) == 1
    assert not result['done']
    ledger = json.loads((watched / 'spacr_watch/watch_ledger.json').read_text())
    entry = ledger['fields'][result['failed'][0]]
    assert 'Collection native Z input' in entry['error']
    assert 'collection_checkpoint' not in entry
    assert not (watched / 'spacr_watch/merged').exists()
    recovered = core._watch_folder_and_analyse(settings, Recorder())
    assert len(recovered['done']) == 1 and not recovered['failed']


@pytest.mark.parametrize('output', ['none', 'removed_merged'])
def test_native_z_inputs_alone_cannot_complete_an_analysis(tmp_path, output):
    """Input fingerprints cannot substitute for a merged or measured result."""
    watched, _batch, _rows = _inputs(tmp_path)
    settings = dict(_mask(watched), **_fast(watched))

    def analyse(folder, recipe):
        """Leave either no result or remove the one result after writing it."""
        if output == 'removed_merged':
            Recorder()(folder, recipe)
            shutil.rmtree(Path(folder) / 'merged')

    result = core._watch_folder_and_analyse(settings, analyse)
    assert len(result['failed']) == 1 and not result['done']
    ledger = json.loads((watched / 'spacr_watch/watch_ledger.json').read_text())
    entry = ledger['fields'][result['failed'][0]]
    assert 'no completed analysis artifacts' in entry['error']
    assert 'collection_checkpoint' not in entry
    assert not (watched / 'spacr_watch/merged').exists()


def test_native_z_nested_map_snapshots_keep_relative_source_identity(tmp_path):
    """Nested acquired files keep ledger paths while staged volume names stay flat."""
    watched, _batch, rows = _inputs(tmp_path)
    nested = watched / 'incoming'
    nested.mkdir()
    for row in rows:
        shutil.move(watched / row['target'], nested / row['target'])
    result = core._watch_folder_and_analyse(
        dict(_mask(watched), **_fast(watched)), Recorder())
    assert len(result['done']) == 1 and not result['failed']
    ledger = json.loads((watched / 'spacr_watch/watch_ledger.json').read_text())
    snapshots = ledger['fields'][result['done'][0]]['snapshot_sha256']
    assert set(snapshots) == {f'incoming/{row["target"]}' for row in rows}


@pytest.mark.parametrize('stop_after', [1, 2])
def test_stop_during_volume_staging_rebuilds_fresh_inputs(
        tmp_path, monkeypatch, stop_after):
    """A cancelled partial ZYX stage is discarded before the next attempt."""
    import spacr.tiff_io as tiff_io

    watched, _batch, rows = _inputs(tmp_path)
    settings = dict(_mask(watched), **_fast(watched))
    token = CancellationToken()
    writer = tiff_io.write_tiff
    writes = []

    def cancel_after_write(path, array, **kwargs):
        """Press Stop after the first or final complete derived TIFF write."""
        writer(path, array, **kwargs)
        writes.append(path)
        if len(writes) == stop_after:
            token.cancel('stop during native Z staging')

    with monkeypatch.context() as patch:
        patch.setattr(tiff_io, 'write_tiff', cancel_after_write)
        with installed_token(token), pytest.raises(PipelineCancelled):
            core._watch_folder_and_analyse(settings, Recorder())
    key = 'plate1_A01_0001_001'
    field = watched / 'spacr_watch' / 'fields' / key
    assert len(list(field.glob('*.tif'))) == stop_after
    first = json.loads((watched / 'spacr_watch/watch_ledger.json').read_text())
    assert first['fields'][key]['status'] == 'interrupted'
    recorder = Recorder()
    resumed = core._watch_folder_and_analyse(settings, recorder)
    assert resumed['done'] == [key] and len(recorder.calls) == 1
    assert len(list(field.glob('*.tif'))) == 2
    assert {p.name for p in (field / '.watch_planar').glob('*.tif')} == {
        row['target'] for row in rows}


@pytest.mark.parametrize('change', ['missing', 'sparse', 'one_plane', 'no_map',
                                    'timelapse', 't_stack', 'classify',
                                    'feedback', 'spacing', 'measure_spacing',
                                    'measure_crops', 'measure_default_crops',
                                    'measure_time',
                                    'illumination', 'metadata'])
def test_native_z_preflight_refuses_unsafe_layouts(tmp_path, change):
    """No watcher workspace is written for unsupported volume recipes."""
    watched, _batch, rows = _inputs(tmp_path)
    settings = dict(_mask(watched), **_fast(watched))
    if change == 'missing':
        (watched / rows[0]['target']).unlink()
        result = core._watch_folder_and_analyse(settings, lambda *_: None)
        assert result['incomplete'] and not result['done']
        return
    if change in ('sparse', 'one_plane'):
        import csv

        path = watched / convert.MAP_FILENAME
        with path.open(newline='') as handle:
            mapped = list(csv.DictReader(handle))
        mapped = [row for row in mapped if (
            row['z'] == '1' if change == 'one_plane' else not (
                row['channel'] == '1' and row['z'] == '2'))]
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=convert.MAP_COLUMNS)
            writer.writeheader()
            writer.writerows(mapped)
    elif change == 'no_map':
        (watched / convert.MAP_FILENAME).unlink()
    elif change == 'timelapse':
        settings['timelapse'] = True
    elif change == 't_stack':
        settings['t_stack'] = True
    elif change == 'classify':
        settings['watch_pipeline'] = 'mask_measure_classify'
    elif change == 'feedback':
        settings.update(watch_pipeline='mask_measure', microscope_feedback=True)
    elif change in ('measure_spacing', 'measure_crops',
                    'measure_default_crops', 'measure_time'):
        measure = dict(MEASURE)
        if change == 'measure_crops':
            measure.update(anisotropy=2, save_png=True)
        elif change == 'measure_default_crops':
            measure.update(anisotropy=2)
            measure.pop('save_png')
        elif change == 'measure_time':
            measure.update(anisotropy=2, timelapse=True)
        recipe = tmp_path / 'measure.json'
        recipe.write_text(json.dumps(measure))
        settings.update(watch_pipeline='mask_measure',
                        watch_measure_settings=str(recipe))
    elif change == 'illumination':
        settings['illumination_correction'] = True
    elif change == 'metadata':
        settings['metadata_type'] = 'auto'
    else:
        settings['anisotropy'] = None
    with pytest.raises((ValueError, ZStackError)):
        core._watch_folder_and_analyse(settings, lambda *_: None)
    assert not (watched / 'spacr_watch').exists()
