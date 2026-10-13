"""The live fixed-map native time series uses the same original planes as batch Mask."""

import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from spacr import convert, core
from spacr.cancellation import PipelineCancelled
from spacr.zstack import UnknownAnisotropyError
from tests.conftest import MISSING_CHANNEL_AXIS, check_cellpose_eval_call
from tests.test_native_tzyx_batch_f548 import _settings
from tests.test_watch_nested_f548 import _fast
from tests.test_watch_timelapse_series_f548 import _converted_series

pytest_plugins = ['tests.test_cov_object_masks_sam']


def _watch(folder, **changes):
    settings = {**_settings(folder, cell_channel=None), **_fast(folder),
                'watch_pipeline': 'mask'}
    settings.update(changes)
    return settings


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model(monkeypatch):
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
        converted = check_cellpose_eval_call(
            x, channel_axis, z_axis=z_axis, do_3D=do_3D)
        assert do_3D and z_axis == 0 and channel_axis == -1
        seen.append(converted[0].copy())
        labels = np.zeros(converted[0].shape[:-1], np.uint16)
        labels[:, 5:17, 5:17] = 1
        return labels, None, None

    monkeypatch.setattr(spacr_object.cp_models.CellposeModel, 'eval', volume_eval)
    return seen


def test_native_series_waits_for_declared_planes_and_matches_batch(tmp_path,
                                                                  fake_model, monkeypatch):
    source, rows = _converted_series(tmp_path)
    batch, watched = tmp_path / 'batch', tmp_path / 'watched'
    batch.mkdir()
    watched.mkdir()
    shutil.copy2(source / convert.MAP_FILENAME, batch / convert.MAP_FILENAME)
    shutil.copy2(source / convert.MAP_FILENAME, watched / convert.MAP_FILENAME)
    original = {}
    for row in rows:
        name = row['target']
        original[name] = _digest(source / name)
        shutil.copy2(source / name, batch / name)
        if int(row['t']) == 1:
            shutil.copy2(source / name, watched / name)
    seen = _model(monkeypatch)
    core.preprocess_generate_masks(_settings(batch, cell_channel=None))
    batch_calls = len(seen)
    assert batch_calls == 2
    first = core._watch_folder_and_analyse(_watch(watched))
    assert not first['done'] and len(first['incomplete']) == 1
    assert len(seen) == batch_calls
    assert not (watched / 'spacr_watch/fields').exists()
    for row in rows:
        if int(row['t']) == 2:
            shutil.copy2(source / row['target'], watched / row['target'])
    result = core._watch_folder_and_analyse(_watch(watched))
    assert len(result['done']) == 1 and not result['failed']
    assert len(seen) == batch_calls + 2
    field = watched / 'spacr_watch/fields' / result['done'][0]
    assert {path.name for path in field.glob('*.tif')} == set(original)
    assert {_name: _digest(field / _name) for _name in original} == original
    assert {_name: _digest(watched / _name) for _name in original} == original
    with (field / convert.MAP_FILENAME).open(newline='', encoding='utf-8') as handle:
        scoped = list(csv.DictReader(handle))
    assert {row['target'] for row in scoped} == set(original)
    receipt = json.loads((field / 'stack/.spacr_volume_series_ingest.json').read_text())
    assert receipt['axes'] == 'TZYXC'
    assert receipt['map_sha256'] == _digest(field / convert.MAP_FILENAME)
    assert set(receipt['inputs']) == set(original)
    names = sorted(path.name for path in (batch / 'merged').glob('*.npy'))
    assert names == sorted(path.name for path in
                           (watched / 'spacr_watch/merged').glob('*.npy'))
    for name in names:
        np.testing.assert_array_equal(np.load(batch / 'merged' / name),
                                      np.load(watched / 'spacr_watch/merged' / name))
    resumed = core._watch_folder_and_analyse(_watch(watched))
    assert resumed['done'] == result['done'] and len(seen) == batch_calls + 2


@pytest.mark.parametrize('change, match', [
    ({'watch_pipeline': 'mask_measure'}, 'Mask only'),
    ({'watch_normalization_pool': 'fixed_map', 'batch_size': 2}, 'fixed_map'),
    ({'timelapse': True}, 't_stack'),
    ({'t_axis_order': 'ZTYX'}, 'TZYX'),
    ({'frame_interval_s': None}, 'frame interval'),
    ({'anisotropy': None}, 'anisotropy'),
    ({'microscope_feedback': True}, 'feedback'),
    ({'plot': True}, 'plain Cellpose'),
    ({'nucleus_channel': 2}, 'object channels'),
])
def test_native_series_refuses_unsupported_mode_before_workspace(tmp_path,
                                                                  change, match):
    source, _rows = _converted_series(tmp_path)
    with pytest.raises((ValueError, UnknownAnisotropyError), match=match):
        core._watch_folder_and_analyse(_watch(source, **change))
    assert not (source / 'spacr_watch').exists()


def test_native_series_collection_stop_resumes_without_repeating_inference(
        tmp_path, fake_model, monkeypatch):
    source, _rows = _converted_series(tmp_path)
    seen = _model(monkeypatch)
    collect = core._watch_collect

    def interrupted(*args, **kwargs):
        collect(*args, **kwargs)
        raise PipelineCancelled('Stop after per-time links')

    with monkeypatch.context() as patch:
        patch.setattr(core, '_watch_collect', interrupted)
        with pytest.raises(PipelineCancelled):
            core._watch_folder_and_analyse(_watch(source))
    assert len(seen) == 2
    result = core._watch_folder_and_analyse(_watch(source))
    assert len(result['done']) == 1 and len(seen) == 2
    assert {path.name for path in (source / 'spacr_watch/merged').glob('*.npy')} == {
        'plate1_A01_1_1.npy', 'plate1_A01_1_2.npy'}


@pytest.mark.parametrize('kind', ['orphan', 'linked_reserved', 'hidden_output'])
def test_native_series_resume_excludes_only_regular_reserved_mask_workspace(
        tmp_path, fake_model, monkeypatch, kind):
    source, _rows = _converted_series(tmp_path)
    seen = _model(monkeypatch)
    collect = core._watch_collect

    def interrupted(*args, **kwargs):
        collect(*args, **kwargs)
        raise PipelineCancelled('Stop after per-time links')

    with monkeypatch.context() as patch:
        patch.setattr(core, '_watch_collect', interrupted)
        with pytest.raises(PipelineCancelled):
            core._watch_folder_and_analyse(_watch(source))
    ledger_path = source / 'spacr_watch/watch_ledger.json'
    ledger = json.loads(ledger_path.read_text())
    key = next(iter(ledger['fields']))
    original = ledger['fields'][key]['collection_checkpoint']['artifacts']
    field = source / 'spacr_watch/fields' / key
    output = field / 'masks/nucleus_mask_stack'
    assert output.is_dir() and original and len(seen) == 2
    name = '.scientific-output' if kind == 'hidden_output' else '.spacr-native-mask-orphan'
    scratch = output / name
    if kind == 'linked_reserved':
        scratch.symlink_to(field / 'stack', target_is_directory=True)
    else:
        scratch.mkdir(mode=0o700)
        np.save(scratch / 'data.npy', np.arange(24, dtype=np.float32))
    result = core._watch_folder_and_analyse(_watch(source))
    resumed = json.loads(ledger_path.read_text())['fields'][key]
    assert len(seen) == 2
    assert resumed['collection_checkpoint']['artifacts'] == original
    assert all(_digest(field / relative) == digest for relative, digest in original.items())
    if kind == 'orphan':
        assert result['done'] == [key] and not result['failed']
        assert scratch.is_dir() and (scratch / 'data.npy').is_file()
        assert not any('.spacr-native-mask-' in name for name in original)
        assert {path.name for path in (source / 'spacr_watch/merged').glob('*.npy')} == {
            'plate1_A01_1_1.npy', 'plate1_A01_1_2.npy'}
    else:
        assert result['failed'] == [key] and not result['done']
        expected = 'linked directory' if kind == 'linked_reserved' else 'artifacts changed'
        assert expected in resumed['error']


def test_two_native_series_scope_convert_rows_and_time_outputs_by_field(
        tmp_path, fake_model, monkeypatch):
    source, rows = _converted_series(tmp_path, wells=('A01', 'A02'))
    seen = _model(monkeypatch)
    result = core._watch_folder_and_analyse(_watch(source))
    assert len(result['done']) == 2 and len(seen) == 4
    for key in result['done']:
        well = 'A01' if '_A01_' in key else 'A02'
        field = source / 'spacr_watch/fields' / key
        with (field / convert.MAP_FILENAME).open(newline='', encoding='utf-8') as handle:
            scoped = list(csv.DictReader(handle))
        assert {row['target'] for row in scoped} == {
            row['target'] for row in rows if row['well'] == well}
        assert {path.name for path in (field / 'merged').glob('*.npy')} == {
            f'plate1_{well}_1_1.npy', f'plate1_{well}_1_2.npy'}
    assert {path.name for path in (source / 'spacr_watch/merged').glob('*.npy')} == {
        f'plate1_{well}_1_{time}.npy' for well in ('A01', 'A02')
        for time in (1, 2)}


@pytest.mark.parametrize('damage', [
    'map', 'plane', 'receipt', 'receipt_digest', 'receipt_oversize',
    'missing_frame', 'extra_plane', 'missing_stack', 'missing_masks',
    'missing_stack_frame', 'linked_mask_folder',
])
def test_native_series_checkpoint_rejects_changed_staging_without_reanalysis(
        tmp_path, fake_model, monkeypatch, damage):
    source, rows = _converted_series(tmp_path)
    seen = _model(monkeypatch)
    collect = core._watch_collect

    def interrupted(*args, **kwargs):
        collect(*args, **kwargs)
        raise PipelineCancelled('Stop after per-time links')

    with monkeypatch.context() as patch:
        patch.setattr(core, '_watch_collect', interrupted)
        with pytest.raises(PipelineCancelled):
            core._watch_folder_and_analyse(_watch(source))
    field = next((source / 'spacr_watch/fields').iterdir())
    if damage == 'map':
        (field / convert.MAP_FILENAME).write_text('changed map')
    elif damage == 'plane':
        name = rows[0]['target']
        (field / name).write_bytes(b'changed plane')
    elif damage == 'receipt':
        (field / 'stack/.spacr_volume_series_ingest.json').write_text('{}')
    elif damage == 'receipt_digest':
        path = field / 'stack/.spacr_volume_series_ingest.json'
        receipt = json.loads(path.read_text())
        receipt['stacks']['plate1_A01_1_1.npy'] = '0' * 64
        path.write_text(json.dumps(receipt))
    elif damage == 'receipt_oversize':
        (field / 'stack/.spacr_volume_series_ingest.json').write_bytes(
            b'x' * (16 * 1024 * 1024 + 1))
    elif damage == 'missing_frame':
        (field / 'merged/plate1_A01_1_2.npy').unlink()
    elif damage == 'extra_plane':
        (field / 'unexpected.tif').write_bytes((field / rows[0]['target']).read_bytes())
    elif damage == 'missing_stack':
        shutil.rmtree(field / 'stack')
    elif damage == 'missing_masks':
        shutil.rmtree(field / 'masks')
    elif damage == 'missing_stack_frame':
        (field / 'stack/plate1_A01_1_2.npy').unlink()
    else:
        shutil.rmtree(field / 'masks/nucleus_mask_stack')
        (field / 'masks/nucleus_mask_stack').symlink_to(field / 'stack',
                                                       target_is_directory=True)
    result = core._watch_folder_and_analyse(_watch(source))
    assert len(result['failed']) == 1 and len(seen) == 2
    ledger = json.loads((source / 'spacr_watch/watch_ledger.json').read_text())
    assert 'Collection' in ledger['fields'][result['failed'][0]]['error']


def test_native_series_refuses_missing_or_one_frame_map_before_workspace(tmp_path):
    source, rows = _converted_series(tmp_path)
    map_path = source / convert.MAP_FILENAME
    original = map_path.read_bytes()
    map_path.unlink()
    with pytest.raises(ValueError, match='fixed Convert map'):
        core._watch_folder_and_analyse(_watch(source))
    map_path.write_bytes(original)
    with map_path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        headings, kept = reader.fieldnames, [row for row in reader if row['t'] == '1']
    with map_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=headings)
        writer.writeheader()
        writer.writerows(kept)
    for row in rows:
        if row['t'] != '1':
            (source / row['target']).unlink()
    with pytest.raises(ValueError, match='two mapped timepoints'):
        core._watch_folder_and_analyse(_watch(source))
    assert not (source / 'spacr_watch').exists()


def test_native_series_refuses_map_change_between_manifest_and_scope(tmp_path,
                                                                     monkeypatch):
    source, _rows = _converted_series(tmp_path)
    original = core._watch_map_bytes
    calls = 0

    def changed(folder):
        nonlocal calls
        calls += 1
        data = original(folder)
        return data if calls == 1 else data + b'\n'

    monkeypatch.setattr(core, '_watch_map_bytes', changed)
    with pytest.raises(ValueError, match='changed during native series preflight'):
        core._watch_folder_and_analyse(_watch(source))
    assert calls == 2 and not (source / 'spacr_watch').exists()


def test_native_series_restart_discards_partial_private_ingest_before_checkpoint(
        tmp_path, fake_model, monkeypatch):
    source, rows = _converted_series(tmp_path)
    original = {row['target']: _digest(source / row['target']) for row in rows}
    seen = _model(monkeypatch)

    def interrupted(field_dir, settings):
        field = Path(field_dir)
        assert {row['target'] for row in rows} == {
            path.name for path in field.glob('*.tif')}
        (field / 'stack').mkdir()
        (field / 'masks').mkdir()
        (field / 'stack/.spacr_volume_series_ingest.json').write_text(
            '{"version": 1, "interrupted": true}')
        (field / 'masks/partial.npz').write_bytes(b'partial publication')
        raise KeyboardInterrupt('interrupted before collection checkpoint')

    with pytest.raises(KeyboardInterrupt, match='before collection checkpoint'):
        core._watch_folder_and_analyse(_watch(source), interrupted)
    field = next((source / 'spacr_watch/fields').iterdir())
    ledger = json.loads((source / 'spacr_watch/watch_ledger.json').read_text())
    assert ledger['fields'][field.name]['status'] == 'running'
    assert 'collection_checkpoint' not in ledger['fields'][field.name]
    assert (field / 'masks/partial.npz').exists()
    assert not seen

    result = core._watch_folder_and_analyse(_watch(source))
    assert result['done'] == [field.name] and len(seen) == 2
    assert not (field / 'masks/partial.npz').exists()
    assert {_name: _digest(field / _name) for _name in original} == original
    assert {path.name for path in (source / 'spacr_watch/merged').glob('*.npy')} == {
        'plate1_A01_1_1.npy', 'plate1_A01_1_2.npy'}
