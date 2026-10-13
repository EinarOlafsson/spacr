"""Recursive acquisition discovery keeps companions and scientific fields separate."""
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from spacr import core
from tests.test_watch_folder_and_analyse import (
    Recorder, _name, _settings, _ledger, _channels, _rows, real_pipeline,
    MASK, MEASURE,
)


def _write(folder, name, value=5):
    """Write a real readable acquisition image in an arbitrary nested folder."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    tifffile.imwrite(path, np.full((16, 16), value, np.uint16))
    return path


def _fast(folder):
    """Use short settle/idle windows without replacing the watcher clock."""
    return _settings(folder, watch_settle_seconds=0.01,
                     watch_poll_seconds=0.01, watch_idle_minutes=0.001)


def test_scan_preserves_top_level_names_and_prunes_outputs_hidden_and_links(tmp_path):
    root = _write(tmp_path, _name('A01', 1))
    nested = _write(tmp_path / 'plate' / 'well', _name('A02', 1))
    for folder in ['spacr_watch', 'orig', 'stack', 'masks', 'merged',
                   'measurements', 'results', 'test', 'nucleus_mask_stack', '.hidden']:
        _write(tmp_path / 'plate' / folder, _name('B01', 1))
    _write(tmp_path, '.' + _name('B02', 1))
    (tmp_path / 'alias.tif').symlink_to(root)
    (tmp_path / 'linked-folder').symlink_to(nested.parent, target_is_directory=True)
    (nested.parent / 'loop').symlink_to(tmp_path, target_is_directory=True)
    assert core._watch_images(str(tmp_path)) == sorted([
        root.name, str(nested.relative_to(tmp_path))])


def test_nested_complete_fields_keep_flat_metadata_names_and_resume(tmp_path):
    originals = {}
    for folder, well in [(tmp_path, 'A01'), (tmp_path / 'plate' / 'A02', 'A02')]:
        for channel in [1, 2]:
            path = _write(folder, _name(well, channel), channel)
            originals[path] = path.read_bytes()
    recorder = Recorder()
    result = core._watch_folder_and_analyse(_fast(tmp_path), recorder)
    assert result['done'] == ['plate1_A01_0001_001', 'plate1_A02_0001_001']
    assert not result['failed'] and not result['incomplete']
    assert recorder.calls[1][1] == [_name('A02', 1), _name('A02', 2)]
    nested = _ledger(tmp_path)['plate1_A02_0001_001']
    assert set(nested['files']) == {str(Path('plate/A02') / _name('A02', n)) for n in [1, 2]}
    resumed = Recorder()
    assert core._watch_folder_and_analyse(_fast(tmp_path), resumed)['done'] == result['done']
    assert not resumed.calls
    assert all(path.read_bytes() == data for path, data in originals.items())


def test_cross_folder_channels_never_complete_a_field(tmp_path, capsys):
    _write(tmp_path / 'first', _name('A01', 1))
    _write(tmp_path / 'second', _name('A01', 2))
    recorder = Recorder()
    result = core._watch_folder_and_analyse(_fast(tmp_path), recorder)
    assert not recorder.calls and not result['done']
    assert result['incomplete'] == ['plate1_A01_0001_001']
    assert 'different acquisition folders or duplicate channels' in capsys.readouterr().out


def test_duplicate_field_id_in_two_complete_folders_is_not_silently_merged(tmp_path):
    for folder in [tmp_path, tmp_path / 'other']:
        for channel in [1, 2]:
            _write(folder, _name('A01', channel))
    recorder = Recorder()
    result = core._watch_folder_and_analyse(_fast(tmp_path), recorder)
    assert not recorder.calls and not result['done']
    assert result['incomplete'] == ['plate1_A01_0001_001']
    assert not (tmp_path / 'spacr_watch/merged').exists()


def test_duplicate_numeric_channel_spellings_do_not_count_as_companions(tmp_path):
    _write(tmp_path / 'well', _name('A01', 1))
    _write(tmp_path / 'well', _name('A01', 1).replace('C01', 'C1'))
    recorder = Recorder()
    result = core._watch_folder_and_analyse(_fast(tmp_path), recorder)
    assert not recorder.calls and not result['done']
    assert result['incomplete'] == ['plate1_A01_0001_001']


@pytest.mark.parametrize('nested', [False, True])
@pytest.mark.parametrize('missing', ['absent', 'truncated'])
def test_nested_incomplete_companion_waits(tmp_path, missing, nested):
    folder = tmp_path / 'A01' if nested else tmp_path
    _write(folder, _name('A01', 1))
    if missing == 'truncated':
        (folder / _name('A01', 2)).write_bytes(b'II*\0partial TIFF')
    recorder = Recorder()
    result = core._watch_folder_and_analyse(_fast(tmp_path), recorder)
    assert not recorder.calls and not result['done']
    assert result['incomplete'] == ['plate1_A01_0001_001']


def test_nested_companion_arrival_allows_the_waiting_field(tmp_path):
    folder = tmp_path / 'A01'
    _write(folder, _name('A01', 1))
    first = core._watch_folder_and_analyse(_fast(tmp_path), Recorder())
    assert first['incomplete']
    _write(folder, _name('A01', 2))
    recorder = Recorder()
    second = core._watch_folder_and_analyse(_fast(tmp_path), recorder)
    assert second['done'] == ['plate1_A01_0001_001']
    assert len(recorder.calls) == 1


def test_nested_actual_preprocessing_and_measure_matches_flat_batch(tmp_path, real_pipeline):
    from spacr.measure import measure_crop
    batch, watched = tmp_path / 'batch', tmp_path / 'watched'
    batch.mkdir()
    watched.mkdir()
    raw = {}
    for channel, image in enumerate(_channels(0), start=1):
        name = _name('A01', channel)
        tifffile.imwrite(batch / name, image)
        folder = watched / 'plate1/A01'
        folder.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(folder / name, image)
        raw[folder / name] = (folder / name).read_bytes()
    core.preprocess_generate_masks(dict(MASK, src=str(batch)))
    measure_crop(dict(MEASURE, src=str(batch / 'merged')))
    config = tmp_path / 'measure.json'
    config.write_text(json.dumps(MEASURE))
    result = core.preprocess_generate_masks(dict(
        MASK, **_fast(watched), watch_pipeline='mask_measure',
        watch_measure_settings=str(config)))
    assert len(result['done']) == 1 and not result['failed']
    for path in (batch / 'merged').glob('*.npy'):
        np.testing.assert_array_equal(np.load(path), np.load(watched / 'spacr_watch/merged' / path.name))
    for table in ['cell', 'nucleus']:
        assert _rows(batch / 'measurements/measurements.db', table) == _rows(
            watched / 'spacr_watch/measurements/measurements.db', table)
    assert all(path.read_bytes() == data for path, data in raw.items())
