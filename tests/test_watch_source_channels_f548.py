"""Raw watch fields wait until selected channel positions have known identities."""
import json
import shutil

import numpy as np
import pytest
import tifffile

from spacr import core, io
from spacr.utils import _get_regex
from tests.test_watch_folder_and_analyse import Recorder, _name
from tests.test_watch_nested_f548 import _fast


def _write(folder, name, value):
    """Write a real readable single-channel TIFF."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    tifffile.imwrite(path, np.full((16, 16), value, np.uint16))
    return path


def test_yokogawa_nonconsecutive_selection_waits_for_intermediate_channels(tmp_path):
    settings = dict(_fast(tmp_path), channels=[0, 3])
    recorder = Recorder()
    names = {_name('A01', channel): channel for channel in range(1, 5)}
    for channel in (1, 2):
        _write(tmp_path, _name('A01', channel), channel * 7)
    first = core._watch_folder_and_analyse(settings, recorder)
    assert not first['done'] and first['incomplete'] and not recorder.calls
    _write(tmp_path, _name('A01', 4), 28)
    second = core._watch_folder_and_analyse(settings, recorder)
    assert not second['done'] and second['incomplete'] and not recorder.calls
    _write(tmp_path, _name('A01', 3), 21)
    result = core._watch_folder_and_analyse(settings, recorder)
    assert len(result['done']) == len(recorder.calls) == 1
    assert recorder.calls[0][1] == sorted(names)
    ledger = json.loads((tmp_path / 'spacr_watch/watch_ledger.json').read_text())
    assert set(ledger['fields'][result['done'][0]]['files']) == set(names)
    assert core._watch_folder_and_analyse(settings, recorder)['done'] == result['done']
    assert len(recorder.calls) == 1


@pytest.mark.parametrize('metadata_type,origin', [
    ('arrayscan', 0), ('micromanager_mda', 0),
])
def test_zero_based_vendor_origin_is_explicit(metadata_type, origin):
    assert core._watch_source_channels(dict(
        metadata_type=metadata_type, channels=[0, 3])) == {
            str(value) for value in range(origin, origin + 4)}


def test_arrayscan_zero_based_channel_order_matches_batch_raw_stack(tmp_path):
    watched, batch = tmp_path / 'watched', tmp_path / 'batch'
    watched.mkdir()
    batch.mkdir()
    names = [f'A01f01d{channel}.tif' for channel in range(4)]
    for channel, name in enumerate(names):
        _write(batch, name, (channel + 1) * 9)
    recorder = Recorder()
    settings = dict(_fast(watched), metadata_type='arrayscan', channels=[0, 3])
    for channel in (0, 1, 3):
        shutil.copy2(batch / names[channel], watched / names[channel])
    assert core._watch_folder_and_analyse(settings, recorder)['incomplete']
    assert not recorder.calls
    shutil.copy2(batch / names[2], watched / names[2])
    result = core._watch_folder_and_analyse(settings, recorder)
    assert len(result['done']) == len(recorder.calls) == 1
    field = watched / 'spacr_watch/fields' / result['done'][0]
    pattern = _get_regex('arrayscan', 'tif', None)
    io._rename_and_organize_image_files(str(field), pattern, 1, 'arrayscan',
                                        ['.tif'], save_original_images=True)
    io._rename_and_organize_image_files(str(batch), pattern, 1, 'arrayscan',
                                        ['.tif'], save_original_images=True)
    watch_stack = next((field / 'stack').glob('*.npy'))
    batch_stack = next((batch / 'stack').glob('*.npy'))
    np.testing.assert_array_equal(np.load(watch_stack), np.load(batch_stack))
    assert np.load(watch_stack)[0, 0, [0, 3]].tolist() == [9, 36]


def test_one_based_yokogawa_refuses_c00_even_with_expected_companions(tmp_path):
    _write(tmp_path, _name('A01', 1).replace('C01', 'C00'), 1)
    _write(tmp_path, _name('A01', 1), 2)
    _write(tmp_path, _name('A01', 2), 3)
    recorder = Recorder()
    with pytest.raises(ValueError, match='below the documented origin'):
        core._watch_folder_and_analyse(_fast(tmp_path), recorder)
    assert not recorder.calls
    assert not (tmp_path / 'spacr_watch/merged').exists()


def test_a_late_lower_channel_stops_resume_without_replacing_results(tmp_path):
    for channel in (1, 2):
        _write(tmp_path, _name('A01', channel), channel)
    recorder = Recorder()
    done = core._watch_folder_and_analyse(_fast(tmp_path), recorder)
    result = tmp_path / 'spacr_watch/merged' / (done['done'][0] + '.npy')
    before = result.read_bytes()
    _write(tmp_path, _name('A01', 1).replace('C01', 'C00'), 9)
    with pytest.raises(ValueError, match='below the documented origin'):
        core._watch_folder_and_analyse(_fast(tmp_path), recorder)
    assert len(recorder.calls) == 1 and result.read_bytes() == before


def test_known_numeric_convention_refuses_a_named_channel(tmp_path):
    for channel in (1, 2):
        _write(tmp_path, _name('A01', channel), channel)
    _write(tmp_path, _name('A01', 1).replace('C01', 'CXX'), 3)
    recorder = Recorder()
    with pytest.raises(ValueError, match='nonnumeric channel'):
        core._watch_folder_and_analyse(_fast(tmp_path), recorder)
    assert not recorder.calls


def test_custom_numeric_origin_requires_conversion_map_before_output(tmp_path):
    for channel in (0, 1):
        _write(tmp_path, f'A01_f1_ch{channel}.tif', channel + 1)
    recorder = Recorder()
    settings = dict(_fast(tmp_path), metadata_type='custom',
                    custom_regex=r'(?P<wellID>A01)_f(?P<fieldID>1)_ch(?P<chanID>\d+)')
    with pytest.raises(ValueError, match='conversion_map.csv'):
        core._watch_folder_and_analyse(settings, recorder)
    assert not recorder.calls and not (tmp_path / 'spacr_watch').exists()


@pytest.mark.parametrize('channels', [[-1], [0, 0], [False], '[0, nope]', []])
def test_invalid_raw_selected_positions_fail_before_workspace(tmp_path, channels):
    with pytest.raises(ValueError, match='distinct non-negative zero-based'):
        core._watch_folder_and_analyse(dict(_fast(tmp_path), channels=channels),
                                       Recorder())
    assert not (tmp_path / 'spacr_watch').exists()
