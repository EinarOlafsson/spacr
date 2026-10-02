"""Map-backed watches refuse channel schemas that raw ingest would compact."""
from pathlib import Path
import shutil

import numpy as np
import pytest

from spacr import core, convert, io
from spacr.utils import _get_regex
from tests.test_watch_folder_and_analyse import Recorder, _name
from tests.test_watch_nested_f548 import _fast, _write
from tests.test_watch_conversion_manifest_f548 import _map, _rows


@pytest.mark.parametrize('indices', [[0, 4], [-1, 0], [0, 0], [False], [0.0], [], 'not-a-list', '[0, 4]', None])
def test_invalid_selected_positions_fail_before_watch_outputs(tmp_path, indices):
    _map(tmp_path, (1, 2, 3, 4))
    with pytest.raises(ValueError, match='distinct zero-based positions'):
        core._watch_folder_and_analyse(dict(_fast(tmp_path), channels=indices), Recorder())
    assert not (tmp_path / 'spacr_watch').exists()


@pytest.mark.parametrize('indices', [[0, 3], '[0, 3]', (3, 0)])
def test_known_convert_schema_accepts_valid_nonconsecutive_selections(tmp_path, indices):
    _map(tmp_path, (1, 2, 3, 4))
    for channel in (1, 2, 3, 4):
        _write(tmp_path, _name('A01', channel), channel)
    recorder = Recorder()
    result = core._watch_folder_and_analyse(dict(_fast(tmp_path), channels=indices), recorder)
    assert len(result['done']) == 1 and len(recorder.calls) == 1


@pytest.mark.parametrize('kind', ['uniform_sparse', 'mixed_dense', 'mixed_plates'])
def test_map_schema_that_would_shift_channel_positions_is_rejected(tmp_path, kind):
    if kind == 'uniform_sparse':
        _map(tmp_path, (1, 4))
    else:
        rows = _map(tmp_path, (1, 2), ('A01', 'A02'))
        rows = [row for row in rows if not (row['well'] == 'A02' and row['channel'] == 2)]
        if kind == 'mixed_plates':
            for row in rows:
                if row['well'] == 'A02':
                    row['plate'] = 'plate2'
                    row['target'] = convert.target_name('plate2', 'A02', 1, row['channel'])
        _rows(tmp_path, rows)
    with pytest.raises(ValueError, match='sparse or mixed fields would shift channel positions'):
        core._watch_folder_and_analyse(_fast(tmp_path), Recorder())
    assert not (tmp_path / 'spacr_watch').exists()


def test_real_convert_sparse_field_map_refused_without_source_changes(tmp_path):
    raw, output = tmp_path / 'raw', tmp_path / 'converted'
    for well, channels in [('A01', (1, 2, 3, 4)), ('A02', (1, 4))]:
        for channel in channels:
            _write(raw / well, f'field01_C{channel}.tif', channel)
    converted = convert.convert_folder({'src': str(raw), 'dst': str(output), 'preview_rows': 0})
    assert converted.is_complete
    before = {path: path.read_bytes() for folder in (raw, output) for path in folder.rglob('*') if path.is_file()}
    recorder = Recorder()
    with pytest.raises(ValueError, match='complete C01..CN channel schema'):
        core._watch_folder_and_analyse(dict(_fast(output), channels=[0, 3]), recorder)
    assert not recorder.calls and not (output / 'spacr_watch').exists()
    assert all(path.read_bytes() == content for path, content in before.items())


def test_valid_converted_fields_preserve_actual_batch_stack_order(tmp_path):
    raw, output, batch = tmp_path / 'raw', tmp_path / 'converted', tmp_path / 'batch'
    for well in ('A01', 'A02'):
        for channel in (1, 2, 3, 4):
            _write(raw / well, f'field01_C{channel}.tif', channel * 7)
    result = convert.convert_folder({'src': str(raw), 'dst': str(output), 'preview_rows': 0})
    assert result.is_complete
    batch.mkdir()
    for path in output.glob('*.tif'):
        shutil.copy2(path, batch / path.name)
    pattern = _get_regex('cellvoyager', 'tif', None)
    observed = []

    def preprocess(folder, settings):
        """Run actual static raw stacking and retain the original positional selection."""
        assert settings['channels'] == [0, 3]
        count = io._rename_and_organize_image_files(
            folder, pattern, 1, 'cellvoyager', ['.tif'],
            timelapse=False, save_original_images=True)
        assert count == 4
        observed.extend(Path(folder, 'stack').glob('*.npy'))

    settings = dict(_fast(output), channels=[0, 3])
    watched = core._watch_folder_and_analyse(settings, preprocess)
    io._rename_and_organize_image_files(str(batch), pattern, 1, 'cellvoyager', ['.tif'],
                                       timelapse=False, save_original_images=True)
    assert len(watched['done']) == len(observed) == 2
    for path in observed:
        data = np.load(path)
        np.testing.assert_array_equal(data, np.load(batch / 'stack' / path.name))
        assert data.shape[-1] == 4
        assert data[0, 0, [0, 3]].tolist() == [7, 28]
    assert settings['channels'] == [0, 3]
